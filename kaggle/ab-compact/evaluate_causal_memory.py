#!/usr/bin/env python3
"""CPU-only audit of the causal record and its installed dropped-message hook."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUDIT = ROOT / "tests/fixtures/arc-causal-audit-v1.json"
MAX_CHARS = 3000
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import compaction_replay as cr


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(text: str) -> str:
    return " ".join(str(text).lower().split())


def messages_by_level(path: Path) -> list[list[dict]]:
    segments: list[list[dict]] = []
    current: list[dict] = []
    level = None
    for turn in cr.parse(path):
        found = re.findall(r"Current state: step \d+, level (\d+)", turn.frame)
        next_level = int(found[-1]) if found else level
        if current and next_level is not None and level is not None and next_level != level:
            segments.append(current)
            current = []
        level = next_level
        current.append({"role": "user", "content": turn.frame})
        last_call = "x"
        for kind, payload in turn.events:
            if kind == "assistant":
                message = dict(payload)
                calls = message.get("tool_calls") or []
                last_call = str(calls[0].get("id", "x")) if len(calls) == 1 else "x"
                current.append(message)
            elif kind == "tool":
                current.append({"role": "tool", "tool_call_id": last_call, "content": payload})
            else:
                current.append({"role": "user", "content": payload})
    if current:
        segments.append(current)
    return segments


def retained(fact, output: str) -> bool:
    """Require the exact fact record; nearby outcomes cannot satisfy a different action."""
    return canonical(fact.text) in canonical(output)


def _turn_messages(source: Path, index: int) -> list[dict]:
    turn = cr.parse(source)[index]
    messages = [{"role": "user", "content": turn.frame}]
    last_call = "x"
    for kind, payload in turn.events:
        if kind == "assistant":
            message = dict(payload)
            calls = message.get("tool_calls") or []
            last_call = str(calls[0].get("id", "x")) if len(calls) == 1 else "x"
            messages.append(message)
        elif kind == "tool":
            messages.append({"role": "tool", "tool_call_id": last_call, "content": payload})
        else:
            messages.append({"role": "user", "content": payload})
    return messages


def _independent_audit(causal, snippet) -> dict:
    cases = json.loads(AUDIT.read_text())
    totals = {arm: {"expected": 0, "found": 0} for arm in ("snippet", "naive", "causal")}
    exclusions = {"expected": 0, "passed": 0}
    rows = []
    for case in cases:
        source = cr.RUNS / case["run"] / "transcripts" / case["file"]
        messages = _turn_messages(source, case["turn_index"])
        game_id = re.sub(r"_p\d+$", "", source.stem)
        facts = causal.extract_facts(messages, game_id=game_id)
        outputs = {
            "snippet": snippet.digest(messages)[-MAX_CHARS:],
            "naive": causal.naive_record(facts, MAX_CHARS),
            "causal": causal.bounded_record(facts, MAX_CHARS),
        }
        result = {"id": case["id"], "facts": len(facts), "arms": {}}
        for arm, output in outputs.items():
            found = all(all(canonical(part) in canonical(output) for part in expectation["contains_all"])
                        for expectation in case.get("expect", []))
            totals[arm]["expected"] += len(case.get("expect", []))
            totals[arm]["found"] += sum(
                all(canonical(part) in canonical(output) for part in expectation["contains_all"])
                for expectation in case.get("expect", []))
            result["arms"][arm] = found
        for exclusion in case.get("exclude", []):
            exclusions["expected"] += 1
            absent = not any(f.category == exclusion["category"] and
                             all(canonical(part) in canonical(f.text) for part in exclusion.get("contains_all", []))
                             for f in facts)
            exclusions["passed"] += int(absent)
        rows.append(result)
    for arm, cell in totals.items():
        cell["recall"] = cell["found"] / cell["expected"] if cell["expected"] else None
    return {"fixture": str(AUDIT), "cases": len(cases), "retention": totals,
            "candidate_exclusions": exclusions, "rows": rows}


def _live_replay(causal) -> dict:
    ta = cr.tool_agent()
    games = [game for run in cr.CONTROL_RUNS for game in cr.recorded_games(run)]
    expected = {re.sub(r"_p\d+$", "", game.stem) for game in games}
    expected_runs = {(str(game.parent.parent), re.sub(r"_p\d+$", "", game.stem)) for game in games}
    base = {game: cr.replay(game, ta) for game in games}
    with tempfile.TemporaryDirectory(prefix="arc-causal-events-") as temp:
        events = Path(temp) / "events.jsonl"
        causal.install(MAX_CHARS, ta, events)
        try:
            treatment = {game: cr.replay(game, ta, causal) for game in games}
        finally:
            causal.uninstall(ta)
        rows = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
    observed = {row.get("game_id", "") for row in rows if row.get("kind") == "digest"}
    observed_runs = {(row.get("runtime_dir", ""), row.get("game_id", "")) for row in rows
                     if row.get("kind") == "digest"}
    identity = observed == expected and observed_runs == expected_runs
    causal_lengths = [len(digest) for stats in treatment.values() for _, digest in stats.digests]
    base_history = [count for stats in base.values() for count in stats.history_msgs_per_request]
    treatment_history = [count for stats in treatment.values() for count in stats.history_msgs_per_request]
    request_tokens = [count for stats in treatment.values() for count in stats.request_tokens]
    return {
        "games": len(games),
        "expected_game_ids": len(expected),
        "observed_game_ids": len(observed),
        "observed_game_runs": len(observed_runs),
        "identity_gate": identity,
        "missing_game_ids": sorted(expected - observed),
        "unexpected_game_ids": sorted(observed - expected),
        "missing_game_runs": len(expected_runs - observed_runs),
        "unexpected_game_runs": len(observed_runs - expected_runs),
        "digest_events": sum(row.get("kind") == "digest" for row in rows),
        "max_record_chars": max(causal_lengths, default=0),
        "max_request_tokens": max(request_tokens, default=0),
        "mean_history_messages_control": statistics.fmean(base_history),
        "mean_history_messages_treatment": statistics.fmean(treatment_history),
        "mean_history_messages_displaced": statistics.fmean(base_history) - statistics.fmean(treatment_history),
        "dropped_messages_seen": sum(stats.patch_seen for stats in treatment.values()),
    }


def run() -> dict:
    causal = load("causal_memory_eval", HERE / "causal_memory.py")
    snippet = load("compaction_eval", HERE / "compaction.py")
    totals = {arm: {category: {"facts": 0, "retained": 0} for category in causal.CATEGORIES}
              for arm in ("snippet", "naive", "causal")}
    lengths = {arm: 0 for arm in totals}
    source_facts = 0
    segments = 0
    for run_name in cr.CONTROL_RUNS:
        games = cr.recorded_games(run_name)
        if len(games) != 25:
            raise RuntimeError(f"{run_name}: expected 25 public games, found {len(games)}")
        for game_path in games:
            game_id = re.sub(r"_p\d+$", "", game_path.stem)
            for messages in messages_by_level(game_path):
                facts = causal.extract_facts(messages, game_id=game_id)
                if not facts:
                    continue
                source_facts += len(facts)
                segments += 1
                outputs = {"snippet": snippet.digest(messages)[-MAX_CHARS:],
                           "naive": causal.naive_record(facts, MAX_CHARS),
                           "causal": causal.bounded_record(facts, MAX_CHARS)}
                for arm, output in outputs.items():
                    lengths[arm] = max(lengths[arm], len(output))
                for fact in facts:
                    for arm, output in outputs.items():
                        totals[arm][fact.category]["facts"] += 1
                        totals[arm][fact.category]["retained"] += int(retained(fact, output))
    rates = {}
    for arm, categories in totals.items():
        rates[arm] = {}
        for category, cell in categories.items():
            rates[arm][category] = {**cell, "retention": cell["retained"] / cell["facts"] if cell["facts"] else None}
    audit = _independent_audit(causal, snippet)
    live = _live_replay(causal)
    gates = {
        "identity_from_events": live["identity_gate"],
        "record_bounded": live["max_record_chars"] <= MAX_CHARS,
        "request_within_context_budget": live["max_request_tokens"] <= 31744,
        "audit_exclusions": audit["candidate_exclusions"]["passed"] == audit["candidate_exclusions"]["expected"],
        "hook_exercised": live["digest_events"] > 0 and live["dropped_messages_seen"] > 0,
    }
    return {
        "decision": "CPU_CORRECTED_NO_GPU_AUTHORIZED" if all(gates.values()) else "CPU_INVALID_NO_GPU_AUTHORIZED",
        "population": {"runs": list(cr.CONTROL_RUNS), "level_segments": segments, "candidate_extracted_facts": source_facts},
        "mechanical_retention_diagnostic_not_decision_evidence": rates,
        "max_output_chars": lengths,
        "independent_hand_label_audit": audit,
        "installed_hook_replay": live,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
