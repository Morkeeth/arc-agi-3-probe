#!/usr/bin/env python3
"""CPU-only comparison of causal records with the current rolling snippet digest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
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
    value = " ".join(str(text).lower().split())
    for field in ("board_changed", "reward", "level_completed", "game_over", "run_complete", "stop_reason", "error"):
        value = re.sub(
            rf"['\"]?{field}['\"]?\s*[:=]\s*['\"]?([^,'\"}}\s]+)",
            lambda match: f"{field}={match.group(1).strip().lower()}",
            value,
        )
    return value


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
        for kind, payload in turn.events:
            if kind == "assistant":
                current.append(dict(payload))
            elif kind == "tool":
                current.append({"role": "tool", "content": payload})
            else:
                current.append({"role": "user", "content": payload})
    if current:
        segments.append(current)
    return segments


def retained(fact, output: str) -> bool:
    haystack = canonical(output)
    if fact.category != "action_effect":
        return canonical(fact.evidence) in haystack
    match = re.match(r"action\((.*?)\) => (.*)", fact.text)
    if not match:
        return False
    action, outcome = canonical(match.group(1)), canonical(match.group(2))
    action_at = haystack.find(action)
    while action_at >= 0:
        outcome_at = haystack.find(outcome, action_at, action_at + 600)
        if outcome_at >= 0:
            return True
        action_at = haystack.find(action, action_at + 1)
    return False


def run() -> dict:
    causal = load("causal_memory_eval", HERE / "causal_memory.py")
    snippet = load("compaction_eval", HERE / "compaction.py")
    totals = {arm: {category: {"facts": 0, "retained": 0} for category in causal.CATEGORIES}
              for arm in ("snippet", "causal")}
    max_lengths = {"snippet": 0, "causal": 0}
    max_estimated_tokens = {"snippet": 0, "causal": 0}
    represented: dict[str, dict[str, set[str]]] = {
        arm: defaultdict(set) for arm in ("snippet", "causal")
    }
    available: dict[str, set[str]] = defaultdict(set)
    segments = 0
    source_facts = 0

    for run_name in cr.CONTROL_RUNS:
        games = cr.recorded_games(run_name)
        if len(games) != 25:
            raise RuntimeError(f"{run_name}: expected 25 public games, found {len(games)}")
        for game_path in games:
            game_key = f"{run_name}/{game_path.stem}"
            for messages in messages_by_level(game_path):
                facts = causal.extract_facts(messages)
                # Count a factual statement once per level, even if the agent repeated the same sentence.
                unique = {}
                for fact in facts:
                    unique[(fact.category, fact.evidence, fact.text)] = fact
                facts = list(unique.values())
                if not facts:
                    continue
                segments += 1
                source_facts += len(facts)
                outputs = {
                    "snippet": snippet.digest(messages)[-MAX_CHARS:],
                    "causal": causal.bounded_record(facts, MAX_CHARS),
                }
                for arm, output in outputs.items():
                    max_lengths[arm] = max(max_lengths[arm], len(output))
                    max_estimated_tokens[arm] = max(max_estimated_tokens[arm], (len(output) + 2) // 3)
                for fact in facts:
                    available[game_key].add(fact.category)
                    for arm, output in outputs.items():
                        cell = totals[arm][fact.category]
                        cell["facts"] += 1
                        if retained(fact, output):
                            cell["retained"] += 1
                            represented[arm][game_key].add(fact.category)

    rates = {}
    macro = {}
    for arm in ("snippet", "causal"):
        rates[arm] = {}
        active = []
        for category, cell in totals[arm].items():
            rate = cell["retained"] / cell["facts"] if cell["facts"] else None
            rates[arm][category] = {**cell, "retention": rate}
            if rate is not None:
                active.append(rate)
        macro[arm] = sum(active) / len(active)

    all_available = {
        arm: sum(1 for game, categories in available.items() if categories <= represented[arm][game])
        for arm in ("snippet", "causal")
    }
    non_regression = all(
        rates["causal"][category]["retention"] >= rates["snippet"][category]["retention"]
        for category in rates["causal"]
        if rates["causal"][category]["retention"] is not None
    )
    margin = macro["causal"] - macro["snippet"]
    identity_gate = True  # asserted separately by the concurrent integration test in the same gate command
    win = margin >= 0.10 and non_regression and max_lengths["causal"] <= MAX_CHARS and identity_gate
    return {
        "population": {"runs": list(cr.CONTROL_RUNS), "game_runs": len(available),
                       "distinct_game_ids": len({key.split("/", 1)[1].rsplit("_p0", 1)[0] for key in available}),
                       "level_segments": segments, "source_facts": source_facts},
        "budget": {"max_chars": MAX_CHARS, "max_estimated_tokens": 1000},
        "per_category": rates,
        "macro_retention": macro,
        "causal_margin": margin,
        "non_regression": non_regression,
        "max_output_chars": max_lengths,
        "max_output_estimated_tokens": max_estimated_tokens,
        "game_runs_with_all_available_categories_represented": all_available,
        "event_identity_gate": "covered by test_event_identity_distinguishes_concurrent_games_in_one_runtime_dir",
        "gpu_hypothesis_gate": "WIN" if win else "STOP",
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
