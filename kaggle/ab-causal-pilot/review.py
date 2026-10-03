"""Read a saved matched pair; never run models, rebuild notebooks, or alter trial evidence.

The frozen verdict remains authoritative. This adds a per-game reading surface and
links exact emitted memory to input and an observed conversation continuation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot_frozen_verdict", HERE / "verdict.py")
verdict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verdict)
MARKERS = {"control": "Compacted earlier turns", "treatment": "Causal record from earlier dropped turns"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def reference(root, path, line=None):
    result = {"file": str(path.relative_to(root))}
    if line is not None:
        result["line"] = line
    else:
        result["sha256"] = sha(path)
    return result


def rows(path):
    for line, text in enumerate(path.read_text().splitlines(), 1):
        if text.strip():
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line}: expected an object")
            yield line, value


def text_content(messages):
    """Only input content; never tool schemas, metadata, or serialized field names."""
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(p["text"] for p in content if isinstance(p, dict)
                         and p.get("type") == "text" and isinstance(p.get("text"), str))
    return "\n".join(parts)


def continuation(request, later):
    """Require byte-equivalent message prefix and same turn before showing behavior.

    Duck's response rows repeat input messages, so they contain no model answer.
    The next request can retain the answer and tool observation. If trimming or a
    turn change breaks that exact relation, leave it unknown rather than guess.
    """
    before, after = request.get("messages", []), later.get("messages", [])
    if (not before or later.get("event") != "request"
            or request.get("analysis_step") is None
            or request.get("analysis_step") != later.get("analysis_step")
            or len(after) <= len(before) or after[:len(before)] != before):
        return None
    tail = after[len(before):]
    if tail[0].get("role") != "assistant":
        return None
    calls = tail[0].get("tool_calls") or []
    ids = {call.get("id") for call in calls if isinstance(call.get("id"), str) and call["id"]}
    observations = [m for m in tail[1:] if m.get("role") == "tool"
                    and isinstance(m.get("tool_call_id"), str) and m["tool_call_id"] in ids]
    return {"assistant": tail[0], "tool_observations": observations,
            "meaning": "Recorded continuation after this input; no claim that memory caused the action."}


def inspect_evidence(root, arm):
    games = {game: {"records": [], "issues": []} for game in verdict.EXPECTED_GAMES}
    issues, synthetic, sources = [], False, []
    paths = [root / "causal_pilot_config.json", root / "benchmark.json",
             root / "pilot_prompt_records.jsonl",
             root / ("compaction_events.jsonl" if arm == "control" else "causal_memory_events.jsonl")]
    requests = {game: [] for game in games}
    for path in sorted(root.rglob("*_requests.jsonl")):
        # Identity must be in the basename, not an unrelated ancestor directory.
        matches = [g for g in games if re.fullmatch(re.escape(g) + r"_p0_requests\.jsonl", path.name)]
        if len(matches) != 1:
            issues.append(f"unrecognized request filename: {path.relative_to(root)}")
            continue
        game = matches[0]
        paths.append(path)
        try:
            entries = list(rows(path))
            for line, row in entries:
                if row.get("event") != "request":
                    continue
                messages = row.get("messages")
                if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                    raise ValueError(f"{path.name}:{line}: malformed request messages")
                for message in messages:
                    calls = message.get("tool_calls") or []
                    if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
                        raise ValueError(f"{path.name}:{line}: malformed tool calls")
            synthetic |= any(row.get("dry_run_only") is True for _, row in entries)
            requests[game].append((path, entries))
        except (OSError, ValueError) as exc:
            games[game]["issues"].append(str(exc))
    for game, streams in requests.items():
        if len(streams) > 1:
            games[game]["issues"].append("ambiguous duplicate request streams; no continuation selected")
    for path in paths:
        if not path.is_file():
            issues.append(f"missing {path.relative_to(root)}")
            continue
        sources.append(reference(root, path))
        try:
            values = [read_json(path)] if path.suffix == ".json" else [row for _, row in rows(path)]
            for value in values:
                synthetic |= value.get("dry_run_only") is True or value.get("schema") == "dry-run-only"
                synthetic |= any("synthetic local dry run" in str(g.get("solver_note", ""))
                                 for g in value.get("game_runs", []))
        except (OSError, ValueError, AttributeError) as exc:
            issues.append(f"unreadable {path.relative_to(root)}: {exc}")
    record_file = root / "pilot_prompt_records.jsonl"
    if record_file.is_file():
        try:
            for line, record in rows(record_file):
                if record.get("kind") != "record_emitted":
                    continue
                game, exact = record.get("game_id"), record.get("exact_line")
                if not isinstance(game, str) or game not in games:
                    issues.append(f"record line {line} has unknown game")
                    continue
                if (not isinstance(exact, str) or MARKERS[arm] not in exact
                        or hashlib.sha256(exact.encode()).hexdigest() != record.get("record_sha256")):
                    games[game]["issues"].append(f"record line {line} fails exact hash/marker binding")
                    continue
                item = {"exact_memory": exact, "emitted_at": reference(root, record_file, line),
                        "input_at": None, "continuation_at": None, "continuation": None}
                streams = requests[game]
                if len(streams) == 1:
                    path, entries = streams[0]
                    for index, (request_line, request) in enumerate(entries):
                        if request.get("event") != "request" or exact not in text_content(request.get("messages", [])):
                            continue
                        item["input_at"] = reference(root, path, request_line)
                        item["analysis_step"] = request.get("analysis_step")
                        item["request_index_within_turn"] = request.get("request_index_within_turn")
                        for next_line, later in entries[index + 1:]:
                            if later.get("event") != "request":
                                continue
                            item["continuation"] = continuation(request, later)
                            if item["continuation"] is not None:
                                item["continuation_at"] = reference(root, path, next_line)
                            break
                        break
                games[game]["records"].append(item)
        except (OSError, ValueError) as exc:
            issues.append(f"unreadable record stream: {exc}")
    return {"games": games, "issues": issues, "synthetic": synthetic, "sources": sources}


def inspect_frozen(root, arm):
    try:
        # Validate strict JSON before handing it to the unchanged frozen inspector.
        if (root / "benchmark.json").is_file():
            read_json(root / "benchmark.json")
        return verdict.inspect_run(root, arm)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        return {"arm": arm, "valid": False, "problems": [f"unreadable run: {exc}"]}


def review(control, treatment):
    roots = {"control": control.resolve(), "treatment": treatment.resolve()}
    inspected = {arm: inspect_frozen(root, arm) for arm, root in roots.items()}
    evidence = {arm: inspect_evidence(root, arm) for arm, root in roots.items()}
    issues = []
    for arm, summary in inspected.items():
        # The sealed inspector predates strict JSON finite-number handling.
        for field in ("scores", "levels"):
            if any(not math.isfinite(n) for n in summary.get(field, {}).values()):
                issues.append(f"{arm}: non-finite {field}; review is invalid")
        issues += [f"{arm}: {p}" for p in evidence[arm]["issues"]]
        issues += [f"{arm} {g}: {p}" for g, row in evidence[arm]["games"].items() for p in row["issues"]]
    decision = verdict.decide(*inspected.values()) if not issues else {"verdict": "INVALID", "why": "review evidence errors"}
    synthetic = any(e["synthetic"] for e in evidence.values())
    game_rows = []
    for game in verdict.EXPECTED_GAMES:
        row = {"game": game}
        for arm in roots:
            summary = inspected[arm]
            row[arm] = {"score": summary.get("scores", {}).get(game),
                        "levels": summary.get("levels", {}).get(game),
                        "stop": summary.get("stop_reasons", {}).get(game),
                        **evidence[arm]["games"][game]}
        c, t = row["control"]["score"], row["treatment"]["score"]
        row["score_delta"] = t - c if c is not None and t is not None and math.isfinite(c) and math.isfinite(t) else None
        game_rows.append(row)
    return {"schema": "arc-causal-pair-review-v1", "mode": "SYNTHETIC_REVIEW" if synthetic else "SAVED_PAIR_REVIEW",
            "meaning": "Offline reading of supplied artifacts; no model run, runtime attestation, transfer claim, or submission permission.",
            "behavioral_result": None if synthetic else decision,
            "synthetic_rule_path": decision if synthetic else None,
            "seal_sha256": sha(HERE / "seal.json"), "verdict_source_sha256": sha(HERE / "verdict.py"),
            "roots": {arm: str(root) for arm, root in roots.items()}, "inspection": inspected,
            "evidence_issues": issues, "sources": {arm: e["sources"] for arm, e in evidence.items()},
            "games": game_rows}


def render(result):
    synthetic = result["mode"] == "SYNTHETIC_REVIEW"
    lines = ["# ARC KAGGLE matched pair", "",
             "SYNTHETIC SAVED FIXTURE — no behavioral result." if synthetic else "SAVED PAIR — local artifact inspection only.", "",
             "Question: did causal memory improve play across the same 25 games?", "",
             "No game is excluded. Memory followed by an action is inspectable evidence, not proof that memory caused the action.", ""]
    if not synthetic:
        lines += [f"Frozen decision: **{result['behavioral_result']['verdict']}**", result['behavioral_result']['why'], ""]
    lines += ["Next: resolve invalid evidence; a real PROMISING pair permits only a separately approved second pair. No submission.", ""]
    for arm, inspected in result["inspection"].items():
        lines += [f"{arm.title()} validity: {inspected['valid']}."]
        lines += [f"- {p}" for p in inspected.get("problems", [])]
    lines += [f"- {p}" for p in result["evidence_issues"]]
    for game in result["games"]:
        lines += ["", f"## {game['game']}", "", f"Score change: {game['score_delta']}."]
        for arm in ("control", "treatment"):
            row = game[arm]
            lines += ["", f"### {arm.title()}", "", f"Score {row['score']}; levels {row['levels']}; stop {row['stop']}.", ""]
            if not row["records"]:
                lines += ["No valid emitted memory record found."]
            bound = [r for r in row["records"] if r["input_at"]]
            acted = [r for r in bound if r["continuation"]]
            lines += [f"Memory records: {len(row['records'])}; request-bound: {len(bound)}; with continuation: {len(acted)}.",
                      "One inspectable example follows; every record is retained in review.json."]
            examples = (acted or bound or row["records"])[:1]
            for record in examples:
                lines += ["Exact memory:", "", "    " + record["exact_memory"].replace("\n", "\n    "), ""]
                for key in ("emitted_at", "input_at", "continuation_at"):
                    ref = record[key]
                    if ref:
                        path = Path(result["roots"][arm]) / ref["file"]
                        lines += [f"{key.replace('_', ' ').title()}: [{ref['file']}:{ref['line']}](<{path}:{ref['line']}>)."]
                if record["input_at"] is None:
                    lines += ["Exact record not found in an unambiguous model input."]
                if record["continuation"] is None:
                    lines += ["Next action: unknown. No exact conversation continuation retained."]
                else:
                    lines += ["", "Recorded next assistant/tool messages:", "", "    " + json.dumps(record["continuation"], indent=2).replace("\n", "\n    ")]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the evidence roots")
    args = parser.parse_args()
    output = args.output.resolve()
    if any(output == root.resolve() or root.resolve() in output.parents for root in (args.control, args.treatment)):
        parser.error("output must be outside both input roots")
    if output.exists():
        parser.error("output already exists; choose a new directory")
    result = review(args.control, args.treatment)
    # Never serialize NaN/Infinity as valid outcomes.
    payload = json.dumps(result, indent=2, allow_nan=False)
    output.mkdir(parents=True)
    (output / "review.json").write_text(payload + "\n")
    (output / "REVIEW.md").write_text(render(result))
    decision = result["behavioral_result"] or result["synthetic_rule_path"]
    print(json.dumps({"mode": result["mode"], "behavioral_result": result["behavioral_result"], "review": str(output / "REVIEW.md")}))
    return 2 if decision["verdict"] == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
