"""Apply every frozen causal-memory pilot verdict branch to two pulled run directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAL = json.loads((HERE / "seal.json").read_text())
EXPECTED_GAMES = SEAL["settings"]["games"]
TERMINAL = {"won", "gave_up", "cancelled"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contains(root: Path, marker: str) -> bool:
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 50_000_000:
            continue
        if "prompt" not in str(path).lower() and "model_call" not in path.name.lower():
            continue
        try:
            if marker in path.read_text(errors="replace"):
                return True
        except OSError:
            pass
    return False


def _events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def inspect_run(root: Path, arm: str) -> dict:
    problems: list[str] = []
    cfg_path = root / "causal_pilot_config.json"
    bench_path = root / "benchmark.json"
    if not cfg_path.is_file():
        return {"arm": arm, "path": str(root), "valid": False, "problems": ["missing causal_pilot_config.json"]}
    if not bench_path.is_file():
        return {"arm": arm, "path": str(root), "valid": False, "problems": ["missing benchmark.json"]}
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("schema") != "arc-causal-pilot-v2" or cfg.get("arm") != arm:
        problems.append("runtime arm/schema does not match")
    for field in ("settings", "datasets", "model", "frozen_input_hashes"):
        if cfg.get(field) != SEAL.get(field):
            problems.append(f"runtime {field} differs from seal")

    benchmark = json.loads(bench_path.read_text())
    games = benchmark.get("game_runs") or []
    ids = [str(game.get("game_id", "")) for game in games]
    if ids != EXPECTED_GAMES:
        problems.append(f"game identities/order differ: {ids}")
    incomplete = [game.get("game_id") for game in games if game.get("state") not in TERMINAL or game.get("final_score") is None]
    if incomplete:
        problems.append(f"incomplete games: {incomplete}")
    errors = [game.get("game_id") for game in games if str(game.get("solver_note", "")).lower().startswith("error:")]
    if errors:
        problems.append(f"solver errors: {errors}")

    event_name = "compaction_events.jsonl" if arm == "control" else "causal_memory_events.jsonl"
    events = _events(root / event_name)
    digests = [event for event in events if event.get("kind") == "digest"]
    if not digests:
        problems.append(f"missing digest proof in {event_name}")
    too_long = [event for event in digests if int(event.get("chars", 0) or 0) > int(SEAL["settings"]["memory_record_max_chars"])]
    if too_long:
        problems.append(f"memory record exceeded {SEAL['settings']['memory_record_max_chars']} chars")
    marker = "Compacted earlier turns" if arm == "control" else "Causal record from earlier dropped turns"
    if not _contains(root, marker):
        problems.append(f"exact prompt-input proof missing marker: {marker}")

    scores = {game["game_id"]: float(game["final_score"]) for game in games if game.get("final_score") is not None}
    levels = {game["game_id"]: int(game.get("levels_completed", 0) or 0) for game in games}
    return {
        "arm": arm,
        "path": str(root),
        "valid": not problems,
        "problems": problems,
        "benchmark_sha256": _sha(bench_path),
        "config_sha256": _sha(cfg_path),
        "games": ids,
        "scores": scores,
        "levels": levels,
        "digest_events": len(digests),
        "max_record_chars": max((int(event.get("chars", 0) or 0) for event in digests), default=0),
    }


def decide(control: dict, treatment: dict) -> dict:
    if not control.get("valid") or not treatment.get("valid"):
        return {"verdict": "INVALID", "why": "one or both runs failed the frozen validity/proof checks"}
    if control["games"] != treatment["games"]:
        return {"verdict": "INVALID", "why": "arms do not contain the same game identities in the same order"}
    ids = control["games"]
    deltas = [treatment["scores"][game] - control["scores"][game] for game in ids]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    median = statistics.median(deltas)
    c_score, t_score = sum(control["scores"].values()), sum(treatment["scores"].values())
    c_levels, t_levels = sum(control["levels"].values()), sum(treatment["levels"].values())
    metrics = {
        "paired_score_deltas": dict(zip(ids, deltas)),
        "median_paired_score_delta": median,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "control_total_score": c_score,
        "treatment_total_score": t_score,
        "control_total_levels": c_levels,
        "treatment_total_levels": t_levels,
    }
    if (t_levels < c_levels and t_score <= c_score) or (median < 0 and losses > wins):
        return {"verdict": "FAILED", "why": "treatment met a frozen failure condition", "metrics": metrics}
    if median > 0 and wins > losses and t_levels >= c_levels:
        return {"verdict": "PROMISING", "why": "treatment met every frozen promising condition", "metrics": metrics}
    return {"verdict": "INCONCLUSIVE", "why": "valid pair met neither failed nor promising conditions", "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = inspect_run(args.control, "control")
    treatment = inspect_run(args.treatment, "treatment")
    result = {"schema": "arc-causal-pilot-verdict-v2", "control": control, "treatment": treatment,
              "decision": decide(control, treatment)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0 if result["decision"]["verdict"] != "INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
