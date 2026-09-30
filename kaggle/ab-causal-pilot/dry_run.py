"""Exercise the sealed pair's real artifact inspection and verdict path without Kaggle or GPU.

The generated scores are synthetic and are labelled as such. This proves packaging and evaluation
wiring only; it is not ARC gameplay evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(root: Path, arm: str, verdict) -> None:
    root.mkdir(parents=True)
    settings = verdict.SEAL["settings"]
    _write_json(root / "causal_pilot_config.json", {
        "schema": "arc-causal-pilot-v2",
        "arm": arm,
        "observed_settings": settings,
        "dry_run_only": True,
    })
    games = []
    for index, game in enumerate(verdict.EXPECTED_GAMES):
        control_score = float(index % 4)
        score = control_score + (1.0 if arm == "treatment" and index < 13 else 0.0)
        levels = 2 + index % 3 + (1 if arm == "treatment" and index < 13 else 0)
        games.append({
            "game_id": game,
            "state": "gave_up",
            "final_score": score,
            "levels_completed": levels,
            "solver_note": "synthetic local dry run; no model or game was executed",
        })
    _write_json(root / "benchmark.json", {"schema": "dry-run-only", "game_runs": games})

    marker = "Compacted earlier turns" if arm == "control" else "Causal record from earlier dropped turns"
    event_name = "compaction_events.jsonl" if arm == "control" else "causal_memory_events.jsonl"
    events, records = [], []
    for game in verdict.EXPECTED_GAMES:
        exact = f"- {marker}: dry-run record for {game}"
        events.append({"kind": "digest", "game_id": game, "chars": len(exact), "dry_run_only": True})
        records.append({
            "kind": "record_emitted",
            "game_id": game,
            "exact_line": exact,
            "record_sha256": hashlib.sha256(exact.encode()).hexdigest(),
            "dry_run_only": True,
        })
        request = {"event": "request", "messages": [{"role": "user", "content": f"frame\n{exact}\nend"}],
                   "dry_run_only": True}
        (root / f"{game}_p0_requests.jsonl").write_text(json.dumps(request, sort_keys=True) + "\n")
    (root / event_name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in events))
    (root / "pilot_prompt_records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )


def run(output: Path) -> dict:
    build = _load("causal_pilot_build_dry", HERE / "build.py")
    seal = build.build()
    verdict = _load("causal_pilot_verdict_dry", HERE / "verdict.py")
    if output.exists():
        shutil.rmtree(output)
    control, treatment = output / "control", output / "treatment"
    _fixture(control, "control", verdict)
    _fixture(treatment, "treatment", verdict)
    control_result = verdict.inspect_run(control, "control")
    treatment_result = verdict.inspect_run(treatment, "treatment")
    decision = verdict.decide(control_result, treatment_result)
    result = {
        "schema": "arc-causal-pilot-local-dry-run-v1",
        "authorization": "LOCAL_DRY_RUN_ONLY_NO_GPU_PUSH_UPLOAD_SUBMISSION_OR_SPEND",
        "synthetic_outcomes": True,
        "seal_sha256": hashlib.sha256((HERE / "seal.json").read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256((HERE / "comparison-baseline.json").read_bytes()).hexdigest(),
        "control": control_result,
        "treatment": treatment_result,
        "decision_path_exercised": decision,
        "proof": {
            "denominator_games": len(verdict.EXPECTED_GAMES),
            "control_valid": control_result["valid"],
            "treatment_valid": treatment_result["valid"],
            "control_prompt_bound_games": len(control_result["prompt_input_games"]),
            "treatment_prompt_bound_games": len(treatment_result["prompt_input_games"]),
            "outside_policy_equal": json.loads((HERE / "pair-diff.json").read_text())["outside_policy_equal"],
            "generated_notebooks": seal["generated"],
        },
        "meaning": "Packaging and the real verdict path work locally. The synthetic decision is not a gameplay result.",
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "dry-run-result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"proof": result["proof"], "result": str(args.output / "dry-run-result.json")}, indent=2))
    return 0 if all((result["proof"]["control_valid"], result["proof"]["treatment_valid"],
                     result["proof"]["outside_policy_equal"])) else 2


if __name__ == "__main__":
    raise SystemExit(main())
