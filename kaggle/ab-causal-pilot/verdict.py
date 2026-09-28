"""Apply every frozen causal-memory pilot verdict branch to two pulled run directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAL = json.loads((HERE / "seal.json").read_text())
EXPECTED_GAMES = SEAL["settings"]["games"]
TERMINAL = {"won", "gave_up"}
INFRA_PATTERNS = {
    "gpu_oom": re.compile(r"out of memory|OutOfMemoryError", re.I),
    "server_restart": re.compile(r"watchdog.{0,80}restart|restarting.{0,80}(?:vllm|server)", re.I),
    "connection_failure": re.compile(r"connection refused|ConnectionError|failed to connect|server disconnected", re.I),
    "request_timeout": re.compile(r"ReadTimeout|request.{0,40}timed out|timeout waiting for", re.I),
    "kernel_failure": re.compile(r"notebook.{0,40}(?:error|failed)|KernelWorkerStatus\.ERROR|watchdog exhausted", re.I),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _game_from_value(value: str) -> str | None:
    matches = [game for game in EXPECTED_GAMES if game in value]
    return matches[0] if len(matches) == 1 else None


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _strings(nested)


def _request_payloads(root: Path) -> dict[str, list[str]]:
    payloads = {game: [] for game in EXPECTED_GAMES}
    for path in root.rglob("*_requests.jsonl"):
        game = _game_from_value(str(path))
        if game is None or not path.is_file() or path.stat().st_size > 200_000_000:
            continue
        try:
            for line in path.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") != "request":
                    continue
                payloads[game].append("\n".join(_strings(row.get("messages", []))))
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


def _bound_prompt_games(root: Path, marker: str) -> tuple[set[str], set[str], list[str]]:
    emitted: dict[str, set[str]] = {game: set() for game in EXPECTED_GAMES}
    malformed: list[str] = []
    for index, event in enumerate(_events(root / "pilot_prompt_records.jsonl")):
        if event.get("kind") != "record_emitted":
            continue
        game = str(event.get("game_id", "") or "")
        exact = event.get("exact_line")
        digest = str(event.get("record_sha256", "") or "")
        if game not in EXPECTED_GAMES or not isinstance(exact, str) or marker not in exact:
            malformed.append(f"event {index} has no bound game/exact record")
            continue
        if hashlib.sha256(exact.encode()).hexdigest() != digest:
            malformed.append(f"event {index} record hash mismatch")
            continue
        emitted[game].add(exact)
    requests = _request_payloads(root)
    event_games = {game for game, records in emitted.items() if records}
    request_games = {
        game for game, records in emitted.items()
        if records and any(record in payload for record in records for payload in requests[game])
    }
    return event_games, request_games, malformed


def _events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _digest_games(events: list[dict]) -> set[str]:
    found = set()
    for event in events:
        if event.get("kind") != "digest":
            continue
        game = str(event.get("game_id", "") or "")
        if game not in EXPECTED_GAMES:
            game = _game_from_value(str(event.get("runtime_dir", "") or "")) or ""
        if game in EXPECTED_GAMES:
            found.add(game)
    return found


def _infrastructure_failures(root: Path, games: list[dict]) -> dict[str, int]:
    counts = {name: 0 for name in INFRA_PATTERNS}
    for path in [*root.glob("*.log"), *root.glob("*.txt")]:
        if not path.is_file() or path.name == "config_proof.txt" or path.stat().st_size > 100_000_000:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for name, pattern in INFRA_PATTERNS.items():
            counts[name] += len(pattern.findall(text))
    for game in games:
        note = str(game.get("solver_note", "") or "")
        for name, pattern in INFRA_PATTERNS.items():
            counts[name] += len(pattern.findall(note))
        if game.get("state") == "cancelled":
            counts["cancelled_games"] = counts.get("cancelled_games", 0) + 1
    return {name: count for name, count in counts.items() if count}


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
    observed = cfg.get("observed_settings")
    if observed != SEAL.get("settings"):
        problems.append("observed runtime settings differ from seal")

    benchmark = json.loads(bench_path.read_text())
    games = benchmark.get("game_runs") or []
    ids = [str(game.get("game_id", "")) for game in games]
    if ids != EXPECTED_GAMES:
        problems.append(f"game identities/order differ: {ids}")
    def is_number(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    incomplete = [game.get("game_id") for game in games if game.get("state") not in TERMINAL]
    if incomplete:
        problems.append(f"incomplete games: {incomplete}")
    missing_outcomes = [
        game.get("game_id") for game in games
        if not is_number(game.get("final_score")) or not is_number(game.get("levels_completed"))
        or not str(game.get("state", "")).strip()
    ]
    if missing_outcomes:
        problems.append(f"missing numeric score/level or stop reason: {missing_outcomes}")
    errors = [game.get("game_id") for game in games if str(game.get("solver_note", "")).lower().startswith("error:")]
    if errors:
        problems.append(f"solver errors: {errors}")

    event_name = "compaction_events.jsonl" if arm == "control" else "causal_memory_events.jsonl"
    events = _events(root / event_name)
    digests = [event for event in events if event.get("kind") == "digest"]
    if not digests:
        problems.append(f"missing digest proof in {event_name}")
    digest_games = _digest_games(events)
    missing_digest = sorted(set(EXPECTED_GAMES) - digest_games)
    if missing_digest:
        problems.append(f"per-game digest proof missing: {missing_digest}")
    too_long = [event for event in digests if int(event.get("chars", 0) or 0) > int(SEAL["settings"]["memory_record_max_chars"])]
    if too_long:
        problems.append(f"memory record exceeded {SEAL['settings']['memory_record_max_chars']} chars")
    marker = "Compacted earlier turns" if arm == "control" else "Causal record from earlier dropped turns"
    emitted_games, prompt_games, malformed_records = _bound_prompt_games(root, marker)
    if malformed_records:
        problems.append(f"malformed prompt-record proof: {malformed_records}")
    missing_emitted = sorted(set(EXPECTED_GAMES) - emitted_games)
    if missing_emitted:
        problems.append(f"per-game emitted-record proof missing: {missing_emitted}")
    missing_prompt = sorted(set(EXPECTED_GAMES) - prompt_games)
    if missing_prompt:
        problems.append(f"per-game request-bound exact prompt proof missing: {missing_prompt}")

    scores = {game["game_id"]: float(game["final_score"]) for game in games if is_number(game.get("final_score"))}
    levels = {game["game_id"]: int(game["levels_completed"]) for game in games if is_number(game.get("levels_completed"))}
    stop_reasons = {game["game_id"]: str(game.get("state", "")) for game in games}
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
        "stop_reasons": stop_reasons,
        "digest_events": len(digests),
        "digest_games": sorted(digest_games),
        "emitted_record_games": sorted(emitted_games),
        "prompt_input_games": sorted(prompt_games),
        "max_record_chars": max((int(event.get("chars", 0) or 0) for event in digests), default=0),
        "infrastructure_failures": _infrastructure_failures(root, games),
        "observed_settings": observed,
    }


def decide(control: dict, treatment: dict) -> dict:
    if not control.get("valid") or not treatment.get("valid"):
        return {"verdict": "INVALID", "why": "one or both runs failed the frozen validity/proof checks"}
    if control["games"] != treatment["games"]:
        return {"verdict": "INVALID", "why": "arms do not contain the same game identities in the same order"}
    if control.get("infrastructure_failures", {}) != treatment.get("infrastructure_failures", {}):
        return {
            "verdict": "INVALID",
            "why": "infrastructure failures affected the arms unequally",
            "control_infrastructure_failures": control.get("infrastructure_failures", {}),
            "treatment_infrastructure_failures": treatment.get("infrastructure_failures", {}),
        }
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
