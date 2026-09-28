from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

from tests import compaction_replay as cr

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "kaggle/ab-causal-pilot"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_pair_is_current_and_only_policy_differs():
    build = _load("causal_pair_build", HERE / "build.py")
    before = {path: path.read_bytes() for path in HERE.rglob("*") if path.is_file() and path.name != "__pycache__"}
    seal = build.build()
    after = {path: path.read_bytes() for path in HERE.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    for path, payload in before.items():
        if "__pycache__" not in path.parts:
            assert after[path] == payload
    diff = json.loads((HERE / "pair-diff.json").read_text())
    assert diff["outside_policy_equal"] is True
    assert diff["normalized_changed_cells"] == [13]
    assert diff["unexpected"] == []
    assert seal["authorization"].startswith("LOCAL_PREPARATION_ONLY")


def test_treatment_embeds_corrected_source_and_installs_real_drop_path(tmp_path, monkeypatch):
    build = _load("causal_pair_hook", HERE / "build.py")
    ta = cr.tool_agent()
    original = ta.ToolAgent._persistent_history_messages
    pkg = tmp_path / "deadprobe"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("ARC_CAUSAL_EVENTS", str(tmp_path / "events.jsonl"))
    source = build._causal_block(build.CAUSAL.read_text())
    scope = {"_pkg": pkg, "Path": Path}
    try:
        exec(compile(source, "causal-treatment-block", "exec"), scope)
        assert ta.ToolAgent._persistent_history_messages is not original
        assert ta.ToolAgent._persistent_history_messages.__module__ == "deadprobe.causal_memory"
        assert ta.ToolAgent._ensure_session.__module__ == "deadprobe.causal_memory"
        assert (pkg / "causal_memory.py").read_bytes() == build.CAUSAL.read_bytes()
    finally:
        scope["_causal_memory"].uninstall(ta)
    assert ta.ToolAgent._persistent_history_messages is original


def _summary(scores, levels):
    games = [f"g{i}" for i in range(len(scores))]
    return {"valid": True, "games": games, "scores": dict(zip(games, scores)), "levels": dict(zip(games, levels))}


def test_every_verdict_branch_is_executable():
    verdict = _load("causal_pair_verdict", HERE / "verdict.py")
    invalid = verdict.decide({"valid": False}, {"valid": True})
    assert invalid["verdict"] == "INVALID"

    control = _summary([1, 1, 1], [1, 1, 1])
    failed = verdict.decide(control, _summary([0, 0, 1], [0, 1, 1]))
    assert failed["verdict"] == "FAILED"

    promising = verdict.decide(control, _summary([2, 2, 1], [2, 1, 1]))
    assert promising["verdict"] == "PROMISING"

    inconclusive = verdict.decide(control, _summary([2, 0, 1], [2, 1, 0]))
    assert inconclusive["verdict"] == "INCONCLUSIVE"

    unequal_infrastructure = dict(_summary([2, 2, 1], [2, 1, 1]), infrastructure_failures={"gpu_oom": 1})
    assert verdict.decide(control, unequal_infrastructure)["verdict"] == "INVALID"


def test_pulled_run_inspection_requires_config_event_and_exact_prompt(tmp_path):
    verdict = _load("causal_pair_inspect", HERE / "verdict.py")
    for arm, marker, event_name in (
        ("control", "Compacted earlier turns", "compaction_events.jsonl"),
        ("treatment", "Causal record from earlier dropped turns", "causal_memory_events.jsonl"),
    ):
        run = tmp_path / arm
        (run / "prompts").mkdir(parents=True)
        config = {
            "schema": "arc-causal-pilot-v2",
            "arm": arm,
            "observed_settings": verdict.SEAL["settings"],
        }
        (run / "causal_pilot_config.json").write_text(json.dumps(config))
        games = [{"game_id": game, "state": "gave_up", "final_score": 1.0, "levels_completed": 1,
                  "solver_note": "tokens=1"} for game in verdict.EXPECTED_GAMES]
        (run / "benchmark.json").write_text(json.dumps({"game_runs": games}))
        events = []
        records = []
        for game in verdict.EXPECTED_GAMES:
            event = {"kind": "digest", "chars": 2999}
            if arm == "treatment":
                event["game_id"] = game
            else:
                event["runtime_dir"] = f"/kaggle/working/runs/{game}_p0"
            events.append(event)
            exact = f"- {marker}: exact record {game}"
            records.append({"kind": "record_emitted", "game_id": game,
                            "record_sha256": hashlib.sha256(exact.encode()).hexdigest(),
                            "exact_line": exact})
            request = {"event": "request", "messages": [{"role": "user", "content": f"frame\n{exact}\nend"}]}
            (run / f"{game}_p0_requests.jsonl").write_text(json.dumps(request) + "\n")
        (run / event_name).write_text("".join(json.dumps(event) + "\n" for event in events))
        (run / "pilot_prompt_records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        inspected = verdict.inspect_run(run, arm)
        assert inspected["valid"] is True, inspected["problems"]

    missing = tmp_path / "treatment" / f"{verdict.EXPECTED_GAMES[-1]}_p0_requests.jsonl"
    missing.write_text(json.dumps({"event": "request", "messages": [{"role": "user", "content": "no record"}]}) + "\n")
    inspected = verdict.inspect_run(tmp_path / "treatment", "treatment")
    assert inspected["valid"] is False
    assert any("per-game request-bound exact prompt proof missing" in problem for problem in inspected["problems"])


def test_one_unbound_digest_and_prompt_cannot_stand_for_25_games(tmp_path):
    verdict = _load("causal_pair_adversarial_binding", HERE / "verdict.py")
    run = tmp_path / "treatment"
    (run / "prompts").mkdir(parents=True)
    config = {"schema": "arc-causal-pilot-v2", "arm": "treatment",
              "observed_settings": verdict.SEAL["settings"]}
    (run / "causal_pilot_config.json").write_text(json.dumps(config))
    games = [{"game_id": game, "state": "gave_up", "final_score": 1.0, "levels_completed": 1,
              "solver_note": "tokens=1"} for game in verdict.EXPECTED_GAMES]
    (run / "benchmark.json").write_text(json.dumps({"game_runs": games}))
    (run / "causal_memory_events.jsonl").write_text(json.dumps({"kind": "digest", "chars": 20}) + "\n")
    exact = "- Causal record from earlier dropped turns: one marker"
    record = {"kind": "record_emitted", "game_id": verdict.EXPECTED_GAMES[0], "exact_line": exact,
              "record_sha256": hashlib.sha256(exact.encode()).hexdigest()}
    (run / "pilot_prompt_records.jsonl").write_text(json.dumps(record) + "\n")
    request = {"event": "request", "messages": [{"role": "user", "content": exact}]}
    (run / f"{verdict.EXPECTED_GAMES[0]}_p0_requests.jsonl").write_text(json.dumps(request) + "\n")
    inspected = verdict.inspect_run(run, "treatment")
    assert inspected["valid"] is False
    assert any("per-game digest proof missing" in problem for problem in inspected["problems"])
    assert any("per-game emitted-record proof missing" in problem for problem in inspected["problems"])
    assert any("per-game request-bound exact prompt proof missing" in problem for problem in inspected["problems"])


def test_copied_expected_constants_are_not_observed_runtime_proof(tmp_path):
    verdict = _load("causal_pair_adversarial_config", HERE / "verdict.py")
    run = tmp_path / "control"
    run.mkdir()
    copied = {"schema": "arc-causal-pilot-v2", "arm": "control", "settings": verdict.SEAL["settings"]}
    (run / "causal_pilot_config.json").write_text(json.dumps(copied))
    (run / "benchmark.json").write_text(json.dumps({"game_runs": []}))
    inspected = verdict.inspect_run(run, "control")
    assert inspected["valid"] is False
    assert "observed runtime settings differ from seal" in inspected["problems"]


def test_missing_levels_and_cancelled_infrastructure_are_invalid(tmp_path):
    verdict = _load("causal_pair_adversarial_outcomes", HERE / "verdict.py")
    run = tmp_path / "control"
    run.mkdir()
    config = {"schema": "arc-causal-pilot-v2", "arm": "control",
              "observed_settings": verdict.SEAL["settings"]}
    (run / "causal_pilot_config.json").write_text(json.dumps(config))
    games = [{"game_id": game, "state": "gave_up", "final_score": 1.0, "solver_note": "tokens=1"}
             for game in verdict.EXPECTED_GAMES]
    games[0]["state"] = "cancelled"
    games[0]["solver_note"] = "Server disconnected; watchdog exhausted"
    (run / "benchmark.json").write_text(json.dumps({"game_runs": games}))
    inspected = verdict.inspect_run(run, "control")
    assert inspected["valid"] is False
    assert any("missing numeric score/level or stop reason" in problem for problem in inspected["problems"])
    assert any("incomplete games" in problem for problem in inspected["problems"])
    assert inspected["infrastructure_failures"]["connection_failure"] == 1
    assert inspected["infrastructure_failures"]["kernel_failure"] == 1
    assert inspected["infrastructure_failures"]["cancelled_games"] == 1
