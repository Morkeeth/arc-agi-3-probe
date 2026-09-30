from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import sys
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
    baseline = json.loads((HERE / "comparison-baseline.json").read_text())
    assert baseline["denominator"]["count"] == 25
    assert baseline["missing_or_unpaired_games"] == "INVALID"


def test_local_dry_run_exercises_real_verdict_path(tmp_path):
    dry = _load("causal_pair_dry_run", HERE / "dry_run.py")
    result = dry.run(tmp_path / "dry-run")
    assert result["synthetic_outcomes"] is True
    assert result["authorization"].startswith("LOCAL_DRY_RUN_ONLY")
    assert result["proof"] == {
        "denominator_games": 25,
        "control_valid": True,
        "treatment_valid": True,
        "control_prompt_bound_games": 25,
        "treatment_prompt_bound_games": 25,
        "outside_policy_equal": True,
        "generated_notebooks": result["proof"]["generated_notebooks"],
    }
    assert (tmp_path / "dry-run/dry-run-result.json").is_file()


def test_treatment_embeds_corrected_source_and_installs_real_drop_path(tmp_path, monkeypatch):
    build = _load("causal_pair_hook", HERE / "build.py")
    ta = cr.tool_agent()
    original = ta.ToolAgent._persistent_history_messages
    # Other hook tests load their own temporary deadprobe package. The real notebook has one package
    # root, so remove those test-only modules before reproducing that import shape here.
    for name in list(sys.modules):
        if name == "deadprobe" or name.startswith("deadprobe."):
            monkeypatch.delitem(sys.modules, name, raising=False)
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


def test_control_proof_identity_uses_real_shared_artifacts_state_filename(tmp_path, monkeypatch):
    """The real fork stores all states in one artifacts dir; identity must come from the filename."""
    build = _load("causal_pair_control_identity", HERE / "build.py")
    ta = cr.tool_agent()
    original_ensure = ta.ToolAgent._ensure_session
    spec = importlib.util.spec_from_file_location(
        "deadprobe.compaction", ROOT / "kaggle/ab-compact/compaction.py"
    )
    compaction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compaction)
    monkeypatch.setitem(sys.modules, "deadprobe.compaction", compaction)
    monkeypatch.setenv("ARC_COMPACT_EVENTS", str(tmp_path / "compaction_events.jsonl"))
    compaction.install(3000, ta)

    control = build._notebook("control")
    hook = build._cell_source(control["cells"][build.HOOK_CELL])
    identity = hook[
        hook.index('_pilot_state_suffix = "_tool_runtime_state.json"'):
        hook.index("_pilot_original_lines =")
    ]
    scope = {"Path": Path, "_pilot_re": re, "_pilot_tool_agent": ta, "sys": sys}
    try:
        exec(compile(identity, "control-proof-identity", "exec"), scope)
        agent = cr.new_agent(ta, context_window=2048)
        state_path = tmp_path / "artifacts" / "ft09-0d8bbf25_p0_tool_runtime_state.json"
        agent._ensure_session(state_path)
        assert agent._session_runtime_dir == state_path.parent
        assert agent._pilot_game_id == "ft09-0d8bbf25"

        messages = [{"role": "system", "content": "s"}]
        for index in range(8):
            messages += [
                {"role": "user", "content": f"Current state: step {index}, level 1. " + "x" * 900},
                {"role": "assistant", "content": f"turn {index}"},
                {"role": "tool", "content": f"result {index}"},
            ]
        agent._persistent_history_messages(messages)
        events = [json.loads(line) for line in (tmp_path / "compaction_events.jsonl").read_text().splitlines()]
        assert events and events[0]["kind"] == "digest"
        assert events[0]["game_id"] == "ft09-0d8bbf25"
        assert events[0]["runtime_dir"] == str(state_path.parent)
    finally:
        compaction.uninstall(ta)
        ta.ToolAgent._ensure_session = original_ensure


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
