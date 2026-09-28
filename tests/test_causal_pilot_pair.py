from __future__ import annotations

import importlib.util
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
            "settings": verdict.SEAL["settings"],
            "datasets": verdict.SEAL["datasets"],
            "model": verdict.SEAL["model"],
            "frozen_input_hashes": verdict.SEAL["frozen_input_hashes"],
        }
        (run / "causal_pilot_config.json").write_text(json.dumps(config))
        games = [{"game_id": game, "state": "gave_up", "final_score": 1.0, "levels_completed": 1,
                  "solver_note": "tokens=1"} for game in verdict.EXPECTED_GAMES]
        (run / "benchmark.json").write_text(json.dumps({"game_runs": games}))
        (run / event_name).write_text(json.dumps({"kind": "digest", "chars": 2999}) + "\n")
        (run / "prompts" / "one-prompt.log").write_text(f"[MODEL INPUT]\n{marker}: exact record\n")
        inspected = verdict.inspect_run(run, arm)
        assert inspected["valid"] is True, inspected["problems"]

    missing = tmp_path / "treatment" / "prompts" / "one-prompt.log"
    missing.write_text("no memory marker")
    inspected = verdict.inspect_run(tmp_path / "treatment", "treatment")
    assert inspected["valid"] is False
    assert any("exact prompt-input proof missing" in problem for problem in inspected["problems"])
