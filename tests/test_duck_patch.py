"""Integration: the patch against the real Duck harness code, no LLM.

Runs the real _HarnessGameSession from jeroencottaar/taaf-kaggle-source-share
on the PR01 fake game through TAAF's real GameAPI. The LLM is replaced by a
scripted analyzer that wastes actions the way a confused model would.
Skipped unless DUCK_SRC points at the unzipped dataset's src/ directory.
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("DUCK_SRC"), reason="set DUCK_SRC to run against the Duck source")

ENVS = Path(__file__).resolve().parent / "envs"
TARGET = {"action": "MOUSE", "row": 12 * 4 + 1, "col": 12 * 4 + 1}


class ScriptedAnalyzer:
    """One step_env call per analyze(), from a fixed script."""

    def __init__(self, script):
        self.script = list(script)
        self.payloads = []
        self.generated_tokens = 0
        self.stop = lambda: None

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kw):
        from inference.agent.tool_agent import AnalyzerTurnResult

        if not self.script:
            self.stop()  # script done: end the game the way a soft deadline would
            return AnalyzerTurnResult(step_executed=False, retryable_failure=False)
        p = step_env(self.script.pop(0))
        self.payloads.append(p)
        return AnalyzerTurnResult(step_executed=bool(p.get("executed")))


def run(mode, script, tmp_path, runtime_s=20, **cfg_kw):
    import arc_agi
    import inference.agent.tool_agent as ta
    import inference.framework.solver as s
    import taaf.game_api as g

    from deadprobe import duck_patch

    duck_patch.uninstall()
    if mode != "off":
        duck_patch.install(mode, duck_patch.ProbeConfig(log=False, **cfg_kw), solver_module=s, tool_agent_module=ta)
    game = g.GameAPI(
        env_name="pr01",
        arcade_spec=g.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENVS)),
    )
    game.start_game()
    analyzer = ScriptedAnalyzer(script)
    solver = s.HarnessSolver(
        label="t", model="none", analyzer_factory=lambda game, i: analyzer, max_actions_per_game=40, max_runtime_s_per_game=runtime_s
    )
    analyzer.stop = solver._stop_event.set
    solver.job_dir = tmp_path
    solver._play_one(game, 0, 0)
    duck_patch.uninstall()
    return game.game_run, analyzer


WASTEFUL = [{"action": "UP"}] * 4 + [{"action": "RIGHT"}, TARGET]


def test_control_arm_spends_every_wasted_action(tmp_path):
    run_, _ = run("off", WASTEFUL, tmp_path)
    assert run_.actions_per_level[0] == 6
    assert run_.levels_completed == 1


def test_guard_skips_repeat_noops_for_free(tmp_path):
    run_, an = run("guard", WASTEFUL, tmp_path)
    # UP twice (the second sighting of the timer tick promotes it to HUD),
    # two repeats skipped, RIGHT, click: 4 real actions against 6
    assert run_.actions_per_level[0] == 4
    assert run_.levels_completed == 1
    skipped = [p for p in an.payloads if "known" in str(p) or "skipped" in str(p)]
    assert len(skipped) == 2
    assert all(not p.get("executed") for p in skipped)


def test_probe_spends_budget_up_front(tmp_path):
    run_, an = run("probe", [TARGET], tmp_path, budget_level1=4, max_clicks=0)
    # probe: UP, DOWN, LEFT, RIGHT = 4 actions, then the scripted click
    assert run_.actions_per_level[0] == 5
    assert run_.levels_completed == 1


def test_probe_can_win_the_level_by_itself(tmp_path):
    run_, an = run("probe", [], tmp_path, budget_level1=8, max_clicks=3)
    assert run_.levels_completed >= 1
    assert an.payloads == []  # the analyzer never needed to act on level 1


def test_prompt_gets_probe_note_and_known_noops():
    import inference.agent.tool_agent as ta
    import inference.framework.solver as s

    from deadprobe import duck_patch

    duck_patch.uninstall()
    orig = ta.ToolAgent._build_user_prompt
    ta.ToolAgent._build_user_prompt = lambda self, action_num, **kw: "BASE"
    try:
        duck_patch.install("guard", duck_patch.ProbeConfig(log=False), solver_module=s, tool_agent_module=ta)
        agent = ta.ToolAgent.__new__(ta.ToolAgent)
        agent.__dict__["_deadprobe_note"] = "Level-start probe (2 actions ...)"
        out = ta.ToolAgent._build_user_prompt(agent, 0, valid_actions=["UP"])
        assert out.startswith("BASE\nLevel-start probe")
        assert "_deadprobe_note" not in agent.__dict__  # shown once
    finally:
        duck_patch.uninstall()
        ta.ToolAgent._build_user_prompt = orig


class LoopingAnalyzer(ScriptedAnalyzer):
    """A model that keeps asking for the same dead action."""

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kw):
        from inference.agent.tool_agent import AnalyzerTurnResult

        p = step_env({"action": "UP"})
        self.payloads.append(p)
        return AnalyzerTurnResult(step_executed=bool(p.get("executed")))


def test_guard_against_a_stuck_model_is_bounded(tmp_path, monkeypatch):
    """Risk check: the guard refuses, the model asks again, the loop spins.

    Without a bound this burns LLM turns until the per-game time cap. With
    max_consecutive_skips=3, every fourth request goes through, so the model
    sees a real (dead) result and the wasted turns are capped at 3 per action.
    """
    import tests.test_duck_patch as me

    monkeypatch.setattr(me, "ScriptedAnalyzer", LoopingAnalyzer)
    run_, an = run("guard", [], tmp_path, runtime_s=1.0)
    spent = run_.actions_per_level[0]
    refused = sum(1 for p in an.payloads if not p.get("executed"))
    assert spent >= 3
    assert refused <= 3 * spent
    assert refused >= 3 * (spent - 2) - 3  # the bound is actually exercised
