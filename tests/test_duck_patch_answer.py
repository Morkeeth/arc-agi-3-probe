"""Mode guard+answer: the paper's next steps 1 and 2, offline on PR01 against the real Duck session.

1. honest_board_changed: a timer tick no longer reads as board progress.
2. A refusal is answered inside the batch, so the rest of the batch runs and
   the model is not called again just to re-send it.

Duck's model acts through a Python script that calls action([...]) with a
list, and each list is one step_env batch. In plain guard mode a refused
action raises: as the first action it voids the whole batch, later in the
batch it silently cuts the batch short. Either way the model needs another
call to re-send what did not run. This file measures that with a scripted
model, not a GPU run. It says nothing about scores.
"""

import os

import pytest

from tests.test_duck_patch import TARGET, run

pytestmark = pytest.mark.skipif(not os.environ.get("DUCK_SRC"), reason="set DUCK_SRC to run against the Duck source")

PLAN = [{"action": "UP"}] * 4 + [{"action": "RIGHT"}, TARGET]


class ReplanningAnalyzer:
    """A model that sends its whole remaining plan as one batch per call.

    After each call it drops what ran, what was answered as a known no-op,
    and an action the guard refused by name. That is a generous model: it
    never re-asks for an action it was told is dead.
    """

    def __init__(self, script):
        self.plan = list(script)
        self.calls = 0
        self.payloads = []
        self.generated_tokens = 0
        self.stop = lambda: None

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kw):
        from inference.agent.tool_agent import AnalyzerTurnResult

        if not self.plan:
            self.stop()
            return AnalyzerTurnResult(step_executed=False, retryable_failure=False)
        self.calls += 1
        p = step_env({"actions": list(self.plan)})
        self.payloads.append(p)
        ran = int(p.get("executed_count") or 0) if p.get("executed") else 0
        answered = len(p.get("skipped_known_noops") or [])
        refused = 1 if "already changed nothing" in str(p.get("error", "")) else 0
        del self.plan[: ran + answered + refused]
        return AnalyzerTurnResult(step_executed=bool(p.get("executed")))


def play(mode, tmp_path, monkeypatch, script=PLAN, **kw):
    import tests.test_duck_patch as base

    monkeypatch.setattr(base, "ScriptedAnalyzer", ReplanningAnalyzer)
    return run(mode, script, tmp_path, **kw)


def test_control_spends_every_action_in_one_call(tmp_path, monkeypatch):
    run_, an = play("off", tmp_path, monkeypatch)
    assert run_.levels_completed == 1
    assert run_.actions_per_level[0] == 6
    assert an.calls == 1


def test_guard_saves_actions_but_costs_model_calls(tmp_path, monkeypatch):
    run_, an = play("guard", tmp_path, monkeypatch)
    assert run_.levels_completed == 1
    assert run_.actions_per_level[0] == 4
    assert an.calls == 4  # cut short once, then two refusals as first action, then the rest


def test_answer_saves_actions_and_model_calls(tmp_path, monkeypatch):
    run_, an = play("guard+answer", tmp_path, monkeypatch)
    assert run_.levels_completed == 1
    assert run_.actions_per_level[0] == 4  # as guard: two repeats skipped, no action spent
    assert an.calls == 1  # as control: one batch, one model call
    p = an.payloads[0]
    assert p["executed"] is True
    assert p["executed_count"] == 4
    assert p["skipped_known_noops"] == ["UP", "UP"]
    assert "UP" not in p["executed_actions"][2:]  # the answered UPs are not reported as run
    assert "already changed nothing" in p["stop_detail"]


def test_answer_reports_the_skip_to_the_model(tmp_path, monkeypatch):
    """What the model reads goes through ToolAgent's compaction, which drops unknown keys."""
    import inference.agent.tool_agent as ta

    _, an = play("guard+answer", tmp_path, monkeypatch)
    agent = ta.ToolAgent.__new__(ta.ToolAgent)
    compact = ta.ToolAgent._compact_action_result(agent, an.payloads[0])
    assert compact["executed_count"] == 4
    assert "UP, UP" in compact["stop_detail"]


def test_all_answered_batch_is_not_executed(tmp_path, monkeypatch):
    # UP, UP makes UP a known no-op; then a batch of only UPs spends nothing
    script = [{"action": "UP"}, {"action": "UP"}, {"action": "UP"}, {"action": "RIGHT"}, TARGET]

    class OneAtATime(ReplanningAnalyzer):
        def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kw):
            from inference.agent.tool_agent import AnalyzerTurnResult

            if not self.plan:
                self.stop()
                return AnalyzerTurnResult(step_executed=False)
            self.calls += 1
            p = step_env({"actions": [self.plan.pop(0)]})
            self.payloads.append(p)
            return AnalyzerTurnResult(step_executed=bool(p.get("executed")))

    import tests.test_duck_patch as base

    monkeypatch.setattr(base, "ScriptedAnalyzer", OneAtATime)
    run_, an = run("guard+answer", script, tmp_path)
    third = an.payloads[2]
    assert third["executed"] is False
    assert third["skipped_known_noops"] == ["UP"]
    assert "no action was spent" in third["error"]
    assert run_.actions_per_level[0] == 4


def _single_up_board_changed(mode, tmp_path, monkeypatch):
    class Singles(ReplanningAnalyzer):
        def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kw):
            from inference.agent.tool_agent import AnalyzerTurnResult

            if not self.plan:
                self.stop()
                return AnalyzerTurnResult(step_executed=False)
            p = step_env(self.plan.pop(0))
            self.payloads.append(p)
            return AnalyzerTurnResult(step_executed=bool(p.get("executed")))

    import tests.test_duck_patch as base

    monkeypatch.setattr(base, "ScriptedAnalyzer", Singles)
    # DOWN then UP: both only tick the timer. The second tick promotes the timer row to HUD.
    _, an = run(mode, [{"action": "DOWN"}, {"action": "UP"}], tmp_path)
    return an.payloads[1]["board_changed"]


def test_timer_tick_reads_as_progress_in_guard(tmp_path, monkeypatch):
    assert _single_up_board_changed("guard", tmp_path, monkeypatch) is True


def test_timer_tick_is_not_progress_with_answer(tmp_path, monkeypatch):
    assert _single_up_board_changed("guard+answer", tmp_path, monkeypatch) is False
