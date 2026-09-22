import numpy as np
import pytest

from deadprobe.grid import GAMEPLAY
from deadprobe.memo import NoOpMemo
from deadprobe.probe import StepOut, plan_candidates, probe_level


class FakeGame:
    """Pure-python fake: 64x64, timer row 0, player on row 30, token at (50, 50)."""

    def __init__(self):
        self.x, self.t, self.level, self.state = 10, 60, 0, "NOT_FINISHED"
        self.actions = 0

    def grid(self):
        g = np.zeros((64, 64), dtype=int)
        g[0, : self.t] = 8
        g[30, self.x] = 9
        g[50, 50] = 11
        return g

    def step(self, aid, data):
        self.actions += 1
        self.t -= 1
        if aid == 4 and self.x < 63:
            self.x += 1
        if aid == 6 and data == {"x": 50, "y": 50}:
            self.level += 1
        return StepOut(self.grid(), self.level, self.state)


def test_candidates_order_and_clicks():
    g = FakeGame()
    c = plan_candidates(g.grid(), [1, 2, 4, 6], max_clicks=2)
    assert [a for a, _ in c[:3]] == [1, 2, 4]
    assert (6, {"x": 50, "y": 50}) in c


def test_probe_finds_dead_and_live():
    g = FakeGame()
    memo = NoOpMemo()
    rep = probe_level(g.step, g.grid(), [1, 2, 4], 0, budget=10, memo=memo)
    # UP and DOWN only tick the timer; the second tick promotes it to HUD and
    # the post pass re-judges the first one.
    assert rep.dead == ["UP", "DOWN"]
    assert rep.live_actions == ["RIGHT"]
    assert rep.cost == 3 == g.actions  # every probe action is a real action
    assert rep.stop_reason == "exhausted"
    assert "UP=hud only" in rep.note()


def test_probe_respects_budget():
    g = FakeGame()
    rep = probe_level(g.step, g.grid(), [1, 2, 3, 4, 5, 7], 0, budget=2)
    assert rep.cost == 2 and rep.stop_reason == "budget"


def test_probe_stops_on_level_up():
    g = FakeGame()
    rep = probe_level(g.step, g.grid(), [6], 0, budget=10, max_clicks=3)
    assert rep.stop_reason == "level_up"
    assert rep.results[-1].kind == GAMEPLAY


def test_probe_skip_list():
    g = FakeGame()
    rep = probe_level(g.step, g.grid(), [1, 2, 4], 0, skip={"1"})
    assert [r.action_id for r in rep.results] == [2, 4]


def test_probe_on_real_engine_game(pr01_env):
    """PR01 through the official toolkit: UP and DOWN dead, LEFT and RIGHT live."""
    from arcengine import GameAction

    env = pr01_env
    obs = env.reset()

    def step(aid, data):
        o = env.step(GameAction.from_id(aid), data=data or {})
        return StepOut(o.frame[-1], o.levels_completed, o.state.name)

    memo = NoOpMemo()
    avail = [a for a in obs.available_actions if a != 0]
    rep = probe_level(step, obs.frame[-1], avail, obs.levels_completed, budget=4, max_clicks=0, memo=memo)
    assert rep.dead == ["UP", "DOWN"]
    assert rep.live_actions == ["LEFT", "RIGHT"]
    # the timer row was learned as HUD, so a repeat UP is now a known no-op
    cur = env.step(GameAction.ACTION1).frame[-1]
    assert memo.is_known_noop(cur, 1)


def test_probe_click_wins_level_on_real_engine(pr01_env):
    from arcengine import GameAction

    env = pr01_env
    obs = env.reset()

    def step(aid, data):
        o = env.step(GameAction.from_id(aid), data=data or {})
        return StepOut(o.frame[-1], o.levels_completed, o.state.name)

    rep = probe_level(step, obs.frame[-1], [6], 0, budget=5, max_clicks=3)
    assert rep.stop_reason == "level_up"
