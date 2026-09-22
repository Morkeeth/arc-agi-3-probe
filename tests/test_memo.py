import numpy as np

from deadprobe.memo import NoOpMemo, action_key


def frame(x, t=40):
    g = np.zeros((64, 64), dtype=int)
    g[0, :t] = 8
    g[30, x] = 9
    return g


def test_action_key_includes_click_cell():
    assert action_key(1) == "1"
    assert action_key(6, {"x": 3, "y": 7}) == "6@7,3"
    assert action_key(6, {"row": 7, "col": 3}) == "6@7,3"


def test_repeat_noop_is_known_and_counted():
    m = NoOpMemo()
    m.record(frame(10, 40), 1, None, frame(10, 39))  # UP: only timer moved
    assert m.is_known_noop(frame(10, 39), 1)  # same board, timer masked
    m.record(frame(10, 39), 1, None, frame(10, 38))
    assert m.n_repeat_noop == 1
    assert m.stats()["noop"] == 2


def test_dead_is_state_dependent():
    m = NoOpMemo()
    # RIGHT at the wall does nothing; RIGHT elsewhere moves
    m.record(frame(63, 40), 4, None, frame(63, 39))
    m.record(frame(10, 39), 4, None, frame(11, 38))
    assert m.is_known_noop(frame(63, 20), 4)
    assert not m.is_known_noop(frame(11, 20), 4)
    assert not m.is_known_noop(frame(12, 20), 4)  # never seen: unknown, not dead


def test_live_evidence_wins_over_noop():
    m = NoOpMemo()
    m.record(frame(5), 5, None, frame(5, 39))
    m.record(frame(5), 5, None, frame(6, 39))  # nondeterministic: once it worked
    assert not m.is_known_noop(frame(5), 5)


def test_reset_is_never_blocked():
    m = NoOpMemo()
    m.record(frame(5), 0, None, frame(5))
    assert not m.is_known_noop(frame(5), 0)


def test_hud_growth_reindexes_old_entries():
    def f(x, top, bottom):
        g = np.zeros((64, 64), dtype=int)
        g[0, :top] = 8  # timer
        g[63, :bottom] = 7  # a second HUD counter, first seen later
        g[30, x] = 9
        return g

    m = NoOpMemo()
    m.record(f(10, 40, 20), 1, None, f(10, 39, 20))  # learns the top band only
    assert m.is_known_noop(f(10, 30, 20), 1)
    m.record(f(20, 39, 20), 2, None, f(20, 39, 19))  # learns the bottom band
    # the first entry, keyed before the bottom band existed, must still match
    assert m.is_known_noop(f(10, 5, 3), 1)
    assert m.known_noops(f(20, 1, 1)) == ["2"]


def test_level_change_counts_as_live():
    m = NoOpMemo()
    m.record(frame(5), 6, {"x": 1, "y": 1}, frame(5), level_changed=True)
    assert not m.is_known_noop(frame(5), 6, {"x": 1, "y": 1})
