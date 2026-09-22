import pytest

from deadprobe.score import breakeven_saving, game_score, level_score


def test_level_score_matches_official_formula():
    assert level_score(10, 10) == 100
    assert level_score(10, 20) == 25
    assert level_score(20, 10) == 115  # cap
    assert level_score(10, 0) == 0


def test_game_score_weights_and_cap():
    # two levels, weights 1 and 2, only level 1 done at baseline
    assert game_score([10, 10], [10, 5], 1) == pytest.approx(100 * 1 / 3)
    # both done, level 2 at double baseline
    assert game_score([10, 10], [10, 20], 2) == pytest.approx((100 + 25 * 2) / 3)


def test_probe_cost_hurts_when_it_saves_nothing():
    # ls20 level 1 baseline is 22 (public games list). An agent at 44 actions
    # scores 25. Six probe actions that save nothing drop it to about 19.4.
    assert breakeven_saving(22, 44, 6) == pytest.approx(level_score(22, 50) - 25)
    assert breakeven_saving(22, 44, 6) < -5
