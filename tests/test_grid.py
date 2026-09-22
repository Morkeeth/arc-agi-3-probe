import numpy as np

from deadprobe.grid import GAMEPLAY, HUD_ONLY, NONE, classify_change, click_targets, state_key


def board():
    g = np.zeros((64, 64), dtype=int)
    g[0:2, 0:40] = 8  # timer bar flush with the top edge
    g[30:34, 30:34] = 9  # player
    g[50, 50] = 11  # small token
    return g


def test_no_change():
    g = board()
    assert classify_change(g, g.copy()).kind == NONE


def test_timer_tick_is_a_hud_candidate_not_a_hud_fact():
    # One sighting is not proof. classify_change flags the side; only the memo
    # promotes it after a second sighting.
    a = board()
    b = a.copy()
    b[0:2, 39] = 0
    c = classify_change(a, b)
    assert c.kind == GAMEPLAY and c.thin_edge == "top2" and c.n_cells == 2


def test_player_move_plus_timer_is_gameplay():
    a = board()
    b = a.copy()
    b[0:2, 39] = 0
    b[30:34, 30] = 0
    b[30:34, 34] = 9
    c = classify_change(a, b)
    assert c.kind == GAMEPLAY and c.n_gameplay_cells == 10  # timer cells are not masked yet


def test_thick_edge_change_is_gameplay():
    # a change against the edge that is thicker than the HUD band is real
    a = board()
    b = a.copy()
    b[0:10, 58:64] = 3  # 6 wide: thicker than the 4-cell band
    assert classify_change(a, b).kind == GAMEPLAY


def test_known_hud_mask_blanks_cells():
    a = board()
    b = a.copy()
    b[20, 20] = 4  # interior, would be gameplay
    mask = np.zeros_like(a, dtype=bool)
    mask[20, 20] = True
    assert classify_change(a, b, hud_mask=mask).kind == HUD_ONLY


def test_state_key_ignores_masked_cells():
    a = board()
    b = a.copy()
    b[0:2, 39] = 0
    assert state_key(a) != state_key(b)
    mask = np.zeros_like(a, dtype=bool)
    mask[0:2, :] = True
    assert state_key(a, mask) == state_key(b, mask)


def test_click_targets_skip_hud_and_prefer_small():
    t = click_targets(board(), max_targets=3)
    assert t[0] == (50, 50)  # 1-cell token first
    assert (31, 31) in t or (32, 32) in t or (31, 32) in t or (32, 31) in t
    assert all(r >= 2 for r, _ in t)  # never the timer strip


def test_resized_grid_is_gameplay():
    assert classify_change(np.zeros((8, 8)), np.zeros((16, 16))).kind == GAMEPLAY
