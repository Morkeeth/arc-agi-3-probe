"""NoOpMemo: remember which (board state, action) pairs changed nothing.

This is the zero-cost half of the lever. It spends no extra actions. It only
watches actions the agent takes anyway, and answers one question: has this
exact action already been tried in this exact board state and done nothing?

Dead is state dependent. LEFT against a wall is a no-op; LEFT one cell away is
not. So the memo never marks an action dead globally. The key is the board with
learned HUD cells blanked, plus the action (with click coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .grid import DEFAULT_HUD_THICKNESS, GAMEPLAY, HUD_ONLY, Change, as_grid, classify_change, edge_band, state_key


def action_key(action_id: int, data: dict[str, Any] | None = None) -> str:
    """ACTION6 keys include the clicked cell; simple actions do not."""
    if int(action_id) == 6:
        d = data or {}
        return f"6@{int(d.get('y', d.get('row', 0)))},{int(d.get('x', d.get('col', 0)))}"
    return str(int(action_id))


@dataclass
class NoOpMemo:
    hud_thickness: int = DEFAULT_HUD_THICKNESS
    hud_mask: np.ndarray | None = None
    noops: dict[str, set[str]] = field(default_factory=dict)  # state key -> action keys
    live: dict[str, set[str]] = field(default_factory=dict)
    n_observed: int = 0
    n_noop: int = 0
    n_repeat_noop: int = 0  # no-ops the memo already knew about: pure waste
    n_skipped: int = 0
    max_records: int = 2000  # about 8 MB per game at 64x64
    _records: list = field(default_factory=list, repr=False)

    def _learn_hud(self, before: np.ndarray, after: np.ndarray, change: Change) -> bool:
        """Grow the HUD mask from a HUD-only change. True if it grew.

        The whole edge band is masked (see grid.edge_band), so a timer bar
        matches across ticks it has not shown yet.
        """
        if change.kind != HUD_ONLY or before.shape != after.shape:
            return False
        diff = before != after
        if self.hud_mask is None or self.hud_mask.shape != diff.shape:
            self.hud_mask = np.zeros_like(diff, dtype=bool)
        fresh = diff & ~self.hud_mask
        if not fresh.any():
            return False
        rows, cols = np.nonzero(fresh)
        band = edge_band(rows, cols, diff.shape, self.hud_thickness)
        self.hud_mask |= band if band is not None else fresh
        return True

    def key(self, grid) -> str:
        return state_key(grid, self.hud_mask)

    def classify(self, before, after) -> Change:
        return classify_change(before, after, hud_thickness=self.hud_thickness, hud_mask=self.hud_mask)

    def record(self, before, action_id: int, data: dict[str, Any] | None, after, *, level_changed: bool = False) -> Change:
        """Record one real transition. Returns its classification."""
        b, a = as_grid(before), as_grid(after)
        change = self.classify(b, a)
        ak = action_key(action_id, data)
        is_live = level_changed or change.kind == GAMEPLAY
        self.n_observed += 1
        if not is_live:
            if self.is_known_noop(b, action_id, data):
                self.n_repeat_noop += 1
            self.n_noop += 1
        grew = self._learn_hud(b, a, change)
        self._records.append((b.astype(np.int8), ak, is_live))
        if len(self._records) > self.max_records:
            del self._records[: len(self._records) - self.max_records]
        if grew:
            self._reindex()
        else:
            self._index(b, ak, is_live)
        return change

    def _index(self, grid: np.ndarray, ak: str, is_live: bool) -> None:
        k = self.key(grid)
        (self.live if is_live else self.noops).setdefault(k, set()).add(ak)

    def _reindex(self) -> None:
        """The HUD mask grew, so every stored key is stale. Rebuild."""
        self.noops, self.live = {}, {}
        for g, ak, is_live in self._records:
            self._index(g, ak, is_live)

    def is_known_noop(self, grid, action_id: int, data: dict[str, Any] | None = None) -> bool:
        if int(action_id) == 0:  # never block RESET
            return False
        ak = action_key(action_id, data)
        k = self.key(grid)
        return ak in self.noops.get(k, ()) and ak not in self.live.get(k, ())

    def known_noops(self, grid) -> list[str]:
        k = self.key(grid)
        return sorted(self.noops.get(k, set()) - self.live.get(k, set()))

    def stats(self) -> dict[str, int]:
        return {
            "observed": self.n_observed,
            "noop": self.n_noop,
            "repeat_noop": self.n_repeat_noop,
            "skipped": self.n_skipped,
        }
