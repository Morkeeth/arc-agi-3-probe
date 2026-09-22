"""NoOpMemo: remember which (board state, action) pairs changed nothing.

This is the zero-cost half of the lever. It spends no extra actions. It only
watches actions the agent takes anyway, and answers one question: has this
exact action already been tried in this exact board state and done nothing?

Two rules keep it honest.

1. Dead is state dependent. LEFT against a wall is a no-op; LEFT one cell away
   is not. So an action is never marked dead globally. The key is the board
   with learned HUD cells blanked, plus the action (with click coordinates).
2. A thin change against a frame edge is a HUD candidate, not a HUD fact. It
   becomes HUD only after promote_after sightings. One sighting could be a
   cursor marker. Measured on the public games: vc33 and su15 flip one corner
   cell on every click, and that really is a counter, but the promotion rule
   is what tells that case apart from a marker that means something.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .grid import (
    DEFAULT_HUD_THICKNESS,
    GAMEPLAY,
    Change,
    as_grid,
    classify_change,
    hud_candidate_cells,
    state_key,
)


def action_key(action_id: int, data: dict[str, Any] | None = None) -> str:
    """ACTION6 keys include the clicked cell; simple actions do not."""
    if int(action_id) == 6:
        d = data or {}
        return f"6@{int(d.get('y', d.get('row', 0)))},{int(d.get('x', d.get('col', 0)))}"
    return str(int(action_id))


@dataclass
class NoOpMemo:
    hud_thickness: int = DEFAULT_HUD_THICKNESS
    promote_after: int = 2  # sightings of one edge side before it counts as HUD
    max_records: int = 1000  # about 8 MB per game at 64x64, before and after
    hud_mask: np.ndarray | None = None
    noops: dict[str, set[str]] = field(default_factory=dict)  # state key -> action keys
    live: dict[str, set[str]] = field(default_factory=dict)
    n_observed: int = 0
    n_noop: int = 0
    n_repeat_noop: int = 0  # no-ops the memo already knew about: pure waste
    n_skipped: int = 0
    _sides: Counter = field(default_factory=Counter, repr=False)
    _pending: dict = field(default_factory=dict, repr=False)
    _records: list = field(default_factory=list, repr=False)

    # --- learning -------------------------------------------------------
    def _promote(self, before: np.ndarray, after: np.ndarray, change: Change) -> bool:
        """Maybe grow the HUD mask from a repeated thin edge change.

        Candidate cells are kept per side until the side is promoted. Then the
        mask takes all of them plus their same-color components: a shrinking
        bar in one go, without blanking unrelated objects on the same row.
        """
        if change.thin_edge is None or before.shape != after.shape:
            return False
        side = change.thin_edge
        diff = before != after
        if self.hud_mask is not None and self.hud_mask.shape == diff.shape:
            diff &= ~self.hud_mask
        if not diff.any():
            return False
        cand = hud_candidate_cells(before, diff)
        pend = self._pending.get(side)
        self._pending[side] = cand if pend is None or pend.shape != cand.shape else (pend | cand)
        self._sides[side] += 1
        if self._sides[side] < self.promote_after:
            return False
        if self.hud_mask is None or self.hud_mask.shape != cand.shape:
            self.hud_mask = np.zeros_like(cand, dtype=bool)
        self.hud_mask |= self._pending[side]
        return True

    def key(self, grid) -> str:
        return state_key(grid, self.hud_mask)

    def classify(self, before, after) -> Change:
        return classify_change(before, after, hud_thickness=self.hud_thickness, hud_mask=self.hud_mask)

    # --- recording ------------------------------------------------------
    def record(
        self,
        before,
        action_id: int,
        data: dict[str, Any] | None,
        after,
        *,
        level_changed: bool = False,
    ) -> Change:
        """Record one real transition. Returns its classification.

        The classification uses the mask as it stands after this transition,
        so the second tick of a timer is already called HUD.
        """
        b, a = as_grid(before), as_grid(after)
        ak = action_key(action_id, data)
        first = self.classify(b, a)
        grew = self._promote(b, a, first)
        change = self.classify(b, a) if grew else first
        is_live = bool(level_changed) or change.kind == GAMEPLAY
        self.n_observed += 1
        if not is_live:
            if self.is_known_noop(b, action_id, data):
                self.n_repeat_noop += 1
            self.n_noop += 1
        self._records.append((b.astype(np.int8), a.astype(np.int8), ak, bool(level_changed)))
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
        """The HUD mask grew, so every stored key and verdict is stale."""
        self.noops, self.live = {}, {}
        for b, a, ak, level_changed in self._records:
            is_live = level_changed or self.classify(b, a).kind == GAMEPLAY
            self._index(b, ak, is_live)

    # --- reading --------------------------------------------------------
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
            "hud_cells": int(self.hud_mask.sum()) if self.hud_mask is not None else 0,
        }
