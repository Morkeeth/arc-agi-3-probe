"""Active probe: spend a small fixed budget of real actions at level start.

Every probe action is a real, scored action. RESET also counts (checked in
arc_agi/scorecard.py: inc_reset_count adds to actions). So the probe never
resets to undo itself. It walks forward and records what each action did
relative to the board just before it.

The core takes a step callable so it can drive the toolkit env, Duck's
session, or a test fake with the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .grid import DEFAULT_HUD_THICKNESS, GAMEPLAY, as_grid, click_targets
from .memo import NoOpMemo, action_key

SIMPLE_IDS = (1, 2, 3, 4, 5, 7)
NAMES = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 6: "MOUSE", 7: "ACTION7"}


@dataclass(frozen=True)
class StepOut:
    """What a step callable must return."""

    grid: Any
    levels_completed: int
    state: str  # "NOT_FINISHED", "WIN", "GAME_OVER", ...


StepFn = Callable[[int, dict[str, Any] | None], StepOut]


@dataclass
class ProbeResult:
    action_id: int
    data: dict[str, Any] | None
    kind: str  # none, hud_only, gameplay
    n_cells: int

    @property
    def display(self) -> str:
        if self.action_id == 6 and self.data:
            return f"MOUSE(row={self.data['y']}, col={self.data['x']})"
        return NAMES.get(self.action_id, f"ACTION{self.action_id}")

    @property
    def live(self) -> bool:
        return self.kind == GAMEPLAY


@dataclass
class ProbeReport:
    results: list[ProbeResult] = field(default_factory=list)
    stop_reason: str = "budget"  # budget, exhausted, level_up, game_over, win

    @property
    def cost(self) -> int:
        return len(self.results)

    @property
    def dead(self) -> list[str]:
        return [r.display for r in self.results if not r.live]

    @property
    def live_actions(self) -> list[str]:
        return [r.display for r in self.results if r.live]

    def note(self) -> str:
        """One line for the LLM prompt. Facts only, with the state caveat."""
        if not self.results:
            return ""
        parts = [f"{r.display}={'changed gameplay' if r.live else r.kind.replace('_', ' ')}" for r in self.results]
        return (
            f"Level-start probe ({self.cost} actions, already spent, in this order): "
            + "; ".join(parts)
            + ". Each result is relative to the board just before that action; an action with no effect here may work elsewhere."
        )


def plan_candidates(
    grid,
    available: list[int],
    *,
    max_clicks: int = 3,
    hud_thickness: int = DEFAULT_HUD_THICKNESS,
    skip: set[str] | None = None,
) -> list[tuple[int, dict[str, Any] | None]]:
    """Order: simple actions by id, then clicks on the smallest objects."""
    skip = skip or set()
    out: list[tuple[int, dict[str, Any] | None]] = []
    for aid in SIMPLE_IDS:
        if aid in available and action_key(aid) not in skip:
            out.append((aid, None))
    if 6 in available and max_clicks > 0:
        for r, c in click_targets(grid, max_targets=max_clicks, hud_thickness=hud_thickness):
            data = {"x": int(c), "y": int(r)}
            if action_key(6, data) not in skip:
                out.append((6, data))
    return out


def probe_level(
    step: StepFn,
    grid,
    available: list[int],
    levels_completed: int,
    *,
    budget: int = 6,
    max_clicks: int = 3,
    memo: NoOpMemo | None = None,
    skip: set[str] | None = None,
) -> ProbeReport:
    """Try each candidate once, in order, until budget or a level change.

    memo, when given, records every probe transition, so the zero-cost guard
    knows about them later.
    """
    memo = memo if memo is not None else NoOpMemo()
    report = ProbeReport()
    cur = as_grid(grid)
    candidates = plan_candidates(cur, available, max_clicks=max_clicks, hud_thickness=memo.hud_thickness, skip=skip)
    if not candidates:
        report.stop_reason = "exhausted"
        return report
    seen: list[tuple[Any, Any, bool]] = []  # before, after, level_up, for the post pass
    for aid, data in candidates:
        if report.cost >= budget:
            report.stop_reason = "budget"
            break
        out = step(aid, data)
        nxt = as_grid(out.grid)
        level_up = out.levels_completed > levels_completed
        change = memo.record(cur, aid, data, nxt, level_changed=level_up)
        kind = GAMEPLAY if level_up else change.kind
        report.results.append(ProbeResult(aid, data, kind, change.n_cells))
        seen.append((cur, nxt, level_up))
        cur = nxt
        if out.state == "WIN":
            report.stop_reason = "win"
            break
        if level_up:
            report.stop_reason = "level_up"
            break
        if out.state == "GAME_OVER":
            report.stop_reason = "game_over"
            break
    else:
        report.stop_reason = "exhausted"
    # Cold start: the first action of a game is judged before the memo has
    # learned the HUD. Judge every probe result again with the final mask.
    for result, (before, after, level_up) in zip(report.results, seen):
        result.kind = GAMEPLAY if level_up else memo.classify(before, after).kind
    return report
