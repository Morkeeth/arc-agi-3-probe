"""Grid helpers: change classification, HUD masking, click targets.

A grid is a 2D integer array (the last frame of a FrameDataRaw, or Duck's
tuple-of-tuples grid). Everything here is pure and has no engine imports.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

# A change that touches a frame edge and is at most this many cells thick is
# treated as HUD (timer bar, step counter). Duck's own prompt warns about the
# same thing: a strip flush against an edge that shrinks every step.
DEFAULT_HUD_THICKNESS = 4

NONE = "none"
HUD_ONLY = "hud_only"
GAMEPLAY = "gameplay"


def as_grid(grid) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int16)
    if arr.ndim == 3:  # a stack of frames: take the final visible one
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D grid, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class Change:
    kind: str  # NONE, HUD_ONLY or GAMEPLAY
    n_cells: int
    n_gameplay_cells: int
    bbox: tuple[int, int, int, int] | None  # r0, c0, r1, c1 inclusive, all changed cells
    thin_edge: str | None = None  # side name when the unmasked cells hug one edge

    @property
    def gameplay_changed(self) -> bool:
        return self.kind == GAMEPLAY


def thin_edge_side(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int], thickness: int) -> str | None:
    """Name the edge when the cells form one thin band touching it.

    A timer bar, a step counter and a click counter all look like this. A
    single sighting is not proof: see NoOpMemo, which promotes a side to HUD
    only after it has been seen alone more than once.
    """
    h, w = shape
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
    if r0 == 0 and r1 < thickness:
        return f"top{r1 + 1}"
    if r1 == h - 1 and r0 > h - 1 - thickness:
        return f"bottom{h - r0}"
    if c0 == 0 and c1 < thickness:
        return f"left{c1 + 1}"
    if c1 == w - 1 and c0 > w - 1 - thickness:
        return f"right{w - c0}"
    return None


def _is_edge_strip(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int], thickness: int) -> bool:
    return thin_edge_side(rows, cols, shape, thickness) is not None


def hud_candidate_cells(before, changed: np.ndarray) -> np.ndarray:
    """The changed cells plus the same-color component each one sat in.

    A shrinking bar is one component, so the whole bar is covered after one
    tick. Unrelated objects on the same row are not, which matters: bp35 has a
    live click target on row 1 under the top HUD band.
    """
    g = as_grid(before)
    out = changed.copy()
    seen_colors: set[int] = set()
    for r, c in zip(*np.nonzero(changed)):
        color = int(g[r, c])
        if color in seen_colors:
            continue
        seen_colors.add(color)
        for cells in components(g, ignore=[]):
            if (int(r), int(c)) in cells and int(g[cells[0]]) == color:
                for y, x in cells:
                    out[y, x] = True
                break
    return out


def classify_change(
    before,
    after,
    *,
    hud_thickness: int = DEFAULT_HUD_THICKNESS,
    hud_mask: np.ndarray | None = None,
) -> Change:
    """Classify the difference between two grids.

    Only cells in hud_mask are treated as HUD. A thin edge band is reported in
    Change.thin_edge as a candidate, but still counts as gameplay until the
    caller has learned it. Calling a change dead on one sighting is how a
    cursor marker gets mistaken for a timer.
    """
    a, b = as_grid(before), as_grid(after)
    if a.shape != b.shape:
        # A resize is always a real change (new level, new camera).
        return Change(GAMEPLAY, int(max(a.size, b.size)), int(max(a.size, b.size)), None)
    diff = a != b
    n = int(diff.sum())
    if n == 0:
        return Change(NONE, 0, 0, None)
    rows, cols = np.nonzero(diff)
    bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
    gp = diff if hud_mask is None else (diff & ~hud_mask)
    n_gp = int(gp.sum())
    if n_gp == 0:
        return Change(HUD_ONLY, n, 0, bbox)
    gr, gc = np.nonzero(gp)
    side = thin_edge_side(gr, gc, a.shape, hud_thickness) if hud_thickness > 0 else None
    return Change(GAMEPLAY, n, n_gp, bbox, side)


def state_key(grid, hud_mask: np.ndarray | None = None) -> str:
    """Stable hash of a grid with HUD cells blanked out."""
    g = as_grid(grid).copy()
    if hud_mask is not None and hud_mask.shape == g.shape:
        g[hud_mask] = -1
    h = hashlib.blake2b(g.tobytes(), digest_size=10)
    h.update(str(g.shape).encode())
    return h.hexdigest()


def background_color(grid) -> int:
    g = as_grid(grid)
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[counts.argmax()])


def components(grid, *, ignore: Iterable[int] = ()) -> list[list[tuple[int, int]]]:
    """4-connected same-color components, skipping colors in ignore."""
    g = as_grid(grid)
    h, w = g.shape
    skip = set(int(x) for x in ignore)
    seen = np.zeros_like(g, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) in skip:
                continue
            color = g[r, c]
            cells = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] == color:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            out.append(cells)
    return out


def click_targets(
    grid,
    *,
    max_targets: int = 3,
    hud_thickness: int = DEFAULT_HUD_THICKNESS,
    min_cells: int = 1,
    max_frac: float = 0.25,
) -> list[tuple[int, int]]:
    """Pick up to max_targets (row, col) cells to click, one per object.

    Objects are non-background components that do not sit in an edge HUD band
    and are not huge (walls, floors). Smallest objects first. That order is a
    guess, not a measured result: the census run on 2026-09-22 found 41 of 56
    chosen click cells did nothing, so this chooser is the weak part.
    Each target is the member cell nearest the component centroid.
    """
    g = as_grid(grid)
    h, w = g.shape
    bg = background_color(g)
    picks: list[tuple[int, tuple[int, int]]] = []
    for cells in components(g, ignore=[bg]):
        n = len(cells)
        if n < min_cells or n > max_frac * g.size:
            continue
        rs = np.array([c[0] for c in cells])
        cs = np.array([c[1] for c in cells])
        if hud_thickness > 0 and _is_edge_strip(rs, cs, (h, w), hud_thickness):
            continue
        cy, cx = rs.mean(), cs.mean()
        best = min(cells, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
        picks.append((n, best))
    picks.sort(key=lambda t: (t[0], t[1]))
    return [p for _, p in picks[:max_targets]]
