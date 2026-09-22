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

    @property
    def gameplay_changed(self) -> bool:
        return self.kind == GAMEPLAY


def edge_band(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int], thickness: int) -> np.ndarray | None:
    """If the cells form one thin band touching a frame edge, return a mask of
    that whole band (full width or height, same thickness). Else None.

    Masking the whole band, not only the cells seen changing, is what lets a
    shrinking timer bar match across ticks it has not shown yet.
    """
    h, w = shape
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
    band = np.zeros(shape, dtype=bool)
    if r0 == 0 and r1 < thickness:
        band[0 : r1 + 1, :] = True
    elif r1 == h - 1 and r0 > h - 1 - thickness:
        band[r0:, :] = True
    elif c0 == 0 and c1 < thickness:
        band[:, 0 : c1 + 1] = True
    elif c1 == w - 1 and c0 > w - 1 - thickness:
        band[:, c0:] = True
    else:
        return None
    return band


def _is_edge_strip(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int], thickness: int) -> bool:
    """True when the cells form one thin band touching one frame edge."""
    return edge_band(rows, cols, shape, thickness) is not None


def classify_change(
    before,
    after,
    *,
    hud_thickness: int = DEFAULT_HUD_THICKNESS,
    hud_mask: np.ndarray | None = None,
) -> Change:
    """Classify the difference between two grids.

    hud_mask: optional boolean mask of cells already known to be HUD. Those
    cells never count as gameplay. Remaining changed cells are HUD only when
    they form one thin strip against an edge.
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
    if hud_thickness > 0 and _is_edge_strip(gr, gc, a.shape, hud_thickness):
        return Change(HUD_ONLY, n, 0, bbox)
    return Change(GAMEPLAY, n, n_gp, bbox)


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
    and are not huge (walls, floors). Smallest objects first: in the public
    games buttons and tokens are small, while large regions are usually
    scenery. Each target is the member cell nearest the component centroid.
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
