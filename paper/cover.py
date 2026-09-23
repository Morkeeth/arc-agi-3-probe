"""Cover image for the paper: one game frame from a Duck run, in the official 16-colour ARC palette.

    python paper/cover.py                       # render paper/cover.png from paper/cover_frame.json
    python paper/cover.py --events <p0_events.jsonl>   # re-extract the frame from a run's events file first

Source frame: the last action frame of sb26 in run control-s2r (runs/ab-2026-09-23/control-s2r/artifacts/
sb26-7fbdac44_p0_events.jsonl). That artifact is gitignored (runs/*), so the one frame the cover uses is
copied into paper/cover_frame.json, which is tracked. With --events the frame is re-extracted and must
equal the tracked copy, or the script fails. The palette is arc_agi.rendering.COLOR_MAP, never retyped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAME = HERE / "cover_frame.json"
PNG = HERE / "cover.png"
DEFAULT_EVENTS = HERE.parent / "runs" / "ab-2026-09-23" / "control-s2r" / "artifacts" / "sb26-7fbdac44_p0_events.jsonl"


def palette() -> dict[int, str]:
    from arc_agi.rendering import COLOR_MAP  # the toolkit's palette, 0 to 15

    if sorted(COLOR_MAP) != list(range(16)):
        raise SystemExit(f"COLOR_MAP has keys {sorted(COLOR_MAP)}, expected 0..15")
    return {k: v[:7] for k, v in COLOR_MAP.items()}  # drop the alpha byte


def extract(events: Path) -> dict:
    rows = [json.loads(ln) for ln in events.read_text().splitlines() if ln.strip()]
    acts = [r for r in rows if r.get("type") == "action"]
    last = acts[-1]
    return {
        "source": "/".join(events.parts[events.parts.index("runs"):]) if "runs" in events.parts else events.name,
        "game": events.name.split("_")[0],
        "action_num": last["action_num"],
        "action": last["action_display"],
        "level": last["level"],
        "state": last["state"],
        "board": last["board"],
    }


def render(frame: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    pal = palette()
    board = frame["board"]
    h, w = len(board), len(board[0])
    fig = plt.figure(figsize=(12, 6.75), dpi=160)  # 1920 x 1080
    fig.patch.set_facecolor("#111111")
    ax = fig.add_axes([0.46, 0.06, 0.50, 0.88])
    ax.imshow(board, cmap=ListedColormap([pal[i] for i in range(16)]), vmin=-0.5, vmax=15.5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#444444")
    tx = fig.text
    tx(0.04, 0.84, "Model time, not actions", color="#FFFFFF", fontsize=24, weight="bold", va="top")
    tx(0.04, 0.72, "The binding constraint for a local LLM agent\non ARC-AGI-3 under Kaggle limits", color="#DDDDDD", fontsize=14, va="top")
    tx(0.04, 0.50, f"Game {frame['game']}, level {frame['level']}, action {frame['action_num']}: {frame['action']}\n"
       f"State {frame['state']}. Frame {h} x {w}, 16-colour ARC palette.",
       color="#AAAAAA", fontsize=10, va="top", family="monospace")
    tx(0.04, 0.30, "Every game-run stopped at the 7,920 s cap unless it won.\nModel time was 99% of wall time.",
       color="#88D8F1", fontsize=12, va="top")
    tx(0.04, 0.10, "ARC Prize 2026 paper track. Oscar Morke.", color="#888888", fontsize=10, va="top")
    fig.savefig(out, facecolor=fig.get_facecolor())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", nargs="?", const=str(DEFAULT_EVENTS), help="re-extract the frame from this events file")
    ap.add_argument("--out", default=str(PNG))
    a = ap.parse_args(argv)
    if a.events:
        fresh = extract(Path(a.events))
        if FRAME.exists():
            old = json.loads(FRAME.read_text())
            if old != fresh:
                print(f"FAIL {FRAME.name} differs from the frame in {a.events}", file=sys.stderr)
                return 1
        FRAME.write_text(json.dumps(fresh, separators=(",", ":")) + "\n")
    if not FRAME.exists():
        print(f"FAIL {FRAME} missing; run with --events <p0_events.jsonl>", file=sys.stderr)
        return 1
    render(json.loads(FRAME.read_text()), Path(a.out))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
