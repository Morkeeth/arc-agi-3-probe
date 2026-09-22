"""Figure 1: per-game score, plain Duck against Duck + guard.

Reads the numbers from the run files, so the figure cannot drift from them:
  runs/duck-control/summary.txt, runs/duck-deadprobe/summary.txt  (score, levels)
  runs/duck-deadprobe/log.txt  (the "deadprobe: game=... memo=" lines, guard skips)

Run from the repo root:  python paper/ARC-PAPER-FIG1-AB.py
Writes paper/ARC-PAPER-FIG1-AB.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_numbers import load_runs  # noqa: E402

R = load_runs(HERE.parent)
games = ["tu93", "sc25", "cd82", "ls20", "g50t"]
ctrl = [R.control.games[g].score for g in games]
ctrl_lv = [R.control.games[g].levels for g in games]
grd = [R.guard.games[g].score for g in games]
grd_lv = [R.guard.games[g].levels for g in games]
skips = [R.memo[g]["skipped"] for g in games]
title = (
    f"One run per arm, {len(games)} public ARC-AGI-3 games. "
    f"Mean {R.control.mean:.2f} vs {R.guard.mean:.2f}, "
    f"{sum(ctrl_lv)} levels each." if sum(ctrl_lv) == sum(grd_lv) else
    f"Mean {R.control.mean:.2f} vs {R.guard.mean:.2f}, levels {sum(ctrl_lv)} vs {sum(grd_lv)}."
)

BG = "#fcfcfb"; T1 = "#0b0b0b"; T2 = "#52514e"; C1 = "#2a78d6"; C2 = "#eb6834"
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
x = range(len(games)); w = 0.36
b1 = ax.bar([i - w / 2 - 0.01 for i in x], ctrl, w, color=C1, label="Plain Duck (submitted)")
b2 = ax.bar([i + w / 2 + 0.01 for i in x], grd, w, color=C2, label="Duck + repeat-no-op guard")
for bars, lv in ((b1, ctrl_lv), (b2, grd_lv)):
    for b, l in zip(bars, lv):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{b.get_height():.2f}\n{l} lvl",
                ha="center", va="bottom", fontsize=7.5, color=T2)
ax.set_xticks(list(x)); ax.set_xticklabels([f"{g}\n{s} guard skips" for g, s in zip(games, skips)], fontsize=8.5, color=T1)
ax.set_ylabel("Game score (Kaggle log)", color=T2, fontsize=9); ax.set_ylim(0, 7)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#d6d5d0")
ax.tick_params(colors=T2, labelsize=8); ax.yaxis.grid(True, color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.set_title(title, fontsize=10, color=T1, loc="left")
fig.tight_layout(); fig.savefig(HERE / "ARC-PAPER-FIG1-AB.png", facecolor=BG)
print(f"wrote {HERE / 'ARC-PAPER-FIG1-AB.png'}: {title}")
