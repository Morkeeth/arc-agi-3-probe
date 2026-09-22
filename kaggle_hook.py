"""Paste this into the Duck notebook's section 6 "Customization hook" cell.

Two ways to get the module onto Kaggle, neither of which needs the TAAF source
dataset to change:
  A. attach a private dataset built from this repo (see ARC-PREP-RETURN.md), or
  B. paste deadprobe/*.py into one notebook cell above this one.

Set MODE to pick the arm. Run the control arm first, on the same games and the
same model, or the comparison means nothing.
"""

MODE = "off"  # off (control) | guard | probe | probe+guard

import sys
from pathlib import Path

# Option A. Kaggle mounts a dataset at /kaggle/input/<slug> or at
# /kaggle/input/datasets/<owner>/<slug>; find it rather than guess.
for _init in Path("/kaggle/input").rglob("deadprobe/__init__.py"):
    sys.path.insert(0, str(_init.parent.parent))
    break

if MODE != "off":
    from deadprobe import duck_patch

    duck_patch.install(MODE, duck_patch.ProbeConfig(budget_level1=6, budget_later=3, max_clicks=3))

# Offline ablation: a short game list and a hard action cap keep a run inside
# one GPU session. Remove both for a run you intend to submit.
# bm.games = [g for g in bm.games if g.env_name.split("-")[0] in ("ls20", "vc33", "sb26", "cd82", "ka59")]
# bm.solver.max_actions_per_game = 300
bm.label = f"{bm.label}-deadprobe-{MODE}"
