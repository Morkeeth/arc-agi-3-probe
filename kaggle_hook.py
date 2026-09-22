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

# Option A. The dataset holds the module files flat (no folder, so no
# zip or tar handling is involved). Find duck_patch.py wherever Kaggle mounted
# it and rebuild the package in /kaggle/working.
import shutil

if MODE != "off":
    _hit = next(Path("/kaggle/input").rglob("duck_patch.py"), None)
    if _hit is None:
        raise RuntimeError("deadprobe dataset not mounted: attach oscarmorke/arc-agi-3-probe")
    _pkg = Path("/kaggle/working/deadprobe_pkg/deadprobe")
    _pkg.mkdir(parents=True, exist_ok=True)
    for _f in _hit.parent.glob("*.py"):
        shutil.copy(_f, _pkg / _f.name)
    sys.path.insert(0, str(_pkg.parent))

if MODE != "off":
    from deadprobe import duck_patch

    duck_patch.install(MODE, duck_patch.ProbeConfig(budget_level1=6, budget_later=3, max_clicks=3))

# Offline ablation on 5 public games. Both arms use the same list. Section 7
# rebuilds bm.games from the bundled files in an interactive run, so the staged
# notebooks apply DEADPROBE_GAMES there, right after that rebuild.
# Set DEADPROBE_GAMES = None to play all 25 public games.
DEADPROBE_GAMES = ("sc25", "g50t", "cd82", "tu93", "ls20")
bm.label = f"{bm.label}-deadprobe-{MODE}"
