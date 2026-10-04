# How to rerun the 4 Oct night comparison

Local only. No Kaggle call, no model, no network. About 15 minutes on the M1 Pro when the machine is busy.

One command, from any directory. It writes to a new folder and refuses to overwrite.

```sh
cd /Users/morkeeth/CODE/.worktrees/arc-night-compare-20261004 && PY=/Users/morkeeth/CODE/arc-agi-3-probe/.venv/bin/python && D=runs/night-compare-rerun-$(date +%Y%m%d-%H%M) && $PY night_compare.py scorer-check && $PY night_compare.py arm --arm plain --out $D/plain.json && $PY night_compare.py arm --arm guard --out $D/guard.json && $PY night_compare.py verdict --dir $D
```

What you get in the new folder:

- `plain.json`: baseline arm, 500 walks (25 games x seeds 0 to 19), one row per walk.
- `guard.json`: change arm, same games and seeds.
- `verdict.json`: both arms, per-seed and per-game tables, drift check, verdict by the preregistered rule.

The walks are deterministic. Scores, levels and actions must match the saved files exactly. Only `wall_s` changes.

Needs: the environment files at `~/CODE/arc-agi-3-probe-data/environment_files` and the venv at `~/CODE/arc-agi-3-probe/.venv` (arc_agi and arcengine already installed).

Preregistration: `~/.local/state/night-run/2026-10-04/receipts/ARC-START.md`.
