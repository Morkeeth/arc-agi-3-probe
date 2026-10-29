# ARC-AGI-3 Probe

**Experiments that record which actions leave an ARC-AGI-3 game state unchanged, so an agent stops repeating the same no-op in the same state.**

- Problem: ARC-AGI-3 agents waste actions on moves that change nothing.
- For: people building agents on the Duck harness; a small Python package (`deadprobe`), an offline harness and Kaggle notebook builders.
- What it found: in the recorded runs the dead-action guard did not beat its control (mean 7.42 against 8.59). The results stay in `runs/` because they narrow the next experiment.
- Try it: `pip install -e '.[dev,arc]'`, then `python harness.py ab --steps 300 --seed 1` with the competition environment files. Started September 2026.


Experiments for reducing wasted actions in ARC-AGI-3 agents. The core package records actions that leave a game state unchanged, then lets an agent avoid repeating the same no-op in the same state.

This repository contains:

- `deadprobe/`: the state comparison, no-op memory, action probes, and scoring code.
- `harness.py`: an offline census, random walk, and guarded-versus-unguarded comparison.
- `kaggle/`: reproducible Kaggle notebook builders and experiment inputs.
- `runs/`: recorded experiment outputs and the rules used to evaluate them.
- `tests/`: unit and integration checks for the probe and experiment builders.

## Run locally

Python 3.11 or later is required. The ARC environment files are not included. Download them from the ARC-AGI-3 Kaggle competition after accepting its rules.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,arc]'

export ARC_ENV_DIR=/absolute/path/to/environment_files
python harness.py census --games ls20,vc33
python harness.py walk --steps 300 --seed 1
python harness.py ab --steps 300 --seed 1
```

Run the test suite with:

```bash
python -m pytest
```

The `ab` command uses a random policy. It checks the mechanism and measures wasted actions; it is not the Duck solver and is not evidence of leaderboard performance.

## Recorded results

The tested dead-action guard did not beat its control. Across three 25-game runs, the recorded mean was 7.42 for the guarded arm and 8.59 for control, a difference of -1.17. See [`runs/ab-2026-09-23/verdict.json`](runs/ab-2026-09-23/verdict.json), which is computed by [`ab_verdict.py`](ab_verdict.py).

Increasing the sequence cap from 8 to 16 also lost in the recorded test: 6.52 versus the same 8.59 control mean, with fewer completed levels. See [`runs/l4-2026-09-23/verdict.json`](runs/l4-2026-09-23/verdict.json), computed by [`l4_verdict.py`](l4_verdict.py).

These are experiment results from the checked-in run files, not public Kaggle leaderboard scores. The repository keeps unsuccessful results because they narrow the next experiment.

## Reproduce a result

The files under `runs/` contain the recorded inputs and outputs used by the verdict scripts. To recompute the two summaries above:

```bash
python ab_verdict.py runs/ab-2026-09-23
python l4_verdict.py
```

Some historical Kaggle experiments depend on competition data, attached datasets, GPU types, or notebook versions that Kaggle controls. Read the matching `runs/RULE-*.md` file before rerunning one of them.
