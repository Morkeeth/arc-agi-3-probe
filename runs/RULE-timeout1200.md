# RULE: lane T, analyzer_timeout 1200 s against 900 s, 25 public games

Written 2026-09-23 about 19:20 UTC, before any lane T run. Nothing below changes after the first push. The verdict comes from `l4_verdict.py --treat to1200`. Nobody types a verdict number by hand.

## What is compared

- Treatment `to1200`: `oscarmorke/ab-to1200-s1` and `-s2`, built by `kaggle/ab-to/build.py` from `ab-q38-control-s1`. One line changes: `bm.solver.analyzer_timeout = 1200.0` in place of `900.0`. The run label AB_ID changes too. sha256 (first 12): s1 `9e9e237ef326`, s2 `4bd52937ea97`.
- Why this lever: it is the only code difference between our base and the top-scoring public notebook `chiakazirim/duck-qwen3-8-tuned` (notebook diff, 23 Sep).
- Control: the same 3 control runs as lane 4 (`control-s1`, `control-s2r`, `control-s3r`, mean 8.59, 38.0 levels per run).

## Decision rule

- WIN if the to1200 mean score is strictly above 8.59 AND its levels per run are at least 38.0. Read only with 2 valid runs.
- LEDGER keep if the gap is above the control range (4.96) and levels do not fall. Kill if it is below control by more than that range. Pending otherwise. "Lever acted" is not a kill test here: a timeout changes only calls that run long, and actions may not move.
- Submission: a WIN makes `ab-to1200-s1` the candidate for the next day's submission through the daily loop. Nothing is submitted from this run by a machine.

## Sealed predictions

- T1: the mean score moves by less than the control range. LEDGER reads pending. RULE WIN at about 45%.
- T2: model calls per game-run stay within 10% of control (a timeout cuts off few calls).
- T3: if calls hit 900 s in control, the longer timeout gives those turns an answer instead of a retry. Levels per run rise by at most 2.

## Budget

`kaggle quota` 19:1x UTC: 23.93 h used, 6.07 h left, refresh 2026-09-26 00:00 UTC. 2 runs are about 5 GPU hours. This uses the rest of the week's quota.

## Amendments

None yet.
