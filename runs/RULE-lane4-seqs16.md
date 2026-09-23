# RULE: lane 4, vLLM 16 sequences against 8, 25 public games

Written 2026-09-23 about 16:40 UTC, before any lane 4 run. Nothing below changes after the first push. A change before that goes in a dated amendment. The verdict comes from `ab_verdict.py` functions on the run files. Nobody types a verdict number by hand.

## What is compared

- Treatment `seqs16`: `oscarmorke/ab-l4-seqs16-s1` and `-s2`. Built by `kaggle/ab-l4/build.py` (commit `7a348df`) from the same base as RULE-guard-answer.md. One line differs from the base: `"TAAF_VLLM_MAX_NUM_SEQS": "16"` in place of `"8"`. MODE is `off`, and the read-only counters are installed. sha256 (first 12): seqs16-s1 `f1eb716d8010`, seqs16-s2 `68e7ad727269`.
- Control `seqs8`: the 3 valid control runs of RULE-guard-answer.md (`runs/ab-2026-09-23/control-s1`, `control-s2r`, `control-s3r`). They are the same base with 8 sequences and MODE off. They ran on 23 Sep between 07:52 and 15:53 UTC. The only other difference is the hook's print line and label (`build.py` of each lane). Reusing them saves about 7 GPU hours. The cost: control and treatment do not run in the same time window.
- Games: the 25 public games, in the fork's fixed order.

## Metrics

Per run: mean `final_score` over 25 games, levels summed, and per game actions, model calls and wall seconds (counters). Per arm: the mean of run means, levels per run (mean), actions per game-run, wall seconds per action.

Valid run: as in RULE-guard-answer.md, plus the proof line must show `max_num_seqs=16`. A failure before the proof line is an infrastructure failure and is pushed again (A1 there).

## Decision rule

- WIN if the seqs16 mean score is strictly above the seqs8 mean (8.59) AND seqs16 levels per run are at least seqs8 levels per run (38.0). Read only with 2 valid seqs16 runs.
- LEDGER keep if the seqs16 mean beats seqs8 by more than the seqs8 range (4.96), and levels per run do not fall. Kill if it is below seqs8 by more than that range, or if actions per game-run do not rise (the lever did not act). Pending otherwise.
- Submission: a WIN makes `ab-l4-seqs16-s1` the candidate for the next day's one submission, through the daily loop. Nothing is submitted tonight.

## Sealed predictions

- Q1, the lever acts: actions per game-run rise by 20% or more against seqs8 (seqs8: 11,976 actions over 75 game-runs, 159.7 per game-run).
- Q2, wall seconds per action fall (seqs8: 49.45).
- Q3, time binds: levels per run rise. If Q1 holds and Q3 does not, the model acts faster but not better.
- Q4, my guess: the score gap is inside the seqs8 range, so LEDGER reads pending. RULE WIN at about 55%.
- Q5, risk: 16 sequences may not fit the fixed 5 GiB KV cache. Then vLLM queues requests or fails at start. A start failure is an infrastructure failure.

## Budget

`kaggle quota` 16:3x UTC: 19.19 h used, 10.81 h left, refresh 2026-09-26 00:00 UTC. 2 runs are about 5 GPU hours.

## Amendments

None yet.
