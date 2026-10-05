# ARC night comparison, 4 to 5 Oct 2026: random policy, guard off against guard on

## One paragraph

The authorized next experiment (causal-memory GPU pair) did not run. It needs Kaggle GPU and Oscar's approval. The fallback was the smallest one-change comparison the committed local harness already holds: a random policy on the 25 public games, with and without the no-op guard, 20 seeds. By the preregistered rule the verdict is BETTER, at the exact line (15 of 20 seeds). The effect is tiny: mean local score 0.0933 against 0.0795 on a 0 to 100 scale. This is a local score from a policy with no model. It says nothing about Duck and nothing about the leaderboard. It does not reopen the closed GPU guard verdicts.

## What ran and what did not

| Thing | Tonight |
|---|---|
| Actual run | Yes. 1,000 offline walks (2 arms x 25 games x 20 seeds), CPU, M1 Pro |
| Local score | Yes. `deadprobe/score.py`, mirror of the official formula |
| Official submission | No |
| Leaderboard movement | No. None claimed, none checked |
| Kaggle call of any kind | No |
| Model call | No |

## Preregistration summary

Full text: `~/.local/state/night-run/2026-10-04/receipts/ARC-START.md`, written before any arm ran. Addendum 1 was written after the baseline arm and before the change arm.

- Base: `18eab01` (newest relevant commit, contains main `d05798a`).
- Baseline: `harness._random_walk(..., guard=False)`, 200 steps per game. Recorded local score 0.1905, source `runs/ab.json`, commit c48569d.
- Change: `guard=True`. One argument differs. No source line in `harness.py` or `deadprobe/` changed (sha256 of both files is stored in each raw file).
- Task set: 25 public games in `seal.json` order. Seeds 0 to 19. Denominator 500 paired walks per arm. No exclusions.
- Metric: per-seed mean game score over 25 games. 20 paired deltas (guard minus plain).
- Rule: BETTER if mean delta above 0, 15 or more of 20 seeds positive, and guard levels at least plain levels. WORSE if mean delta below 0 and 15 or more seeds negative. Else INCONCLUSIVE.
- Budget: 60 minutes of compute. Used: about 15.5 minutes (baseline 174.5 s, determinism rerun 15.7 s, change arm 729.5 s). No cut was needed.

## Result

| | Plain (baseline) | Guard (change) |
|---|---|---|
| Walks | 500 | 500 |
| Mean local score (0 to 100) | 0.0795 | 0.0933 |
| Levels cleared, total | 19 | 25 |
| Walks with a level | 19 | 25 |
| Actions spent | 100,025 | 100,021 |
| Re-rolls by the guard | 0 | 203,539 |
| Timeouts (walk over 120 s) | 0 | 0 |
| Walks ended by the 200 step cap | 500 | 500 |
| Wall seconds, sum | 174.5 | 729.5 |
| Wall seconds, slowest walk | 1.8 | 20.7 |

- Mean of the 20 per-seed deltas: +0.0138. Median: +0.0170.
- Seeds positive 15, negative 4, tied 1. The line was 15.
- Verdict by the preregistered rule: **BETTER**.

Source for every number: `verdict.json` and `drift-diagnosis.json` in this folder, computed from `plain.json` and `guard.json`.

## Read this before using the verdict

- It passed at the exact line. One seed the other way gives INCONCLUSIVE.
- The mean effect is inside its own noise. Standard error of the mean delta is 0.0141, so the mean is 0.98 standard errors from zero (post hoc, `drift-diagnosis.json`). The sign test passes (two-sided p = 0.041). A t-test would not.
- 21 of 25 games score zero in both arms on all 20 seeds. The whole signal comes from four games: sp80, cd82, lf52, m0r0. On those four the guard mean is higher. No game went down.
- Every walk in both arms ended at the 200 step cap. No walk won a game. No walk timed out. So the comparison did not measure a timeout. It measured what a random policy does with 200 actions.
- The guard arm took 4.2 times the wall time. The guard re-rolled 203,539 times and each re-roll hashes the board. The machine was also busy (load average about 27). Wall time does not enter the score here. On a GPU run with a time cap it would. That is the closed result, and it stays closed.
- The policy is random. Duck with a model is a different policy. The 23 Sep GPU verdict for the guard (LOSE, ledger kill, `runs/ab-2026-09-23/verdict.json`) is unchanged by this.

## Drift check

- As preregistered (seed 0 must match the recorded run): **FAILED**. Seed 0 tonight has mean 0.0 and sp80 does not clear.
- Cause: the recorded run was seed 1. My assumption of seed 0 was wrong. `runs/ab.json` does not store its seed.
- Evidence: tonight's seed 1 matches the recorded file on 25 of 25 games in both arms across six fields (score, levels, actions, noop, repeat_noop, observed). Means are equal to the last digit: plain 0.19047619047619047, guard 0.043086598113367895. Every other seed matches on 6 to 9 games. Source: `drift-diagnosis.json`.
- So there is no code or environment drift between 22 Sep and tonight. The walks are deterministic: a second run of seeds 0 and 1 matched on 50 of 50 walks.
- A side finding. The recorded single seed (0.19 against 0.04, guard looks worse) is the most negative of the 20 seeds. The old local number was the worst case, and one sp80 clear decided it.

## Scorer check

`scorer-check.json`. The scorer was shown to go both ways on one known case, the sp80 level-1 clear recorded in `runs/walk.json`:

- Level 1 cleared at the baseline action count: 4.7619. Matches hand arithmetic and the recorded score.
- Same level with twice the actions: 1.1905. Different from the right answer, and matches hand arithmetic.
- Level not cleared: 0.0.
- `pytest tests/test_score.py tests/test_memo.py`: 12 passed.

## Per-seed table

Mean local score over 25 games. Levels are totals over 25 games.

| Seed | Plain | Guard | Delta | Plain levels | Guard levels |
|---|---|---|---|---|---|
| 0 | 0.0000 | 0.0234 | +0.0234 | 0 | 1 |
| 1 | 0.1905 | 0.0431 | -0.1474 | 1 | 1 |
| 2 | 0.0000 | 0.0138 | +0.0138 | 0 | 1 |
| 3 | 0.1905 | 0.1987 | +0.0083 | 1 | 2 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| 5 | 0.0000 | 0.0401 | +0.0401 | 0 | 1 |
| 6 | 0.0208 | 0.0411 | +0.0203 | 1 | 1 |
| 7 | 0.1905 | 0.1979 | +0.0074 | 1 | 2 |
| 8 | 0.0538 | 0.1071 | +0.0534 | 2 | 1 |
| 9 | 0.0924 | 0.0233 | -0.0691 | 1 | 2 |
| 10 | 0.3269 | 0.3678 | +0.0410 | 2 | 2 |
| 11 | 0.2050 | 0.2139 | +0.0088 | 2 | 2 |
| 12 | 0.0342 | 0.0958 | +0.0615 | 1 | 1 |
| 13 | 0.0000 | 0.0411 | +0.0411 | 0 | 1 |
| 14 | 0.0531 | 0.0174 | -0.0357 | 2 | 2 |
| 15 | 0.1905 | 0.1369 | -0.0536 | 1 | 1 |
| 16 | 0.0000 | 0.0502 | +0.0502 | 0 | 1 |
| 17 | 0.0112 | 0.0113 | +0.0001 | 2 | 1 |
| 18 | 0.0212 | 0.0529 | +0.0317 | 1 | 1 |
| 19 | 0.0100 | 0.1905 | +0.1805 | 1 | 1 |

## Per-game table

Mean local score over 20 seeds. Levels are totals over 20 seeds. Wall is the sum over 20 walks. Timeouts are for both arms together.

| Game | Plain | Guard | Plain levels | Guard levels | Guard re-rolls | Plain wall s | Guard wall s | Timeouts |
|---|---|---|---|---|---|---|---|---|
| tn36 | 0 | 0 | 0 | 0 | 0 | 13.1 | 14.3 | 0 |
| lf52 | 0.0042 | 0.0207 | 1 | 3 | 19,014 | 5.8 | 18.0 | 0 |
| cn04 | 0 | 0 | 0 | 0 | 784 | 2.6 | 4.3 | 0 |
| bp35 | 0 | 0 | 0 | 0 | 566 | 12.9 | 22.4 | 0 |
| wa30 | 0 | 0 | 0 | 0 | 12,386 | 2.2 | 4.8 | 0 |
| lp85 | 0 | 0 | 0 | 0 | 30,877 | 16.9 | 171.8 | 0 |
| r11l | 0 | 0 | 0 | 0 | 578 | 11.1 | 23.2 | 0 |
| tu93 | 0 | 0 | 0 | 0 | 2,940 | 2.0 | 3.9 | 0 |
| sp80 | 1.4936 | 1.7753 | 13 | 17 | 430 | 4.4 | 8.9 | 0 |
| m0r0 | 0.0174 | 0.0184 | 1 | 1 | 197 | 3.7 | 10.1 | 0 |
| vc33 | 0 | 0 | 0 | 0 | 30,534 | 12.6 | 130.7 | 0 |
| ar25 | 0 | 0 | 0 | 0 | 287 | 3.4 | 4.3 | 0 |
| ka59 | 0 | 0 | 0 | 0 | 405 | 4.2 | 4.8 | 0 |
| sc25 | 0 | 0 | 0 | 0 | 418 | 6.0 | 6.5 | 0 |
| sk48 | 0 | 0 | 0 | 0 | 127 | 5.5 | 5.0 | 0 |
| dc22 | 0 | 0 | 0 | 0 | 1,590 | 6.7 | 9.0 | 0 |
| cd82 | 0.4728 | 0.5182 | 4 | 4 | 957 | 3.4 | 5.1 | 0 |
| ft09 | 0 | 0 | 0 | 0 | 30,877 | 20.8 | 143.1 | 0 |
| g50t | 0 | 0 | 0 | 0 | 1,978 | 4.3 | 5.5 | 0 |
| ls20 | 0 | 0 | 0 | 0 | 3 | 5.2 | 4.6 | 0 |
| re86 | 0 | 0 | 0 | 0 | 6 | 2.4 | 2.4 | 0 |
| s5i5 | 0 | 0 | 0 | 0 | 30,379 | 8.2 | 65.3 | 0 |
| sb26 | 0 | 0 | 0 | 0 | 7,815 | 7.7 | 16.6 | 0 |
| su15 | 0 | 0 | 0 | 0 | 30,391 | 7.0 | 42.0 | 0 |
| tr87 | 0 | 0 | 0 | 0 | 0 | 2.6 | 3.0 | 0 |

## Limits

- Random policy, no model. A plumbing-level proxy. The harness docstring says the same: only a Duck run on a GPU settles the real question.
- 200 actions per game. No game was won. Scores are near the floor.
- One comparison, one rule, passed at the line. No replication.
- The 25 public games only. No held-out games.
- The authorized question (does a causal record help Duck) is untouched. It still needs the GPU pair.

## Files

- `plain.json`, `guard.json`: raw rows, one per walk (score, levels, actions, blocked, wall_s, timeout, at_step_cap).
- `verdict.json`: arm totals, per-seed and per-game tables, preregistered drift check, verdict.
- `drift-diagnosis.json`: seed match counts and post hoc statistics.
- `scorer-check.json`, `plain.log`, `guard.log`.
- `HOW-TO-RERUN.md`: one command.
- Code: `night_compare.py` on branch `night/arc-compare-20261004`.

## Addendum, 2026-10-05 about 05:10 CEST: a correction and a second look at the GPU pair

Nothing above was edited. Full text: `ARC-RETURN-2.md` (copy in this folder).

- Correction. The first paragraph says the causal experiment "needs Kaggle GPU". True. But the preregistration also said no local model runtime was on this machine. That was wrong. `mlx_lm` 0.31.3 was installed in another lane's worktree (`~/CODE/.worktrees/sim-night-compare-20261004/.venv-model`) about 23 minutes before my check. My check read one PATH and two interpreters and I wrote "on this machine". See Addendum 4 in `PREREG-ARC-START.md`.
- The numbers in this file do not depend on that line. The random-policy comparison, its verdict and its limits are unchanged.
- A local-model version of the causal comparison is not worth an hour. Measured on this Mac from the other lane's server log: about 250 prompt tokens per second on Qwen2.5-7B 4-bit. The Kaggle agent sends about 20,000 prompt tokens per request and about 57 requests per game. That is over an hour per game for one arm, with a different and much weaker model. The GPU pair is the experiment.
- The GPU pair's cheap local proof was rerun tonight: exit 0, 25 games, both arms valid, notebook hashes equal to the seal. Result copy: `pair-dry-run-result-20261005.json`. Scores in that file are synthetic.
- Three things would stop the written sequence before the first push. They are listed in `ARC-RETURN-2.md`.
