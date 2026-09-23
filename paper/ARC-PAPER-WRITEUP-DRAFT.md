<!--
DRAFT for the ARC Prize 2026 Paper Track. Not submitted, not published.
Writeup body = from the title line to the end of "Sources". Kaggle limit: 1,500 words. Our limit: 1,450.
Word count method (paper/verify_numbers.py, check "words"): take the body, drop HTML comments and
table separator rows, replace every "|" with a space, then count whitespace-separated tokens that
contain at least one letter or digit. Markdown marks alone (#, -, **) are not words.
Every number in the body is recomputed by paper/verify_numbers.py from runs/. Run it after any edit.
Figure: paper/ARC-PAPER-FIG1-AB.png, made by paper/ARC-PAPER-FIG1-AB.py from the same run files.
Before submission: paper/PAPER-SUBMIT-CHECKLIST.md.
-->

# What does a dead action cost an LLM agent in ARC-AGI-3?

### A census of no-op actions, a free guard against repeating them, and an A/B with a decision rule fixed in advance that came back null

Oscar Morke. ARC-AGI-3 submission ref 56470466 (plain Duck baseline).

## 1. What was submitted

The submitted notebook is `oscarmorke/duck-control`, version 1. It is the public Duck harness by jeroencottaar (Tufa Labs), from the Kaggle dataset `jeroencottaar/taaf-kaggle-source-share` (TAAF). We added one hook cell, set to `off`, which installs nothing; its game filter applies to offline runs only (`kaggle/duck-control/`). It runs Qwen3.6-27B-FP8 through vLLM (`runs/duck-control/taaf_setup_env.json`) on one RTX PRO 6000 (the GPU check line in `runs/duck-control/log.txt`). The vLLM wheelhouse and model snapshot are datasets by driessmit1. The agent is their work, not ours.

Our work is a measurement and a guard. The guard was tested as an ablation and was not submitted, because our decision rule said no.

The submission was made on 2026-09-22 at 17:22 UTC. It is COMPLETE and scored 1.22 on the public board (`kaggle competitions submissions arc-prize-2026-arc-agi-3 -v`).

## 2. The question

In ARC-AGI-3, every action counts, including RESET and actions that change nothing (`arc_agi/scorecard.py` in github.com/arcprize/ARC-AGI). Game scores run from 0 to 100%, with human action counts as the reference (Kaggle ARC-AGI-3 Evaluation page). Per level, the ratio of human to agent actions is squared, and later levels weigh more (docs.arcprize.org/methodology; our `deadprobe/score.py`). The square makes waste expensive. Twice the human action count keeps a quarter of the level score.

A dead action leaves the gameplay state unchanged. We asked two things. How common are dead actions on the public games? Does a mechanical guard that refuses to repeat a known dead action help an LLM agent?

## 3. Method

Code: repo `arc-agi-3-probe`, commit `9f8b1de`.

**Classifier** (`deadprobe/grid.py`). It compares frames before and after an action and labels the change none, HUD or gameplay. A thin edge change becomes HUD only after two sightings. HUD cells are masked by connected component, so a live object on the same row stays visible.

**Memo** (`deadprobe/memo.py`). It stores (masked board state, action) pairs that changed nothing. Dead is judged per state, never globally.

**Guard** (`deadprobe/duck_patch.py`, mode `guard`). It patches Duck at runtime from the notebook's hook cell, leaving the TAAF source unchanged. When the model asks for an action that already did nothing in this exact board state, the guard refuses it before it reaches the game. No action is spent. The known no-ops for the current board are added to the prompt as facts. Actions are never removed from the valid list. After three refusals in a row, one repeat goes through, so a stuck model cannot spin until the time cap. A test went red without that bound.

**Tests.** 34 pass with the Duck source present (re-run 2026-09-22 on a fresh download of TAAF). One plays a fake game, built on the real `arcengine`, through the real Duck session with a scripted model. Control spends 6 actions on it and guard spends 4.

## 4. Census: how common are dead actions?

All 25 public games, offline, official toolkit. Source: `runs/census.json`, `runs/walk.json`.

- At the level-1 start, 14 of 81 simple actions changed nothing. 8 of 25 games had at least one. In sc25, UP, DOWN, LEFT and RIGHT were all dead.
- 20 of 56 click targets from our heuristic did nothing. The chooser is weak, and two games flip a counter cell on every click, so both counts are lower bounds.
- A random policy, 200 steps per game: 2,366 of 4,955 observed transitions were no-ops. 1,650 repeated a no-op already seen in the same board state. The guard can skip those for free.
- Four games had zero no-ops: ls20, re86, tn36 and tr87.

## 5. The decision rule

Fixed before either GPU run finished, and not registered publicly: recommend Duck plus guard only if its mean score over the 5 games beats control and it clears at least as many levels. A tie or a loss means submit plain Duck.

The games were sc25, g50t, cd82 and tu93, which had many dead actions, and ls20, which had none and acts as an internal control.

## 6. Result

Both arms ran as private Kaggle notebooks, identical except for the hook mode. Sources: `runs/duck-control/summary.txt`, `runs/duck-deadprobe/summary.txt`, and the `deadprobe:` memo lines in `runs/duck-deadprobe/log.txt`.

| Game | Control levels | Control score | Guard levels | Guard score | Guard skips |
|---|---|---|---|---|---|
| tu93 | 2 of 9 (56 actions) | 5.40 | 2 of 9 (200 actions) | 0.51 | 84 |
| sc25 | 2 of 6 | 0.94 | 0 of 6 | 0.00 | 19 |
| cd82 | 0 of 6 | 0.00 | 1 of 6 | 0.71 | 35 |
| ls20 | 0 of 7 | 0.00 | 1 of 7 | 0.55 | 0 |
| g50t | 0 of 7 | 0.00 | 0 of 7 | 0.00 | 8 |
| Mean | 4 levels | 1.27 | 4 levels | 0.35 | 146 total |

The guard cleared the same number of levels and lost on mean score. By the rule, we submitted plain Duck.

The median points the other way: 0.00 for control, 0.51 for guard. Control's mean comes mostly from tu93, where it cleared 2 levels in 56 actions. The guard arm won ls20, where it skipped nothing, so that win is sampling. We read the result as null, not as harm. Figure 1 shows the per-game scores.

## 7. What the guard costs

The guard refused 146 repeated no-ops. Each refusal saved one real action. The model kept asking for known dead actions anyway: 84 refusals in tu93. The prompt fact did not stop the repeats. A further 10 known repeats ran (memo lines). The three-refusal bound lets some through by design.

"Zero cost" holds for actions only. A refusal returns an error to the model's action script. If nothing else in that script ran, Duck calls the model again. The logs do not count those calls. Token totals cannot measure that cost here: both arms ran to the cap, so tokens track throughput (control 101.53, guard 109.45 generated tokens per second, `summary.txt`). All 10 games ended `gave_up` at Duck's per-game time cap of 7,920 s (both `log.txt` files). Time was the binding limit, not actions. A guard that saves actions but spends time moves cost from one budget to the other.

We think this is why the effect is small. The metric charges actions only on cleared levels. When an agent clears few levels before a time cap, a saved second is worth more than a saved action.

## 8. Limits

- One run per arm, on 5 of 25 public games. Sampling swings per game are larger than any plausible guard effect.
- Only guard mode ran on a GPU. The level-start probe modes were tested offline and never run with Duck.
- Duck reports board change with a full-grid diff. A timer tick makes every action look live, so the model gets a false progress signal. We left that unchanged in the A/B.
- Hidden state: two boards that look alike can differ inside the game, so the guard could skip an action that would have worked.
- The classifier compares final frames only. Some games answer with several animation frames.

## 9. Follow-up: three repeats on a stronger base

We then tested mode `guard+answer`. It adds the HUD-masked board diff, and it answers a refusal inside the batch with no new model call. The rule was committed before any run (`runs/RULE-guard-answer.md`). The base was the public Qwen3.8 fork `wuliao0/duck-qwen3-8-anim-base`, on all 25 public games, with 3 repeats per arm.

Control scored 8.59 (runs 6.43, 11.40 and 7.94) and cleared 114 levels. guard+answer scored 7.42 and cleared 107. The rule says no. The control range, 4.96, is larger than the gap. Model calls were fixed by the time cap (4,254 against 4,200), so the skipped no-ops bought no extra turns, and real actions fell by 28%. Under a time cap, the guard cannot help.

## 10. Process

A human set the goal, the question and the decision rule, and gave each go or stop. A fleet of Claude Code sessions did the build, the tests, the Kaggle runs and this write-up. Each claim was checked at the object before use.

The author is a builder and product person with a hackathon background, who works by directing AI coding agents. The process had failures, and the checks caught them. One agent predicted that a stale notebook pushed earlier would fail on import, and it did (`ModuleNotFoundError`). Another found that the game filter in the hook cell had no effect, because a later cell rebuilt the game list. Both were caught before any number was used. The author made the submission. Outward acts stay with the human.

## Sources

- Repo `arc-agi-3-probe`, commit `9f8b1de`: `deadprobe/`, `harness.py`, `kaggle_hook.py`, `kaggle/`, `runs/*.json`, `runs/duck-*/log.txt`.
- Same repo, commit `0a1bf51`: `runs/duck-*/summary.txt`, `runs/duck-*/taaf_setup_env.json`.
- Branch `paper/arc-writeup`: `paper/verify_numbers.py` rechecks every number; `paper/ARC-PAPER-FIG1-AB.py` draws Figure 1.
- Duck: kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner. TAAF: kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share.
- Toolkit: github.com/arcprize/ARC-AGI and github.com/arcprize/ARCEngine.
