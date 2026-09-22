<!--
DRAFT for the ARC Prize 2026 Paper Track. Generated 2026-09-22. Not submitted, not published.
Writeup body = from the title line to the end of "Sources". It must stay under 1,500 words.
Figure: ARC-PAPER-FIG1-AB.png (same folder), made by ARC-PAPER-FIG1-AB.py. Usable as the cover image.
-->

# What does a dead action cost an LLM agent in ARC-AGI-3?

### A census of no-op actions, a free guard against repeating them, and a pre-registered A/B that came back null

Oscar Morke. ARC-AGI-3 submission ref 56470466 (plain Duck baseline).

## 1. What was submitted

The submitted notebook is `oscarmorke/duck-control`, version 1. It is the public Duck harness by jeroencottaar (Tufa Labs), run without changes, from the Kaggle dataset `jeroencottaar/taaf-kaggle-source-share` (TAAF). It runs Qwen3.6-27B-FP8 through vLLM on an RTX Pro 6000 (`runs/duck-control/taaf_setup_env.json`). The vLLM wheelhouse and the model snapshot are datasets by driessmit1. The agent is their work. This paper does not claim it.

Our work is a measurement and a guard. The guard was tested as an ablation and was not submitted, because our pre-registered rule said no.

The submission was made on 2026-09-22 at 17:22 UTC. At the time of writing its status was PENDING, with no public score (`kaggle competitions submissions arc-prize-2026-arc-agi-3 -v`).

## 2. The question

In ARC-AGI-3, every action counts, including RESET and actions that change nothing (`arc_agi/scorecard.py` in github.com/arcprize/ARC-AGI). Game scores run from 0 to 100%, with human action counts as the reference, and are capped at 100% (Kaggle ARC-AGI-3 Evaluation page). Per level, the ratio of human to agent actions is squared, and later levels weigh more (docs.arcprize.org/methodology; our `deadprobe/score.py`). The square makes waste expensive. Twice the human action count keeps a quarter of the level score.

A dead action leaves the gameplay state unchanged. We asked two things. How common are dead actions on the public games? Does a mechanical guard that refuses to repeat a known dead action help an LLM agent?

## 3. Method

Code: repo `arc-agi-3-probe`, commit `9f8b1de`.

**Classifier** (`deadprobe/grid.py`). It compares frames before and after an action and labels the change none, HUD or gameplay. A thin edge change becomes HUD only after two sightings. HUD cells are masked by connected component, so a live object on the same row stays visible.

**Memo** (`deadprobe/memo.py`). It stores (masked board state, action) pairs that changed nothing. Dead is judged per state, never globally.

**Guard** (`deadprobe/duck_patch.py`, mode `guard`). It installs into Duck at runtime from the notebook's hook cell. The TAAF source is unchanged. When the model asks for an action that already did nothing in this exact board state, the guard refuses it before it reaches the game. No action is spent, and Duck asks the model again. The known no-ops for the current board are added to the prompt as facts. Actions are never removed from the valid list. After three refusals in a row, one repeat goes through, so a stuck model cannot spin until the time cap. A test went red without that bound.

**Tests.** 34 pass with the Duck source present (re-run 2026-09-22). One plays a fake game, built on the real `arcengine`, through the real Duck session with a scripted model. Control spends 6 actions on it and guard spends 4.

## 4. Census: how common are dead actions?

All 25 public games, offline, official toolkit. Recounted from `runs/census.json` and `runs/walk.json` for this paper.

- At the level-1 start, 14 of 81 simple actions changed nothing. 8 of 25 games had at least one. In sc25, UP, DOWN, LEFT and RIGHT were all dead.
- 20 of 56 click targets from our heuristic did nothing. The chooser is weak, and two games flip a counter cell on every click, so both counts are lower bounds.
- A random policy, 200 steps per game: 2,366 of 4,955 observed transitions were no-ops. 1,650 repeated a no-op already seen in the same board state. The guard can skip those for free.
- Four games had zero no-ops: ls20, re86, tn36 and tr87.

A random-policy A/B gave means of 0.19 (plain) and 0.04 (guard) over 25 games (`runs/ab.json`). Only sp80 cleared a level, in both arms. This shows only that the plumbing runs.

## 5. The pre-registered rule

Written before either GPU result existed (`ARC-PREP-RETURN.md`): recommend Duck plus guard only if its mean score over the 5 games beats control and it clears at least as many levels. A tie or a loss means submit plain Duck.

The games were sc25, g50t, cd82 and tu93, which had many dead actions, and ls20, which had none and acts as an internal control.

## 6. Result

Both arms ran as private Kaggle notebooks with the same notebook, model, caps and games. Sources: `runs/duck-control/summary.txt`, `runs/duck-deadprobe/summary.txt`, and the `deadprobe:` memo lines in `runs/duck-deadprobe/log.txt`.

| Game | Control levels | Control score | Guard levels | Guard score | Guard skips |
|---|---|---|---|---|---|
| tu93 | 2 of 9 (56 actions) | 5.40 | 2 of 9 (200 actions) | 0.51 | 84 |
| sc25 | 2 of 6 | 0.94 | 0 of 6 | 0.00 | 19 |
| cd82 | 0 of 6 | 0.00 | 1 of 6 | 0.71 | 35 |
| ls20 | 0 of 7 | 0.00 | 1 of 7 | 0.55 | 0 |
| g50t | 0 of 7 | 0.00 | 0 of 7 | 0.00 | 8 |
| Mean | 4 levels | 1.27 | 4 levels | 0.35 | 146 total |

The guard cleared the same number of levels and lost on mean score. By the rule, we submitted plain Duck.

The median points the other way: 0.00 for control, 0.51 for guard. Control's mean comes mostly from tu93, where it cleared 2 levels in 56 actions. The guard arm won ls20, where it skipped nothing, so that win is sampling. We read the result as null, not as harm. Figure 1 (`ARC-PAPER-FIG1-AB.png`) shows the per-game scores.

## 7. What the guard costs

The guard refused 146 repeated no-ops. Each refusal saved one real action. The model kept asking for known dead actions anyway: 84 refusals in tu93. The prompt fact did not stop the repeats.

"Zero cost" holds for actions only. Each refusal triggers a new model call (`duck_patch.py`). Token totals cannot measure that cost here: both arms ran to the cap, so tokens track throughput (control 101.5, guard 109.5 generated tokens per second, `summary.txt`). All 10 games ended `gave_up` at Duck's per-game time cap (both `log.txt` files; the cap is 7,920 s per `ARC-PREP-RETURN.md`). Time was the binding limit, not actions. A guard that saves actions but spends time moves cost from one budget to the other.

We think this is why the effect is small. The metric charges actions only on cleared levels. When an agent clears few levels before a time cap, a saved second is worth more than a saved action.

## 8. Limits

- One run per arm, on 5 of 25 public games. Sampling swings per game are larger than any plausible guard effect.
- Only guard mode ran on a GPU. The level-start probe modes were tested offline and never run with Duck.
- Duck reports board change with a full-grid diff. A timer tick makes every action look live, so the model gets a false progress signal. We left that unchanged in the A/B.
- Hidden state: two boards that look the same can differ inside the game, and the guard could skip an action that would have worked. Not seen in tests.
- The classifier compares final frames only. Some games answer with several animation frames.
- The public score of the submission was unknown at write time.

## 9. What we would do next

1. Set `honest_board_changed=True`. It swaps Duck's full-grid diff for the HUD-masked one, so a timer tick no longer reads as progress. One flag, zero actions.
2. Answer a refusal without a new model call, so the guard saves time as well as actions.
3. Run at least 3 seeds per arm on all 25 public games before claiming a direction.
4. Run the level-start probe arm and measure its action cost against the level baseline.

## 10. Process

A human set the goal, the question and the decision rule, and gave each go or stop. AI coding agents, an orchestrated fleet of Claude Code sessions, did the build, the tests, the Kaggle runs and this write-up. Each lane returned a report with its sources, and claims were checked at the object before use.

The author is a builder and product person with a hackathon background, who works by directing AI coding agents. The process had failures, and the checks caught them. One agent predicted that a stale notebook pushed earlier would fail on import, and it did (`ModuleNotFoundError`, `ARC-PREP-RETURN.md`). Another found that the game filter in the hook cell had no effect, because a later cell rebuilt the game list. Both were caught before any number was used. The submission was made from the author's Kaggle account. Outward acts such as submitting stay with the human.

## Sources

- Repo `arc-agi-3-probe`, commit `9f8b1de`: `deadprobe/`, `harness.py`, `kaggle_hook.py`, `kaggle/`, `runs/`.
- `ARC-PREP-RETURN.md` (2026-09-22) and `ARC-SCOPE-RETURN.md` (2026-09-21): run notes, the decision rule, the tests.
- Duck: kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner. TAAF: kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share.
- Toolkit: github.com/arcprize/ARC-AGI and github.com/arcprize/ARCEngine.
