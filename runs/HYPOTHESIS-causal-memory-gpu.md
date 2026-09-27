# SEALED GPU HYPOTHESIS: causal memory

Sealed after the CPU gate passed on 28 September 2026. This file authorizes no launch.

## Hypothesis

Replacing the rolling 3,000-character raw-snippet digest with a 3,000-character causal record will preserve the action effects, rejected hypotheses, object/goal constraints, and unfinished tests that the agent needs to progress. It should recover the level loss of the failed compaction arm without increasing the prompt-memory budget.

The CPU result qualifies this for a future GPU experiment. It does not show that the model will use the record well or improve its ARC score.

## One changed lever

Keep the frozen control base, model, context window, request limits, games, seeds, time limit, and evaluator unchanged. Replace only the failed arm's raw snippet construction with `causal_memory.extract_facts` plus `causal_memory.bounded_record` at 3,000 characters. Install the stable state-path game identity before writing evidence.

## Required pre-launch evidence

1. A candidate notebook is built from the same frozen base and differs only in the hook block and arm identifier.
2. The embedded causal-memory source matches the reviewed source byte for byte.
3. A CPU replay still passes the sealed 1,000-token-estimator budget and all four category non-regression checks.
4. Two concurrent games in one runtime directory write distinct full game IDs.
5. No failed-arm notebook or result is rewritten.

## Future GPU rule

Run two valid treatment repeats over all 25 public games. Keep only if:

- both repeats complete without a configuration failure;
- mean levels are at least 40 per run;
- mean local score exceeds the frozen control mean `8.589519100442978`;
- event evidence covers at least 24 distinct game IDs in each repeat;
- the causal record stays within 3,000 characters; and
- no candidate run is discarded after its score is visible.

Kill if mean levels are 38 or lower, if score does not beat the frozen control, if identity or budget evidence fails, or if the causal hook changes any other experimental lever.

## Explicit stop

No notebook was created or uploaded. No GPU was launched. No Kaggle submission, merge, publication, or spend was made by this CPU run.
