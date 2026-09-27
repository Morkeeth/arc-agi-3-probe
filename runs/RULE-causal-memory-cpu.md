# RULE: causal memory CPU gate

Sealed before the comparison is run.

## Question

At the same 3,000-character budget, does a structured causal record preserve more actionable facts from recorded public-game turns than the failed arm's rolling snippet digest?

## Population

All 75 recorded public-game transcripts from `control-s1`, `control-s2r`, and `control-s3r` under `runs/ab-2026-09-23`. No model calls and no generated substitute data.

## Facts fixed before evaluation

Facts must be tied to an observed turn and belong to exactly one of these categories:

1. `action_effect`: an executed action paired with an observed result.
2. `rejected_hypothesis`: an explicit failed, blocked, wrong, or no-effect interpretation.
3. `constraint`: an object, goal, action, or interaction constraint stated from an observed turn.
4. `unfinished_test`: an explicit next test, check, probe, inspection, or experiment.

The extractor is deterministic. It uses visible assistant text, tool calls, and tool results from the recorded turn. It does not use scores, run labels, later outcomes, or a model judge.

## Budget and comparison

- Candidate: bounded causal record, at most 3,000 characters (the fork's estimator treats this as at most 1,000 prompt tokens).
- Control: current `compaction.digest(...)`, truncated to its final 3,000 characters exactly as the failed arm did.
- Both receive the same parsed messages in the same order.
- A fact is retained only when its normalized evidence anchor is present in the bounded output.
- Report per-category retention, macro-average retention, games with every available category retained, character count, and observed-turn traceability.

## Win rule

Seal a new GPU hypothesis only if all are true:

1. Candidate macro-average retention is at least 10 percentage points above control.
2. Candidate is no worse than control in any category with at least one source fact.
3. Candidate output never exceeds 3,000 characters / 1,000 estimated prompt tokens.
4. Every retained candidate record includes its observed turn.
5. The event-identity integration check shows two concurrent games in the same runtime directory produce two distinct stable game IDs.

Otherwise stop. Do not upload a notebook, start a GPU, submit, merge, publish, or spend.
