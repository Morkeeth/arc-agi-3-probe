# ARC KAGGLE forced-step CPU comparison

Frozen before execution: `RULE-forced-steps-cpu-20261010.md`. Full model prompts, responses, proposed code, per-arm SDK actions, mechanics events and timing: `forced-steps-cpu-20261010.json`. The branch commit before this new mechanism preserves the prior optional-choice failures.

**Valid, bounded proxy comparison; gameplay promotion bar failed.** On public offline `ls20-9607627b`, seeds 0 and 1, one isolated Claude Sonnet proposal per seed was shared by both arms. Each proposal chose LEFT, LEFT, UP, UP and supplied `def predict(grid, action): return None`. Both arms took the same four post-opening SDK actions on each seed. Plain and forced each completed **0 levels** on each seed; paired level deltas were **0, 0**. No game win was observed in this four-action window. This is not a Kaggle score.

The forced framework registered the agent-supplied abstaining predictor once per seed, then made four `test`, four `plan`, and four `execute` calls per seed. All eight tests observed a real SDK action. Every prediction abstained, all plans returned `untrusted`, and every execute received an empty plan; there was no planned SDK action. Those calls prove the wrapper ran, not that the agent learned mechanics. Forced CPU time was about 0.20 seconds per seed versus about 0.005 seconds for plain action execution, excluding the shared model proposal and SDK setup. Model calls and action caps were matched; inference tokens and CPU time were not equalized.

This one-shot test is done. Do not promote the forced wrapper to paid GPU on call counts. The information bottleneck is predictor construction: the model saw two past transitions and the current letter grid, then explicitly abstained rather than giving a full-grid transition function. A future mechanism would need to change what can be inferred and independently verify non-abstaining predictions before asking whether planning improves gameplay. That would require a new frozen rule; no retry, pod, paid compute, Kaggle kernel, submission or deploy happened here.

## Ten-line diagnosis across earlier evidence

1. The frozen 25-game private treatment and control completed real gameplay, but the treatment transcripts contain zero `register/test/plan/execute` calls; this is a non-adoption result, not a mechanics-model test.
2. The treatment's global evidence file was empty and could be overwritten by the final game, so the retained per-game transcript is the stronger adoption source.
3. The scripted installed-path control executed real transitions and recorded prospective evidence, proving the recorder can work when called; it did not prove agent use.
4. The first valid Haiku CPU proxy took one SDK action and two `status` calls in three decisions, with zero mechanics-model calls.
5. The checkpoint Haiku proxy made five decisions and two SDK actions, including four `view` calls, but still zero `register/test/plan/execute` calls; two code snippets errored.
6. The corrected-history/schema attempt was invalid: workspace-context contamination and then a closed model client prevented a valid decision.
7. A separate one-shot direct-model attempt returned provider 404 before its first decision; it is model availability failure, not agent refusal.
8. The later valid, isolated Claude-login proxy made five executable decisions with six `view` and five `status` calls, zero model calls and zero further SDK actions.
9. Today's forced-step proxy called all four stages on two seeds, but the agent supplied only an abstaining predictor, so plans remained untrusted and gameplay tied 0–0 levels.
10. The missing causal link is a supported, prospectively checked transition model that changes decisions and level completion under matched budgets; no paid pilot or paper performance claim follows yet.
