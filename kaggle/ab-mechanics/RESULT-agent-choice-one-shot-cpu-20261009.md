# ARC KAGGLE one-shot CPU agent-choice verdict

Frozen rule: `RULE-agent-choice-one-shot-cpu-20261009.md`. Attempt artifact: `agent-choice-one-shot-cpu-20261009.json`. Prior failed and invalid traces are preserved.

**Not executed.** The public offline SDK game `ls20-9607627b` seed 0 loaded and the two real past `LEFT` history entries were prepared. The direct model API then returned `404 NOT_FOUND`: `models/gemini-2.5-flash` is no longer available to new users on this account and the provider names `models/gemini-3.8-flash` as the replacement. The attempt produced no model response, emitted Python, agent decision, further SDK action, or mechanics call. Parent mechanics counts and events are empty. This is an API model-availability failure, **not** evidence that an agent declined register/test or that the policy fits or fails.

The installed schema/history/view preflight passed three focused tests immediately before the one attempt. Per the frozen rule and coordinator limit, there was no retry or model swap. No pod, paid GPU, Kaggle kernel, or submission was started. A later run would need a new model pin and separately frozen one-shot rule; this result grants no GPU promotion.
