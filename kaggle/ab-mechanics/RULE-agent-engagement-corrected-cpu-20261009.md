# ARC KAGGLE third CPU adoption rule, frozen before run

This is the **final prompt/interface correction** on the same local public-game proxy, not a Kaggle-model run or a score claim. Preserve `agent-engagement-cpu-20261009.json` and `agent-engagement-checkpoint-cpu-20261009.json` as failed baselines. The frozen 25-game Kaggle treatment remains untouched.

The only corrections are to input fidelity: show the proxy the exact installed `ToolAgent._tools` Python tool schema, state that `mechanics.view()` returns `{'level': int, 'grid': list[list[int]]}`, make the installed schema itself say NumPy is unavailable, and carry real SDK history entries so `history`, `transitions`, `last_transition`, and `previous_frame` work as in the installed sandbox. The agent still receives the advisory checkpoint after a gameplay-changing transition. No harness-inserted mechanics calls, candidate predictor, or outcome is permitted.

Use public offline `ls20-9607627b`, seed 0; the same first two real `LEFT` transitions are past context. Maximum five proxy decisions, four further real SDK actions, 20 seconds per sandbox call. Retain full prompt/response/emitted code, installed tool schema, SDK history and action trace, mechanics counts/events, source hashes, model identity and usage. No paid GPU, Runpod resource, Kaggle retry, or submission.

Primary pass: agent-emitted code actually invokes installed parent-owned `mechanics.register` and `mechanics.test`, and at least one prospective test executes a real SDK action with an `observed` event. Full-chain pass: after primary pass, agent-emitted code also invokes `mechanics.plan` and `mechanics.execute` through that route. A view/status call, text mention, or scripted control is not adoption. Sandbox errors cannot be counted as successful calls.

If this corrected proxy still chooses only view/status or normal actions and not register/test, **stop prompt tuning**. Verdict: this mechanics policy is not fitting the agent under this local proxy; it needs a different mechanism and a separately designed test before any paid GPU. Do not translate proxy failure into a claim about hidden Kaggle performance.
