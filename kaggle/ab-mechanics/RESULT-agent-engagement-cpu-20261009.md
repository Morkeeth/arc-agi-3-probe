# ARC KAGGLE bounded CPU engagement result

Frozen rule: `RULE-agent-engagement-cpu-20261009.md`. Full model prompts/responses, emitted Python, installed-sandbox outputs, actual SDK actions, and per-game mechanics state: `agent-engagement-cpu-20261009.json`.

**Primary adoption gate: FAIL.** In three Claude Haiku proxy-agent decisions on public offline `ls20-9607627b` seed 0, the agent emitted zero `mechanics.register`, `mechanics.test`, `mechanics.plan`, or `mechanics.execute` calls. It called `mechanics.status` twice, took one real `UP` SDK action, and spent the other decisions inspecting the board. The installed parent-owned mechanics state retained zero models and zero prospective evidence events. Sandbox errors: zero. This is agent non-adoption in this bounded proxy slice, not a claim that the full Kaggle model would behave identically.

**Plumbing control:** the existing scripted `local_gate.py` passed on the same public game: six actual SDK transitions, 36 live frame reads, prospective comparison and negative trust gates. This proves the installed route can record evidence when calls are made. It does **not** count as agent adoption.

Two earlier attempts are preserved but excluded: `agent-engagement-cpu-20261009-invalid-feedback.json` failed to return sandbox stdout to the model; `agent-engagement-cpu-20261009-invalid-color.json` used the wrong numeric-to-character map. The final runner uses the installed `ARC_COLOR_CHARS` and returns stdout. The harness is still a bounded proxy, not the original full tool-agent loop: it uses Claude Haiku rather than the Kaggle model, omits image input, and does not carry the full game-agent history/segmentation. Do not promote a GPU run from this result.

Next build change to test before paid compute: make a small, explicit mechanics decision checkpoint after a changed transition, and give the agent a sanctioned way to derive a full-grid predictor without raw-grid leakage or guesswork. Predeclare and compare agent call traces against this failed baseline; do not count a forced harness call as success.
