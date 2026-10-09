# ARC KAGGLE advisory checkpoint CPU probe, frozen before run

Comparison baseline: the preserved failed three-decision proxy trace `agent-engagement-cpu-20261009.json` (zero agent-chosen register/test/plan/execute). Do not relabel the frozen 25-game Kaggle treatment. This is a **new, local proxy-agent** condition, not the exact Kaggle model or a hidden score.

Change only the available affordance and advisory: parent-owned `mechanics.view()` returns a copy of the current SDK numeric grid inside Python, never a future frame or trust claim; after a real gameplay-changing transition the prompt asks the agent to decide whether to model now. It explicitly allows normal action play. The harness must not insert register/test/plan/execute calls into model code.

Use the same public offline game `ls20-9607627b`, seed 0, and the same two past `LEFT` observations as context. Maximum five proxy-agent decisions, four subsequent real SDK actions, 20 seconds per sandbox call, no paid GPU, pod, Kaggle retry, or submission. Keep exact prompt/response/code, SDK transitions, mechanics counts/events, source hashes, model identity and usage.

**Primary adoption pass:** the agent-emitted code actually reaches the installed parent-owned `mechanics.register` and `mechanics.test`; at least one prospective test executes one real SDK action and leaves an `observed` per-game event. Counting text mentions, `view`, `status`, `probe`, or a scripted control is invalid. **Full-chain pass:** after the primary gate, the agent also chooses `mechanics.plan` and `mechanics.execute` through the installed route. If the trust condition was never met, say full chain was not reached. A proxy pass does not establish that the Kaggle model will adopt or that scores improve.

The source-control test must first show two actual SDK states through the installed view path, including that altering the returned copy cannot alter parent state. Fail closed on API/auth or sandbox errors. Preserve all invalid attempts, but only a run with the correct installed `ARC_COLOR_CHARS` and tool-output feedback can be scored.
