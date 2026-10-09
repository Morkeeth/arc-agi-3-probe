# ARC KAGGLE agent-engagement CPU probe, frozen before execution

This is a local **proxy-agent** test, not the 96 GB Kaggle model, a Kaggle score, or a Runpod trial. It uses the installed mechanics sandbox and the public offline SDK game `ls20-9607627b`, seed 0. Two fixed `LEFT` observations from the retained treatment opening are context only; no future board is shown to the agent. The agent receives the existing mechanics policy, a compact current observation, and can return Python code. The harness does not insert mechanics calls into that code. Maximum three agent decisions, four further SDK actions, and 20 seconds per sandbox call. No competition submission.

Primary pass: within the limit, the agent itself emits `mechanics.register(...)` and `mechanics.test(...)`, both execute through the installed parent-owned sandbox, and a per-game evidence record contains a real prospective `observed` comparison from the SDK. A text mention, `status`, `probe`, or a scripted harness call is not a pass.

Secondary full-chain pass: the agent additionally chooses `mechanics.plan(...)` and `mechanics.execute(...)` after the required prospective trust checks, and the recorded SDK actions match the parent evidence. If trust cannot be earned in this slice, report this as not reached, not failure of the planner.

Record the full model prompt and response, emitted code, sandbox result, SDK action trace, mechanics events, source hashes, model identity, and usage. Do not infer score gain or same-model adoption from a proxy-agent pass. Any API/auth failure or refusal to produce usable code is `not executed`, not a model-adoption result.
