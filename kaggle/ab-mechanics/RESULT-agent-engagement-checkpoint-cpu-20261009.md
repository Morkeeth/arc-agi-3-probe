# ARC KAGGLE checkpoint CPU verdict

Frozen rule: `RULE-agent-engagement-checkpoint-cpu-20261009.md`. Full prompt, response, emitted code, sandbox result, SDK action, source hashes and per-game mechanics state: `agent-engagement-checkpoint-cpu-20261009.json`. The earlier failed proxy trace is preserved separately; the frozen 25-game Kaggle treatment is untouched.

**Adoption gate did not flip.** On public offline `ls20-9607627b`, seed 0, the Claude Haiku proxy made five decisions and two real SDK actions. It called the new parent-owned `mechanics.view()` four times. It chose zero `mechanics.register`, zero `mechanics.test`, zero `mechanics.plan`, and zero `mechanics.execute` calls. The parent-owned evidence has zero models and zero prospective events. Both executed actions changed only edge cells, not interior gameplay. The scored trial's model API usage reported about $0.025 at list price; no Runpod resource was created.

Two of five emitted snippets errored in the installed sandbox: one attempted disallowed `numpy`, and one treated the documented `{'level', 'grid'}` view as a list. These are real agent mistakes; they do not count as mechanics adoption. The full-chain condition was not reached because the agent never registered or tested a predictor.

**Installed-path controls passed:** the view returns the current parent SDK grid for two distinct states and child mutation cannot change parent state; the installed checkpoint branch fires only after changed gameplay (2 focused tests green). The existing scripted mechanics control still passes with six real SDK transitions and 36 live SDK frame reads. That control proves tool viability, not agent choice.

**Boundary:** this is a proxy model without the Kaggle model's image input or full ToolAgent history/segmentation. It cannot forecast the frozen Kaggle score. The next useful local change is to expose an unambiguous view shape in the agent prompt and carry real transition history into the proxy harness; then test agent choice again under a newly frozen rule. Do not promote a paid GPU run from this result.
