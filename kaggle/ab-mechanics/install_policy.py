"""Install mechanics policy into the pinned baseline source, once, with exact anchors."""
from pathlib import Path
import hashlib, json, sys
HERE = Path(__file__).resolve().parent
PROMPT = '''\nMechanics experiment: use the mechanics object in Python to learn executable dynamics. First inspect actual frames/transitions. Write a self-contained pure function source `def predict(grid, action): ...` returning the complete predicted pixel grid or None when unsupported; register with mechanics.register(source). Never fit to future observations. mechanics.probe(actions) suggests a single discriminating action across candidates; mechanics.test(action) commits predictions before one real action and reports full-frame mismatches. Two changed, distinct-state prospective matches permit mechanics.plan(model_id, full_target_grid, actions, depth=3, nodes=64), followed by mechanics.execute(model_id, plan['actions']). Search and execution are bounded and stop on mismatch or level change. mechanics.status() shows evidence. No-op matches grant no trust. This confidence is conditional on observed state: hidden phases can still invalidate it. Spend at most two additional diagnostic actions per level; prefer tests that advance the game. If extraction, timer dynamics, modelling cost or hidden state prevents reliable full-grid predictions, continue the original action policy immediately. Do not force a model on every game.\n'''
def install(root):
    agent = Path(root)/'inference/agent'
    sandbox = agent/'python_tool_sandbox.py'
    tool = agent/'tool_agent.py'
    s,t = sandbox.read_text(),tool.read_text()
    if 'MECHANICS_POLICY_INSTALLED' in s:
        raise RuntimeError('Policy already installed')
    runtime = (HERE/'mechanics_runtime.py').read_text().replace(chr(34)*3, chr(39)*3)
    def replace_once(text, old, new):
        if text.count(old)!=1: raise RuntimeError('Pinned anchor mismatch: '+old[:90])
        return text.replace(old,new,1)
    # Insert trusted runtime into bootstrap, not into model-generated source.
    s = replace_once(s, '    def _send(payload):', '\n'.join('    '+line if line else '' for line in runtime.splitlines())+'\n\n    def _send(payload):')
    s = replace_once(s, '        _persistence_enabled = bool(initial.get("persistence_enabled"))', '        mechanics = Mechanics((initial.get("state") or {}).get("mechanics_state"), lambda: runtime_globals["current_frame"], action)\n        runtime_globals["mechanics"] = mechanics\n\n        _persistence_enabled = bool(initial.get("persistence_enabled"))')
    s = s.replace('                    **_collect_retention_payload(),', '                    "mechanics_state": mechanics.state,\n                    **_collect_retention_payload(),')
    s = replace_once(s, '                    "result": message.get("result"),', '                    "mechanics_state": message.get("mechanics_state"),\n                    "result": message.get("result"),')
    s += '\n# MECHANICS_POLICY_INSTALLED\n'
    t = replace_once(t, '            initial_state=_serialized_runtime_state(),', '            initial_state={**_serialized_runtime_state(), "mechanics_state": getattr(self, "_mechanics_states", {}).get(str(state_path))},')
    t = replace_once(t, '        action_results = [\n            item\n            for item in sandbox_result.get("action_results") or []', '        if not hasattr(self, "_mechanics_states"):\n            self._mechanics_states = {}\n        if sandbox_result.get("mechanics_state") is not None:\n            self._mechanics_states[str(state_path)] = sandbox_result["mechanics_state"]\n            (state_path.parent / "mechanics-evidence.json").write_text(json.dumps(sandbox_result["mechanics_state"]))\n\n        action_results = [\n            item\n            for item in sandbox_result.get("action_results") or []')
    # Existing prefix mechanism keeps the original system instructions intact.
    t = replace_once(t, '    _prefix = os.environ.get("ARC3_SYSTEM_PROMPT_PREFIX", "").strip()', '    _prefix = os.environ.get("ARC3_SYSTEM_PROMPT_PREFIX", "").strip() + '+repr(PROMPT))
    compile(s,str(sandbox),'exec'); compile(t,str(tool),'exec')
    sandbox.write_text(s); tool.write_text(t)
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (sandbox,tool)}
if __name__=='__main__': print(json.dumps(install(sys.argv[1]),indent=2))
