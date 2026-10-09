"""Install parent-owned mechanics; generated code receives only an RPC client."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
PROMPT = '''\nMechanics experiment: use the mechanics object in Python to learn executable dynamics. First inspect actual frames/transitions. `mechanics.view()` returns `{'level': int, 'grid': list[list[int]]}`: a copy of the current SDK numeric grid and level inside Python; it does not reveal a future state or certify a model. Use `mechanics.view()['grid']` only for compact derived inspection and for building a self-contained pure function source `def predict(grid, action): ...` returning the complete predicted pixel grid or None when unsupported; register with mechanics.register(source). Never fit to future observations. mechanics.probe(actions) suggests a single discriminating action across candidates; mechanics.test(action) commits predictions before one real action and reports full-frame mismatches. Two changed, distinct-state prospective matches permit mechanics.plan(model_id, full_target_grid, actions, depth=3, nodes=64), followed by mechanics.execute(model_id, plan['actions']). Search and execution are bounded and stop on mismatch or level change. mechanics.status() shows evidence. No-op matches grant no trust. This confidence is conditional on observed state: hidden phases can still invalidate it. Spend at most two additional diagnostic actions per level; prefer tests that advance the game. If extraction, timer dynamics, modelling cost or hidden state prevents reliable full-grid predictions, continue the original action policy immediately. Do not force a model on every game.\n'''
CLIENT = '''        class MechanicsClient:
            def _call(self, method, *args):
                _send({"type":"mechanics", "method":method, "args":list(args)})
                reply = _recv()
                if reply.get("type") != "mechanics_result":
                    raise RuntimeError(reply.get("error", "Mechanics request failed"))
                _refresh_state(reply["state"])
                action_results.extend(reply.get("action_results", []))
                return reply.get("result")
            def register(self, source): return self._call("register", source)
            def probe(self, actions): return self._call("probe", actions)
            def test(self, action): return self._call("test", action)
            def plan(self, name, target, actions, depth=3, nodes=64): return self._call("plan", name, target, actions, depth, nodes)
            def execute(self, name, actions): return self._call("execute", name, actions)
            def status(self): return self._call("status")
            def view(self): return self._call("view")
        runtime_globals["mechanics"] = MechanicsClient()
'''
def install(root, mode='treatment'):
    agent=Path(root)/'inference/agent';sandbox=agent/'python_tool_sandbox.py';tool=agent/'tool_agent.py'
    s,t=sandbox.read_text(),tool.read_text()
    def once(text,old,new):
        if text.count(old)!=1:raise RuntimeError('Pinned anchor mismatch: '+old[:70])
        return text.replace(old,new,1)
    # Both arms report only executions recorded by the parent action handler.
    s=once(s,'"action_results": list(message.get("action_results") or host_action_results),','"action_results": list(host_action_results),')
    if mode=='treatment':
        s=once(s,'from inference.utils.grid_utils import ARC_COLOR_CHARS','from inference.utils.grid_utils import ARC_COLOR_CHARS\nfrom inference.agent.mechanics_runtime import MechanicsHost')
        s=once(s,'        _persistence_enabled = bool(initial.get("persistence_enabled"))',CLIENT+'\n        _persistence_enabled = bool(initial.get("persistence_enabled"))')
        s=once(s,'    animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,','    mechanics_state: dict[str, Any] | None = None,\n    mechanics_frame_handler: Callable[[], dict[str, Any]] | None = None,\n    animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,')
        s=once(s,'        host_action_results: list[dict[str, Any]] = []','        host_action_results: list[dict[str, Any]] = []\n        mechanics_host = MechanicsHost(mechanics_state, initial_state, action_handler, mechanics_frame_handler)')
        branch='''            if msg_type == "mechanics":
                start = len(mechanics_host.action_results)
                try:
                    result = mechanics_host.dispatch(message)
                    reply = {"type":"mechanics_result", "result":result, "state":mechanics_host.current}
                except Exception as exc:
                    reply = {"type":"mechanics_error", "error":str(exc)}
                observed = mechanics_host.action_results[start:]
                host_action_results.extend(observed)
                reply["action_results"] = observed
                _send_json_line(process.stdin, reply)
                continue
'''
        s=once(s,'            if msg_type == "animation":',branch+'            if msg_type == "animation":')
        s=once(s,'                raw_action_result = action_result_payload.get("action_result") or {}','                mechanics_host.current = json.loads(json.dumps(action_result_payload.get("state") or {}))\n                raw_action_result = action_result_payload.get("action_result") or {}')
        s=once(s,'                    "result": message.get("result"),','                    "mechanics_state": mechanics_host.mechanics.state,\n                    "result": message.get("result"),')
        t=once(t,'        sandbox_result = run_sandboxed_python(','''        def _mechanics_live_frame():
            solver = getattr(self._step_env_callback, "__self__", None)
            game = getattr(solver, "game", None)
            env = getattr(game, "env", None)
            observation = getattr(env, "observation_space", None)
            if observation is None or not observation.frame:
                raise RuntimeError("Live SDK observation unavailable; use the original action policy")
            raw = observation.frame[-1]
            grid = raw.tolist() if hasattr(raw, "tolist") else [list(row) for row in raw]
            return {"grid": grid, "level": int(observation.levels_completed)}
        if not hasattr(self, "_mechanics_states"):
            self._mechanics_states = {}
        mechanics_state = self._mechanics_states.setdefault(str(state_path), {"models":{},"events":[],"level":None})
        sandbox_result = run_sandboxed_python(''')
        t=once(t,'            initial_state=_serialized_runtime_state(),','            initial_state=_serialized_runtime_state(),\n            mechanics_state=mechanics_state,\n            mechanics_frame_handler=_mechanics_live_frame,')
        t=once(t,'        action_results = [\n            item\n            for item in sandbox_result.get("action_results") or []','        (state_path.parent / "mechanics-evidence.json").write_text(json.dumps(mechanics_state))\n\n        action_results = [\n            item\n            for item in sandbox_result.get("action_results") or []')
        t=once(t,'    _prefix = os.environ.get("ARC3_SYSTEM_PROMPT_PREFIX", "").strip()','    _prefix = os.environ.get("ARC3_SYSTEM_PROMPT_PREFIX", "").strip() + '+repr(PROMPT))
        checkpoint='''        if self._last_action_gameplay_changed is True:
            lines.append(
                "Mechanics decision checkpoint: the last real action changed gameplay. "
                "Choose whether a full-grid predictor is viable now. `mechanics.view()` gives the current numeric frame inside Python; "
                "if you have a supported rule, register it and use `mechanics.test(action)` for a prospective real action. "
                "If not, continue the ordinary action policy. This is a choice, not a requirement to call mechanics."
            )
'''
        t=once(t,'        # recording is independent of display: the guard needs the ledger even',checkpoint+'        # recording is independent of display: the guard needs the ledger even')
        t=once(t,'"Only letter-coded board views and lightweight metadata are exposed; raw numeric color IDs are not available."',
               '"The current numeric grid is available only through `mechanics.view()` inside Python for prediction work; do not print the full grid."')
        t=once(t,'"The raw numeric grid is not available. Use `.segmentation` as the primary view; use `.ascii` only to read a small, specific region. "',
               '"For prediction work, `mechanics.view()[\'grid\']` is a copy of the current numeric grid; never print it. Use `.segmentation` as the primary view and `.ascii` only for small regions. The sandbox has no numpy; use Python builtins or allowed pure-Python modules. "')
        (agent/'mechanics_runtime.py').write_text((HERE/'mechanics_runtime.py').read_text())
    elif mode!='control':raise ValueError('Unknown arm')
    compile(s,str(sandbox),'exec');compile(t,str(tool),'exec')
    sandbox.write_text(s);tool.write_text(t)
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (sandbox,tool)}
if __name__=='__main__':print(json.dumps(install(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'treatment'),indent=2))
