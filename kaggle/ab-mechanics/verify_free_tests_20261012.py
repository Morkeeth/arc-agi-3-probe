"""Exercise changed gates and patch the exact frozen bundled source on CPU."""
import ast
import importlib.util
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

from free_tests_20261012 import HERE, BASE, RECEIPTS, COMPACT_PATCH, FORCED_PATCH

def literal_assignment(source, filename):
    for node in ast.walk(ast.parse('\n'.join(line for line in source.splitlines() if not line.lstrip().startswith('!')))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'write_text':
            if filename in ast.unparse(node.func.value) and isinstance(node.args[0], ast.Constant):
                yield node.args[0].value

def main():
    base = json.loads((BASE / 'experiment.ipynb').read_text())
    with tempfile.TemporaryDirectory(prefix='arc-free-preflight-') as directory:
        root = Path(directory)
        with zipfile.ZipFile(RECEIPTS.parent / 'assets/taaf-kaggle-source-bundle-copy.zip') as archive:
            archive.extractall(root)
        patch = root / 'harness.patch'
        patch.write_text(''.join(base['cells'][1]['source']).split('\n', 1)[1].rstrip() + '\n')
        subprocess.run(['git', 'apply', str(patch)], cwd=root / 'src', check=True)
        notebook = json.loads((HERE / 'free-forced-mechanics-20261012/experiment.ipynb').read_text())
        installer = list(literal_assignment(''.join(notebook['cells'][2]['source']), 'install_policy.py'))[-1]
        namespace = {'__file__': str(HERE / 'install_policy.py'), '__name__': 'installer_preflight'}
        exec(compile(installer, 'installed_policy.py', 'exec'), namespace)
        namespace['install'](root / 'src/ARC3-Inference', 'treatment')
        tool_path = root / 'src/ARC3-Inference/inference/agent/tool_agent.py'
        source = tool_path.read_text()
        compile(source, str(tool_path), 'exec')
        tree = ast.parse(source)
        handle = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_handle_action')
        opening_gate = next(n for n in handle.body if isinstance(n, ast.If) and 'opening_success' in ast.unparse(n.test))
        runtime_spec = importlib.util.spec_from_file_location('mechanics_under_test', HERE / 'mechanics_runtime.py')
        runtime = importlib.util.module_from_spec(runtime_spec)
        runtime_spec.loader.exec_module(runtime)
        import sys
        inference = SimpleNamespace(agent=SimpleNamespace(mechanics_runtime=runtime))
        sys.modules['inference'] = inference
        sys.modules['inference.agent'] = inference.agent
        # Only relocate the event output for the CPU check.
        exec(FORCED_PATCH.replace('/kaggle/working/mechanics-calls.jsonl', str(root / 'calls.jsonl')), {})
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction
        arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                        environments_dir='/Users/morkeeth/CODE/arc-agi-3-probe-data/environment_files')
        game = arcade.make('ls20-9607627b', seed=0)
        state = {'models': {}, 'events': [], 'level': None}
        events = []
        def frame():
            observation = game.observation_space
            return {'grid': observation.frame[-1].tolist(), 'level': int(observation.levels_completed)}
        gate_code = compile(ast.Module(body=[opening_gate], type_ignores=[]), 'exact_installed_opening_gate', 'exec')
        def action_handler(actions, **kwargs):
            exec(gate_code, {'mechanics_state': state})
            assert len(actions) == 1
            action = actions[0]['action']
            game.step(getattr(GameAction, action))
            events.append(action)
            return {'state': {'current_frame': frame()}, 'action_result': {'executed': True, 'executed_count': 1}}
        host = runtime.MechanicsHost(state, {'current_frame': frame()}, action_handler, frame)
        try:
            action_handler([{'action': 'ACTION1'}])
            raise AssertionError('Raw opening action passed')
        except RuntimeError:
            pass
        assert not events
        host.dispatch({'type': 'mechanics', 'method': 'register', 'args': ['def predict(grid, action):\n    return grid']})
        result = host.dispatch({'type': 'mechanics', 'method': 'test', 'args': ['ACTION1']})
        assert len(events) == 1 and result['status'] == 'observed'
        assert state['counts']['opening_success'] == 1
        action_handler([{'action': 'ACTION2'}])
        assert len(events) == 2
        # A level reset must not force a second opening test.
        host.mechanics.state['level'] = -1
        host.mechanics._scope()
        assert state['counts']['opening_success'] == 1
        # Run the compact wrapper on the actual pinned ToolAgent methods' API.
        class Agent:
            def _ensure_session(self, path): self._session_runtime_dir = Path(path).parent
            def analyze(self, path, *args, **kwargs):
                return self._build_user_prompt(current_frame=SimpleNamespace(level=1))
            def _build_user_prompt(self, *args, **kwargs): return 'Current frame is unchanged'
        sys.modules['inference.agent.tool_agent'] = SimpleNamespace(ToolAgent=Agent)
        exec(COMPACT_PATCH.replace('/kaggle/working/compact-events.jsonl', str(root / 'compact.jsonl')), {})
        agent = Agent()
        agent._compact_scope = (str(root), 1)
        agent._history_messages = []
        for i in range(6):
            agent._history_messages.extend([{'role':'user','content':str(i)},
                {'role':'assistant','content':'claim-'+str(i)}, {'role':'tool','content':'observed-'+str(i)}])
        prompt = agent.analyze(root / 'state.json')
        assert len(agent._history_messages) == 6 and agent._compact_dropped_this_turn == 12
        assert 'observed-0' in prompt and 'unverified' in prompt
        agent._build_user_prompt(current_frame=SimpleNamespace(level=2))
        assert agent._compact_record == ''
        report = {'status':'PASS', 'installed_source_compiled':True,
                  'forced_opening_raw_action_blocked':True, 'real_sdk_actions':len(events),
                  'registered_predictor_real_test_observed':True, 'later_raw_actions_allowed':True,
                  'level_change_preserves_opening_completion':True,
                  'compact_wrapper_dropped_messages':12, 'compact_kept_user_blocks':2,
                  'compact_clears_on_level_change':True,
                  'scope':'Installed-source gate and real SDK action check; compact wrapper API check, not model gameplay or a score.'}
        (RECEIPTS / 'preflight.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))

if __name__ == '__main__':
    main()
