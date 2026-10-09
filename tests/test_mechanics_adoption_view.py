"""Installed parent-owned current-frame view, exercised through child RPC."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

HERE = Path(__file__).resolve().parents[1] / 'kaggle' / 'ab-mechanics'
sys.path.insert(0, str(HERE / 'upstream' / 'ARC3-Inference'))
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.tool_agent import ToolAgent
from inference.utils.grid_utils import ARC_COLOR_CHARS


def test_installed_view_is_current_and_cannot_be_forged():
    world = {'grid': [[1, 0, 0]], 'level': 0}
    calls = []

    def frame():
        return {'grid': [list(row) for row in world['grid']], 'level': world['level']}

    def act(actions, **kwargs):
        assert actions == [{'action': 'RIGHT'}]
        calls.append(actions)
        world['grid'] = [[0, 1, 0]]
        return {'action_result': {'executed': True, 'executed_count': 1},
                'state': {'current_frame': frame(), 'history': [], 'valid_actions': ['RIGHT']}}

    result = run_sandboxed_python(
        code="before=mechanics.view(); before['grid'][0][0]=9\n"
             "again=mechanics.view(); action([{'action':'RIGHT'}]); after=mechanics.view()\n"
             "result={'again':again,'after':after}",
        timeout_seconds=10,
        initial_state={'current_frame': frame(), 'history': [], 'valid_actions': ['RIGHT']},
        action_handler=act, mechanics_frame_handler=frame,
    )
    assert not result['error']
    assert result['result']['again'] == {'level': 0, 'grid': [[1, 0, 0]]}
    assert result['result']['after'] == {'level': 0, 'grid': [[0, 1, 0]]}
    assert len(calls) == 1
    assert result['mechanics_state']['counts']['view'] == 3


def test_installed_checkpoint_fires_only_after_changed_gameplay():
    path = HERE / 'upstream' / 'ARC3-Inference' / 'inference' / 'agent' / 'tool_agent.py'
    tree = ast.parse(path.read_text())
    blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and ast.unparse(node.test) == 'self._last_action_gameplay_changed is True'
              and any('Mechanics decision checkpoint' in ast.unparse(child) for child in node.body)]
    assert len(blocks) == 1
    installed_block = compile(ast.fix_missing_locations(ast.Module(body=blocks, type_ignores=[])),
                              str(path), 'exec')
    for changed in (False, True):
        scope = {'self': SimpleNamespace(_last_action_gameplay_changed=changed), 'lines': []}
        exec(installed_block, scope)
        assert bool(scope['lines']) is changed
        if changed:
            assert 'choice, not a requirement' in scope['lines'][0]


def test_exact_installed_schema_and_real_sdk_history_are_available():
    class NoSession:
        def _ensure_session(self, path):
            pass

    schema = ToolAgent._tools(NoSession(), HERE / 'proxy-state.json')[0]
    description = schema['function']['description']
    assert "mechanics.view()['grid']" in description
    assert 'no numpy' in description
    assert schema['function']['parameters']['required'] == ['code']

    arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir='/Users/morkeeth/CODE/arc-agi-3-probe-data/environment_files')
    game = arcade.make('ls20-9607627b', seed=0)

    def frame(step):
        grid = game.observation_space.frame[-1].tolist()
        return {'grid': grid, 'level': int(game.observation_space.levels_completed),
                'step': step, 'shape': [64, 64],
                'ascii': '\n'.join(''.join(ARC_COLOR_CHARS[v] for v in row) for row in grid)}

    history = [{'action': '', 'frame': frame(0), 'result': {}}]
    for step in (1, 2):
        game.step(GameAction.ACTION3)
        history.append({'action': 'ACTION3', 'frame': frame(step), 'result': {'executed': True}})
    current = frame(2)
    result = run_sandboxed_python(
        code="result={'history':len(history),'transitions':len(transitions),"
             "'last':last_transition.action,'previous_diff':previous_frame.ascii != current_frame.ascii}",
        timeout_seconds=10,
        initial_state={'current_frame': current, 'history': history, 'valid_actions': ['ACTION3']},
        action_handler=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('no action expected')),
    )
    assert not result['error']
    assert result['result'] == {'history': 3, 'transitions': 2, 'last': 'ACTION3', 'previous_diff': True}
