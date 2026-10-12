"""Build two private single-variable screens from the frozen 12 Oct Intel control.

No launch, paid compute, or submission in this builder. The existing control
has the parent-only action reporting fix; treatments retain every budget cell.
"""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / 'submit-20261012'
RECEIPTS = Path('/Users/morkeeth/.local/state/fleet/arc-runpod-20261011/free-tests-20261012')
CONTROL_OUTPUT = RECEIPTS.parent / 'private-validation-output/benchmark.json'

COMPACT_PATCH = r'''
import json
from pathlib import Path
from inference.agent.tool_agent import ToolAgent

_original_analyze = ToolAgent.analyze
_original_prompt = ToolAgent._build_user_prompt

def _visible_text(message):
    content = message.get('content') or ''
    if isinstance(content, list):
        content = ' '.join(str(p.get('text', '')) for p in content if isinstance(p, dict))
    return ' '.join(str(content).split())

def _compact_analyze(self, state_path, *args, **kwargs):
    self._ensure_session(state_path)
    history = self._history_messages
    starts = [i for i, m in enumerate(history) if m.get('role') == 'user']
    cut = starts[-2] if len(starts) > 2 else 0
    dropped = history[:cut]
    parts = []
    for message in dropped:
        role = message.get('role')
        if role == 'assistant':
            text = _visible_text(message)
            if text:
                parts.append('Earlier model claim/plan (unverified): ' + text[:400])
            for call in (message.get('tool_calls') or [])[:3]:
                arguments = call.get('function', {}).get('arguments', '{}')
                try:
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    code = str(arguments.get('code', ''))
                    parts.append('Earlier proposed Python (not execution proof): ' + code[:200])
                except (ValueError, TypeError, AttributeError):
                    pass
        elif role == 'tool':
            parts.append('Observed earlier tool response: ' + _visible_text(message)[:700])
    previous = getattr(self, '_compact_record', '')
    self._compact_record = (previous + '\n' + '\n'.join(parts))[-4000:]
    self._history_messages = history[cut:]
    self._compact_dropped_this_turn = len(dropped)
    return _original_analyze(self, state_path, *args, **kwargs)

def _compact_prompt(self, *args, **kwargs):
    frame = kwargs.get('current_frame')
    scope = (str(self._session_runtime_dir), getattr(frame, 'level', None))
    if getattr(self, '_compact_scope', None) != scope:
        self._compact_record = ''
        self._compact_scope = scope
    prompt = _original_prompt(self, *args, **kwargs)
    record = getattr(self, '_compact_record', '')
    if record:
        prompt += '\nCompact decision record (past observations; revise against current frame):\n' + record
    path = Path('/kaggle/working/compact-events.jsonl')
    with path.open('a') as file:
        file.write(json.dumps({'runtime_dir': scope[0], 'level': scope[1],
            'dropped_messages': getattr(self, '_compact_dropped_this_turn', 0),
            'retained_messages': len(self._history_messages), 'record_chars': len(record)}) + '\n')
    return prompt

ToolAgent.analyze = _compact_analyze
ToolAgent._build_user_prompt = _compact_prompt
print('COMPACT_DECISION_PROOF retained_user_blocks=2 record_chars=4000 no_extra_model_calls')
'''

FORCED_PROMPT = '''
At the start of each NEW GAME, before any action, inspect the actual frame, write
a self-contained def predict(grid, action) for one supported action, register it
with mechanics.register(source), then call mechanics.test(action) for your first
real action. Its prediction is committed before the action and checked against
the actual next frame. A mismatch is evidence to revise, never permission to
claim a solved mechanic. After that first prospective test use the original
action policy freely. This mandatory opening spends one of the SAME action
budget, not an extra action. mechanics.status alone does not satisfy the gate.
'''

FORCED_PATCH = r'''
import json
from pathlib import Path
from inference.agent import mechanics_runtime

_dispatch = mechanics_runtime.MechanicsHost.dispatch
def _record_dispatch(self, message):
    before = len(self.action_results)
    counts = self.mechanics.state.setdefault('counts', {})
    counts['_test_active'] = message.get('method') == 'test'
    try:
        result = _dispatch(self, message)
    finally:
        counts['_test_active'] = False
    state = self.mechanics.state
    observed = self.action_results[before:]
    if message.get('method') == 'test' and any(r.get('executed') for r in observed):
        state.setdefault('counts', {})['opening_success'] = 1
    with Path('/kaggle/working/mechanics-calls.jsonl').open('a') as file:
        file.write(json.dumps({'method': message.get('method'),
            'level': state.get('level'), 'models': len(state.get('models', {})),
            'real_action_results': observed,
            'last_event_status': (state.get('events') or [{}])[-1].get('status'),
            'result': result}, default=str) + '\n')
    return result
mechanics_runtime.MechanicsHost.dispatch = _record_dispatch
print('FORCED_MECHANICS_PROOF register_and_test_before_first_action')
'''

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sources(notebook):
    return [''.join(c['source']) for c in notebook['cells']]

def build():
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    base = json.loads((BASE / 'experiment.ipynb').read_text())
    live_files = list((RECEIPTS / 'control-live').glob('*.ipynb'))
    assert len(live_files) == 1
    live = json.loads(live_files[0].read_text())
    assert sources(base) == sources(live), 'Live control differs from local base'
    benchmark = json.loads(CONTROL_OUTPUT.read_text())
    rows = benchmark['game_runs']
    assert len(rows) == 25 and len({r['game_id'] for r in rows}) == 25
    assert all(r['state'] in ('won', 'gave_up') and isinstance(r['final_score'], (int, float)) for r in rows)
    summary = {'games': 25, 'wins': sum(r['state'] == 'won' for r in rows),
               'levels': sum(r['levels_completed'] for r in rows),
               'mean_score': sum(r['final_score'] for r in rows) / 25,
               'actions': sum(len(r['history']) for r in rows)}
    seal = {'control_ref': 'oscarmorke/arc3-intel-control-submission-20261012',
            'control_version': 1, 'notebook_sha256': digest(BASE / 'experiment.ipynb'),
            'cell_sources_sha256': hashlib.sha256(''.join(sources(base)).encode()).hexdigest(),
            'benchmark_sha256': digest(CONTROL_OUTPUT), 'control_summary': summary,
            'games': [r['game_id'] for r in rows], 'arms': {},
            'interpretation': 'One stochastic private screen per treatment versus a frozen historical run; no significance or public-score claim.',
            'compact_gate': '25 valid pairs, wins >= control and (levels > control or mean paired score delta > 0), actual context drops required',
            'mechanics_gate': '25 valid pairs, >=20 games with registered predictor and observed real test action, wins and levels >= control and mean paired score delta > 0',
            'platform_timeout_seconds': 10800, 'runpod_spend': 0, 'competition_submission': False}
    for arm in ('compact', 'forced-mechanics'):
        n = copy.deepcopy(base)
        setup = ''.join(n['cells'][2]['source'])
        if arm == 'compact':
            patch = COMPACT_PATCH
        else:
            # This replaces the existing control install with the same toolkit
            # treatment, and enforces only its first-game prospective test.
            installer = (HERE / 'install_policy.py').read_text()
            installer = installer.replace("PROMPT = '''", 'PROMPT = ' + repr(FORCED_PROMPT) + " + '''", 1)
            anchor = '            raw_payload = self._step_env_callback(step_arguments)'
            gate = '''            if not mechanics_state.get('counts', {}).get('opening_success'):
                if not mechanics_state.get('counts', {}).get('_test_active') or not mechanics_state.get('models'):
                    raise RuntimeError('First game action requires mechanics.register(predictor), then mechanics.test(action)')
'''
            inject = "        t=once(t," + repr(anchor) + ',' + repr(gate + anchor) + ')\n'
            installer = installer.replace("        t=once(t,'        sandbox_result = run_sandboxed_python(',", inject + "        t=once(t,'        sandbox_result = run_sandboxed_python(',", 1)
            evidence_anchor = '        (state_path.parent / "mechanics-evidence.json").write_text(json.dumps(mechanics_state))'
            evidence_append = '''
        with Path('/kaggle/working/mechanics-games.jsonl').open('a') as evidence_file:
            evidence_file.write(json.dumps({'runtime_dir': str(state_path.parent),
                'counts': mechanics_state.get('counts', {}),
                'observed_tests': sum(e.get('status') == 'observed' for e in mechanics_state.get('events', []))}) + '\\n')
'''
            installer = installer.replace('        t=once(t,' + repr('    _prefix = os.environ.get("ARC3_SYSTEM_PROMPT_PREFIX", "").strip()'),
                '        t=once(t,' + repr(evidence_anchor) + ',' + repr(evidence_anchor + evidence_append) + ')\n' +
                '        t=once(t,' + repr('    _prefix = os.environ.get("ARC3_SYSTEM_PROMPT_PREFIX", "").strip()'), 1)
            old = "subprocess.run([sys.executable,str(policy_dir/'install_policy.py'),str(BUNDLE_DIR/'src/ARC3-Inference'),'control'],check=True,timeout=30)"
            new = "(policy_dir/'install_policy.py').write_text(" + repr(installer) + ")\n" + old.replace("'control'", "'treatment'")
            assert setup.count(old) == 1
            setup = setup.replace(old, new)
            patch = FORCED_PATCH
        # Install Python wrapper before unpickling the solver imports tool_agent.
        setup += "\nexperiment_patch=Path('/kaggle/working/experiment_patch.py')\nexperiment_patch.write_text(" + repr(patch) + ")\n"
        n['cells'][2]['source'] = setup.splitlines(True)
        preload = "exec(compile(Path('/kaggle/working/experiment_patch.py').read_text(), 'experiment_patch.py', 'exec'))\n"
        n['cells'][7]['source'] = (preload + ''.join(n['cells'][7]['source'])).splitlines(True)
        changed = [i for i, (a, b) in enumerate(zip(sources(base), sources(n))) if a != b]
        assert changed == [2, 7]
        # Common guard: private notebooks cannot accidentally be submitted.
        n['cells'][0]['source'] = ("assert not __import__('os').environ.get('KAGGLE_IS_COMPETITION_RERUN', '').lower() in ('1','true'), 'Private experiment only'\n" + ''.join(n['cells'][0]['source'])).splitlines(True)
        first = ''.join(n['cells'][0]['source'])
        input_mapping = '''def resolve_input(canonical, flat):
    for path in (Path(canonical), Path(flat)):
        if path.is_dir(): return path
    return Path(canonical)
INPUT_ROOTS={ref:resolve_input('/kaggle/input/datasets/'+ref,
    '/kaggle/input/'+ref.split('/')[-1]) for ref in EXPECTED_INPUTS}
COMPETITION_INPUT=resolve_input('/kaggle/input/competitions/arc-prize-2026-arc-agi-3',
    '/kaggle/input/arc-prize-2026-arc-agi-3')
'''
        first = first.replace("for ref,rows in EXPECTED_INPUTS.items():\n", input_mapping + "for ref,rows in EXPECTED_INPUTS.items():\n", 1)
        first = first.replace("root=Path('/kaggle/input/datasets')/ref", 'root=INPUT_ROOTS[ref]')
        first = first.replace("Path('/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy')", "INPUT_ROOTS['dfranzen/taaf-kaggle-source-bundle-copy']")
        mount_check = "for ref,rows in EXPECTED_INPUTS.items():\n"
        readiness = '''mount_deadline=time.monotonic()+120
while time.monotonic()<mount_deadline:
    missing=[]
    for ref,rows in EXPECTED_INPUTS.items():
        for row in rows:
            path=INPUT_ROOTS[ref]/row['name']
            if not path.is_file() or path.stat().st_size!=row.get('totalBytes',0):
                missing.append({'path':str(path),'expected_bytes':row.get('totalBytes',0),
                    'actual_bytes':path.stat().st_size if path.is_file() else None})
    Path('/kaggle/working/input-preflight.json').write_text(json.dumps({'missing':missing,
        'input_roots':[str(p) for p in Path('/kaggle/input').glob('*')]}))
    if not missing: break
    time.sleep(5)
print('INPUT_PREFLIGHT='+Path('/kaggle/working/input-preflight.json').read_text())
'''
        assert first.count(mount_check) == 1
        first = first.replace(mount_check, readiness + mount_check, 1)
        n['cells'][0]['source'] = first.splitlines(True)
        # GPU sessions expose legacy flat data mounts; CPU mounts are nested.
        # Resolve only the two observed layouts, then keep exact bytes/hashes.
        setup = ''.join(n['cells'][2]['source'])
        setup = setup.replace("'/kaggle/input/datasets/dfranzen/pennyroyal-v253'", "str(INPUT_ROOTS['dfranzen/pennyroyal-v253'])")
        setup = setup.replace("'/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'", "str(INPUT_ROOTS['dfranzen/taaf-kaggle-source-bundle-copy'])")
        n['cells'][2]['source'] = setup.splitlines(True)
        for index in (4, 10):
            source = ''.join(n['cells'][index]['source'])
            source = source.replace('"/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"', "str(COMPETITION_INPUT/'arc_agi_3_wheels')")
            n['cells'][index]['source'] = source.splitlines(True)
        for c in n['cells']:
            if c['cell_type'] == 'code':
                source = ''.join(c['source'])
                if not source.startswith('%%writefile '):
                    # Shell magics are inherited unchanged from the live base.
                    python_lines = [line[:len(line)-len(line.lstrip())] + 'pass\n'
                                    if line.lstrip().startswith('!') else line
                                    for line in source.splitlines(True)]
                    ast.parse(''.join(python_lines))
        dest = HERE / ('free-' + arm + '-20261012')
        dest.mkdir(exist_ok=True)
        notebook = dest / 'experiment.ipynb'
        notebook.write_text(json.dumps(n, indent=1) + '\n')
        metadata = json.loads((BASE / 'kernel-metadata.json').read_text())
        metadata.update(id='oscarmorke/arc3-intel-' + arm + '-20261012', title='ARC3 Intel ' + arm + ' 20261012', is_private=True)
        (dest / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        diff = ''.join(difflib.unified_diff(''.join(sources(base)).splitlines(True), ''.join(sources(n)).splitlines(True), fromfile='frozen-control', tofile=arm))
        (dest / 'source.diff').write_text(diff)
        seal['arms'][arm] = {'path': str(dest), 'ref': metadata['id'], 'notebook_sha256': digest(notebook), 'changed_cells': [0, 2, 4, 7, 10],
            'common_repair': 'Observed GPU flat mounts versus CPU nested mounts; exact expected sizes and bundle hashes still required'}
    (RECEIPTS / 'freeze.json').write_text(json.dumps(seal, indent=2) + '\n')
    print(json.dumps(seal, indent=2))

if __name__ == '__main__':
    build()
