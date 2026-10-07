"""CPU boundary tests: no vLLM, model request, Kaggle allocation or claimed GPU success."""
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'kaggle/ab-causal-startup-v3'
spec = importlib.util.spec_from_file_location('build_smoke', HERE / 'build_smoke.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_notebook_has_no_gameplay_and_enforces_full_preflight():
    meta = json.loads((HERE / 'smoke/kernel-metadata.json').read_text())
    n = json.loads((HERE / 'smoke' / meta['code_file']).read_text())
    assert len(n['cells']) == 2
    assert 'assert not TRUE_SUBMISSION' in ''.join(n['cells'][1]['source'])
    setup = ''.join(n['cells'][1]['source'])
    assert "TAAF_KAGGLE_FAST_START='0'" in setup
    assert 'timeout=3000' in setup and 'timeout=150' in setup
    assert 'shell=True' not in setup and 'setup_commands.json' not in setup
    wrappers = [x.value for x in ast.walk(ast.parse(setup)) if isinstance(x, ast.Constant) and isinstance(x.value, str) and 'startup_smoke_result.json' in x.value]
    assert len(wrappers) == 1
    ast.parse(wrappers[0])
    assert "'session_timeout_required':3600" in wrappers[0]
    assert not any('bm.run(' in ''.join(c['source']) for c in n['cells'])


@pytest.mark.parametrize('fails,cleanup_ok,skipped,expected', [(False, True, False, True), (True, True, False, False), (False, False, False, False), (False, True, True, False)])
def test_wrapper_executes_and_cleans_up_on_success_and_failure(tmp_path, fails, cleanup_ok, skipped, expected):
    # Model-serving effect is stubbed; wrapper exception and teardown logic are real.
    source = '''from pathlib import Path
WORKING_DIR = Path(__file__).parent
PREFLIGHT_PATH = WORKING_DIR / 'preflight.json'
SETUP_PROVENANCE = WORKING_DIR / 'provenance.json'
def stop_owned_server():
    WORKING_DIR.joinpath('cleanup-called').write_text('yes')
    return CLEANUP
'''.replace('CLEANUP', repr({'stopped': cleanup_ok, 'port_open': not cleanup_ok, 'survivors': [] if cleanup_ok else [42]}))
    (tmp_path / 'preflight.json').write_text(json.dumps(dict.fromkeys(['text', 'tool', 'vision', 'c1_throughput', 'c4_throughput', 'native_mtp'], {}) | ({'skipped': True} if skipped else {})))
    (tmp_path / 'provenance.json').write_text('{}')
    if fails:
        source += '\nraise RuntimeError("startup failed")\n'
    base = "exec(compile(source,str(original),'exec'),{'__name__':'__main__','__file__':str(original)})"
    scope = {'source': source, 'original': tmp_path / 'serving_setup.py', 'json': json}
    if expected:
        exec(m.smoke_wrapper(base), scope)
    else:
        with pytest.raises(RuntimeError):
            exec(m.smoke_wrapper(base), scope)
    assert (tmp_path / 'cleanup-called').exists()
    result = json.loads((tmp_path / 'startup_smoke_result.json').read_text())
    assert result['passed'] is expected
    assert result['gameplay_executed'] is False


def test_source_drift_refused(tmp_path):
    bad = tmp_path / 'serving_setup.py'
    bad.write_text('changed source')
    with pytest.raises(ValueError, match='diagnosed source'):
        m.build(bad, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def load_real_wait(patched, ready_at, dies_at=None):
    source_path = os.environ.get('ARC_DIAGNOSED_SERVING_SOURCE')
    if not source_path:
        pytest.skip('set ARC_DIAGNOSED_SERVING_SOURCE to replay exact archived startup function')
    source = Path(source_path).read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == m.PINNED_SOURCE_SHA256
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_bounded_timeout', 'wait_for_server')]
    clock = SimpleNamespace(now=0.0)
    def request_json(url, timeout):
        if clock.now < ready_at:
            raise ConnectionRefusedError()
        return {'data': [{'id': 'pinned-model'}]}
    def sleep(seconds):
        clock.now += seconds
    scope = {'Any': Any, 'math': math, 'time': SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep),
             'SERVER_READY_TIMEOUT': 2400 if patched else 1500, 'SETUP_TOTAL_TIMEOUT': 2700 if patched else 1800,
             'alive': lambda *args: dies_at is None or clock.now < dies_at, 'tail_log': lambda n: '',
             'request_json': request_json, 'BASE_URL': 'http://local-test', 'SERVED_MODEL_NAME': 'pinned-model', 'print': lambda *args, **kwargs: None}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), 'exec'), scope)
    return scope['wait_for_server'], clock


def test_real_wait_original_fails_and_candidate_accepts_late_readiness():
    wait, clock = load_real_wait(False, 1600)
    with pytest.raises(TimeoutError):
        wait({'pid': 1, 'start_ticks': 1}, 1800)
    assert clock.now == 1500
    wait, clock = load_real_wait(True, 1600)
    result = wait({'pid': 1, 'start_ticks': 1}, 2700)
    assert result['ready_seconds'] == 1600


def test_real_wait_remains_bounded_and_process_death_is_not_retried():
    wait, clock = load_real_wait(True, 3000)
    with pytest.raises(TimeoutError):
        wait({'pid': 1, 'start_ticks': 1}, 1700)
    assert clock.now == 1700
    wait, clock = load_real_wait(True, 1600, dies_at=30)
    with pytest.raises(RuntimeError, match='exited'):
        wait({'pid': 1, 'start_ticks': 1}, 2700)
    assert clock.now == 30

@pytest.mark.parametrize('times_out', [False, True])
def test_generated_direct_cell_runs_only_bounded_known_source(tmp_path, times_out):
    import subprocess
    import sys
    source_path = os.environ.get('ARC_DIAGNOSED_SERVING_SOURCE')
    if not source_path:
        pytest.skip('requires exact archived serving source')
    mount = tmp_path / 'mount'
    mount.mkdir()
    (mount / 'serving_setup.py').write_bytes(Path(source_path).read_bytes())
    work = tmp_path / 'work'; work.mkdir()
    meta = json.loads((HERE / 'smoke/kernel-metadata.json').read_text())
    notebook = json.loads((HERE / 'smoke' / meta['code_file']).read_text())
    cell = ''.join(notebook['cells'][1]['source'])
    def mounted_path(value):
        if value == '/kaggle/input/datasets/keithtyser':
            return mount.parent / 'known'
        if value == '/kaggle/input':
            return tmp_path / 'absent'
        return Path(value)
    known = mount.parent / 'known' / 'duck-qwen38-nvfp4-mtp-vllm-smoke-v1'
    known.parent.mkdir()
    mount.rename(known)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if times_out and len(calls) == 1:
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    scope = dict(TRUE_SUBMISSION=False, Path=mounted_path, WORKING_DIR=work,
                 os=os, sys=sys, subprocess=SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired))
    if times_out:
        with pytest.raises(subprocess.TimeoutExpired):
            exec(cell, scope)
    else:
        exec(cell, scope)
    assert [kw['timeout'] for _, kw in calls] == ([3000, 150] if times_out else [3000])
    assert all(cmd[:2] == [sys.executable, '-c'] for cmd, _ in calls)
    assert calls[0][1]['env']['TAAF_KAGGLE_FAST_START'] == '0'
    if times_out:
        assert 'startup_cleanup_only' in calls[1][0][2]
