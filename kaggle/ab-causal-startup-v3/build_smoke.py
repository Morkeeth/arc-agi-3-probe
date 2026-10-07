"""Prepare one startup-only kernel. This module never calls Kaggle or allocates GPU."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINNED_SOURCE_SHA256 = '037c041c9bd9dcffa9084b32f47af9cf1bf35849d5eaf2e1a3422daac098b2e2'
SESSION_TIMEOUT_SECONDS = 3600


def smoke_wrapper(wrapper):
    old = "exec(compile(source,str(original),'exec'),{'__name__':'__main__','__file__':str(original)})"
    if wrapper.count(old) != 1:
        raise ValueError('unexpected startup wrapper')
    new = '''scope = {'__name__': '__main__', '__file__': str(original)}
setup_completed = False
try:
    exec(compile(source, str(original), 'exec'), scope)
    setup_completed = True
finally:
    cleanup = scope['stop_owned_server']() if 'stop_owned_server' in scope else {'stopped': False, 'error': 'cleanup unavailable'}
    record = {'setup_completed': setup_completed, 'cleanup': cleanup, 'gameplay_executed': False}
    if setup_completed:
        record['preflight'] = json.loads(scope['PREFLIGHT_PATH'].read_text())
        record['provenance'] = json.loads(scope['SETUP_PROVENANCE'].read_text())
    record['passed'] = bool(setup_completed and cleanup.get('stopped') and cleanup.get('port_open') is False and cleanup.get('survivors') == [] and not record.get('preflight', {}).get('skipped', False) and {'text', 'tool', 'vision', 'c1_throughput', 'c4_throughput', 'native_mtp'} <= record.get('preflight', {}).keys())
    scope.get('WORKING_DIR', original.parent).joinpath('startup_smoke_result.json').write_text(json.dumps(record, sort_keys=True, indent=2))
    if setup_completed and not record['passed']:
        raise RuntimeError('startup smoke did not pass preflight and cleanup')
'''
    return wrapper.replace(old, new).replace("'session_timeout_required':8400", "'session_timeout_required':3600")


def build(source_path, destination=None):
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != PINNED_SOURCE_SHA256:
        raise ValueError('serving source is not the diagnosed source')
    dest = destination or HERE / 'smoke'
    meta = json.loads((HERE / 'control/kernel-metadata.json').read_text())
    path = HERE / 'control' / meta['code_file']
    notebook = json.loads(path.read_text())
    original = copy.deepcopy(notebook)
    setup = ''.join(notebook['cells'][9]['source'])
    candidates = [n.value for n in ast.walk(ast.parse(setup)) if isinstance(n, ast.Constant) and isinstance(n.value, str) and 'startup_patch_receipt.json' in n.value]
    if len(candidates) != 1:
        raise ValueError('expected one startup wrapper')
    wrapper = candidates[0]
    if PINNED_SOURCE_SHA256 not in wrapper:
        raise ValueError('wrapper source identity differs')
    updated = smoke_wrapper(wrapper)
    cleanup = """import hashlib,json,os
from pathlib import Path
original=Path(os.environ['TAAF_KAGGLE_BUNDLE_DIR'])/'serving_setup.py'
source=original.read_text()
assert hashlib.sha256(source.encode()).hexdigest()==SOURCE_HASH, 'cleanup source drift'
scope={'__name__':'startup_cleanup_only','__file__':str(original)}
exec(compile(source,str(original),'exec'),scope)
result=scope['stop_owned_server']()
scope['WORKING_DIR'].joinpath('startup_timeout_cleanup.json').write_text(json.dumps(result,sort_keys=True))
""".replace('SOURCE_HASH',repr(PINNED_SOURCE_SHA256))
    direct = """# Invoke only the hash-pinned serving source. No mounted command list is executed.
assert not TRUE_SUBMISSION, 'startup smoke is not a competition submission'
source_slug = 'duck-qwen38-nvfp4-mtp-vllm-smoke-v1'
candidates = [Path('/kaggle/input/datasets/keithtyser') / source_slug, Path('/kaggle/input') / source_slug]
matching = [p for p in candidates if (p / 'serving_setup.py').is_file()]
assert len(matching) == 1, f'Expected one known source mount, got {matching}'
BUNDLE_DIR = matching[0]
assert __import__('hashlib').sha256((BUNDLE_DIR / 'serving_setup.py').read_bytes()).hexdigest() == SOURCE_HASH, 'source drift'
SETUP_ENV_PATH = WORKING_DIR / 'startup-smoke-env.json'
SETUP_ENV_PATH.write_text('{}')
env = os.environ.copy()
env.update(TAAF_KAGGLE_BUNDLE_DIR=str(BUNDLE_DIR), TAAF_KAGGLE_WORKING_DIR=str(WORKING_DIR), TAAF_KAGGLE_SETUP_ENV=str(SETUP_ENV_PATH), TAAF_KAGGLE_FAST_START='0')
try:
    subprocess.run([sys.executable, '-c', WRAPPER], check=True, env=env, cwd=WORKING_DIR, timeout=3000)
except subprocess.TimeoutExpired:
    # Killing the setup child does not prove its separately-sessioned vLLM server stopped.
    # Use the same source's identity-bound cleanup, never a broad process kill.
    subprocess.run([sys.executable, '-c', CLEANUP], check=True, env=env, cwd=WORKING_DIR, timeout=150)
    raise
""".replace('SOURCE_HASH',repr(PINNED_SOURCE_SHA256)).replace('WRAPPER',repr(updated)).replace('CLEANUP',repr(cleanup))
    # Two code cells only: shared environment profile and direct bounded serving preflight.
    notebook['cells'] = [copy.deepcopy(original['cells'][3]), {'cell_type':'code', 'metadata':{}, 'source':direct.splitlines(keepends=True), 'outputs':[], 'execution_count':None}]
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            ast.parse(''.join(cell['source']))
    slug = 'arc-causal-startup-only-v3'
    meta.update(id='oscarmorke/' + slug, title=slug, code_file=slug + '.ipynb')
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / meta['code_file']
    output.write_text(json.dumps(notebook, indent=1) + '\n')
    (dest / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    package = {'launch_authorized': False, 'session_timeout_seconds_required': SESSION_TIMEOUT_SECONDS,
               'maximum_requested_gpu_hours': 1.0, 'kernel_count': 1, 'retries': 0,
               'source_sha256': PINNED_SOURCE_SHA256, 'base_notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'notebook_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'metadata_sha256': hashlib.sha256((dest / 'kernel-metadata.json').read_bytes()).hexdigest(),
               'retained_base_cells': [3], 'direct_serving_cell': True, 'arbitrary_setup_commands_executed': False, 'child_timeout_seconds': 3000, 'timeout_cleanup_seconds': 150,
               'success': 'full serving preflights plus stopped owned process, closed port and no survivors; no gameplay score',
               'failure_action': 'stop; no retry or pair launch', 'requires': 'fresh input identity check and explicit root authorization before upload'}
    (dest / 'package.json').write_text(json.dumps(package, indent=2) + '\n')
    return package


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source), indent=2))
