"""Read-only monitor for the one root-launched startup smoke; never pushes/retries runs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REF = 'oscarmorke/arc-causal-startup-only-v3'
OUT = Path.home() / 'tuesday-night-2026-10-06/arc-kaggle/startup-smoke-v3'
TERMINAL = ('COMPLETE', 'ERROR', 'CANCELLED', 'CANCEL_ACKNOWLEDGED')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(name, value):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def verdict(status, output):
    path = output / 'startup_smoke_result.json'
    if not path.exists():
        return {'verdict': 'INVALID', 'reason': 'no startup smoke result', 'status': status, 'gameplay_score': None}
    data = json.loads(path.read_text())
    cleanup = data.get('cleanup', {})
    passed = (status.endswith('COMPLETE') and data.get('setup_completed') is True and data.get('passed') is True
              and data.get('gameplay_executed') is False and cleanup.get('stopped') is True
              and cleanup.get('port_open') is False and cleanup.get('survivors') == [])
    provenance = data.get('provenance', {})
    preflight = data.get('preflight', {})
    passed = passed and {'text', 'tool', 'vision', 'c1_throughput', 'c4_throughput', 'native_mtp'} <= preflight.keys() and not preflight.get('skipped', False)
    patch = output / 'startup_patch_receipt.json'
    patch_data = json.loads(patch.read_text()) if patch.exists() else {}
    passed = passed and patch_data.get('original_sha256') == '037c041c9bd9dcffa9084b32f47af9cf1bf35849d5eaf2e1a3422daac098b2e2'
    passed = passed and patch_data.get('executed_sha256') == '2928a57ceedb250c43cf21d4163fcfcdb0ac0009fbcc654e3eb1e8db1c15208c'
    passed = passed and provenance.get('server_ready_timeout_seconds') == 2400 and provenance.get('setup_total_timeout_seconds') == 2700
    return {'verdict': 'STARTUP_PASSED' if passed else 'INVALID', 'status': status,
            'gameplay_score': None, 'setup_elapsed_seconds': provenance.get('setup_elapsed_seconds'),
            'cleanup': cleanup, 'meaning': 'serving preflight only; no claim about gameplay or memory benefit',
            'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'no_retry': True}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Exclusive process marker prevents duplicate monitors from racing the same outputs.
    with (OUT / 'monitor.pid').open('x') as f:
        f.write(str(os.getpid()) + '\n')
    started = time.monotonic()
    deadline = started + 10800  # monitoring wall time includes queue; not GPU runtime authority
    write('monitor-started.json', {'utc': now(), 'pid': os.getpid(), 'ref': REF, 'max_monitor_seconds': 10800, 'launches': 0})
    while time.monotonic() < deadline:
        response = subprocess.run(['kaggle', 'kernels', 'status', REF], capture_output=True, text=True, timeout=90)
        raw = response.stdout + response.stderr
        state = {'utc': now(), 'pid': os.getpid(), 'ref': REF, 'status_output': raw, 'returncode': response.returncode}
        write('CURRENT.json', state)
        with (OUT / 'status.jsonl').open('a') as f:
            f.write(json.dumps(state) + '\n')
        matches = [s for s in TERMINAL if f'.{s}"' in raw]
        if response.returncode == 0 and matches:
            status = matches[0]
            output = OUT / 'output'; output.mkdir(exist_ok=True)
            with (OUT / 'pull.log').open('w') as log:
                result = subprocess.run(['kaggle', 'kernels', 'output', REF, '-p', str(output)], stdout=log, stderr=subprocess.STDOUT, timeout=900)
            if result.returncode:
                final = {'verdict': 'INVALID', 'reason': 'terminal output download incomplete', 'returncode': result.returncode}
            else:
                final = verdict(status, output)
            final.update(utc=now(), ref=REF, launched_version_reported_by_root=1)
            write('verdict.json', final)
            quota = subprocess.run(['kaggle', 'quota'], capture_output=True, text=True, timeout=90)
            write('quota-final.json', {'utc': now(), 'returncode': quota.returncode, 'output': quota.stdout + quota.stderr})
            write('MONITOR-FINISHED.json', final)
            return
        time.sleep(60)
    write('MONITOR-FINISHED.json', {'utc': now(), 'verdict': 'UNRESOLVED', 'reason': 'bounded monitor elapsed; no run retry', 'ref': REF})


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except BaseException as exc:
        write('MONITOR-ERROR.json', {'utc': now(), 'error': repr(exc), 'ref': REF, 'no_retry': True})
        raise
