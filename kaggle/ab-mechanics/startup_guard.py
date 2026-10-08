"""Bounded serving readiness; standard library only, also exercised on CPU."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def stop_owned(proc, grace=5):
    # The launcher creates this process in its own session. Never kill by name/port.
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait(timeout=5)  # Reap the leader after its process group disappears.
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        proc.poll()
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            proc.wait(timeout=5)
            return
        time.sleep(min(.05, max(0, deadline-time.monotonic())))
    # A dead leader says nothing about its children. Escalate for the whole group.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


_HTTP_PROBE = r'''import json, sys
from urllib.request import Request, urlopen
url, payload, timeout = json.loads(sys.argv[1])
request = Request(url, json.dumps(payload).encode(), {'Content-Type': 'application/json'}) if payload is not None else url
with urlopen(request, timeout=timeout) as response:
    if payload is None:
        ok = response.status == 200
    else:
        value = json.load(response)
        ok = response.status == 200 and any(
            isinstance(c.get('text'), str) and c['text'].strip()
            for c in value.get('choices', []))
print(json.dumps(bool(ok)))
'''


def http_probe(url, timeout, payload=None):
    """A disposable process bounds headers AND body by elapsed time.

    Socket timeouts alone reset when bytes arrive. subprocess.run kills and reaps
    this child on expiration; it cannot keep a request alive after the deadline.
    No shell, threads, retry, or worker grandchildren are involved.
    """
    if timeout <= 0:
        raise TimeoutError('probe_deadline')
    result = subprocess.run(
        [sys.executable, '-I', '-S', '-c', _HTTP_PROBE,
         json.dumps([url, payload, timeout])],
        capture_output=True, text=True, timeout=timeout, check=True)
    return json.loads(result.stdout) is True


def wait_ready(proc, log, receipt, experiment_start, server_start, health, generate,
               *, clock=time.monotonic, sleep=time.sleep, cleanup=stop_owned,
               startup_limit=3600, idle_limit=900, total_limit=10800,
               reserve=1200, interval=3, log_offset=0):
    """Log changes extend the idle allowance, never the absolute deadline.

    Log activity is not proof of useful loader progress. Health AND a nonempty
    generation are required. All probe timeouts consume the same bounded budget.
    """
    deadline = min(server_start + startup_limit, experiment_start + total_limit - reserve)
    last_activity = server_start
    previous = None
    state = {'status': 'startup_pending', 'gameplay_started': False,
             'health_ok': False, 'generation_ok': False,
             'pre_server_seconds': server_start - experiment_start,
             'startup_limit_seconds': startup_limit, 'total_limit_seconds': total_limit,
             'reserve_seconds': reserve}
    try:
        while True:
            now = clock()
            state.update(server_seconds=now-server_start, total_seconds=now-experiment_start,
                         process_returncode=proc.poll())
            if proc.poll() is not None:
                raise RuntimeError('serving_process_exited')
            if now >= deadline:
                raise RuntimeError('startup_deadline')
            p = Path(log)
            size = p.stat().st_size if p.exists() else 0
            if previous is not None and size != previous:
                last_activity = now
            previous = size
            if now-last_activity >= idle_limit:
                raise RuntimeError('startup_log_idle')
            try:
                state['health_ok'] = bool(health(min(5, deadline-clock())))
            except Exception as exc:
                state['last_probe_error'] = str(exc)
                state['health_ok'] = False
            if state['health_ok']:
                # Once health is up, a generation failure fails closed; no retries.
                remaining = deadline-clock()
                if remaining <= 0:
                    raise RuntimeError('startup_deadline')
                state['generation_ok'] = bool(generate(min(120, remaining)))
                if clock() >= deadline:
                    raise RuntimeError('startup_deadline')
                if proc.poll() is not None:
                    raise RuntimeError('serving_process_exited')
                if not state['generation_ok']:
                    raise RuntimeError('generation_probe_failed')
                state['status'] = 'startup_ready'
                break
            sleep(max(0, min(interval, deadline-clock())))
    except BaseException as exc:
        state.update(status='startup_failed', error=str(exc))
        try:
            cleanup(proc)
        except Exception as cleanup_exc:
            state['cleanup_error'] = str(cleanup_exc)
        raise
    finally:
        state.update(server_seconds=clock()-server_start, total_seconds=clock()-experiment_start,
                     process_returncode=proc.poll())
        p = Path(log)
        if p.exists():
            with p.open('rb') as f:
                f.seek(0, 2)
                f.seek(max(log_offset, f.tell()-20000))
                state['serve_log_tail'] = f.read().decode(errors='replace')
        Path(receipt).write_text(json.dumps(state, indent=2))
    return state


def await_server(proc, log, working_dir, experiment_start, server_start, port, model, log_offset=0):
    base = f'http://127.0.0.1:{port}'
    def health(timeout):
        return http_probe(base+'/health', timeout)
    def generate(timeout):
        # 64 avoids the earlier backend minimum-token incompatibility.
        payload = {'model': model, 'prompt': 'Say hello.', 'max_tokens': 64,
                   'temperature': 0}
        return http_probe(base+'/v1/completions', timeout, payload)
    return wait_ready(proc, log, Path(working_dir)/'startup-result.json',
                      experiment_start, server_start, health, generate, log_offset=log_offset)
