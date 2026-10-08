import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('startup_guard', ROOT/'kaggle/ab-mechanics/startup_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class Clock:
    def __init__(self, value=0): self.value = value
    def __call__(self): return self.value
    def sleep(self, seconds): self.value += seconds


class Process:
    pid = -1
    def __init__(self): self.code = None
    def poll(self): return self.code


def scenario(tmp_path, **overrides):
    clock = Clock(100)
    proc = Process()
    log = tmp_path/'serve.log'
    log.write_text('loading 0/38')
    receipt = tmp_path/'result.json'
    cleaned = []
    opts = dict(clock=clock, sleep=clock.sleep, cleanup=lambda p: cleaned.append(p),
                startup_limit=3600, idle_limit=900)
    opts.update(overrides)
    def run(health, generate=lambda timeout: True):
        return guard.wait_ready(proc,log,receipt,0,100,health,generate,**opts)
    return clock, proc, log, receipt, cleaned, run


def test_slow_loader_outlives_original_720_second_deadline(tmp_path):
    clock, proc, log, receipt, cleaned, run = scenario(tmp_path)
    def health(timeout):
        log.write_text(f'loading {clock()}')
        return clock() >= 1400
    state = run(health)
    assert state['status'] == 'startup_ready'
    assert state['pre_server_seconds'] == 100
    assert state['server_seconds'] >= 1300
    assert state['generation_ok'] and not cleaned


def test_busy_logs_cannot_extend_absolute_deadline(tmp_path):
    clock, proc, log, receipt, cleaned, run = scenario(tmp_path, startup_limit=30)
    def health(timeout):
        log.write_text('x'*int(clock()))
        return False
    with pytest.raises(RuntimeError, match='startup_deadline'): run(health)
    data = json.loads(receipt.read_text())
    assert data['server_seconds'] == 30
    assert data['status'] == 'startup_failed'
    assert data['serve_log_tail'] and cleaned == [proc]
    assert not data['gameplay_started']


def test_total_ceiling_includes_installation_and_preserves_reserve(tmp_path):
    clock, proc, log, receipt, cleaned, run = scenario(tmp_path, total_limit=130, reserve=20)
    with pytest.raises(RuntimeError, match='startup_deadline'): run(lambda t: False)
    assert clock() == 110
    assert json.loads(receipt.read_text())['total_seconds'] == 110


def test_idle_loader_stops_early(tmp_path):
    clock, proc, log, receipt, cleaned, run = scenario(tmp_path, idle_limit=9)
    with pytest.raises(RuntimeError, match='startup_log_idle'): run(lambda t: False)
    assert clock() == 109 and cleaned == [proc]


def test_dead_process_fails_without_probes(tmp_path):
    clock, proc, log, receipt, cleaned, run = scenario(tmp_path)
    proc.code = 17
    def forbidden(timeout): raise AssertionError('must not probe')
    with pytest.raises(RuntimeError, match='serving_process_exited'): run(forbidden)
    assert json.loads(receipt.read_text())['process_returncode'] == 17


@pytest.mark.parametrize('result', [False, 'exception'])
def test_health_alone_never_allows_gameplay(tmp_path, result):
    clock, proc, log, receipt, cleaned, run = scenario(tmp_path)
    def generation(timeout):
        if result == 'exception': raise TimeoutError('generation timeout')
        return result
    with pytest.raises((RuntimeError, TimeoutError)): run(lambda t: True, generation)
    assert json.loads(receipt.read_text())['status'] == 'startup_failed'
    assert cleaned == [proc]


def test_probe_time_is_charged_to_deadline(tmp_path):
    clock, proc, log, receipt, cleaned, run = scenario(tmp_path, startup_limit=10)
    def generation(timeout):
        assert timeout <= 10
        clock.sleep(timeout)
        return True
    with pytest.raises(RuntimeError, match='startup_deadline'): run(lambda t: True, generation)
    assert cleaned == [proc]


def test_real_owned_process_cleanup(tmp_path):
    proc = subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], start_new_session=True)
    start = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match='startup_deadline'):
            guard.wait_ready(proc,tmp_path/'serve.log',tmp_path/'result.json',start,start,
                             lambda t: False,lambda t: True,startup_limit=.05,interval=.01)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()


def test_matched_arms_embed_current_guard_and_keep_gameplay_gated():
    notebooks = [json.loads((ROOT/f'kaggle/ab-mechanics/{arm}/experiment.ipynb').read_text())
                 for arm in ('control','treatment')]
    src = [''.join(n['cells'][6]['source']) for n in notebooks]
    assert src[0] == src[1]
    assert (ROOT/'kaggle/ab-mechanics/startup_guard.py').read_text() in src[0]
    assert "assert startup_result['status']=='startup_ready'" in src[0]
    assert 'releasing the benchmark' not in src[0]
    assert 'SERVER_STARTUP_TIMEOUT' not in src[0]
    assert [i for i,(a,b) in enumerate(zip(notebooks[0]['cells'],notebooks[1]['cells'])) if a != b] == [0,2,10]


def test_http_health_and_generation_with_cpu_server(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            calls.append(self.path)
            self.send_response(200); self.end_headers()
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, data))
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'choices':[{'text':'hello'}]}).encode())
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        now = time.monotonic()
        state = guard.await_server(Process(),tmp_path/'serve.log',tmp_path,now,now,
                                   server.server_port,'fake-cpu-model')
        assert state['status'] == 'startup_ready'
        assert calls[0] == '/health'
        assert calls[1] == ('/v1/completions', {'model':'fake-cpu-model','prompt':'Say hello.',
                                               'max_tokens':64,'temperature':0})
    finally:
        server.shutdown(); server.server_close(); thread.join()
