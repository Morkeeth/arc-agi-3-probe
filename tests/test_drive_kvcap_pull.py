"""The night driver must judge a run on a complete pull, never on a half download.

26 Sep 2026: `kaggle kernels output` timed out on kaggleusercontent.com mid-download
(drive.log lines 16, 83, 191, 258). ctx16k-s2 arrived without its kernel log and
without vllm-openai-server.log, was marked final COMPLETE anyway, and the verdict
called it invalid for missing proofs the notebook had written. Seed 2 of kv7 was
also pushed after seed 1 had been given up.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AB = "ctx16k-s2"
TIMEOUT = ("HTTPSConnectionPool(host='www.kaggleusercontent.com', port=443): Max retries exceeded "
           "(Caused by ConnectTimeoutError(..., 'Connection to www.kaggleusercontent.com timed out.'))")


@pytest.fixture
def drv(tmp_path, monkeypatch):
    monkeypatch.setenv("KVCAP_OUT", str(tmp_path / "out"))
    monkeypatch.delenv("KVCAP_DRY", raising=False)
    spec = importlib.util.spec_from_file_location("drive_kvcap_t", ROOT / "kaggle" / "ab-kvcap" / "drive_kvcap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT.mkdir(parents=True, exist_ok=True)
    mod.STATE.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "k_status", lambda ab: "COMPLETE")
    mod.mark(f"push-{AB}", "version 1")
    return mod


def _fake_kaggle(monkeypatch, drv, complete: bool, bulk_ok: bool = True):
    """Fake `kaggle kernels output`. complete=False: the verdict-file download times out.
    bulk_ok=False: the verdict files arrive, the bulk artifacts time out."""
    real_run = subprocess.run
    calls = []

    def run(cmd, *a, **kw):
        if not (isinstance(cmd, list) and cmd[:3] == ["kaggle", "kernels", "output"]):
            return real_run(cmd, *a, **kw)
        dest = Path(cmd[cmd.index("-p") + 1])
        dest.mkdir(parents=True, exist_ok=True)
        pattern = cmd[cmd.index("--file-pattern") + 1] if "--file-pattern" in cmd else ".*"
        calls.append(pattern)
        verdict_stage = "benchmark" in pattern and "?!" not in pattern
        text, rc = "", 0
        if pattern == ".*" or verdict_stage:
            (dest / "benchmark.json").write_text("{}")
            (dest / "deadprobe_counters.jsonl").write_text('{"calls": 1}\n')
            if complete:
                (dest / "vllm-openai-server.log").write_text("reserved 5.0 GiB memory for KV Cache\nRunning: 3 reqs\n")
                (dest / "vllm-metrics-final.prom").write_text("x 1\n")
                (dest / f"ab-{AB}.log").write_text(f"DEADPROBE_LIVE ab={AB} seed=2 kv=5.0\nCTX16K_PROOF env=16384\n")
            else:
                text, rc = TIMEOUT + "\n", 1
        if pattern == ".*" or not verdict_stage:
            (dest / "artifacts").mkdir(exist_ok=True)
            if not bulk_ok:
                text, rc = TIMEOUT + "\n", 1
        out = kw.get("stdout")
        if hasattr(out, "write"):
            out.write(text)
            return subprocess.CompletedProcess(cmd, rc)
        return subprocess.CompletedProcess(cmd, rc, stdout=text, stderr="")

    monkeypatch.setattr(drv.subprocess, "run", run)
    return calls


def test_partial_download_is_not_final(drv, monkeypatch):
    _fake_kaggle(monkeypatch, drv, complete=False)
    drv.handle(AB)
    assert not drv.marked(f"final-{AB}"), "a pull that timed out was accepted as the final run"
    assert "incomplete" in drv.LOG.read_text()


def test_complete_download_is_final_with_its_proof(drv, monkeypatch):
    _fake_kaggle(monkeypatch, drv, complete=True)
    drv.handle(AB)
    assert drv.read(f"final-{AB}") == "COMPLETE"
    assert f"ab={AB}" in (drv.OUT / AB / "config_proof.txt").read_text()


def test_incomplete_pull_retries_then_is_set_aside_not_judged(drv, monkeypatch):
    _fake_kaggle(monkeypatch, drv, complete=False)
    for _ in range(drv.PULL_TRIES):
        drv.handle(AB)
    assert drv.read(f"final-{AB}") == "PULL_INCOMPLETE"
    # The verdict globs "*-s[0-9]". A set-aside dir must not match, so it is not judged invalid.
    assert not (drv.OUT / AB).exists()
    assert any(p.name.startswith(f"{AB}-pullincomplete-") for p in drv.OUT.iterdir())


def test_seed_two_waits_for_a_valid_seed_one(drv):
    s1 = "kv7-s1"
    assert drv.s2_ready(s1) == (False, "s1 not final")
    drv.mark(f"final-{s1}", "ERROR")
    ok, why = drv.s2_ready(s1)
    assert not ok and "ERROR" in why
    drv.mark(f"final-{s1}", "PULL_INCOMPLETE")
    assert drv.s2_ready(s1)[0] is False
    drv.mark(f"final-{s1}", "COMPLETE")
    assert drv.s2_ready(s1) == (True, "s1 COMPLETE")


def test_error_run_waits_for_a_full_pull_before_deciding(drv, monkeypatch):
    monkeypatch.setattr(drv, "k_status", lambda ab: "ERROR")
    pushed = []
    monkeypatch.setattr(drv, "push_once", lambda ab, retry=False: pushed.append((ab, retry)))
    _fake_kaggle(monkeypatch, drv, complete=False)
    drv.handle(AB)
    assert not drv.marked(f"final-{AB}") and pushed == [], "decided on a pull that timed out"
    _fake_kaggle(monkeypatch, drv, complete=True)
    drv.handle(AB)
    assert pushed == [(AB, True)], "after a full pull, the documented single retry"
    assert len([p for p in drv.OUT.iterdir() if p.name.startswith(f"{AB}-invalid-")]) == 1


def test_verdict_files_are_enough_when_the_bulk_pull_fails(drv, monkeypatch):
    calls = _fake_kaggle(monkeypatch, drv, complete=True, bulk_ok=False)
    drv.handle(AB)
    assert drv.read(f"final-{AB}") == "COMPLETE"
    assert calls and "benchmark" in calls[0] and "?!" not in calls[0], "verdict files first"
    assert "bulk artifacts pull failed" in drv.LOG.read_text()


def test_a_half_pull_is_never_where_the_verdict_looks(drv, monkeypatch):
    _fake_kaggle(monkeypatch, drv, complete=False)
    drv.handle(AB)
    # The verdict globs "*-s[0-9]"; a retrying pull must not sit there.
    assert not (drv.OUT / AB).exists()
    assert not [p for p in drv.OUT.glob("*-s[0-9]")]


def test_every_verdict_file_is_required(drv, monkeypatch):
    _fake_kaggle(monkeypatch, drv, complete=True)
    real = drv.subprocess.run

    def drop_metrics(cmd, *a, **kw):
        r = real(cmd, *a, **kw)
        if isinstance(cmd, list) and cmd[:3] == ["kaggle", "kernels", "output"]:
            (Path(cmd[cmd.index("-p") + 1]) / "vllm-metrics-final.prom").unlink(missing_ok=True)
        return r

    monkeypatch.setattr(drv.subprocess, "run", drop_metrics)
    drv.handle(AB)
    assert not drv.marked(f"final-{AB}"), "a pull without vllm-metrics-final.prom was accepted"


def test_error_run_never_decides_blind(drv, monkeypatch):
    monkeypatch.setattr(drv, "k_status", lambda ab: "ERROR")
    pushed = []
    monkeypatch.setattr(drv, "push_once", lambda ab, retry=False: pushed.append((ab, retry)))
    _fake_kaggle(monkeypatch, drv, complete=False)
    for _ in range(drv.PULL_TRIES):
        drv.handle(AB)
    assert pushed == [], "a retry or fallback was pushed without the kernel log"
    assert drv.read(f"final-{AB}") == "ERROR PULL_FAILED"

@pytest.mark.parametrize('log_content', [None, ''])
def test_zero_exit_without_exact_kernel_log_never_retries(drv, monkeypatch, log_content):
    monkeypatch.setattr(drv, 'k_status', lambda ab: 'ERROR')
    pushed = []
    monkeypatch.setattr(drv, 'push_once', lambda ab, retry=False: pushed.append((ab, retry)))
    _fake_kaggle(monkeypatch, drv, complete=True)
    real = drv.subprocess.run

    def omit_log(cmd, *a, **kw):
        result = real(cmd, *a, **kw)
        if isinstance(cmd, list) and cmd[:3] == ['kaggle', 'kernels', 'output']:
            dest = Path(cmd[cmd.index('-p') + 1])
            log = dest / f'ab-{AB}.log'
            if log_content is None:
                log.unlink(missing_ok=True)
            else:
                log.write_text(log_content)
            # Another run's nonempty log cannot substitute for this run.
            (dest / 'ab-other-s1.log').write_text('OutOfMemoryError\n')
        return result

    monkeypatch.setattr(drv.subprocess, 'run', omit_log)
    for _ in range(drv.PULL_TRIES):
        drv.handle(AB)
    assert pushed == [], 'zero-exit download without exact kernel log must not spend quota'
    assert drv.read(f'final-{AB}') == 'ERROR PULL_FAILED'

@pytest.mark.parametrize('own_log,expected', [
    ('OutOfMemoryError\n', [('kv7-s1', False)]),
    ('Unrelated runtime error\n', [('kv8-s1', True)]),
])
def test_only_exact_kernel_log_controls_oom_fallback(drv, monkeypatch, own_log, expected):
    ab = 'kv8-s1'
    drv.mark(f'push-{ab}', 'version 1')
    monkeypatch.setattr(drv, 'k_status', lambda ab: 'ERROR')
    pushed = []
    monkeypatch.setattr(drv, 'push_once', lambda ab, retry=False: pushed.append((ab, retry)))

    def download(cmd, *a, **kw):
        assert cmd[:3] == ['kaggle', 'kernels', 'output']
        dest = Path(cmd[cmd.index('-p') + 1])
        dest.mkdir(exist_ok=True)
        # Error runs can lack benchmark outputs; their exact log is sufficient.
        (dest / f'ab-{ab}.log').write_text(own_log)
        (dest / 'ab-unrelated-s1.log').write_text('OutOfMemoryError\n')
        return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')

    monkeypatch.setattr(drv.subprocess, 'run', download)
    drv.handle(ab)
    assert pushed == expected

@pytest.mark.parametrize('ab,expected', [('kv7-s1', []), ('kv8-s1', [('kv7-s1', False)])])
def test_server_oom_is_not_hidden_by_kernel_summary(drv, monkeypatch, ab, expected):
    # Real kv7 failure: kernel ends with zero actions; OOM is only in vLLM log.
    drv.mark(f'push-{ab}', 'version 1')
    monkeypatch.setattr(drv, 'k_status', lambda _: 'ERROR')
    pushed = []
    monkeypatch.setattr(drv, 'push_once', lambda ab, retry=False: pushed.append((ab, retry)))
    def download(arm, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f'ab-{arm}.log').write_text('RuntimeError: Public runs produced no actions.\n')
        (dest / 'vllm-openai-server.log').write_text('torch.OutOfMemoryError: CUDA out of memory.\n')
        return True, 'ok'
    monkeypatch.setattr(drv, 'k_output', download)
    drv.handle(ab)
    assert pushed == expected
