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
            (dest / "deadprobe_counters.jsonl").write_text("")
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
