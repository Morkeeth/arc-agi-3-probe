"""Dry runs of the 27 Sep driver (fake clock, fake Kaggle), one per base-selection scenario. CPU only.

Each builds a fake runs/kvcap-2026-09-26 (verdict.json, DONE marker, base runs made from real control runs),
runs kaggle/ab-compact/drive_compact.py with COMPACT_DRY=1, and reads its log."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path.home() / "CODE/arc-agi-3-probe/runs/ab-2026-09-23"
pytestmark = pytest.mark.skipif(not (RUNS / "control-s2r/benchmark.json").exists(), reason="control runs missing")
IGN = shutil.ignore_patterns("transcripts", "prompts", "solver_analysis", "artifacts")


def _kvcap(tmp: Path, keep: str | None, done: bool = True) -> Path:
    d = tmp / "kvcap"
    (d / "state").mkdir(parents=True)
    if done:
        (d / "state/DONE").write_text("2026-09-26T09:00:00Z")
    arms = {"kv": {"mech": "LOSE"}, "ctx16k": {"mech": "LOSE"}}
    submit = {"slug": "oscarmorke/ab-q38-control-s1", "version": 1, "why": "no MECH keep"}
    if keep:
        arms["ctx16k" if keep == "ctx16k" else "kv"] = {"mech": "KEEP"}
        submit = {"ab_id": f"{keep}-s1", "slug": f"oscarmorke/ab-{keep}-s1", "version": None, "why": "MECH keep"}
        for k, src in ((1, "control-s1"), (2, "control-s2r")):
            b = d / f"{keep}-s{k}"
            shutil.copytree(RUNS / src, b, ignore=IGN)
            cp = re.sub(r"ab=control-s\d ", f"ab={keep}-s{k} ", (RUNS / src / "config_proof.txt").read_text())
            (b / "config_proof.txt").write_text(cp)
            gib = {"kv8": "8.0", "kv7": "7.0"}.get(keep, "5.0")
            vl = b / "vllm-openai-server.log"
            vl.write_text(vl.read_text(errors="replace").replace("reserved 5.0 GiB", f"reserved {gib} GiB"))
            if keep == "ctx16k":
                next(b.glob("ab-*.log")).open("a").write("CTX16K_PROOF env=16384 tool_agent=16384\n")
                pf = b / "vllm-metrics-final.prom"
                pf.write_text(re.sub(r"^(vllm:request_prompt_tokens_sum\{[^}]*\}) (\S+)",
                                     lambda m: f"{m.group(1)} {float(m.group(2)) / 2}", pf.read_text(), flags=re.M))
    (d / "verdict.json").write_text(json.dumps({"verdict": {"arms": arms, "submit": submit}}))
    return d


def _run(tmp: Path, scenario: str, kvcap: Path, now: str = "2026-09-27T00:05:00Z") -> str:
    out = tmp / "out"
    env = dict(os.environ, COMPACT_DRY="1", COMPACT_SCENARIO=scenario, COMPACT_NOW=now,
               COMPACT_KVCAP_DIR=str(kvcap), COMPACT_OUT=str(out))
    r = subprocess.run([sys.executable, str(ROOT / "kaggle/ab-compact/drive_compact.py")], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    log = (out / "drive.log").read_text()
    assert "competitions submit" not in log and "kaggle competitions" not in log  # never submits
    assert (out / "state/DONE").exists() and "DRY would run: launchctl bootout" in log
    return log


@pytest.mark.parametrize("keep,base", [(None, "control"), ("kv8", "kv8"), ("kv7", "kv7"), ("ctx16k", "ctx16k")])
def test_base_follows_the_26_sep_verdict(tmp_path, keep, base):
    log = _run(tmp_path, "ok", _kvcap(tmp_path, keep))
    assert f"BASE {base}:" in log
    pushes = re.findall(r"DRY would run: kaggle kernels push -p \S+/(\S+)", log)
    assert pushes == [f"{base}-cmp-s1", f"{base}-cmp-s2"], pushes
    assert f"{base}-cmp-s1 COMPLETE" in log and f"{base}-cmp-s2 COMPLETE" in log
    assert re.search(r"verdict \| RULE (KEEP|KILL|PENDING)", log), log[-3000:]
    assert "verdict | base valid runs 0" not in log
    v = json.loads((tmp_path / "out/verdict.json").read_text())
    assert v["base"] == base and v["stats"]["arm"]["n_valid"] == 2, v["stats"]["arm"].get("invalid")


def test_no_done_marker_waits_until_0100_then_control(tmp_path):
    log = _run(tmp_path, "ok", _kvcap(tmp_path, "kv8", done=False))
    assert "waiting for the 26 Sep driver's DONE marker" in log
    first_push = re.search(r"^(\S+) DRY would run: kaggle kernels push", log, re.M).group(1)
    assert first_push >= "2026-09-27T01:00:00Z"
    assert "BASE kv8:" in log  # the verdict file is read at 01:00 even without DONE


def test_missing_verdict_is_control(tmp_path):
    kv = _kvcap(tmp_path, None)
    (kv / "verdict.json").unlink()
    assert "BASE control: no readable" in _run(tmp_path, "ok", kv)


def test_early_start_sleeps_and_does_nothing_before_0005(tmp_path):
    log = _run(tmp_path, "ok", _kvcap(tmp_path, None), now="2026-09-26T20:00:00Z")
    first = log.splitlines()[1]
    assert "before 27 Sep 00:05 UTC: sleeping" in first
    first_push = re.search(r"^(\S+) DRY would run: kaggle kernels push", log, re.M).group(1)
    assert first_push >= "2026-09-27T00:05:00Z"


def test_error_gets_one_retry(tmp_path):
    log = _run(tmp_path, "error", _kvcap(tmp_path, None))
    assert "control-cmp-s1 ERROR: one retry" in log
    assert len(re.findall(r"kernels push -p \S+/control-cmp-s1\b", log)) == 2


def test_low_quota_never_pushes(tmp_path):
    log = _run(tmp_path, "lowq", _kvcap(tmp_path, None))
    assert "kernels push" not in log and "GAVE UP control-cmp-s1" in log


def test_changed_dataset_blocks_the_push(tmp_path):
    log = _run(tmp_path, "dataset", _kvcap(tmp_path, None))
    assert "kernels push" not in log and "One knob would break" in log
