"""compaction_verdict.py: the sealed decision rule, and the proof checks on fixture runs built from real control runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import compaction_verdict as cv

RUNS = Path.home() / "CODE/arc-agi-3-probe/runs/ab-2026-09-23"


def _s(lpr, mean, calls=56.72, n=2):
    return {"n_valid": n, "runs": [f"x-s{i}" for i in range(1, n + 1)], "levels_per_run": lpr, "mean": mean,
            "calls_per_game_run": calls}


def test_rule_lines():
    b = _s(38.0, 8.59, n=3)
    ok = {"x-s1": 25, "x-s2": 25}
    assert cv.decide(b, _s(40.0, 8.60), ok)["rule"] == "KEEP"
    assert cv.decide(b, _s(40.0, 8.50), ok)["rule"] == "PENDING"  # levels up 2 but mean not above
    assert cv.decide(b, _s(39.5, 9.9), ok)["rule"] == "PENDING"   # +1.5 levels: inside the band
    assert cv.decide(b, _s(37.5, 9.9), ok)["rule"] == "KILL"      # levels down
    assert cv.decide(b, _s(45.0, 12.0), {"x-s1": 25, "x-s2": 19})["rule"] == "KILL"  # proof missing in 6 games
    assert cv.decide(b, _s(45.0, 12.0, n=1), {"x-s1": 25})["rule"] == "PENDING"
    c2 = cv.decide(b, _s(40.0, 9.0, calls=60.0), ok)
    assert c2["c2_holds"] is False and c2["rule"] == "KEEP"  # C2 is reported, not gating


@pytest.mark.skipif(not (RUNS / "control-s2r/benchmark.json").exists(), reason="control runs missing")
def test_end_to_end_on_fixture_runs(tmp_path):
    """Two control runs dressed as arm runs: with proofs the rule reads them; without, the proof check goes red."""
    sha = hashlib.sha256(cv.PATCH.read_bytes()).hexdigest()[:12]
    out = tmp_path / "compact"
    for k, src in ((1, "control-s2r"), (2, "control-s3r")):
        d = out / f"control-cmp-s{k}"
        shutil.copytree(RUNS / src, d, ignore=shutil.ignore_patterns("transcripts", "prompts", "solver_analysis", "artifacts"))
        cp = (RUNS / src / "config_proof.txt").read_text().strip()
        cp = cp.replace("ab=control-s2 ", f"ab=control-cmp-s{k} ").replace("ab=control-s3 ", f"ab=control-cmp-s{k} ")
        cp = cp.replace("sha={", f"sha={{'compaction.py': '{sha}', ")
        (d / "config_proof.txt").write_text(cp + "\n")
        (next(d.glob("ab-*.log"))).open("a").write(f"COMPACT_PROOF max_chars={cv.MAX_CHARS} persist=deadprobe.compaction\n")
        games = [json.loads(x)["game"] for x in (d / "deadprobe_counters.jsonl").read_text().splitlines() if x.strip()]
        n = 25 if k == 1 else 19
        (d / "compaction_events.jsonl").write_text("".join(
            json.dumps({"runtime_dir": f"/kaggle/working/runs/{g}_p0/state", "kind": "digest"}) + "\n" for g in games[:n]))
    ctl = [RUNS / d for d in ("control-s1", "control-s2r", "control-s3r")]
    assert cv.main(["--base", "control", "--dir", str(out), "--base-dirs", *map(str, ctl)]) == 0
    v = json.loads((out / "verdict.json").read_text())["verdict"]
    assert v["n_valid"] == 2, v
    assert v["digest_games"] == {"control-cmp-s1": 25, "control-cmp-s2": 19}
    assert v["rule"] == "KILL" and "fewer than 20" in v["why"]
    # remove the COMPACT_PROOF line from s1: that run turns invalid
    lf = next((out / "control-cmp-s1").glob("ab-*.log"))
    lf.write_text(lf.read_text().replace("COMPACT_PROOF", "XX"))
    assert cv.main(["--base", "control", "--dir", str(out), "--base-dirs", *map(str, ctl)]) == 0
    v = json.loads((out / "verdict.json").read_text())
    assert v["stats"]["arm"]["n_valid"] == 1
    assert any("COMPACT_PROOF" in p for p in v["stats"]["arm"]["invalid"]["control-cmp-s1"])
