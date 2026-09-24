"""Loop A notebooks (runs/RULE-kvcap.md): each differs from control-s1 in one knob and AB_ID, nothing else.

The ctx16k test also runs the notebook's real cell 11 on CPU against the local fork bundle, with the env
that cell 9 leaves behind (LOCAL_ANALYZER_CONTEXT_WINDOW=32768), and reads the value tool_agent froze.
The same run on the control cell 11 must give 32768, so the check can go red.
"""

from __future__ import annotations

import difflib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONTROL = ROOT / "kaggle" / "ab-q38" / "control-s1" / "ab-q38-control-s1.ipynb"
BUNDLE = Path.home() / "CODE" / "arc-agi-3-probe-data" / "fork-src"


def _build():
    spec = importlib.util.spec_from_file_location("kvcap_build", ROOT / "kaggle" / "ab-kvcap" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in json.loads(path.read_text())["cells"]]


def _changed(a: list[str], b: list[str]) -> tuple[list[str], list[str]]:
    la = "\n".join(a).splitlines()
    lb = "\n".join(b).splitlines()
    rem, add = [], []
    for line in difflib.unified_diff(la, lb, lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        (rem if line.startswith("-") else add).append(line[1:])
    return rem, add


def test_one_knob_each():
    dirs = _build()
    assert sorted(d.name for d in dirs) == sorted(f"{a}-s{k}" for a in ("kv8", "kv7", "ctx16k") for k in (1, 2))
    control = _cells(CONTROL)
    for d in dirs:
        arm, ab_id = d.name.rsplit("-", 1)[0], d.name
        nb = d / f"ab-{ab_id}.ipynb"
        got = _cells(nb)
        assert len(got) == len(control), f"{ab_id}: cell count changed"
        rem, add = _changed(control, got)
        id_rem = [x for x in rem if x.startswith("AB_ID")]
        id_add = [x for x in add if x.startswith("AB_ID")]
        assert id_rem == ['AB_ID = "control-s1"'] and id_add == [f'AB_ID = "{ab_id}"'], ab_id
        rem = [x for x in rem if not x.startswith("AB_ID")]
        add = [x for x in add if not x.startswith("AB_ID")]
        if arm in ("kv8", "kv7"):
            val = {"kv8": "8589934592", "kv7": "7516192768"}[arm]
            assert rem == ['    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120",'], (ab_id, rem)
            assert add == [f'    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "{val}",'], (ab_id, add)
        else:
            assert rem == [], (ab_id, rem)  # ctx16k only adds lines
            code = [x for x in add if x.strip() and not x.lstrip().startswith("#")]
            assert [x for x in code if "16384" in x and "os.environ[" in x and "=" in x.split("16384")[0]], add
            assert all("LOCAL_ANALYZER_CONTEXT_WINDOW" in x or "_ctx_" in x or "tool_agent" in x or x.startswith("      f") for x in code), code
            # Placement: first statement of the cell with the benchmark pickle.load, after the setup-command cell.
            i11 = next(i for i, c in enumerate(got) if "benchmark_initial.pkl" in c)
            i9 = next(i for i, c in enumerate(got) if "setup_commands.json" in c)
            assert i9 < i11
            c11 = got[i11]
            assert c11.index('os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = "16384"') < c11.index("pickle.load(file)")
            assert not any("inference" in c and "import" in c for c in got[:i11]), "a harness import before the override"
            assert "16384" not in "".join(got[:i11]), "knob placed where cell 9 overwrites it"
        meta = json.loads((d / "kernel-metadata.json").read_text())
        cmeta = json.loads((CONTROL.parent / "kernel-metadata.json").read_text())
        assert meta.pop("id") == f"oscarmorke/ab-{ab_id}" and meta.pop("title") == f"ab-{ab_id}"
        assert meta.pop("code_file") == f"ab-{ab_id}.ipynb"
        for k in ("id", "title", "code_file"):
            cmeta.pop(k)
        assert meta == cmeta, f"{ab_id}: metadata differs beyond id/title/code_file"


def _run_cell11(nb: Path, tmp_path: Path) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cells = _cells(nb)
    c11 = next(c for c in cells if "benchmark_initial.pkl" in c)
    setup = tmp_path / "taaf_setup_env.json"
    setup.write_text(json.dumps({"LOCAL_ANALYZER_CONTEXT_WINDOW": "32768"}))
    src = BUNDLE / "src"
    prelude = (
        "import json, os, pickle, sys\nfrom pathlib import Path\n"
        f"BUNDLE_DIR = Path({str(BUNDLE)!r}); WORKING_DIR = Path({str(tmp_path)!r}); SETUP_ENV_PATH = Path({str(setup)!r})\n"
        "TRUE_SUBMISSION = False\n"
    )
    epilogue = "\nimport inference.agent.tool_agent as _t\nprint('FROZEN', _t._LOCAL_ANALYZER_CONTEXT_WINDOW)\n"
    env = dict(os.environ, LOCAL_ANALYZER_CONTEXT_WINDOW="32768",  # what cell 9 leaves in os.environ
               PYTHONPATH=os.pathsep.join(str(p) for p in (src / "ARC3-Inference", src / "tufa-arc-agi-framework" / "src", src / "tufa-arc-agi-framework")))
    r = subprocess.run([sys.executable, "-c", prelude + c11 + epilogue], env=env, capture_output=True, text=True, cwd=tmp_path, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout + "\nSETUP=" + setup.read_text()


@pytest.mark.skipif(not (BUNDLE / "benchmark_initial.pkl").exists(), reason="local fork bundle missing")
def test_ctx16k_reaches_tool_agent_and_control_does_not(tmp_path):
    _build()
    out = _run_cell11(ROOT / "kaggle" / "ab-kvcap" / "ctx16k-s1" / "ab-ctx16k-s1.ipynb", tmp_path / "t")
    assert "CTX16K_PROOF env=16384 tool_agent=16384" in out and "FROZEN 16384" in out, out
    assert '"LOCAL_ANALYZER_CONTEXT_WINDOW": "16384"' in out
    ctl = _run_cell11(CONTROL, tmp_path / "c")
    assert "FROZEN 32768" in ctl, ctl
