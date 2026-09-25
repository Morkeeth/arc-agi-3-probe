"""Diff test for the compaction arm notebooks (runs/RULE-compaction.md): the only difference from each base
notebook is the compaction code path in the hook cell, plus AB_ID. Also runs that code path on CPU."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "kaggle" / "ab-compact"
SRC = Path.home() / "CODE/arc-agi-3-probe-data/fork-src/src/ARC3-Inference"


def _build():
    spec = importlib.util.spec_from_file_location("compact_build", HERE / "build.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _build()
CASES = [(b, k) for b in B.BASES for k in B.SEEDS]
HOOK = 13


def _cells(p: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(p.read_text())["cells"]]


def _arm(base: str, k: int) -> Path:
    ab = B.arm_id(base, k)
    return HERE / ab / f"ab-{ab}.ipynb"


@pytest.mark.parametrize("base,k", CASES)
def test_only_the_hook_cell_differs(base, k):
    bnb, anb = B.base_notebook(base), _arm(base, k)
    b, a = json.loads(bnb.read_text()), json.loads(anb.read_text())
    assert {kk: v for kk, v in a.items() if kk != "cells"} == {kk: v for kk, v in b.items() if kk != "cells"}
    assert len(a["cells"]) == len(b["cells"])
    for i, (cb, ca) in enumerate(zip(b["cells"], a["cells"])):
        if i != HOOK:
            assert ca == cb, f"cell {i} differs from the base"
        else:
            assert {kk: v for kk, v in ca.items() if kk != "source"} == {kk: v for kk, v in cb.items() if kk != "source"}
    hb, ha = _cells(bnb)[HOOK], _cells(anb)[HOOK]
    src = (HERE / "compaction.py").read_text()
    want = hb.replace(f'AB_ID = "{B.base_ab_id(base)}"', f'AB_ID = "{B.arm_id(base, k)}"').replace(
        B.ANCHOR, B.ANCHOR + B.block(src), 1)
    assert ha == want  # exactly: AB_ID plus the compaction block after counters.install()
    removed = [l for l in hb.splitlines() if l not in ha.splitlines()]
    assert removed == [f'AB_ID = "{B.base_ab_id(base)}"']


@pytest.mark.parametrize("base,k", CASES)
def test_file_on_disk_is_the_builder_output(base, k):
    text = B.base_notebook(base).read_text()
    assert _arm(base, k).read_text() == B.edit(text, base, k, (HERE / "compaction.py").read_text())
    meta = json.loads((_arm(base, k).parent / "kernel-metadata.json").read_text())
    bmeta = json.loads((B.base_notebook(base).parent / "kernel-metadata.json").read_text())
    slug = f"ab-{B.arm_id(base, k)}"
    assert meta == dict(bmeta, id=f"oscarmorke/{slug}", title=slug, code_file=f"{slug}.ipynb", is_private=True)
    assert meta["is_private"] is True


def test_base_knobs_survive():
    """The base's own knob is still there: compaction goes on top of the winner, not instead of it."""
    kv8 = "".join(_cells(_arm("kv8", 1)))
    assert '"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "8589934592"' in kv8
    assert '"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "7516192768"' in "".join(_cells(_arm("kv7", 1)))
    ctx = _cells(_arm("ctx16k", 1))
    assert "CTX16K_PROOF" in ctx[11] and 'os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = "16384"' in ctx[11]
    ctl = "".join(_cells(_arm("control", 1)))
    assert '"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120"' in ctl and "CTX16K_PROOF" not in ctl


def test_embedded_source_is_the_committed_file():
    src = (HERE / "compaction.py").read_text()
    for base, k in CASES:
        line = next(l for l in _cells(_arm(base, k))[HOOK].splitlines() if l.startswith('(_pkg / "compaction.py").write_text('))
        lit = line[len('(_pkg / "compaction.py").write_text('):-1]
        assert ast.literal_eval(lit) == src
        assert f"COMPACT_MAX_CHARS = {B.MAX_CHARS}" in _cells(_arm(base, k))[HOOK]


@pytest.mark.skipif(not (SRC / "inference/agent/tool_agent.py").exists(), reason="fork source missing")
def test_hook_block_runs_on_cpu(tmp_path, monkeypatch):
    """Execute the inserted lines as the notebook would, against the real fork ToolAgent."""
    sys.path.insert(0, str(SRC))
    import inference.agent.tool_agent as ta
    orig = ta.ToolAgent._persistent_history_messages
    pkg = tmp_path / "deadprobe"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "counters.py").write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("ARC_COMPACT_EVENTS", str(tmp_path / "events.jsonl"))
    for name in [m for m in sys.modules if m == "deadprobe" or m.startswith("deadprobe.")]:
        monkeypatch.delitem(sys.modules, name)
    import importlib
    importlib.invalidate_caches()
    importlib.import_module("deadprobe.counters")  # as in the notebook: the package is imported before the block
    code = B.block((HERE / "compaction.py").read_text())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "hook-block", "exec"), {"_pkg": pkg})
    comp = sys.modules["deadprobe.compaction"]
    try:
        assert f"COMPACT_PROOF max_chars={B.MAX_CHARS}" in out.getvalue()
        assert ta.ToolAgent._persistent_history_messages is not orig
        assert (pkg / "compaction.py").read_text() == (HERE / "compaction.py").read_text()
        a = ta.ToolAgent.__new__(ta.ToolAgent)
        a._context_budget_tokens = 1024
        a._summarized_knowledge = ta._empty_world_model()
        a._last_step_summary = None
        a._history_messages = []
        a._session_runtime_dir = tmp_path / "ft09-0d8bbf25"
        msgs = [{"role": "system", "content": "s"}]
        for i in range(8):
            msgs += [{"role": "user", "content": f"Current state: step {i}, level 1. " + "x" * 900},
                     {"role": "assistant", "content": f"turn {i}"}, {"role": "tool", "content": f"result {i}"}]
        a._persistent_history_messages(msgs)
        ev = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert ev and ev[0]["kind"] == "digest" and ev[0]["runtime_dir"].endswith("ft09-0d8bbf25")
    finally:
        comp.uninstall(ta)
    assert ta.ToolAgent._persistent_history_messages is orig
