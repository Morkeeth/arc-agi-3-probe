"""CPU check of the DRAFT compaction lever against the real fork ToolAgent history code (no model, no GPU)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = Path.home() / "CODE/arc-agi-3-probe-data/fork-src/src/ARC3-Inference"
pytestmark = pytest.mark.skipif(not (SRC / "inference/agent/tool_agent.py").exists(), reason="fork source missing")


def _load():
    sys.path.insert(0, str(SRC))
    import inference.agent.tool_agent as ta
    spec = importlib.util.spec_from_file_location("compaction_draft", ROOT / "kaggle/ab-compact/compaction.py")
    comp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comp)
    return ta, comp


def _agent(ta, budget):
    a = ta.ToolAgent.__new__(ta.ToolAgent)
    a._context_budget_tokens = budget
    a._summarized_knowledge = ta._empty_world_model()
    a._last_step_summary = None
    return a


def _messages(n):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs += [{"role": "user", "content": "frame " + "x" * 800},
                 {"role": "assistant", "content": f"turn {i}: ACTION3 moved the block left"},
                 {"role": "tool", "content": f"result {i}: level 1, board changed"}]
    return msgs


def test_off_is_base_and_on_digests_dropped_turns():
    ta, comp = _load()
    base = _agent(ta, 1500)
    kept_base = base._persistent_history_messages(_messages(12))
    assert len(kept_base) < 36, "the fixture must force truncation"
    assert comp.install(0, ta) == 0 and not comp._ORIG  # knob off: nothing patched
    try:
        comp.install(6000, ta)
        a = _agent(ta, 1500)
        kept = a._persistent_history_messages(_messages(12))
        assert [m["content"] for m in kept] == [m["content"] for m in kept_base]  # same history kept
        c = a._summarized_knowledge[comp.KEY]
        assert "turn 0: ACTION3" in c and "result 0" in c and "frame" not in c
        lines = a._summarized_knowledge_lines()
        assert any("Compacted earlier turns" in x for x in lines)
        a._last_step_summary = {"level_transition": True}
        a._update_summarized_knowledge_from_step_summary()
        assert a._summarized_knowledge[comp.KEY] == ""
    finally:
        comp.uninstall(ta)
    assert not any("Compacted" in x for x in _agent(ta, 1500)._summarized_knowledge_lines())
