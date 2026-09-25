"""The compaction digest on RECORDED histories: the 3 control runs of 23 Sep, 75 games, CPU only, no model.

tests/compaction_replay.py rebuilds each game's messages from its transcript and walks them through the real
fork ToolAgent history code, the same trims `analyze` makes. These tests check the lever does what
runs/RULE-compaction.md says, on the population it will meet on 27 Sep.
"""

from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

import pytest

from tests import compaction_replay as cr

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not (cr.SRC / "inference/agent/tool_agent.py").exists() or not (cr.RUNS / "control-s1/transcripts").is_dir(),
    reason="fork source or recorded control transcripts missing")


def _comp():
    spec = importlib.util.spec_from_file_location("compaction_under_test", ROOT / "kaggle/ab-compact/compaction.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _build():
    spec = importlib.util.spec_from_file_location("compact_build", ROOT / "kaggle/ab-compact/build.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def ta():
    return cr.tool_agent()


@pytest.fixture(scope="module")
def games():
    gs = [g for run in cr.CONTROL_RUNS for g in cr.recorded_games(run)]
    assert len(gs) == 75, f"expected 3 runs x 25 games, got {len(gs)}"
    return gs


@pytest.fixture(scope="module")
def on(ta, games):
    comp = _comp()
    comp.install(_build().MAX_CHARS, ta)
    try:
        yield comp, {g: cr.replay(g, ta, comp) for g in games}
    finally:
        comp.uninstall(ta)


def test_parser_reads_every_turn(games):
    for g in games[:25]:
        turns = cr.parse(g)
        text = g.read_text(errors="replace")
        assert len(turns) == text.count("--- analysis_step="), g.name
        assert sum(1 for t in turns for k, _ in t.events if k == "tool") == text.count("[TOOL RESULT: "), g.name


def test_replay_matches_the_real_prompt_size(games, ta):
    """The replayed history per request must be near the real runs' last snapshot (prompts/*.log message_count)."""
    import re
    real = [int(x) - 1 for run in cr.CONTROL_RUNS for p in (cr.RUNS / run / "prompts").glob("*.log")
            for x in re.findall(r"^message_count: (\d+)", p.read_text(errors="replace"), re.M)[:1]]
    rep = [x for g in games for x in cr.replay(g, ta).history_msgs_per_request]
    assert abs(statistics.fmean(rep) - statistics.fmean(real)) < 4, (statistics.fmean(rep), statistics.fmean(real))


def test_knob_does_not_change_which_history_is_kept(ta, games):
    """With the digest not spliced into prompts, the kept history is identical turn by turn: the patch only reads."""
    comp = _comp()
    base = {g: cr.replay(g, ta).kept_trace for g in games[:25]}
    comp.install(_build().MAX_CHARS, ta)
    try:
        for g in games[:25]:
            assert cr.replay(g, ta, comp, splice=False).kept_trace == base[g], g.name
    finally:
        comp.uninstall(ta)


def test_every_dropped_message_is_seen(on):
    """All four drop paths of analyze (turn start, each request, overflow, turn end) reach the digest."""
    _, r = on
    dropped = sum(len(s.dropped_ids) for s in r.values())
    seen = sum(len(s.digested_ids) for s in r.values())
    assert dropped > 5000, dropped  # the fixture really truncates
    assert seen == dropped and sum(s.same_turn_lost for s in r.values()) == 0
    patch = sum(s.patch_seen for s in r.values())  # counted inside compaction.py, so this can go red
    assert patch == dropped, (patch, dropped)


def test_digest_content_cap_and_proof_population(on):
    comp, r = on
    cap = _build().MAX_CHARS
    games_with_digest = 0
    for g, s in r.items():
        texts = [d for _, d in s.digests if d]
        games_with_digest += bool(texts)
        for d in texts:
            assert len(d) <= cap
            assert "Current state: step" not in d and "Valid actions right now" not in d  # frames are skipped
            assert "[you]" in d or "[result]" in d or "[act]" in d
    # The sealed proof line needs digests in 20 or more of 25 games per run; the recorded runs give 25 of 25.
    for run in cr.CONTROL_RUNS:
        n = sum(1 for g, s in r.items() if g.parent.parent.name == run and any(d for _, d in s.digests))
        assert n >= 24, (run, n)


def test_digest_burden_is_what_the_rule_says(on):
    """RULE-compaction.md states the duplicated digest costs about 2,500 tokens per request, under 10% of the budget."""
    _, r = on
    per_req = [x for s in r.values() for x in s.digest_tokens_per_request]
    mean = statistics.fmean(per_req)
    assert 1500 < mean < 3500, mean
    assert mean / (32768 - 1024) < 0.10


def test_level_filter_and_clear(ta):
    comp = _comp()

    def blk(level, i):
        return [{"role": "user", "content": f"Current state: step {i}, level {level}. frame " + "x" * 900},
                {"role": "assistant", "content": f"L{level} turn {i}",
                 "tool_calls": [{"function": {"name": "python", "arguments": '{"code": "r = action([\\"ACTION3\\"])\\nprint(r)"}'}}]},
                {"role": "tool", "content": f"L{level} result {i}"}]

    d = comp.digest(blk(1, 0) + blk(2, 1), level=2)
    assert "L1" not in d and "L2 turn 1" in d and 'action(["ACTION3"])' in d and "L2 result 1" in d
    comp.install(3000, ta)
    try:
        a = cr.new_agent(ta, 1500)  # budget 1024 tokens: forces drops
        msgs = [{"role": "system", "content": "s"}] + blk(1, 0) + blk(1, 1) + blk(1, 2) + blk(1, 3)
        a._last_step_summary = {"level": 1}
        a._history_messages = a._persistent_history_messages(msgs)
        assert "L1 turn 0" in a._summarized_knowledge[comp.KEY]
        a._last_step_summary = {"level": 2, "level_transition": True}
        a._update_summarized_knowledge_from_step_summary()
        assert a._summarized_knowledge[comp.KEY] == ""
        # old level blocks dropped after the transition do not refill the digest
        msgs = [{"role": "system", "content": "s"}, *a._history_messages] + blk(2, 4) + blk(2, 5) + blk(2, 6)
        a._history_messages = a._persistent_history_messages(msgs)
        assert "L1" not in a._summarized_knowledge.get(comp.KEY, "")
    finally:
        comp.uninstall(ta)
