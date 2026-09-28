from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path

from tests import compaction_replay as cr

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(code: str, call_id: str = "call_1") -> dict:
    return {"role": "assistant", "tool_calls": [{"id": call_id, "function": {
        "name": "python", "arguments": json.dumps({"code": code})}}]}


def test_event_identity_distinguishes_concurrent_games_in_one_runtime_dir(tmp_path):
    ta = cr.tool_agent()
    causal = _load("causal_event_identity", ROOT / "kaggle/ab-compact/causal_memory.py")
    events = tmp_path / "events.jsonl"
    causal.install_event_identity(ta)
    try:
        agents = [cr.new_agent(ta), cr.new_agent(ta)]
        paths = [tmp_path / "tn36-ef4dde99_p0_tool_runtime_state.json",
                 tmp_path / "lf52-271a04aa_p0_tool_runtime_state.json"]

        def emit(agent, path):
            agent._ensure_session(path)
            causal.write_event(agent, events, kind="digest")

        threads = [threading.Thread(target=emit, args=pair) for pair in zip(agents, paths)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        assert {row["game_id"] for row in rows} == {"tn36-ef4dde99", "lf52-271a04aa"}
        assert len({row["runtime_dir"] for row in rows}) == 1
    finally:
        causal.uninstall_event_identity(ta)


def test_effect_is_bound_to_identity_turn_and_call():
    cm = _load("causal_bound", ROOT / "kaggle/ab-compact/causal_memory.py")
    facts = cm.extract_facts([
        {"role": "user", "content": "Current state: step 7, level 2."},
        _call('action(["LEFT"])', "call_left"),
        {"role": "tool", "tool_call_id": "call_left",
         "content": "{'board_changed': False, 'level_completed': False}"},
    ], game_id="game-a")
    effect = next(f for f in facts if f.category == "action_effect")
    assert (effect.game_id, effect.level, effect.turn, effect.tool_call_id, effect.status) == (
        "game-a", 2, 7, "call_left", "observed")
    assert "LEFT" in effect.text and "board_changed=false" in effect.text


def test_multi_action_result_is_excluded_as_ambiguous():
    cm = _load("causal_ambiguous", ROOT / "kaggle/ab-compact/causal_memory.py")
    facts = cm.extract_facts([
        {"role": "user", "content": "Current state: step 2, level 1."},
        _call('action(["LEFT"]); action(["RIGHT"])'),
        {"role": "tool", "tool_call_id": "call_1", "content": "board_changed=true board_changed=false"},
    ])
    assert not [f for f in facts if f.category == "action_effect"]


def test_conditional_sentence_has_one_category_and_is_not_a_fact():
    cm = _load("causal_conditional", ROOT / "kaggle/ab-compact/causal_memory.py")
    facts = cm.extract_facts([
        {"role": "user", "content": "Current state: step 3, level 1."},
        {"role": "assistant", "content": "If LEFT is blocked, next try RIGHT."},
    ])
    assert [(f.category, f.status) for f in facts] == [("hypothesis", "conditional")]


def test_executed_probe_retires_pending_test_as_completed():
    cm = _load("causal_completed", ROOT / "kaggle/ab-compact/causal_memory.py")
    facts = cm.extract_facts([
        {"role": "user", "content": "Current state: step 4, level 1."},
        {"role": "assistant", "content": "Next test: try RIGHT.",
         "tool_calls": _call('action(["RIGHT"])')["tool_calls"]},
        {"role": "tool", "tool_call_id": "call_1", "content": "board_changed=true level_completed=false"},
    ])
    tests = [f for f in facts if f.category == "test"]
    assert len(tests) == 1 and tests[0].status == "completed"
    assert tests[0].tool_call_id == "call_1" and "board_changed=true" in tests[0].evidence


def test_wrong_pair_cannot_receive_retention_credit():
    ev = _load("causal_eval_wrong_pair", ROOT / "kaggle/ab-compact/evaluate_causal_memory.py")
    cm = _load("causal_fact_wrong_pair", ROOT / "kaggle/ab-compact/causal_memory.py")
    fact = cm.Fact("action_effect", "g", 1, 2, "call_left", "observed",
                   'action(["LEFT"]) => board_changed=false', "board_changed=false")
    output = ('[t2 g/l1 call_left effect:observed] action(["LEFT"]) => board_changed=true\n'
              '[t3 g/l1 call_right effect:observed] action(["RIGHT"]) => board_changed=false')
    assert not ev.retained(fact, output)


def test_record_is_bounded_and_naive_arm_is_unlabelled():
    cm = _load("causal_bounded", ROOT / "kaggle/ab-compact/causal_memory.py")
    messages = []
    for turn in range(1, 20):
        messages += [
            {"role": "user", "content": f"Current state: step {turn}, level 1."},
            {"role": "assistant", "content": "Goal constraint: avoid the wall. Next test: try RIGHT.",
             "tool_calls": _call('action(["RIGHT"])')["tool_calls"]},
            {"role": "tool", "tool_call_id": "call_1", "content": "board_changed=true level_completed=false"},
        ]
    facts = cm.extract_facts(messages, game_id="g")
    record = cm.bounded_record(facts, 700)
    naive = cm.naive_record(facts, 700)
    assert len(record) <= 700 and len(naive) <= 700
    assert "effect:observed" in record and "test:completed" in record
    assert "effect:observed" not in naive and "test:completed" not in naive


def test_installed_hook_replays_real_drops_and_emits_identity(tmp_path):
    ta = cr.tool_agent()
    cm = _load("causal_live_hook", ROOT / "kaggle/ab-compact/causal_memory.py")
    events = tmp_path / "events.jsonl"
    game = cr.recorded_games("control-s1")[0]
    cm.install(3000, ta, events)
    try:
        stats = cr.replay(game, ta, cm)
    finally:
        cm.uninstall(ta)
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    expected = game.stem.rsplit("_p0", 1)[0]
    assert stats.patch_seen > 0 and stats.digests
    assert {row["game_id"] for row in rows if row["kind"] == "digest"} == {expected}
    assert all(len(record) <= 3000 for _, record in stats.digests)
