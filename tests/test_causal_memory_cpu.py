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


def test_event_identity_distinguishes_concurrent_games_in_one_runtime_dir(tmp_path, monkeypatch):
    ta = cr.tool_agent()
    causal = _load("causal_event_identity", ROOT / "kaggle/ab-compact/causal_memory.py")
    events = tmp_path / "events.jsonl"
    causal.install_event_identity(ta)
    try:
        agents = [cr.new_agent(ta), cr.new_agent(ta)]
        state_paths = [
            tmp_path / "tn36-ef4dde99_p0_tool_runtime_state.json",
            tmp_path / "lf52-271a04aa_p0_tool_runtime_state.json",
        ]

        def emit(agent, state_path):
            agent._ensure_session(state_path)
            causal.write_event(agent, events, kind="digest", level=1, dropped=1, chars=10)

        threads = [threading.Thread(target=emit, args=pair) for pair in zip(agents, state_paths)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        rows = [json.loads(line) for line in events.read_text().splitlines()]
        assert {row["game_id"] for row in rows} == {"tn36-ef4dde99", "lf52-271a04aa"}
        assert len({row["runtime_dir"] for row in rows}) == 1
    finally:
        causal.uninstall_event_identity(ta)


def test_causal_record_is_bounded_balanced_and_turn_tied():
    cm = _load("causal_memory", ROOT / "kaggle/ab-compact/causal_memory.py")
    messages = []
    for turn in range(1, 10):
        messages.extend(
            [
                {"role": "user", "content": f"Current state: step {turn}, level 1."},
                {
                    "role": "assistant",
                    "content": (
                        f"Turn {turn}: LEFT appears blocked, so that hypothesis is wrong. "
                        f"Goal constraint: the red object must avoid the wall. "
                        f"Next test: try RIGHT and inspect the target."
                    ),
                    "tool_calls": [
                        {
                            "function": {
                                "name": "python",
                                "arguments": json.dumps({"code": 'r = action(["LEFT"])'}),
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "{'board_changed': False, 'reward': 0.0, 'level_completed': False}",
                },
            ]
        )
    facts = cm.extract_facts(messages)
    record = cm.bounded_record(facts, max_chars=700)
    assert len(record) <= 700
    assert {fact.category for fact in facts} == {
        "action_effect",
        "rejected_hypothesis",
        "constraint",
        "unfinished_test",
    }
    for label in ("effect", "reject", "constraint", "test"):
        assert f" {label}]" in record
    assert all(line.startswith("[t") for line in record.splitlines())


def test_observed_action_effect_uses_tool_result_not_assistant_claim():
    cm = _load("causal_memory_effect", ROOT / "kaggle/ab-compact/causal_memory.py")
    messages = [
        {"role": "user", "content": "Current state: step 7, level 2."},
        {
            "role": "assistant",
            "content": "This will certainly finish the level.",
            "tool_calls": [
                {"function": {"name": "python", "arguments": json.dumps({"code": 'action(["SPACE"])'})}}
            ],
        },
        {
            "role": "tool",
            "content": "{'board_changed': False, 'reward': 0.0, 'level_completed': False}",
        },
    ]
    effect = next(f for f in cm.extract_facts(messages) if f.category == "action_effect")
    assert effect.turn == 7
    assert "SPACE" in effect.text
    assert "board_changed=false" in effect.text
    assert "level_completed=false" in effect.text
    assert "certainly finish" not in effect.text
