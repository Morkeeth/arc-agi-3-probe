"""Local artifact-review tests, no model/game/provider execution."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "kaggle/ab-causal-pilot"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pair(tmp_path):
    review = load("pair_review", "review.py")
    fixture = load("saved_fixture", "dry_run.py")
    for arm in ("control", "treatment"):
        fixture._fixture(tmp_path / arm, arm, review.verdict)
    return review, tmp_path / "control", tmp_path / "treatment"


def test_synthetic_scores_never_become_a_behavioral_result(pair):
    review, c, t = pair
    before = {p: p.read_bytes() for root in (c, t) for p in root.rglob("*") if p.is_file()}
    result = review.review(c, t)
    assert result["mode"] == "SYNTHETIC_REVIEW"
    assert result["behavioral_result"] is None
    assert len(result["games"]) == 25
    assert result["synthetic_rule_path"]["verdict"] == "PROMISING"
    assert "PROMISING" not in review.render(result).split("Next:")[0]
    assert before == {p: p.read_bytes() for p in before}


def test_real_logger_shape_binds_only_later_request_continuation(pair):
    review, c, t = pair
    game = review.verdict.EXPECTED_GAMES[0]
    path = t / f"{game}_p0_requests.jsonl"
    request = json.loads(path.read_text())
    request.update(analysis_step=4, request_index_within_turn=1)
    # The source logger repeats input in its response entry, not the model output.
    response = dict(request, event="response", finish_reason="tool_calls")
    assistant = {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                 "function": {"name": "python", "arguments": '{"code":"action([\"LEFT\"])"}'}}]}
    observation = {"role": "tool", "tool_call_id": "call_1", "content": "board_changed: true"}
    later = dict(request, messages=request["messages"] + [assistant, observation], request_index_within_turn=2)
    path.write_text("".join(json.dumps(r) + "\n" for r in (request, response, later)))
    result = review.review(c, t)
    record = result["games"][0]["treatment"]["records"][0]
    assert record["input_at"]["line"] == 1
    assert record["continuation_at"]["line"] == 3
    assert record["continuation"]["assistant"] == assistant
    assert record["continuation"]["tool_observations"] == [observation]


@pytest.mark.parametrize("change", ["trimmed", "different_turn", "response_only", "foreign_tool", "missing_step"])
def test_no_invented_action_binding(pair, change):
    review, _, _ = pair
    before = {"event": "request", "analysis_step": 1, "messages": [{"role": "user", "content": "input"}]}
    after = {"event": "request", "analysis_step": 1, "messages": before["messages"] + [
        {"role": "assistant", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "other", "content": "WIN"}]}
    if change == "trimmed":
        after["messages"] = after["messages"][1:]
    elif change == "different_turn":
        after["analysis_step"] = 2
    elif change == "response_only":
        after["event"] = "response"
    elif change == "missing_step":
        before.pop("analysis_step")
        after.pop("analysis_step")
    result = review.continuation(before, after)
    if change == "foreign_tool":
        assert result["tool_observations"] == []
    else:
        assert result is None


def test_unpaired_game_and_broken_stream_remain_visible(pair):
    review, c, t = pair
    path = t / "benchmark.json"
    value = json.loads(path.read_text())
    value["game_runs"].pop()
    path.write_text(json.dumps(value))
    game = review.verdict.EXPECTED_GAMES[0]
    (t / f"{game}_p0_requests.jsonl").write_text("{broken\n")
    result = review.review(c, t)
    assert result["synthetic_rule_path"]["verdict"] == "INVALID"
    assert len(result["games"]) == 25
    assert result["games"][-1]["treatment"]["score"] is None
    assert result["evidence_issues"]


def test_duplicate_game_streams_cannot_supply_one_lucky_match(pair):
    review, c, t = pair
    game = review.verdict.EXPECTED_GAMES[0]
    (t / "other").mkdir()
    shutil.copy(t / f"{game}_p0_requests.jsonl", t / "other")
    result = review.review(c, t)
    assert result["synthetic_rule_path"]["verdict"] == "INVALID"
    assert result["games"][0]["treatment"]["records"][0]["input_at"] is None


@pytest.mark.parametrize("messages", [None, {}, ["not a message"], [{"role": "assistant", "tool_calls": "bad"}]])
def test_malformed_request_shape_is_invalid_not_a_crash(pair, messages):
    review, c, t = pair
    game = review.verdict.EXPECTED_GAMES[0]
    path = t / f"{game}_p0_requests.jsonl"
    path.write_text(json.dumps({"event": "request", "messages": messages}) + "\n")
    result = review.review(c, t)
    assert result["synthetic_rule_path"]["verdict"] == "INVALID"
    assert result["games"][0]["treatment"]["records"][0]["input_at"] is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_outcomes_cannot_receive_a_decision(pair, invalid):
    review, c, t = pair
    path = t / "benchmark.json"
    value = json.loads(path.read_text())
    value["game_runs"][0]["final_score"] = invalid
    path.write_text(json.dumps(value))
    result = review.review(c, t)
    assert result["synthetic_rule_path"]["verdict"] == "INVALID"
    json.dumps(result, allow_nan=False)


def test_cli_writes_review_and_refuses_evidence_or_existing_output(pair, tmp_path):
    _, c, t = pair
    output = tmp_path / "review"
    cmd = [sys.executable, str(HERE / "review.py"), "--control", str(c), "--treatment", str(t), "--output"]
    first = subprocess.run(cmd + [str(output)], capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert (output / "REVIEW.md").is_file()
    second = subprocess.run(cmd + [str(output)], capture_output=True, text=True)
    assert second.returncode == 2
    nested = subprocess.run(cmd + [str(c / "new")], capture_output=True, text=True)
    assert nested.returncode == 2 and not (c / "new").exists()
