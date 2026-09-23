"""ab_verdict.py: the decision rule branches, and a cross-check against the two real 22 Sep runs."""

import json
from pathlib import Path

import pytest

import ab_verdict as av

ROOT = Path(__file__).resolve().parent.parent


def _write(d: Path, scores, levels, state="gave_up", calls=10, skipped=0, games=av.SUBSET5):
    d.mkdir(parents=True)
    runs = [
        {"game_id": f"{g}-x", "final_score": s, "levels_completed": lv, "state": state, "history": [0] * 5,
         "final_wallclock_seconds": 7920.0, "solver_note": "tokens=1"}
        for g, s, lv in zip(games, scores, levels)
    ]
    (d / "benchmark.json").write_text(json.dumps({"game_runs": runs}))
    (d / "deadprobe_counters.jsonl").write_text(
        "".join(json.dumps({"game": f"{g}-x", "calls": calls, "turns": calls, "skipped": skipped, "time_cap_s": 7920.0}) + "\n" for g in games)
    )


def _run(tmp_path, capsys, *extra):
    av.main([str(tmp_path), "--games", "subset5", "--json", *extra])
    out = capsys.readouterr().out
    return out, json.loads((tmp_path / "verdict.json").read_text())["verdict"]


def test_win_needs_higher_mean_and_no_fewer_levels(tmp_path, capsys):
    for k in (1, 2, 3):
        _write(tmp_path / f"control-s{k}", [1, 0, 0, 0, 0], [1, 0, 0, 0, 0])
        _write(tmp_path / f"answer-s{k}", [2, 0, 0, 0, 0], [1, 0, 0, 0, 0], skipped=4)
    out, v = _run(tmp_path, capsys)
    assert v["rule"] == "WIN"
    assert v["ledger"] == "keep"  # control spread is 0 here


def test_tie_on_mean_is_a_loss(tmp_path, capsys):
    for k in (1, 2, 3):
        _write(tmp_path / f"control-s{k}", [1, 0, 0, 0, 0], [1, 0, 0, 0, 0])
        _write(tmp_path / f"answer-s{k}", [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], skipped=4)
    _, v = _run(tmp_path, capsys)
    assert v["rule"] == "LOSE"


def test_more_score_fewer_levels_is_a_loss(tmp_path, capsys):
    for k in (1, 2, 3):
        _write(tmp_path / f"control-s{k}", [1, 0, 0, 0, 0], [2, 0, 0, 0, 0])
        _write(tmp_path / f"answer-s{k}", [3, 0, 0, 0, 0], [1, 0, 0, 0, 0], skipped=4)
    _, v = _run(tmp_path, capsys)
    assert v["rule"] == "LOSE"


def test_no_verdict_below_three_valid_runs(tmp_path, capsys):
    for k in (1, 2, 3):
        _write(tmp_path / f"control-s{k}", [1, 0, 0, 0, 0], [1, 0, 0, 0, 0])
    for k in (1, 2):
        _write(tmp_path / f"answer-s{k}", [5, 0, 0, 0, 0], [3, 0, 0, 0, 0], skipped=4)
    _write(tmp_path / "answer-s3", [5, 0, 0, 0, 0], [3, 0, 0, 0, 0], state="crashed", skipped=4)
    _, v = _run(tmp_path, capsys)
    assert v["rule"].startswith("PENDING")


def test_mechanism_failure_kills_even_a_win(tmp_path, capsys):
    for k in (1, 2, 3):
        _write(tmp_path / f"control-s{k}", [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], calls=10)
        _write(tmp_path / f"answer-s{k}", [2, 0, 0, 0, 0], [1, 0, 0, 0, 0], calls=12, skipped=4)
    _, v = _run(tmp_path, capsys)
    assert v["rule"] == "WIN" and v["ledger"] == "kill"  # calls per action up 20%


@pytest.mark.skipif(not (ROOT / "runs/duck-control/benchmark.json").exists(), reason="22 Sep run files not on disk")
def test_reads_the_real_22_sep_runs_like_their_summaries(tmp_path, capsys):
    (tmp_path / "control-s1").symlink_to(ROOT / "runs/duck-control")
    (tmp_path / "answer-s1").symlink_to(ROOT / "runs/duck-deadprobe")
    out, v = _run(tmp_path, capsys, "--n-required", "1", "--no-counters")
    assert "control  n=1 mean=1.27" in out  # runs/duck-control/summary.txt: mean score 1.27
    assert "answer   n=1 mean=0.35" in out  # runs/duck-deadprobe/summary.txt: mean score 0.35
    assert v["rule"] == "LOSE"  # the n=1 guard result, as the draft reports it
