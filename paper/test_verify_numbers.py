"""The number check must be able to go red. Run: python -m pytest paper/test_verify_numbers.py

Kept outside tests/ on purpose: the paper cites the pass count of tests/, and this file is not part of it.
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_numbers as vn  # noqa: E402

DRAFT = (HERE / "ARC-PAPER-WRITEUP-DRAFT.md").read_text()


def run(tmp_path, text):
    p = tmp_path / "draft.md"
    p.write_text(text)
    return vn.main(["--draft", str(p)])


def test_real_draft_is_green(tmp_path):
    assert run(tmp_path, DRAFT) == 0


@pytest.mark.parametrize(
    "old,new",
    [
        ("| tu93 | 2 of 9 (56 actions) | 5.40 |", "| tu93 | 2 of 9 (56 actions) | 5.41 |"),  # a table cell
        ("The guard refused 146 repeated", "The guard refused 145 repeated"),  # a prose total
        ("control 101.53, guard 109.45", "control 101.53, guard 109.5"),  # rounding drift
        ("14 of 81 simple actions", "14 of 82 simple actions"),  # census
        ("A further 10 known repeats", "A further 11 known repeats"),  # memo repeat_noop
        ("One flag, zero actions.", "One flag, zero actions, 3 minutes."),  # a new number no claim knows
        ("commit `0a1bf51`: `runs/duck-*/summary.txt`", "commit `9f8b1de`: `runs/duck-*/summary.txt`"),  # the old Sources lie
        ("## Sources", "TODO fill\n\n## Sources"),  # a placeholder
    ],
)
def test_planted_error_goes_red(tmp_path, old, new):
    assert DRAFT.count(old) == 1, old
    assert run(tmp_path, DRAFT.replace(old, new)) == 1


def test_word_limit_goes_red(tmp_path):
    over = vn.WORD_LIMIT - vn.word_count(vn.body_of(DRAFT)) + 1
    padded = DRAFT.replace("## Sources", "padding " * over + "\n\n## Sources")
    assert run(tmp_path, padded) == 1
