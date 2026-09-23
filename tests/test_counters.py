"""Counters on PR01 through the real Duck session, and the staged Kaggle copy matches the source.

The counters are the instrument for RULE-guard-answer.md. They must read the same way in the control
arm (mode off) and the treatment arm, and they must not read 0 on a game that ran.
"""

import filecmp
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
duck = pytest.mark.skipif(not os.environ.get("DUCK_SRC"), reason="set DUCK_SRC to run against the Duck source")


def test_kaggle_dataset_is_the_source():
    """The notebook mounts kaggle/dataset/, not deadprobe/. A stale copy runs old code silently (de2bd8f did)."""
    src = sorted(p.name for p in (ROOT / "deadprobe").glob("*.py"))
    staged = sorted(p.name for p in (ROOT / "kaggle" / "dataset").glob("*.py"))
    assert src == staged
    stale = [n for n in src if not filecmp.cmp(ROOT / "deadprobe" / n, ROOT / "kaggle" / "dataset" / n, shallow=False)]
    assert stale == []


def _play(mode, tmp_path, monkeypatch, analyzer_cls=None):
    import inference.framework.solver as s

    import tests.test_duck_patch as base
    from deadprobe import counters
    from tests.test_duck_patch_answer import PLAN, ReplanningAnalyzer

    monkeypatch.setattr(base, "ScriptedAnalyzer", analyzer_cls or ReplanningAnalyzer)
    orig_run = base.run

    def run_with_counters(mode, script, tmp_path, **kw):
        # install the counters after duck_patch, the order the notebook uses
        from deadprobe import duck_patch

        duck_patch.uninstall()
        orig_install = duck_patch.install

        def install_then_count(*a, **k):
            cfg = orig_install(*a, **k)
            counters.install(solver_module=s, log=False)
            return cfg

        monkeypatch.setattr(duck_patch, "install", install_then_count)
        if mode == "off":
            counters.install(solver_module=s, log=False)
        try:
            return orig_run(mode, script, tmp_path, **kw)
        finally:
            counters.uninstall()

    run_, an = run_with_counters(mode, PLAN, tmp_path)
    recs = [json.loads(line) for line in (tmp_path / "deadprobe_counters.jsonl").read_text().splitlines()]
    assert len(recs) == 1
    return run_, an, recs[0]


@duck
@pytest.mark.parametrize("mode,calls,actions,skipped", [("off", 1, 6, 0), ("guard", 4, 4, 3), ("guard+answer", 1, 4, 2)])
def test_counters_read_the_run(tmp_path, monkeypatch, mode, calls, actions, skipped):
    run_, an, rec = _play(mode, tmp_path, monkeypatch)
    assert an.calls == calls
    assert rec["turns"] == calls + 1  # plus the last turn, where the empty plan stops the game
    assert rec["actions"] == actions == run_.actions_per_level[0]
    assert rec["skipped"] == skipped
    assert rec["levels"] == 1
    assert rec["wall_s"] > 0
    assert rec["calls"] == 0  # the scripted analyzer makes no model request
    assert rec["game"].startswith("pr01")


@duck
def test_counters_count_model_requests(tmp_path, monkeypatch):
    from tests.test_duck_patch_answer import ReplanningAnalyzer

    class ModelAnalyzer(ReplanningAnalyzer):
        """Two model requests per turn, the way a tool loop asks, then acts."""

        def _chat_completion(self, messages=None, **kw):
            return None

        def analyze(self, *a, **kw):
            self._chat_completion([])
            self._chat_completion([])
            return super().analyze(*a, **kw)

    run_, an, rec = _play("guard+answer", tmp_path, monkeypatch, ModelAnalyzer)
    assert rec["turns"] >= 1
    assert rec["calls"] == 2 * rec["turns"]
    assert rec["model_s"] >= 0
    assert "_chat_completion" not in an.__dict__ and "analyze" not in an.__dict__  # restored
