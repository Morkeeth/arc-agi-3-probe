"""Read-only per-game counters for the Duck harness. Installed in every arm, control included.

    from deadprobe import counters
    counters.install()        # after duck_patch.install(...), so this wrapper is outermost

Per game it logs one line and appends one JSON object to <job_dir>/deadprobe_counters.jsonl:

    deadprobe: counters game=<id> state=<s> levels=<n> actions=<n> turns=<n> calls=<n> model_s=<f> wall_s=<f> skipped=<n>

  turns    analyzer.analyze() calls: one agent turn, counted for any analyzer (the scripted test one too)
  calls    ToolAgent._chat_completion() calls: real model requests. 0 when the analyzer has no such method
  model_s  seconds spent inside those requests
  wall_s   seconds from session start to the end of play
  actions  len(game_run.history): actions the engine charged
  skipped  known no-ops the guard refused or answered (0 in the control arm)

It changes no behaviour. It wraps _HarnessGameSession.play only, and wraps two methods on the one
analyzer instance for the length of that play call, then restores them. Instance attributes, not a
proxy object, so duck_patch's _ProbingAnalyzer and its __dict__ notes still reach the real agent.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

_INSTALLED: dict[str, Any] = {}
_LOCK = threading.Lock()


def install(*, solver_module=None, jsonl_name: str = "deadprobe_counters.jsonl", log: bool = True) -> None:
    """Idempotent."""
    if "session_cls" in _INSTALLED:
        return
    if solver_module is None:
        import inference.framework.solver as solver_module  # type: ignore

    S = solver_module._HarnessGameSession
    orig_play = S.play

    def play(self):
        an = self.analyzer
        box = {"turns": 0, "calls": 0, "model_s": 0.0}
        wrapped = []
        orig_analyze = getattr(an, "analyze", None)
        if orig_analyze is not None:

            def analyze(*a, **kw):
                box["turns"] += 1
                return orig_analyze(*a, **kw)

            an.__dict__["analyze"] = analyze
            wrapped.append("analyze")
        orig_chat = getattr(an, "_chat_completion", None)
        if orig_chat is not None:

            def _chat_completion(*a, **kw):
                box["calls"] += 1
                t0 = time.monotonic()
                try:
                    return orig_chat(*a, **kw)
                finally:
                    box["model_s"] += time.monotonic() - t0

            an.__dict__["_chat_completion"] = _chat_completion
            wrapped.append("_chat_completion")
        try:
            return orig_play(self)
        finally:
            for name in wrapped:
                an.__dict__.pop(name, None)
            _emit(self, box, jsonl_name, log)

    S.play = play
    _INSTALLED.update(session_cls=S, orig_play=orig_play, wrapper=play)


def _emit(sess, box: dict, jsonl_name: str, log: bool) -> None:
    try:
        run = sess.game.game_run
        memo = sess.__dict__.get("_deadprobe_memo")
        rec = {
            "game": getattr(run, "game_id", None) or getattr(sess.game, "game_id", "?"),
            "state": getattr(run, "state", None),
            "levels": int(getattr(run, "levels_completed", 0) or 0),
            "actions": len(run.history) if run is not None else 0,
            "turns": box["turns"],
            "calls": box["calls"],
            "model_s": round(box["model_s"], 3),
            "wall_s": round(time.monotonic() - sess.started_at, 3),
            "skipped": int(memo.n_skipped) if memo is not None else 0,
            "time_cap_s": getattr(sess.solver, "max_runtime_s_per_game", None),
        }
        line = "deadprobe: counters " + " ".join(f"{k}={v}" for k, v in rec.items())
        if log:
            print(line, flush=True)
        root = getattr(sess.solver, "job_dir", None)
        if root is not None:
            with _LOCK, open(Path(root) / jsonl_name, "a") as f:
                f.write(json.dumps(rec) + "\n")
        sess.__dict__["_deadprobe_counters"] = rec
    except Exception as exc:  # a counter must never end a game
        print(f"deadprobe: counters error {type(exc).__name__}: {exc}", flush=True)


def uninstall() -> None:
    if "session_cls" not in _INSTALLED:
        return
    S = _INSTALLED.pop("session_cls")
    orig, mine = _INSTALLED.pop("orig_play"), _INSTALLED.pop("wrapper")
    if S.play is mine:  # someone uninstalled underneath us: restoring orig would resurrect their patch
        S.play = orig
