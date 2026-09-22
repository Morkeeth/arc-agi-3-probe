"""Install the probe into the Tufa Labs Duck harness at runtime.

Paste into the Duck notebook's section 6 "Customization hook" cell:

    import sys; sys.path.insert(0, "/kaggle/input/arc-agi-3-probe")
    from deadprobe import duck_patch
    duck_patch.install(mode="guard")      # or "probe", "probe+guard", "off"

No change to the TAAF source dataset is needed. The patch targets three
names in inference/framework/solver.py and one in inference/agent/tool_agent.py
of dataset jeroencottaar/taaf-kaggle-source-share (snapshot 2026-06-12):

- _HarnessGameSession._execute_action: every real action is recorded in a
  per-session NoOpMemo. In guard mode, a repeat of an action that already did
  nothing in this exact board state raises KnownNoOp before it reaches the
  engine. Duck's step_env turns that into an error payload (first action of a
  batch) or stops the batch (later action). No action is spent.
- _HarnessGameSession.play: in probe mode, the analyzer is wrapped so that the
  first analyze() call on each new level runs probe_level() first.
- ToolAgent._build_user_prompt: appends the probe note and the known no-ops for
  the current board to the user turn. The LLM sees facts, not a filtered
  action list. Actions are never removed from valid_actions.

- _HarnessGameSession.step_env (mode guard+answer only): see below.

Modes:
  off           nothing installed (the control arm)
  guard         memo + skip repeats, zero extra actions
  guard+answer  guard, plus two changes (not measured on a GPU yet):
                1. honest_board_changed: board_changed is the HUD-masked
                   gameplay diff, so a timer tick no longer reads as progress.
                2. A refusal is answered inside the batch instead of raised.
                   In guard mode a refused first action voids the whole
                   step_env batch and a later one cuts it short, so the model
                   is called again to re-send what did not run. Here the
                   refused action gets a known answer (nothing changes, no
                   action spent) and the rest of the batch runs. The payload
                   lists the skipped actions in skipped_known_noops and in
                   stop_detail, which survives ToolAgent's compaction.
  probe         level-start probe + prompt note, no skipping
  probe+guard   both
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .memo import NoOpMemo, action_key
from .probe import NAMES, ProbeReport, StepOut, probe_level


class KnownNoOp(RuntimeError):
    pass


@dataclass
class ProbeConfig:
    mode: str = "guard"
    budget_level1: int = 6  # level 1 has the lowest weight: probe hardest there
    budget_later: int = 3
    max_clicks: int = 3
    probe_levels: int = 99  # probe only the first N levels
    honest_board_changed: bool = False  # replace Duck's full-grid flag with the HUD-masked one
    max_consecutive_skips: int = 3  # then let one through, so a stuck model cannot spin until the time cap
    log: bool = True

    @property
    def guard(self) -> bool:
        return "guard" in self.mode

    @property
    def probe(self) -> bool:
        return "probe" in self.mode

    @property
    def answer(self) -> bool:
        return "answer" in self.mode

    @property
    def honest(self) -> bool:
        return self.honest_board_changed or self.answer


_INSTALLED: dict[str, Any] = {}


def _display(action_id: int, data: dict[str, Any] | None) -> str:
    if int(action_id) == 6:
        d = data or {}
        return f"MOUSE(row={d.get('y', 0)}, col={d.get('x', 0)})"
    return NAMES.get(int(action_id), f"ACTION{action_id}")


def _key_display(k: str) -> str:
    if k.startswith("6@"):
        r, c = k[2:].split(",")
        return f"MOUSE(row={r}, col={c})"
    return NAMES.get(int(k), f"ACTION{k}")


def install(mode: str | None = None, cfg: ProbeConfig | None = None, *, solver_module=None, tool_agent_module=None) -> ProbeConfig:
    """Monkeypatch Duck. Idempotent: a second call replaces the config only."""
    cfg = cfg or ProbeConfig()
    if mode is not None:
        cfg.mode = mode
    cfg.mode = os.environ.get("DEADPROBE_MODE", cfg.mode)
    if solver_module is None:
        import inference.framework.solver as solver_module  # type: ignore
    if tool_agent_module is None:
        import inference.agent.tool_agent as tool_agent_module  # type: ignore
    _INSTALLED["cfg"] = cfg
    if cfg.mode == "off":
        uninstall()
        _INSTALLED["cfg"] = cfg
        return cfg
    if "session_cls" in _INSTALLED:
        return cfg

    S = solver_module._HarnessGameSession
    grid_of = solver_module._grid_from_state
    orig_exec = S._execute_action
    orig_play = S.play
    orig_step_env = S.step_env
    TA = tool_agent_module.ToolAgent
    orig_prompt = TA._build_user_prompt

    def memo_of(self) -> NoOpMemo:
        m = self.__dict__.get("_deadprobe_memo")
        if m is None:
            m = NoOpMemo()
            self.__dict__["_deadprobe_memo"] = m
        return m

    def _execute_action(self, action, *, batch_index, batch_size, generated_tokens=None, flush_viewer_payload=True):
        c = _INSTALLED["cfg"]
        memo = memo_of(self)
        before_state = self.game.current_state
        before = np.asarray(grid_of(before_state))
        aid = int(action.id.value)
        data = dict(action.data or {})
        if (
            c.guard
            and not self.__dict__.get("_deadprobe_probing")
            and self.__dict__.get("_deadprobe_skips", 0) < c.max_consecutive_skips
            and memo.is_known_noop(before, aid, data)
        ):
            memo.n_skipped += 1
            self.__dict__["_deadprobe_skips"] = self.__dict__.get("_deadprobe_skips", 0) + 1
            answered = self.__dict__.get("_deadprobe_answered")
            if c.answer and answered is not None:
                answered.append(_display(aid, data))
                return _known_noop_payload(self, action, batch_index, batch_size)
            raise KnownNoOp(
                f"skipped {_display(aid, data)}: it already changed nothing in this exact board state; no action was spent"
            )
        self.__dict__["_deadprobe_skips"] = 0
        lv_before = int(before_state.levels_completed)
        payload = orig_exec(
            self,
            action,
            batch_index=batch_index,
            batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload,
        )
        after_state = self.game.current_state
        after = np.asarray(grid_of(after_state))
        if aid != 0 and before.size and after.size:
            change = memo.record(before, aid, data, after, level_changed=int(after_state.levels_completed) > lv_before)
            payload["gameplay_changed"] = change.gameplay_changed
            if c.honest:
                payload["board_changed"] = change.gameplay_changed
        if payload.get("executed"):
            self.__dict__["_deadprobe_real"] = self.__dict__.get("_deadprobe_real", 0) + 1
        return payload

    def _known_noop_payload(self, action, batch_index, batch_size) -> dict:
        """The answer to a refused action, in the shape of a real one: nothing changed, nothing spent."""
        st = self.game.current_state
        raw = st.raw.state
        return {
            "executed": False,
            "skipped_known_noop": True,
            "action_num": self.action_count,
            "level": solver_module._level_number(self.game),
            "score": int(st.levels_completed),
            "reward": 0.0,
            "state": raw.name,
            "valid_actions": solver_module.to_model_actions(solver_module._engine_action_names(self.game)),
            "board_changed": False,
            "done": False,
            "level_completed": False,
            "game_over": False,
            "run_complete": False,
            "action_name": action.id.name,
            "action_display": _display(int(action.id.value), dict(action.data or {})),
            "batch_index": batch_index,
            "batch_size": batch_size,
            **self.timing_payload(),
        }

    def step_env(self, arguments):
        if not _INSTALLED["cfg"].answer:
            return orig_step_env(self, arguments)
        real_before = self.__dict__.get("_deadprobe_real", 0)
        self.__dict__["_deadprobe_answered"] = answered = []
        try:
            payload = orig_step_env(self, arguments)
        finally:
            self.__dict__.pop("_deadprobe_answered", None)
        if not answered:
            return payload
        real = self.__dict__.get("_deadprobe_real", 0) - real_before
        names = ", ".join(answered)
        payload = dict(payload)
        payload["skipped_known_noops"] = list(answered)
        payload["executed"] = real > 0
        payload["executed_count"] = real
        executed = payload.get("executed_actions")
        if isinstance(executed, list):
            left = list(answered)
            kept = []
            for a in executed:  # drop the answered ones, first match each
                if a in left:
                    left.remove(a)
                else:
                    kept.append(a)
            payload["executed_actions"] = kept
        detail = f"Skipped {names}: each already changed nothing in this exact board state, so no action was spent."
        payload["stop_detail"] = detail
        if real == 0:
            payload["error"] = detail
        return payload

    class _ProbingAnalyzer:
        """Forwards everything to the real analyzer; probes on a new level first."""

        def __init__(self, real, session):
            self._real = real
            self._session = session
            self._probed_level: int | None = None

        def __getattr__(self, name):
            return getattr(self._real, name)

        def analyze(self, *args, **kwargs):
            sess = self._session
            c = _INSTALLED["cfg"]
            lv = int(sess.game.current_state.levels_completed)
            if c.probe and lv != self._probed_level and lv < c.probe_levels and not sess.should_stop():
                self._probed_level = lv
                self._real.__dict__["_deadprobe_note"] = run_session_probe(sess, lv, c).note()
                if sess.should_stop() or solver_module._is_engine_game_over(sess.game):
                    # The probe itself ended the level, the game or the budget.
                    # Hand control back to the play loop without an LLM turn.
                    return tool_agent_module.AnalyzerTurnResult(step_executed=True)
            self._real.__dict__["_deadprobe_session"] = sess
            if "valid_actions" in kwargs:
                kwargs["valid_actions"] = solver_module._engine_action_names(sess.game)
            return self._real.analyze(*args, **kwargs)

    def run_session_probe(sess, lv: int, c: ProbeConfig) -> ProbeReport:
        import arcengine  # type: ignore

        memo = memo_of(sess)
        state = sess.game.current_state

        def step(aid: int, data):
            act = arcengine.ActionInput(id=arcengine.GameAction.from_id(aid), data=data or {})
            sess._execute_action(act, batch_index=1, batch_size=1)
            st = sess.game.current_state
            return StepOut(np.asarray(grid_of(st)), int(st.levels_completed), st.raw.state.name)

        avail = [int(a) for a in state.available_actions if int(a) != 0]
        sess.__dict__["_deadprobe_probing"] = True
        try:
            rep = probe_level(
                step,
                np.asarray(grid_of(state)),
                avail,
                lv,
                budget=c.budget_level1 if lv == 0 else c.budget_later,
                max_clicks=c.max_clicks,
                memo=memo,
                skip=set(memo.known_noops(np.asarray(grid_of(state)))),
            )
        finally:
            sess.__dict__["_deadprobe_probing"] = False
        log = sess.__dict__.setdefault("_deadprobe_log", [])
        log.append({"level": lv + 1, "cost": rep.cost, "dead": rep.dead, "live": rep.live_actions, "stop": rep.stop_reason})
        if c.log:
            print(f"deadprobe: game={sess.game.game_id} level={lv + 1} cost={rep.cost} dead={rep.dead} stop={rep.stop_reason}", flush=True)
        return rep

    def play(self):
        real = self.analyzer
        if _INSTALLED["cfg"].probe or _INSTALLED["cfg"].guard:
            self.analyzer = _ProbingAnalyzer(real, self)
        try:
            return orig_play(self)
        finally:
            self.analyzer = real
            memo = self.__dict__.get("_deadprobe_memo")
            if memo is not None and _INSTALLED["cfg"].log:
                print(f"deadprobe: game={self.game.game_id} memo={memo.stats()}", flush=True)

    def _build_user_prompt(self, action_num, **kwargs):
        text = orig_prompt(self, action_num, **kwargs)
        extra = []
        note = self.__dict__.pop("_deadprobe_note", "")
        if note:
            extra.append(note)
        sess = self.__dict__.get("_deadprobe_session")
        if sess is not None:
            memo = sess.__dict__.get("_deadprobe_memo")
            if memo is not None:
                cur = np.asarray(grid_of(sess.game.current_state))
                dead = memo.known_noops(cur)
                if dead:
                    extra.append(
                        "Already tried in this exact board state with no gameplay effect: "
                        + ", ".join(_key_display(k) for k in dead)
                        + ". Repeating them here spends an action and changes nothing."
                    )
        return text + ("\n" + "\n".join(extra) if extra else "")

    S._execute_action = _execute_action
    S.play = play
    S.step_env = step_env
    TA._build_user_prompt = _build_user_prompt
    _INSTALLED.update(
        session_cls=S, orig_exec=orig_exec, orig_play=orig_play, orig_step_env=orig_step_env, ta_cls=TA, orig_prompt=orig_prompt, run_session_probe=run_session_probe
    )
    return cfg


def uninstall() -> None:
    if "session_cls" not in _INSTALLED:
        return
    S = _INSTALLED.pop("session_cls")
    S._execute_action = _INSTALLED.pop("orig_exec")
    S.play = _INSTALLED.pop("orig_play")
    S.step_env = _INSTALLED.pop("orig_step_env")
    TA = _INSTALLED.pop("ta_cls")
    TA._build_user_prompt = _INSTALLED.pop("orig_prompt")
    _INSTALLED.pop("run_session_probe", None)


__all__ = ["install", "uninstall", "ProbeConfig", "KnownNoOp", "action_key"]
