"""Lever "compaction" (runs/RULE-compaction.md). One knob: ARC_COMPACT_MAX_CHARS, 0 = off = the base.

The base drops old history when a prompt passes the context budget (rolling truncation,
tool_agent.py `_trim_messages_for_context` -> `_drop_oldest_history_block`) and keeps at most 30 assistant
turns (`_PERSISTENT_HISTORY_ASSISTANT_TURNS`). What it drops is gone. The model's own world-model note is
carried, but nothing digests the dropped turns.

With the knob on, every message that leaves the persistent history for good goes into a short digest,
carried in the world-model block of the next user prompt as one "Compacted earlier turns" line, capped at
ARC_COMPACT_MAX_CHARS (the oldest text falls off first).

Where drops are seen. `ToolAgent.analyze` trims in four places: the turn start (new frame added), every
request of the turn, the context-overflow retry, and `_persistent_history_messages` at the end of the turn.
The first three trim a local list, so a wrapper on the last one alone misses most drops (the 24 Sep draft
had that bug). This patch compares the history the turn STARTED with, plus the turn's final messages,
against the history it KEEPS. Anything in the first two and not in the third was dropped this turn, by any
of the four paths. A turn that fails (`preserve_history` False) restores the old history and never calls
`_persistent_history_messages`, so nothing is digested for it, which is correct: nothing was dropped.
Known gap: a message created and dropped inside the same turn (possible only when the whole older history
is already gone) is not seen. The CPU replay counts it (tests/test_compaction_replay.py).

What goes in. Per dropped assistant turn: its visible reply (first 400 characters, not its reasoning) and
up to 3 `action(...)` calls from its python tool code (80 characters each). Per tool result: its first 200
characters. User messages (frames and fixed instructions) are skipped. Only turns from the current level
are kept: the level comes from the "Current state: step N, level L" line of the user prompt that opened
the dropped block. On a level change, run complete or game over, the digest is cleared, like the base
clears its world-model notes.

No extra model call. The digest counts against the same prompt budget, and, like the base's world-model
notes, it is baked into each user prompt that history keeps, so it displaces some raw history.

    from compaction import install; install(MAX_CHARS)    # in the hook cell, before the benchmark runs

Proof: `COMPACT_PROOF max_chars=<n>` printed at install; one JSON line per digest update in
<ARC_COMPACT_EVENTS or /kaggle/working/compaction_events.jsonl> (runtime dir, level, dropped, chars).
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

KEY = "compacted_history"
LEVEL_KEY = "_compacted_level"
ASSISTANT_CHARS = 400
TOOL_CHARS = 200
ACTION_CHARS = 80
ACTIONS_PER_TURN = 3
LEVEL_RE = re.compile(r"Current state: step \d+, level (\d+)")
ACTION_RE = re.compile(r"\baction\((.{0,200}?)\)(?:\s*$|\s*[;\n#])", re.M)
_ORIG: dict[str, Any] = {}
STATS = {"digests": 0, "dropped_messages": 0, "cleared": 0}
_LOCK = threading.Lock()


def _text(m: dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return " ".join(str(c or "").split())


def _actions(m: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"code": args}
        code = str(args.get("code", "")) if isinstance(args, dict) else str(args)
        for a in ACTION_RE.findall(code):
            out.append("action(" + " ".join(a.split())[:ACTION_CHARS] + ")")
            if len(out) >= ACTIONS_PER_TURN:
                return out
    return out


def level_of(m: dict[str, Any]) -> int | None:
    if str(m.get("role", "")).strip() != "user":
        return None
    found = LEVEL_RE.findall(_text(m))
    return int(found[-1]) if found else None


def digest(dropped: list[dict[str, Any]], *, level: int | None = None, start_level: int | None = None) -> str:
    """Digest of dropped messages in order. With `level` set, only turns whose block opened on that level."""
    parts = []
    cur = start_level
    for m in dropped:
        role = str(m.get("role", "")).strip()
        if role == "user":
            lv = level_of(m)
            cur = lv if lv is not None else cur
            continue
        if level is not None and cur is not None and cur != level:
            continue
        if role == "assistant":
            t = _text(m)
            acts = _actions(m)
            if t:
                parts.append(f"[you] {t[:ASSISTANT_CHARS]}")
            if acts:
                parts.append("[act] " + "; ".join(acts))
        elif role == "tool":
            t = _text(m)
            if t:
                parts.append(f"[result] {t[:TOOL_CHARS]}")
    return " | ".join(parts)


def dropped_messages(old_history: list, final_messages: list, kept: list) -> list:
    """Messages in the turn's starting history or final list that are not in the kept history, in order."""
    keep_ids = {id(m) for m in kept}
    seen: set[int] = set()
    out = []
    for m in [*old_history, *final_messages[1:]]:
        i = id(m)
        if i in keep_ids or i in seen:
            continue
        seen.add(i)
        out.append(m)
    return out


def _current_level(agent, messages: list) -> int | None:
    s = getattr(agent, "_last_step_summary", None) or {}
    try:
        if s.get("level") is not None:
            return int(s["level"])
    except (TypeError, ValueError):
        pass
    for m in reversed(messages):
        lv = level_of(m)
        if lv is not None:
            return lv
    return None


def _events_path() -> Path | None:
    p = os.environ.get("ARC_COMPACT_EVENTS", "")
    if p:
        return Path(p)
    w = Path("/kaggle/working")
    return w / "compaction_events.jsonl" if w.is_dir() else None


def _event(agent, **kw) -> None:
    p = _events_path()
    if p is None:
        return
    rec = {"runtime_dir": str(getattr(agent, "_session_runtime_dir", "") or ""), **kw}
    try:
        with _LOCK, p.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def install(max_chars: int | None = None, tool_agent_module=None) -> int:
    if max_chars is None:
        max_chars = int(os.environ.get("ARC_COMPACT_MAX_CHARS", "0") or 0)
    if max_chars <= 0 or _ORIG:
        return max_chars
    if tool_agent_module is None:
        import inference.agent.tool_agent as tool_agent_module
    TA = tool_agent_module.ToolAgent
    _ORIG.update(persist=TA._persistent_history_messages, lines=TA._summarized_knowledge_lines,
                 step=TA._update_summarized_knowledge_from_step_summary)

    def _persistent_history_messages(self, messages, *, tools=None):
        old = list(getattr(self, "_history_messages", None) or [])
        kept = _ORIG["persist"](self, messages, tools=tools)
        dropped = dropped_messages(old, messages, kept)
        if not dropped:
            return kept
        level = _current_level(self, messages)
        start = self._summarized_knowledge.get(LEVEL_KEY)
        d = digest(dropped, level=level, start_level=start if isinstance(start, int) else None)
        for m in dropped:  # remember the level of the last dropped block opener for the next call
            lv = level_of(m)
            if lv is not None:
                self._summarized_knowledge[LEVEL_KEY] = lv
        with _LOCK:
            STATS["dropped_messages"] += len(dropped)
        if d:
            prev = self._summarized_knowledge.get(KEY, "")
            self._summarized_knowledge[KEY] = (f"{prev} | {d}" if prev else d)[-max_chars:]
            with _LOCK:
                STATS["digests"] += 1
            _event(self, kind="digest", level=level, dropped=len(dropped), chars=len(self._summarized_knowledge[KEY]))
        return kept

    def _summarized_knowledge_lines(self):
        lines = _ORIG["lines"](self)
        c = self._summarized_knowledge.get(KEY, "")
        if not c:
            return lines
        extra = f"- Compacted earlier turns (dropped from history, oldest first, truncated): {c}"
        if not lines:
            return ["Working world model carried from earlier turns:", extra]
        return [*lines[:-1], extra, lines[-1]]

    def _update_summarized_knowledge_from_step_summary(self):
        _ORIG["step"](self)
        s = self._last_step_summary or {}
        if (s.get("level_transition") or s.get("run_complete") or s.get("game_over")) and self._summarized_knowledge.get(KEY):
            self._summarized_knowledge[KEY] = ""
            with _LOCK:
                STATS["cleared"] += 1
            _event(self, kind="clear", level=s.get("level"))

    TA._persistent_history_messages = _persistent_history_messages
    TA._summarized_knowledge_lines = _summarized_knowledge_lines
    TA._update_summarized_knowledge_from_step_summary = _update_summarized_knowledge_from_step_summary
    print(f"COMPACT_PROOF max_chars={max_chars} persist={TA._persistent_history_messages.__module__} "
          f"events={_events_path()}", flush=True)
    return max_chars


def uninstall(tool_agent_module) -> None:
    if not _ORIG:
        return
    TA = tool_agent_module.ToolAgent
    TA._persistent_history_messages = _ORIG.pop("persist")
    TA._summarized_knowledge_lines = _ORIG.pop("lines")
    TA._update_summarized_knowledge_from_step_summary = _ORIG.pop("step")
