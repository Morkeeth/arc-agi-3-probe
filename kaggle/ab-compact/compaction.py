"""DRAFT lever "compaction" (runs/RULE-compaction.md). Not in the Kaggle dataset yet, not run on GPU.

The base drops the oldest history when a prompt passes the context budget (rolling truncation,
tool_agent.py:1672-1689 `_trim_messages_for_context` -> `_drop_oldest_history_block`), and keeps at most
30 assistant turns (`_PERSISTENT_HISTORY_ASSISTANT_TURNS`, line 151). What it drops is gone. The model's
own world-model note (lines 1105-1111, printed by `_summarized_knowledge_lines`, line 1236) is carried,
but nothing digests the dropped turns.

This patch is one knob, ARC_COMPACT_MAX_CHARS (0 = off, the base). When `_persistent_history_messages`
drops messages for good, a short digest of them goes into the carried notes, as a "Compacted earlier
turns" line, capped at ARC_COMPACT_MAX_CHARS (oldest text falls off first). The digest keeps the visible
reply of each dropped assistant turn (not its reasoning) and the start of each tool result. Frames
(user messages) are skipped: the current frame is always in the prompt. The digest is cleared on a level
change, like the other notes. No extra model call, so no extra GPU time; the digest counts against the
same prompt budget, so it displaces a little raw history.

    from compaction import install; install(6000)     # in the hook cell, before the benchmark runs
"""

from __future__ import annotations

import os
from typing import Any

KEY = "compacted_history"
ASSISTANT_CHARS = 400
TOOL_CHARS = 200
_ORIG: dict[str, Any] = {}
STATS = {"digests": 0, "dropped_messages": 0}


def _text(m: dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return " ".join(str(c or "").split())


def digest(dropped: list[dict[str, Any]]) -> str:
    parts = []
    for m in dropped:
        role = str(m.get("role", "")).strip()
        t = _text(m)
        if not t:
            continue
        if role == "assistant":
            parts.append(f"[you] {t[:ASSISTANT_CHARS]}")
        elif role == "tool":
            parts.append(f"[result] {t[:TOOL_CHARS]}")
    return " | ".join(parts)


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
        kept = _ORIG["persist"](self, messages, tools=tools)
        keep_ids = {id(m) for m in kept}
        dropped = [m for m in messages[1:] if id(m) not in keep_ids]
        d = digest(dropped)
        if d:
            old = self._summarized_knowledge.get(KEY, "")
            self._summarized_knowledge[KEY] = (f"{old} | {d}" if old else d)[-max_chars:]
            STATS["digests"] += 1
            STATS["dropped_messages"] += len(dropped)
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
        if s.get("level_transition") or s.get("run_complete") or s.get("game_over"):
            self._summarized_knowledge[KEY] = ""

    TA._persistent_history_messages = _persistent_history_messages
    TA._summarized_knowledge_lines = _summarized_knowledge_lines
    TA._update_summarized_knowledge_from_step_summary = _update_summarized_knowledge_from_step_summary
    print(f"COMPACT_PROOF max_chars={max_chars} persist={TA._persistent_history_messages.__module__}", flush=True)
    return max_chars


def uninstall(tool_agent_module) -> None:
    if not _ORIG:
        return
    TA = tool_agent_module.ToolAgent
    TA._persistent_history_messages = _ORIG.pop("persist")
    TA._summarized_knowledge_lines = _ORIG.pop("lines")
    TA._update_summarized_knowledge_from_step_summary = _ORIG.pop("step")
