"""Replay recorded Duck transcripts through the real fork ToolAgent history code, CPU only, no model.

A transcript (runs/<run>/transcripts/<game>_p0.txt) holds every turn: the frame prompt, then per model
response its reasoning, visible reply, python tool call and tool result. This module rebuilds the message
dicts and walks them through the same history steps `ToolAgent.analyze` takes:

  turn start   _trim_messages_for_context([system, *history, frame], preserve_recent=1)   (tool_agent.py:1756)
  each request _trim_messages_for_context(messages)                                       (line 1790)
  turn end     history = _persistent_history_messages(messages)                            (line 2017)

The level change is fed to `_update_summarized_knowledge_from_step_summary` before the turn end, as the
tool dispatch does (line 1584). The compaction digest is spliced into each frame prompt where the base puts
its world-model block ("end of world model."), because the recorded prompt was built without it.

Budget and sizes use the fork's own `_estimate_tokens` (json length / 3) and the Kaggle settings in the
control run's taaf_setup_env.json (context 32768, reply reserve 512, safety margin 512).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SRC = Path.home() / "CODE/arc-agi-3-probe-data/fork-src/src/ARC3-Inference"
RUNS = Path.home() / "CODE/arc-agi-3-probe/runs/ab-2026-09-23"
CONTROL_RUNS = ("control-s1", "control-s2r", "control-s3r")
SECTION_RE = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|THINKING|MODEL RESPONSE META|ASSISTANT|TOOL CALL: [a-z_]+|TOOL RESULT: [a-z_]+|ANALYZER STATUS)\]\n", re.M)
STEP_RE = re.compile(r"^--- analysis_step=\d+ .*---\n", re.M)
CODE_RE = re.compile(r"<parameter=code>\n?(.*?)\n?</parameter>", re.S)
WM_END = "end of world model."


def tool_agent():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import inference.agent.tool_agent as ta
    return ta


@dataclass
class Turn:
    frame: str
    events: list = field(default_factory=list)  # ("assistant", dict) | ("tool", str) | ("user", str)


def parse(path: Path) -> list[Turn]:
    turns: list[Turn] = []
    for block in STEP_RE.split(path.read_text(errors="replace"))[1:]:
        heads = list(SECTION_RE.finditer(block))
        secs = [(h.group(1), block[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(block)].strip())
                for i, h in enumerate(heads)]
        turn = None
        pending = None

        def flush():
            nonlocal pending
            if pending is not None and turn is not None:
                if pending.get("content") or pending.get("reasoning") or pending.get("tool_calls"):
                    turn.events.append(("assistant", pending))
            pending = None

        for label, body in secs:
            if label == "USER PROMPT":
                if turn is None:
                    turn = Turn(frame=body)
                else:
                    flush()
                    turn.events.append(("user", body))
            elif turn is None:
                continue
            elif label == "MODEL RESPONSE META":
                flush()
                pending = {"role": "assistant"}
            elif label == "THINKING" and pending is not None:
                pending["reasoning"] = body
            elif label == "ASSISTANT" and pending is not None:
                pending["content"] = body
            elif label.startswith("TOOL CALL") and pending is not None:
                m = CODE_RE.search(body)
                code = m.group(1) if m else body
                pending.setdefault("tool_calls", []).append(
                    {"id": f"call_{len(turn.events)}", "type": "function",
                     "function": {"name": "python", "arguments": json.dumps({"code": code})}})
            elif label.startswith("TOOL RESULT"):
                flush()
                turn.events.append(("tool", body))
            elif label == "ANALYZER STATUS":
                flush()
        flush()
        if turn is not None:
            turns.append(turn)
    return turns


def _summary(next_frame: str | None) -> dict:
    if next_frame is None:
        return {}
    s: dict = {}
    m = re.findall(r"Current state: step \d+, level (\d+)", next_frame)
    if m:
        s["level"] = int(m[-1])
    if "You have progressed to a new level!" in next_frame:
        s["level_transition"] = True
    if "You have completed the run!" in next_frame:
        s["run_complete"] = True
    if "The game is over." in next_frame:
        s["game_over"] = True
    return s


def new_agent(ta, context_window: int = 32768):
    a = ta.ToolAgent.__new__(ta.ToolAgent)
    a._context_budget_tokens = max(1024, context_window - 512 - 512)
    a._summarized_knowledge = ta._empty_world_model()
    a._last_step_summary = None
    a._history_messages = []
    a._session_runtime_dir = None
    return a


TOOLS = [{"type": "function", "function": {"name": "python", "description": "x" * 1800,
                                           "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                                          "required": ["code"]}}}]


@dataclass
class ReplayStats:
    turns: int = 0
    dropped_ids: set = field(default_factory=set)       # every message that left the history for good
    digested_ids: set = field(default_factory=set)      # of those, the ones the patch saw at turn end
    same_turn_lost: int = 0                             # created and dropped inside one turn: the known gap
    digest_tokens_per_request: list = field(default_factory=list)  # digest copies baked into kept frames
    history_msgs_per_request: list = field(default_factory=list)
    digests: list = field(default_factory=list)         # (turn, digest text) whenever it changed
    kept_trace: list = field(default_factory=list)      # per turn: the kept history as (role, content text)
    patch_seen: int = 0                                 # dropped messages the PATCH counted (its STATS), not this replay


def replay(path: Path, ta, comp=None, context_window: int = 32768, splice: bool = True) -> ReplayStats:
    """Walk one recorded game. With `comp` installed (and splice), the digest goes into each new frame prompt."""
    turns = parse(path)
    a = new_agent(ta, context_window)
    system = {"role": "system", "content": "s" * 24000}  # the fork system prompt is about 24k chars
    st = ReplayStats()
    last_digest = ""
    seen0 = comp.STATS["dropped_messages"] if comp else 0
    digest_chars: dict[int, int] = {}  # id(frame message) -> digest chars baked into it
    alive: list = []  # keep every message object alive, so no id() is ever reused
    for i, t in enumerate(turns):
        st.turns += 1
        frame = t.frame
        digest_now = a._summarized_knowledge.get(comp.KEY, "") if comp else ""
        if splice and digest_now and WM_END in frame:
            line = f"- Compacted earlier turns (dropped from history, oldest first, truncated): {digest_now}"
            frame = frame.replace(WM_END, line + "\n" + WM_END, 1)
        user = {"role": "user", "content": frame}
        digest_chars[id(user)] = len(digest_now) if splice and WM_END in t.frame else 0
        old_hist = list(a._history_messages)
        created = [user]
        msgs = a._trim_messages_for_context([system, *a._history_messages, user], tools=TOOLS, preserve_recent=1)
        for kind, payload in t.events:
            msgs = a._trim_messages_for_context(msgs, tools=TOOLS)
            st.history_msgs_per_request.append(len(msgs) - 1)
            st.digest_tokens_per_request.append(
                sum(digest_chars.get(id(m), 0) for m in msgs) // 3)
            if kind == "assistant":
                m = dict(payload)
            elif kind == "tool":
                m = {"role": "tool", "tool_call_id": "x", "content": payload}
            else:
                m = {"role": "user", "content": payload}
            created.append(m)
            msgs.append(m)
        a._last_step_summary = _summary(turns[i + 1].frame if i + 1 < len(turns) else None) or None
        a._update_summarized_knowledge_from_step_summary()
        final_ids = {id(m) for m in msgs}
        kept = a._persistent_history_messages(msgs, tools=TOOLS)
        keep_ids = {id(m) for m in kept}
        for m in [*old_hist, *created]:
            if id(m) not in keep_ids:
                st.dropped_ids.add(id(m))
                if any(m is x for x in old_hist) or id(m) in final_ids:
                    st.digested_ids.add(id(m))
                else:
                    st.same_turn_lost += 1
        a._history_messages = kept
        alive.extend(created)
        st.kept_trace.append(tuple((m.get("role"), str(m.get("content"))) for m in kept))
        d = a._summarized_knowledge.get(comp.KEY, "") if comp else ""
        if d != last_digest:
            st.digests.append((i, d))
            last_digest = d
    st.patch_seen = (comp.STATS["dropped_messages"] - seen0) if comp else 0
    return st


def recorded_games(run: str = "control-s1") -> list[Path]:
    return sorted((RUNS / run / "transcripts").glob("*_p0.txt"))
