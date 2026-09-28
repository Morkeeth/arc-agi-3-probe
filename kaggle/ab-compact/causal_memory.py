"""Bounded causal memory built only from observed, unambiguous ARC turns.

The live hook consumes messages when the real ToolAgent drops them. Effects are bound to a
game, level, turn and tool-call id; prose remains a hypothesis/test/constraint and is never
silently promoted into an observed effect.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

CATEGORIES = ("action_effect", "hypothesis", "constraint", "test")
LABELS = {"action_effect": "effect", "hypothesis": "hypothesis", "constraint": "constraint", "test": "test"}
STEP_RE = re.compile(r"Current state: step (\d+), level (\d+)")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ACTION_RE = re.compile(r"\baction\((.{0,500}?)\)(?:\s*$|\s*[;\n#])", re.M)
ACTION_WORD_RE = re.compile(r"\b(LEFT|RIGHT|UP|DOWN|SPACE|ACTION\d+|MOUSE)\b", re.I)
RESULT_FIELDS = ("board_changed", "reward", "level_completed", "game_over", "run_complete", "stop_reason", "error")
REJECT_RE = re.compile(r"\b(?:wrong|failed|failure|blocked|no effect|did not|didn't|does not|doesn't|cannot|can't|reject)\b", re.I)
CONSTRAINT_RE = re.compile(r"\b(?:goal|target|constraint|must|need(?:s)? to|cannot|can't|valid actions?|object|avoid|only)\b", re.I)
TEST_RE = re.compile(r"\b(?:next|plan|try|testing|test|check|probe|inspect|verify|experiment)\b", re.I)
CONDITIONAL_RE = re.compile(r"\b(?:if|unless|might|maybe|likely|hypothesis|appears?|seems?)\b", re.I)
STATE_SUFFIX = "_tool_runtime_state.json"
PASS_SUFFIX_RE = re.compile(r"_p\d+$")
KEY = "causal_memory"
STATS = {"digests": 0, "dropped_messages": 0, "cleared": 0}
_EVENT_LOCK = threading.Lock()
_ORIGINAL_ENSURE = None
_ORIG: dict[str, Any] = {}
_EVENTS: Path | None = None


class Fact(NamedTuple):
    category: str
    game_id: str
    level: int
    turn: int
    tool_call_id: str
    status: str
    text: str
    evidence: str


def game_id_from_state_path(state_path: str | Path) -> str:
    name = Path(state_path).name
    stem = name[:-len(STATE_SUFFIX)] if name.endswith(STATE_SUFFIX) else Path(name).stem
    return PASS_SUFFIX_RE.sub("", stem)


def install_event_identity(tool_agent_module) -> None:
    """Small standalone identity probe retained for the concurrent-game unit test."""
    global _ORIGINAL_ENSURE
    if _ORIGINAL_ENSURE is not None:
        return
    cls = tool_agent_module.ToolAgent
    _ORIGINAL_ENSURE = cls._ensure_session

    def _ensure_session(self, state_path):
        result = _ORIGINAL_ENSURE(self, state_path)
        self._causal_game_id = game_id_from_state_path(state_path)
        return result

    cls._ensure_session = _ensure_session


def uninstall_event_identity(tool_agent_module) -> None:
    global _ORIGINAL_ENSURE
    if _ORIGINAL_ENSURE is None:
        return
    tool_agent_module.ToolAgent._ensure_session = _ORIGINAL_ENSURE
    _ORIGINAL_ENSURE = None


def write_event(agent, path: str | Path, **fields) -> None:
    record = {"game_id": str(getattr(agent, "_causal_game_id", "") or ""),
              "runtime_dir": str(getattr(agent, "_session_runtime_dir", "") or ""), **fields}
    with _EVENT_LOCK, Path(path).open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _content(message: dict[str, Any]) -> str:
    value = message.get("content", "")
    if isinstance(value, list):
        value = " ".join(str(part.get("text", "")) for part in value if isinstance(part, dict))
    return " ".join(str(value or "").split())


def _call_actions(message: dict[str, Any]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id", "") or "")
        fn = call.get("function", {})
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"code": args}
        code = str(args.get("code", "")) if isinstance(args, dict) else str(args)
        for raw in ACTION_RE.findall(code):
            action = " ".join(raw.split())
            if action:
                found.append((call_id, action))
    return found


def _result_facts(text: str) -> str | None:
    compact = normalize(text)
    if "traceback (most recent call last)" in compact:
        return None
    values: list[str] = []
    for field in RESULT_FIELDS:
        matches = re.findall(rf"['\"]?{field}['\"]?\s*[:=]\s*['\"]?([^,'\"}}\s]+)", compact, re.I)
        unique = list(dict.fromkeys(value.strip().lower() for value in matches))
        if len(unique) > 1:
            return None
        if unique:
            values.append(f"{field}={unique[0]}")
    if not values:
        changed = list(dict.fromkeys(re.findall(r"\bchanged\s+(\d+|none|false|true)\b", compact)))
        if len(changed) > 1:
            return None
        if changed:
            values.append(f"changed={changed[0]}")
    return " ".join(values[:5]) or None


def _sentences(text: str) -> list[str]:
    return [" ".join(part.split()).strip(" -") for part in SENTENCE_RE.split(text) if part.strip()]


def _action_words(text: str) -> set[str]:
    return {word.upper() for word in ACTION_WORD_RE.findall(text)}


def _classify(sentence: str) -> tuple[str, str] | None:
    """One sentence produces at most one epistemic category."""
    if CONDITIONAL_RE.search(sentence):
        return "hypothesis", "conditional"
    if TEST_RE.search(sentence):
        return "test", "pending"
    if REJECT_RE.search(sentence):
        return "hypothesis", "rejected"
    if CONSTRAINT_RE.search(sentence):
        return "constraint", "claimed"
    return None


class Extractor:
    def __init__(self, game_id: str = ""):
        self.game_id = game_id
        self.level = 0
        self.turn = 0
        self.pending_action: tuple[str, str, int, int] | None = None
        self.facts: list[Fact] = []
        self._seen: set[tuple] = set()

    def _add(self, category: str, status: str, text: str, evidence: str = "", tool_call_id: str = "") -> None:
        clean = " ".join(text.split())[:240]
        anchor = normalize(evidence or clean)[:180]
        key = (category, self.game_id, self.level, self.turn, tool_call_id, status, anchor)
        if clean and key not in self._seen:
            self._seen.add(key)
            self.facts.append(Fact(category, self.game_id, self.level, self.turn, tool_call_id,
                                   status, clean, anchor))
            self.facts = self.facts[-256:]

    def feed(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            role = str(message.get("role", "")).strip()
            text = _content(message)
            if role == "user":
                match = STEP_RE.search(text)
                if match:
                    self.turn, self.level = int(match.group(1)), int(match.group(2))
                self.pending_action = None
                continue
            if role == "assistant":
                actions = _call_actions(message)
                self.pending_action = (*actions[0], self.turn, self.level) if len(actions) == 1 else None
                for sentence in _sentences(text):
                    classified = _classify(sentence)
                    if classified:
                        self._add(classified[0], classified[1], sentence)
                continue
            if role != "tool" or self.pending_action is None:
                continue
            call_id, action, action_turn, action_level = self.pending_action
            result_id = str(message.get("tool_call_id", "") or "")
            if result_id not in ("", "x", call_id):
                continue
            outcome = _result_facts(text)
            self.pending_action = None
            if outcome is None:
                continue
            self.turn, self.level = action_turn, action_level
            self._add("action_effect", "observed", f"action({action}) => {outcome}", outcome, call_id)
            words = _action_words(action)
            if words:
                pending_here = [f for f in self.facts if f.category == "test" and f.status == "pending" and
                                f.game_id == self.game_id and f.level == self.level and f.turn == self.turn]
                replaced: list[Fact] = []
                for fact in self.facts:
                    if (fact.category == "test" and fact.status == "pending" and
                            fact.game_id == self.game_id and fact.level == self.level and
                            ((_action_words(fact.text) & words) or
                             (fact in pending_here and len(pending_here) == 1))):
                        fact = fact._replace(status="completed", tool_call_id=call_id,
                                             evidence=normalize(outcome)[:180])
                    replaced.append(fact)
                self.facts = replaced


def extract_facts(messages: list[dict[str, Any]], game_id: str = "") -> list[Fact]:
    extractor = Extractor(game_id)
    extractor.feed(messages)
    return extractor.facts


def _line(fact: Fact) -> str:
    identity = f" {fact.game_id}/l{fact.level}" if fact.game_id else f" l{fact.level}"
    call = f" {fact.tool_call_id}" if fact.tool_call_id else ""
    return f"[t{fact.turn}{identity}{call} {LABELS[fact.category]}:{fact.status}] {fact.text}"


def bounded_record(facts: list[Fact], max_chars: int = 3000) -> str:
    if max_chars <= 0:
        return ""
    by_category: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        if fact.category in CATEGORIES:
            by_category[fact.category].append(fact)
    for category in CATEGORIES:
        dedup: dict[tuple, Fact] = {}
        for fact in by_category[category]:
            dedup[(fact.game_id, fact.level, fact.turn, fact.tool_call_id, fact.status, normalize(fact.text))] = fact
        by_category[category] = list(dedup.values())[::-1]
    chosen: list[Fact] = []
    used = 0
    index = 0
    while True:
        added = False
        for category in CATEGORIES:
            rows = by_category[category]
            if index >= len(rows):
                continue
            fact = rows[index]
            cost = len(_line(fact)) + (1 if chosen else 0)
            if used + cost <= max_chars:
                chosen.append(fact)
                used += cost
                added = True
        if not added:
            break
        index += 1
    chosen.sort(key=lambda f: (f.level, f.turn, CATEGORIES.index(f.category), f.tool_call_id))
    return "\n".join(_line(fact) for fact in chosen)


def naive_record(facts: list[Fact], max_chars: int = 3000) -> str:
    """Unlabelled last-N-character arm made from the extractor's strings."""
    return "\n".join(f.text for f in facts)[-max_chars:]


def _dropped_messages(old_history: list, final_messages: list, kept: list) -> list:
    keep_ids = {id(m) for m in kept}
    seen: set[int] = set()
    out = []
    for message in [*old_history, *final_messages[1:]]:
        ident = id(message)
        if ident not in keep_ids and ident not in seen:
            seen.add(ident)
            out.append(message)
    return out


def _events_path() -> Path | None:
    if _EVENTS is not None:
        return _EVENTS
    value = os.environ.get("ARC_CAUSAL_EVENTS", "")
    return Path(value) if value else None


def _event(agent, **fields) -> None:
    path = _events_path()
    if path is None:
        return
    try:
        write_event(agent, path, **fields)
    except OSError:
        pass


def prepare_replay(agent, game_id: str, runtime_dir: str | Path) -> None:
    """Give the CPU replay the same identity evidence production gets from `_ensure_session`."""
    agent._causal_game_id = game_id
    agent._session_runtime_dir = Path(runtime_dir)


def install(max_chars: int = 3000, tool_agent_module=None, events_path: str | Path | None = None) -> int:
    """Install the causal record on the real persistent-history drop path."""
    global _EVENTS
    if max_chars <= 0 or _ORIG:
        return max_chars
    if tool_agent_module is None:
        import inference.agent.tool_agent as tool_agent_module
    _EVENTS = Path(events_path) if events_path else None
    TA = tool_agent_module.ToolAgent
    _ORIG.update(persist=TA._persistent_history_messages, lines=TA._summarized_knowledge_lines,
                 step=TA._update_summarized_knowledge_from_step_summary, ensure=TA._ensure_session)

    def _ensure_session(self, state_path):
        result = _ORIG["ensure"](self, state_path)
        self._causal_game_id = game_id_from_state_path(state_path)
        return result

    def _persistent_history_messages(self, messages, *, tools=None):
        old = list(getattr(self, "_history_messages", None) or [])
        kept = _ORIG["persist"](self, messages, tools=tools)
        dropped = _dropped_messages(old, messages, kept)
        if not dropped:
            return kept
        game_id = str(getattr(self, "_causal_game_id", "") or "")
        extractor = getattr(self, "_causal_extractor", None)
        if extractor is None or extractor.game_id != game_id:
            extractor = self._causal_extractor = Extractor(game_id)
        extractor.feed(dropped)
        record = bounded_record(extractor.facts, max_chars)
        self._summarized_knowledge[KEY] = record
        with _EVENT_LOCK:
            STATS["dropped_messages"] += len(dropped)
            STATS["digests"] += 1
        _event(self, kind="digest", level=extractor.level, dropped=len(dropped), chars=len(record),
               fact_count=len(extractor.facts))
        return kept

    def _summarized_knowledge_lines(self):
        lines = _ORIG["lines"](self)
        record = self._summarized_knowledge.get(KEY, "")
        if not record:
            return lines
        extra = "- Causal record from earlier dropped turns:\n" + record
        if not lines:
            return ["Working world model carried from earlier turns:", extra]
        return [*lines[:-1], extra, lines[-1]]

    def _update_summarized_knowledge_from_step_summary(self):
        _ORIG["step"](self)
        summary = self._last_step_summary or {}
        if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
            if self._summarized_knowledge.get(KEY):
                STATS["cleared"] += 1
            self._summarized_knowledge[KEY] = ""
            self._causal_extractor = Extractor(str(getattr(self, "_causal_game_id", "") or ""))
            _event(self, kind="clear", level=summary.get("level"))

    TA._ensure_session = _ensure_session
    TA._persistent_history_messages = _persistent_history_messages
    TA._summarized_knowledge_lines = _summarized_knowledge_lines
    TA._update_summarized_knowledge_from_step_summary = _update_summarized_knowledge_from_step_summary
    return max_chars


def uninstall(tool_agent_module) -> None:
    global _EVENTS
    if not _ORIG:
        return
    TA = tool_agent_module.ToolAgent
    TA._persistent_history_messages = _ORIG.pop("persist")
    TA._summarized_knowledge_lines = _ORIG.pop("lines")
    TA._update_summarized_knowledge_from_step_summary = _ORIG.pop("step")
    TA._ensure_session = _ORIG.pop("ensure")
    _EVENTS = None
