"""Deterministic, bounded causal memory extracted from observed ARC turns.

The record is deliberately not a prose summary. Each line states its observed turn and one of four
fact classes. Action effects come from the tool result, never from the assistant's prediction.
"""

from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

CATEGORIES = ("action_effect", "rejected_hypothesis", "constraint", "unfinished_test")
LABELS = {
    "action_effect": "effect",
    "rejected_hypothesis": "reject",
    "constraint": "constraint",
    "unfinished_test": "test",
}
STEP_RE = re.compile(r"Current state: step (\d+), level (\d+)")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ACTION_RE = re.compile(r"\baction\((.{0,240}?)\)(?:\s*$|\s*[;\n#])", re.M)
RESULT_FIELDS = ("board_changed", "reward", "level_completed", "game_over", "run_complete", "stop_reason", "error")
REJECT_RE = re.compile(r"\b(?:wrong|failed|failure|blocked|no effect|did not|didn't|does not|doesn't|cannot|can't|not the|reject)\b", re.I)
CONSTRAINT_RE = re.compile(r"\b(?:goal|target|constraint|must|need(?:s)? to|cannot|can't|valid actions?|object|avoid|only)\b", re.I)
TEST_RE = re.compile(r"\b(?:next|plan|try|test|check|probe|inspect|verify|experiment)\b", re.I)
STATE_SUFFIX = "_tool_runtime_state.json"
PASS_SUFFIX_RE = re.compile(r"_p\d+$")
_EVENT_LOCK = threading.Lock()
_ORIGINAL_ENSURE = None


class Fact(NamedTuple):
    category: str
    turn: int
    text: str
    evidence: str


def game_id_from_state_path(state_path: str | Path) -> str:
    name = Path(state_path).name
    stem = name[:-len(STATE_SUFFIX)] if name.endswith(STATE_SUFFIX) else Path(name).stem
    return PASS_SUFFIX_RE.sub("", stem)


def install_event_identity(tool_agent_module) -> None:
    """Capture the game identity from the state path before shared runtime-directory evidence is written."""
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
    record = {
        "game_id": str(getattr(agent, "_causal_game_id", "") or ""),
        "runtime_dir": str(getattr(agent, "_session_runtime_dir", "") or ""),
        **fields,
    }
    with _EVENT_LOCK, Path(path).open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _content(message: dict[str, Any]) -> str:
    value = message.get("content", "")
    if isinstance(value, list):
        value = " ".join(str(part.get("text", "")) for part in value if isinstance(part, dict))
    return " ".join(str(value or "").split())


def _actions(message: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"code": args}
        code = str(args.get("code", "")) if isinstance(args, dict) else str(args)
        for raw in ACTION_RE.findall(code):
            action = " ".join(raw.split())
            if action and action not in found:
                found.append(action)
    return found[:3]


def _result_facts(text: str) -> str:
    compact = normalize(text)
    values: list[str] = []
    for field in RESULT_FIELDS:
        match = re.search(rf"['\"]?{field}['\"]?\s*[:=]\s*['\"]?([^,'\"}}\s]+)", compact, re.I)
        if match:
            value = match.group(1).strip().lower()
            values.append(f"{field}={value}")
    if not values:
        changed = re.search(r"\bchanged\s+(\d+|none|false|true)\b", compact)
        if changed:
            values.append(f"changed={changed.group(1)}")
    return " ".join(values[:5]) or compact[:100]


def _sentences(text: str) -> list[str]:
    return [" ".join(part.split()).strip(" -") for part in SENTENCE_RE.split(text) if part.strip()]


def extract_facts(messages: list[dict[str, Any]]) -> list[Fact]:
    facts: list[Fact] = []
    turn = 0
    pending_actions: list[str] = []
    seen: set[tuple[str, int, str]] = set()

    def add(category: str, text: str, evidence: str | None = None) -> None:
        clean = " ".join(text.split())[:220]
        anchor = normalize(evidence if evidence is not None else clean)[:140]
        key = (category, turn, anchor)
        if clean and anchor and key not in seen:
            seen.add(key)
            facts.append(Fact(category, turn, clean, anchor))

    for message in messages:
        role = str(message.get("role", "")).strip()
        text = _content(message)
        if role == "user":
            match = STEP_RE.search(text)
            if match:
                turn = int(match.group(1))
            pending_actions = []
            continue
        if role == "assistant":
            pending_actions = _actions(message)
            for sentence in _sentences(text):
                clipped = sentence[:180]
                if REJECT_RE.search(sentence):
                    add("rejected_hypothesis", clipped)
                if CONSTRAINT_RE.search(sentence):
                    add("constraint", clipped)
                if TEST_RE.search(sentence):
                    add("unfinished_test", clipped)
            continue
        if role == "tool" and pending_actions:
            outcome = _result_facts(text)
            for action in pending_actions:
                add("action_effect", f"action({action}) => {outcome}", evidence=outcome)
            pending_actions = []
    return facts


def bounded_record(facts: list[Fact], max_chars: int = 3000) -> str:
    """Keep a balanced, recent record without splitting facts or exceeding the budget."""
    if max_chars <= 0:
        return ""
    by_category: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        if fact.category in CATEGORIES:
            by_category[fact.category].append(fact)
    for category in CATEGORIES:
        dedup: dict[str, Fact] = {}
        for fact in by_category[category]:
            dedup[normalize(fact.text)] = fact
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
            line = f"[t{fact.turn} {LABELS[category]}] {fact.text}"
            cost = len(line) + (1 if chosen else 0)
            if used + cost <= max_chars:
                chosen.append(fact)
                used += cost
                added = True
        if not added:
            break
        index += 1
    chosen.sort(key=lambda fact: (fact.turn, CATEGORIES.index(fact.category)))
    return "\n".join(f"[t{fact.turn} {LABELS[fact.category]}] {fact.text}" for fact in chosen)
