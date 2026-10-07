"""CPU spike for prospective, state-bound prediction evidence; not a game-policy hook.

A mismatch concerns this captured transition, not the truth of a general rule.
Observed state may omit hidden state. Both future experimental arms must elicit
identical predictions; this module supplies shared instrumentation to both.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def frozen(value: Any) -> Any:
    return json.loads(canonical(value))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Commitment:
    sequence: int
    game_id: str
    level: int
    call_id: str
    rule_id: str
    before_json: str
    action_json: str
    prediction_json: str


class PredictionEvidence:
    def __init__(self):
        self._sequence = 0
        self._pending: dict[tuple[str, str], Commitment] = {}
        self._used: set[tuple[str, str]] = set()
        self.records: list[dict] = []

    def commit(self, *, game_id, level, call_id, rule_id, before, action, prediction):
        if not all(isinstance(v, str) and v.strip() for v in (game_id, call_id, rule_id)):
            raise ValueError('explicit game, call and rule identities required')
        if type(level) is not int or level < 0:
            raise ValueError('invalid level')
        if not isinstance(before, dict) or not before:
            raise ValueError('a nonempty captured state is required')
        if not isinstance(prediction, dict) or not prediction:
            raise ValueError('empty predictions would pass vacuously')
        if not isinstance(action, dict) or not action:
            raise ValueError('one explicit action is required')
        key = (game_id, call_id)
        if key in self._used:
            raise ValueError('call already observed or committed')
        self._sequence += 1
        record = Commitment(self._sequence, game_id, level, call_id, rule_id,
                            canonical(before), canonical(action), canonical(prediction))
        self._pending[key] = record
        self._used.add(key)
        return record

    def observe(self, *, game_id, level, call_id, before, action, outcome):
        # Capture arrival even with no commitment: later backfilling is refused.
        key = (game_id, call_id)
        self._used.add(key)
        self._sequence += 1
        commitment = self._pending.pop(key, None)
        if commitment is None:
            raise ValueError('observation has no prior commitment')
        if (level != commitment.level or canonical(before) != commitment.before_json
                or canonical(action) != commitment.action_json):
            raise ValueError('observation identity, captured state or action differs')
        if not isinstance(outcome, dict):
            raise ValueError('observation must be an object')
        prediction = json.loads(commitment.prediction_json)
        missing = sorted(set(prediction) - set(outcome))
        differences = {k: {'predicted': v, 'observed': outcome[k]} for k, v in prediction.items()
                       if k in outcome and canonical(v) != canonical(outcome[k])}
        record = {'game_id': game_id, 'level': level, 'call_id': call_id,
                  'rule_id': commitment.rule_id, 'commit_sequence': commitment.sequence,
                  'observation_sequence': self._sequence, 'before': json.loads(commitment.before_json),
                  'before_sha256': fingerprint(before), 'action': json.loads(commitment.action_json),
                  'prediction': prediction, 'outcome': frozen(outcome), 'missing': missing,
                  'differences': differences,
                  'verdict': 'INCOMPLETE' if missing else ('MISMATCH' if differences else 'MATCH'),
                  'scope': 'this observed transition only; hidden state and general rule validity untested'}
        self.records.append(frozen(record))
        return record

    def retrieve(self, *, game_id, level, before, max_chars=3000):
        """Only exact observed-state matches; no cross-level or global rule rejection."""
        lines = []
        size = 0
        digest = fingerprint(before)
        for record in reversed(self.records):
            if (record['game_id'], record['level'], record['before_sha256'], record['verdict']) != (game_id, level, digest, 'MISMATCH'):
                continue
            line = canonical(record)
            cost = len(line) + bool(lines)
            if size + cost <= max_chars:
                lines.append(line)
                size += cost
        return '\n'.join(reversed(lines))
