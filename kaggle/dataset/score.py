"""ARC-AGI-3 per-game score, mirrored from arc_agi 0.9.9 scorecard.py.

Per completed level: min(115, (baseline / actions) ** 2 * 100). Level i
(1-indexed) has weight i. Game score is the weighted mean, capped at
max_weights / total_weights * 100 where max_weights sums levels that scored.
Same formula as TAAF game.py _compute_final_score.
"""

from __future__ import annotations


def level_score(baseline: int, actions: int) -> float:
    if actions <= 0:
        return 0.0
    return min(115.0, (baseline / actions) ** 2 * 100.0)


def game_score(baselines: list[int], actions_per_level: list[int], levels_completed: int) -> float:
    total = 0.0
    total_w = 0
    max_w = 0
    for i, base in enumerate(baselines):
        w = i + 1
        total_w += w
        acts = actions_per_level[i] if i < len(actions_per_level) else 0
        s = level_score(base, acts) if i < levels_completed else 0.0
        if s > 0:
            max_w += w
        total += s * w
    if total_w == 0:
        return 0.0
    return min(total / total_w, max_w / total_w * 100.0)


def breakeven_saving(baseline: int, agent_actions: int, probe_cost: int) -> float:
    """How many LLM actions the probe must save to pay for itself on a level.

    With probe: actions = agent_actions - saved + probe_cost. It pays back
    exactly when saved >= probe_cost, because score is monotone in actions.
    Returned for the report: the score delta if the probe saves nothing.
    """
    return level_score(baseline, agent_actions + probe_cost) - level_score(baseline, agent_actions)
