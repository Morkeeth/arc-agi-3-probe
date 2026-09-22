"""Offline harness. No GPU, no network once the environment files are local.

Sub-commands
  census  For each game: reset, then try each available action once from the
          level-1 start state in a fresh copy of the game. Prints which actions
          do nothing at the start. Free: each trial uses its own game instance.
  walk    Random policy for N steps per game with a NoOpMemo watching. Prints
          the no-op rate and the repeat-no-op rate. The repeat rate is what the
          zero-cost guard would have saved for this policy.
  ab      Random policy with and without the guard, scored with the official
          formula. A plumbing check and a weak proxy: the random policy is not
          Duck. Only a Duck run on a GPU settles the real question.

Environment files: --env-dir defaults to the unzipped competition data at
~/CODE/arc-agi-3-probe-data/environment_files. The same 25 public games also
come from the free anonymous key at three.arcprize.org.

Usage:
  python harness.py census --games ls20,vc33
  python harness.py walk --steps 300 --seed 1
  python harness.py ab --steps 300 --seed 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np

from deadprobe.grid import GAMEPLAY, as_grid, click_targets
from deadprobe.memo import NoOpMemo
from deadprobe.probe import NAMES, StepOut
from deadprobe.score import game_score

DEFAULT_ENV_DIR = Path.home() / "CODE" / "arc-agi-3-probe-data" / "environment_files"


def _arcade(env_dir: Path):
    import arc_agi

    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    return arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(env_dir))


def _games(arc, wanted: list[str] | None) -> list:
    envs = arc.get_environments()
    if wanted:
        envs = [e for e in envs if e.game_id.split("-")[0] in wanted]
    return sorted(envs, key=lambda e: e.game_id)


def _step_fn(env):
    from arcengine import GameAction

    def step(aid, data):
        o = env.step(GameAction.from_id(aid), data=data or {})
        return StepOut(o.frame[-1], o.levels_completed, o.state.name)

    return step


def cmd_census(args) -> None:
    """One trial per action, each in its own fresh copy of the game."""
    arc = _arcade(args.env_dir)
    rows = []
    for info in _games(arc, args.games):
        gid = info.game_id.split("-")[0]
        obs = arc.make(info.game_id).reset()
        grid0 = as_grid(obs.frame[-1])
        avail = [a for a in obs.available_actions if a != 0]
        trials: list[tuple[int, dict | None]] = [(a, None) for a in avail if a != 6]
        if 6 in avail:
            trials += [(6, {"x": int(c), "y": int(r)}) for r, c in click_targets(grid0, max_targets=args.clicks)]
        dead, live = [], []
        for aid, data in trials:
            env = arc.make(info.game_id)
            env.reset()
            memo = NoOpMemo()
            out = _step_fn(env)(aid, data)
            change = memo.record(grid0, aid, data, out.grid, level_changed=out.levels_completed > 0)
            name = f"MOUSE(row={data['y']}, col={data['x']})" if data else NAMES.get(aid, f"ACTION{aid}")
            (live if change.kind == GAMEPLAY or out.levels_completed > 0 else dead).append(name)
        rows.append(
            {
                "game": gid,
                "levels": len(info.baseline_actions or []),
                "baseline_l1": (info.baseline_actions or [None])[0],
                "available": avail,
                "dead_at_start": dead,
                "live_at_start": live,
            }
        )
        print(f"{gid}: available={avail} dead_at_start={dead} live={live}", flush=True)
    dead_n = sum(len(r["dead_at_start"]) for r in rows)
    trial_n = sum(len(r["dead_at_start"]) + len(r["live_at_start"]) for r in rows)
    print(f"\ntotal: {dead_n}/{trial_n} trials changed nothing at the level-1 start over {len(rows)} games")
    _dump(args.out, {"census": rows, "dead": dead_n, "trials": trial_n})


def _random_walk(arc, info, steps: int, seed: int, guard: bool):
    """Random policy. With guard=True, a known no-op is re-rolled, not spent."""
    from arcengine import GameAction

    rng = random.Random(seed)
    env = arc.make(info.game_id)
    obs = env.reset()
    memo = NoOpMemo()
    spent = 0
    actions_per_level: list[int] = [0] * len(info.baseline_actions or [])
    level = 0
    blocked = 0
    while spent < steps:
        grid = as_grid(obs.frame[-1])
        avail = [a for a in obs.available_actions if a != 0] or [1]
        choice = None
        for _ in range(8):  # a few re-rolls, then give up and spend it
            aid = rng.choice(avail)
            data = None
            if aid == 6:
                targets = click_targets(grid, max_targets=8) or [(rng.randrange(64), rng.randrange(64))]
                r, c = rng.choice(targets)
                data = {"x": int(c), "y": int(r)}
            if guard and memo.is_known_noop(grid, aid, data):
                blocked += 1
                continue
            choice = (aid, data)
            break
        if choice is None:
            choice = (aid, data)
        aid, data = choice
        nxt = env.step(GameAction.from_id(aid), data=data or {})
        spent += 1
        if level < len(actions_per_level):
            actions_per_level[level] += 1
        memo.record(grid, aid, data, nxt.frame[-1], level_changed=nxt.levels_completed > level)
        level = nxt.levels_completed
        if nxt.state.name in ("WIN",):
            break
        if nxt.state.name == "GAME_OVER":
            nxt = env.step(GameAction.RESET)
            spent += 1
            if level < len(actions_per_level):
                actions_per_level[level] += 1
        obs = nxt
    score = game_score(info.baseline_actions or [], actions_per_level, level)
    return {"actions": spent, "levels": level, "score": score, "blocked": blocked, **memo.stats()}


def cmd_walk(args) -> None:
    arc = _arcade(args.env_dir)
    rows = []
    for info in _games(arc, args.games):
        gid = info.game_id.split("-")[0]
        r = _random_walk(arc, info, args.steps, args.seed, guard=False)
        r["game"] = gid
        rows.append(r)
        pct = 100.0 * r["noop"] / max(1, r["observed"])
        rpt = 100.0 * r["repeat_noop"] / max(1, r["observed"])
        print(f"{gid}: steps={r['observed']} noop={r['noop']} ({pct:.0f}%) repeat_noop={r['repeat_noop']} ({rpt:.0f}%) levels={r['levels']}", flush=True)
    obs = sum(r["observed"] for r in rows)
    print(f"\ntotal: {sum(r['noop'] for r in rows)}/{obs} no-op, {sum(r['repeat_noop'] for r in rows)}/{obs} repeat no-op (free to skip)")
    _dump(args.out, {"walk": rows})


def cmd_ab(args) -> None:
    arc = _arcade(args.env_dir)
    rows = []
    for info in _games(arc, args.games):
        gid = info.game_id.split("-")[0]
        a = _random_walk(arc, info, args.steps, args.seed, guard=False)
        b = _random_walk(arc, info, args.steps, args.seed, guard=True)
        rows.append({"game": gid, "plain": a, "guard": b})
        print(f"{gid}: plain levels={a['levels']} score={a['score']:.2f} | guard levels={b['levels']} score={b['score']:.2f} blocked={b['blocked']}", flush=True)
    pa = sum(r["plain"]["score"] for r in rows) / max(1, len(rows))
    pb = sum(r["guard"]["score"] for r in rows) / max(1, len(rows))
    print(f"\nmean score over {len(rows)} games: plain={pa:.2f} guard={pb:.2f}")
    print("Random policy, not Duck. This checks plumbing and measures waste; it does not decide the lever.")
    _dump(args.out, {"ab": rows, "mean_plain": pa, "mean_guard": pb})


def _dump(path: str | None, payload: dict) -> None:
    if not path:
        return
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["census", "walk", "ab"])
    p.add_argument("--env-dir", type=Path, default=Path(os.environ.get("ARC_ENV_DIR", DEFAULT_ENV_DIR)))
    p.add_argument("--games", type=lambda s: s.split(","), default=None)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clicks", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not args.env_dir.exists():
        raise SystemExit(f"env dir not found: {args.env_dir}. Run: kaggle competitions download arc-prize-2026-arc-agi-3")
    np.random.seed(args.seed)
    {"census": cmd_census, "walk": cmd_walk, "ab": cmd_ab}[args.cmd](args)


if __name__ == "__main__":
    main()
