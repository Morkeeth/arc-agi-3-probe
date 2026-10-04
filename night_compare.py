"""Night comparison, 4 Oct 2026: random policy, guard off against guard on.

Preregistration: ~/.local/state/night-run/2026-10-04/receipts/ARC-START.md.
Local only. No model, no network, no Kaggle call. The random policy is not
Duck. A score from this file is a local score and says nothing about the
leaderboard.

The arms call the unchanged harness._random_walk. The only difference is the
guard argument.

Usage:
  python night_compare.py scorer-check
  python night_compare.py arm --arm plain --out runs/night-compare-20261004/plain.json
  python night_compare.py arm --arm guard --out runs/night-compare-20261004/guard.json
  python night_compare.py verdict --dir runs/night-compare-20261004
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAL = HERE / "kaggle" / "ab-causal-pilot" / "seal.json"
RECORDED = HERE / "runs" / "ab.json"
WALK_TIMEOUT_S = 120
ARMS = {"plain": False, "guard": True}


class WalkTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise WalkTimeout()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal_games() -> list[str]:
    return [g.split("-")[0] for g in json.loads(SEAL.read_text())["settings"]["games"]]


def cmd_arm(args) -> None:
    import harness

    guard = ARMS[args.arm]
    arc = harness._arcade(args.env_dir)
    by_prefix = {e.game_id.split("-")[0]: e for e in arc.get_environments()}
    games = seal_games()[: args.max_games] if args.max_games else seal_games()
    missing = [g for g in games if g not in by_prefix]
    if missing:
        raise SystemExit(f"games missing from env dir: {missing}")
    seeds = list(range(args.seed_from, args.seed_to + 1))
    rows = []
    signal.signal(signal.SIGALRM, _alarm)
    t_all = time.perf_counter()
    for seed in seeds:
        t_seed = time.perf_counter()
        for gid in games:
            info = by_prefix[gid]
            t = time.perf_counter()
            timeout = False
            signal.alarm(WALK_TIMEOUT_S)
            try:
                r = harness._random_walk(arc, info, args.steps, seed, guard=guard)
            except WalkTimeout:
                timeout = True
                r = {"actions": None, "levels": 0, "score": 0.0, "blocked": None}
            finally:
                signal.alarm(0)
            wall = time.perf_counter() - t
            won = bool(r["actions"] is not None and r["actions"] < args.steps)
            rows.append(
                {
                    "arm": args.arm,
                    "game": gid,
                    "game_id": info.game_id,
                    "seed": seed,
                    "score": r["score"],
                    "levels": r["levels"],
                    "levels_total": len(info.baseline_actions or []),
                    "actions": r["actions"],
                    "blocked": r["blocked"],
                    "noop": r.get("noop"),
                    "repeat_noop": r.get("repeat_noop"),
                    "observed": r.get("observed"),
                    "wall_s": round(wall, 4),
                    "timeout": timeout,
                    "at_step_cap": (not timeout) and (not won),
                }
            )
        mean = sum(x["score"] for x in rows if x["seed"] == seed) / len(games)
        print(f"{args.arm} seed {seed}: mean score {mean:.4f} wall {time.perf_counter() - t_seed:.1f}s", flush=True)
    wall_all = time.perf_counter() - t_all
    head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    payload = {
        "schema": "arc-night-compare-arm-v1",
        "arm": args.arm,
        "guard": guard,
        "steps": args.steps,
        "seeds": seeds,
        "games": games,
        "walk_timeout_s": WALK_TIMEOUT_S,
        "started_utc": args.started,
        "wall_s_total": round(wall_all, 2),
        "git_head": head,
        "harness_sha256": _sha(HERE / "harness.py"),
        "memo_sha256": _sha(HERE / "deadprobe" / "memo.py"),
        "score_sha256": _sha(HERE / "deadprobe" / "score.py"),
        "python": sys.version.split()[0],
        "machine": platform.platform(),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out}: {len(rows)} walks, wall {wall_all:.1f}s, timeouts {sum(x['timeout'] for x in rows)}")


def sign_line(n: int) -> int | None:
    """Smallest k such that a two-sided sign test with k of n is below 0.05."""
    for k in range(n // 2 + 1, n + 1):
        tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
        if 2 * tail < 0.05:
            return k
    return None


def cmd_verdict(args) -> None:
    d = Path(args.dir)
    plain = json.loads((d / "plain.json").read_text())
    guard = json.loads((d / "guard.json").read_text())
    games, seeds = plain["games"], plain["seeds"]
    same_design = games == guard["games"] and seeds == guard["seeds"] and plain["steps"] == guard["steps"]
    pk = {(r["game"], r["seed"]): r for r in plain["rows"]}
    gk = {(r["game"], r["seed"]): r for r in guard["rows"]}
    pairs = sorted(set(pk) & set(gk))
    expected = len(games) * len(seeds)

    def arm_stats(rows):
        return {
            "walks": len(rows),
            "mean_score": sum(r["score"] for r in rows) / len(rows),
            "total_levels": sum(r["levels"] for r in rows),
            "walks_with_level": sum(1 for r in rows if r["levels"] > 0),
            "actions": sum(r["actions"] or 0 for r in rows),
            "blocked": sum(r["blocked"] or 0 for r in rows),
            "timeouts": sum(1 for r in rows if r["timeout"]),
            "at_step_cap": sum(1 for r in rows if r["at_step_cap"]),
            "wall_s_sum": round(sum(r["wall_s"] for r in rows), 2),
            "wall_s_max": max(r["wall_s"] for r in rows),
        }

    per_seed = []
    for s in seeds:
        a = sum(pk[(g, s)]["score"] for g in games) / len(games)
        b = sum(gk[(g, s)]["score"] for g in games) / len(games)
        per_seed.append({"seed": s, "plain": a, "guard": b, "delta": b - a,
                         "plain_levels": sum(pk[(g, s)]["levels"] for g in games),
                         "guard_levels": sum(gk[(g, s)]["levels"] for g in games)})
    per_game = []
    for g in games:
        pr = [pk[(g, s)] for s in seeds]
        gr = [gk[(g, s)] for s in seeds]
        per_game.append(
            {
                "game": g,
                "plain_mean": sum(r["score"] for r in pr) / len(seeds),
                "guard_mean": sum(r["score"] for r in gr) / len(seeds),
                "plain_levels": sum(r["levels"] for r in pr),
                "guard_levels": sum(r["levels"] for r in gr),
                "plain_walks_with_level": sum(1 for r in pr if r["levels"] > 0),
                "guard_walks_with_level": sum(1 for r in gr if r["levels"] > 0),
                "guard_blocked": sum(r["blocked"] or 0 for r in gr),
                "plain_wall_s": round(sum(r["wall_s"] for r in pr), 2),
                "guard_wall_s": round(sum(r["wall_s"] for r in gr), 2),
                "timeouts": sum(1 for r in pr + gr if r["timeout"]),
            }
        )
    deltas = [x["delta"] for x in per_seed]
    mean_delta = sum(deltas) / len(deltas)
    pos = sum(1 for x in deltas if x > 0)
    neg = sum(1 for x in deltas if x < 0)
    line = sign_line(len(seeds))
    ps, gs = arm_stats(plain["rows"]), arm_stats(guard["rows"])

    # Drift: seed 0 of the plain arm against the recorded runs/ab.json.
    rec = json.loads(RECORDED.read_text())
    drift_rows = []
    if 0 in seeds:
        for r in rec["ab"]:
            now = pk.get((r["game"], 0))
            if now is None:
                drift_rows.append({"game": r["game"], "issue": "not in tonight's task set"})
                continue
            if abs(now["score"] - r["plain"]["score"]) > 1e-9 or now["levels"] != r["plain"]["levels"] or now["actions"] != r["plain"]["actions"]:
                drift_rows.append({"game": r["game"], "recorded": r["plain"], "tonight": {k: now[k] for k in ("score", "levels", "actions")}})
        seed0_mean = per_seed[seeds.index(0)]["plain"]
    else:
        seed0_mean = None
    drift = {
        "recorded_mean_plain": rec["mean_plain"],
        "recorded_source": "runs/ab.json, commit c48569d",
        "tonight_seed0_mean_plain": seed0_mean,
        "mismatched_games": drift_rows,
        "reproduced": seed0_mean is not None and not drift_rows and abs(seed0_mean - rec["mean_plain"]) < 1e-9,
    }

    invalid = []
    if not same_design:
        invalid.append("arms differ in games, seeds or steps")
    if len(pairs) != expected or ps["walks"] != expected or gs["walks"] != expected:
        invalid.append(f"paired walks {len(pairs)}, expected {expected}")
    for name, st in (("plain", ps), ("guard", gs)):
        if st["timeouts"] > 0.05 * st["walks"]:
            invalid.append(f"{name} timeouts {st['timeouts']} above 5%")
    if invalid:
        verdict = "INVALID"
    elif line is None:
        verdict = "INCONCLUSIVE"
    elif mean_delta > 0 and pos >= line and gs["total_levels"] >= ps["total_levels"]:
        verdict = "BETTER"
    elif mean_delta < 0 and neg >= line:
        verdict = "WORSE"
    else:
        verdict = "INCONCLUSIVE"

    out = {
        "schema": "arc-night-compare-verdict-v1",
        "prereg": "~/.local/state/night-run/2026-10-04/receipts/ARC-START.md",
        "what_this_is": "local score, zero-LLM random policy, 25 public games offline. No submission. No leaderboard claim.",
        "denominator": {"games": len(games), "seeds": len(seeds), "paired_walks": len(pairs), "expected": expected},
        "plain": ps,
        "guard": gs,
        "mean_delta_guard_minus_plain": mean_delta,
        "seeds_positive": pos,
        "seeds_negative": neg,
        "seeds_tied": len(deltas) - pos - neg,
        "sign_line": line,
        "verdict": verdict,
        "invalid_reasons": invalid,
        "drift": drift,
        "per_seed": per_seed,
        "per_game": per_game,
    }
    (d / "verdict.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_seed", "per_game")}, indent=1))


def cmd_scorer_check(args) -> None:
    """Show the scorer can go red: one known case, right and wrong inputs."""
    import harness
    from deadprobe.score import game_score

    arc = harness._arcade(args.env_dir)
    info = {e.game_id.split("-")[0]: e for e in arc.get_environments()}["sp80"]
    base = list(info.baseline_actions)
    rec = [r for r in json.loads((HERE / "runs" / "walk.json").read_text())["walk"] if r["game"] == "sp80"][0]
    total_w = sum(range(1, len(base) + 1))
    # Independent arithmetic, written out, for level 1 cleared in exactly the baseline count.
    by_hand_right = min(100.0 * 1 / total_w, 1 / total_w * 100.0)
    right = game_score(base, [base[0]], 1)
    wrong_actions = game_score(base, [base[0] * 2], 1)  # twice the actions: a quarter of the level score
    by_hand_wrong = 25.0 * 1 / total_w
    not_cleared = game_score(base, [base[0]], 0)
    checks = {
        "sp80_baselines": base,
        "recorded_walk_score": rec["score"],
        "level1_at_baseline": right,
        "by_hand_level1_at_baseline": by_hand_right,
        "level1_at_double_actions": wrong_actions,
        "by_hand_double_actions": by_hand_wrong,
        "level_not_cleared": not_cleared,
        "green_right_matches_hand": abs(right - by_hand_right) < 1e-9,
        "green_wrong_matches_hand": abs(wrong_actions - by_hand_wrong) < 1e-9,
        "red_wrong_differs_from_right": abs(wrong_actions - right) > 1e-6,
        "red_not_cleared_is_zero": not_cleared == 0.0,
        "recorded_clear_is_at_cap": abs(rec["score"] - right) < 1e-9,
    }
    print(json.dumps(checks, indent=1))
    ok = all(v for k, v in checks.items() if k.startswith(("green", "red")))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(checks, indent=1) + "\n")
    raise SystemExit(0 if ok else 1)


def main() -> None:
    import harness

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["arm", "verdict", "scorer-check"])
    p.add_argument("--arm", choices=sorted(ARMS))
    p.add_argument("--env-dir", type=Path, default=harness.DEFAULT_ENV_DIR)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed-from", type=int, default=0)
    p.add_argument("--seed-to", type=int, default=19)
    p.add_argument("--max-games", type=int, default=0)
    p.add_argument("--started", default=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    p.add_argument("--dir", default="runs/night-compare-20261004")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    {"arm": cmd_arm, "verdict": cmd_verdict, "scorer-check": cmd_scorer_check}[args.cmd](args)


if __name__ == "__main__":
    main()
