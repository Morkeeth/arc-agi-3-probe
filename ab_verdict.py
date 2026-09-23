"""The verdict for runs/RULE-guard-answer.md, computed from the run files. No number is typed by hand.

    python ab_verdict.py runs/ab-2026-09-23            # the pre-registered A/B
    python ab_verdict.py runs/ab-2026-09-23 --json     # also write <dir>/verdict.json

Layout: <dir>/<arm>-s<k>/benchmark.json and <dir>/<arm>-s<k>/deadprobe_counters.jsonl,
arm in {control, answer}, k in {1, 2, 3}. These are the files `kaggle kernels output` returns.

Rules, copied from RULE-guard-answer.md. Change them there first, with a dated amendment, never after a run.
  valid run   benchmark.json has every expected game, no game_run in state "crashed",
              no solver_note starting "error:", and one counters record per game
  score       per run: the mean of final_score over its games (float, as Kaggle scores a run)
              per arm: the mean of its valid run means
  levels      per arm: the sum of levels_completed over all its valid runs
  RULE        printed only when both arms have N_REQUIRED valid runs.
              WIN  iff  mean(answer) > mean(control)  and  levels(answer) >= levels(control)
              otherwise LOSE. A tie on mean is a LOSE.
  LEDGER      keep  iff  mean(answer) - mean(control) > spread(control)  and levels do not fall  and mechanism ok
              kill  iff  mean(control) - mean(answer) > spread(control)  or  mechanism fails
              pending otherwise.  spread = max - min of the control run means
  mechanism   ok iff answer skipped > 0 (the lever acted) and answer model calls per action
              <= 1.10 x control model calls per action
  time binds  share of game-runs not "won" whose final_wallclock_seconds >= cap - 120 s
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

N_REQUIRED = 3
ARMS = ("control", "answer")
SUBSET5 = ("tu93", "sc25", "cd82", "ls20", "g50t")
PUBLIC25 = (
    "tn36", "lf52", "cn04", "bp35", "wa30", "lp85", "r11l", "tu93", "sp80", "m0r0", "vc33", "ar25", "ka59",
    "sc25", "sk48", "dc22", "cd82", "ft09", "g50t", "ls20", "re86", "s5i5", "sb26", "su15", "tr87",
)
CALLS_TOLERANCE = 1.10
CAP_SLACK_S = 120.0


def r2(x: float) -> str:
    return str(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass
class Run:
    name: str
    games: dict[str, dict]
    counters: dict[str, dict]
    problems: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems

    @property
    def mean(self) -> float:
        return statistics.fmean(g["final_score"] for g in self.games.values())

    @property
    def levels(self) -> int:
        return sum(int(g["levels_completed"]) for g in self.games.values())

    def subset(self, keys) -> tuple[float, int]:
        gs = [g for k, g in self.games.items() if k in keys]
        return statistics.fmean(g["final_score"] for g in gs), sum(int(g["levels_completed"]) for g in gs)


def load_run(d: Path, expected: tuple[str, ...], need_counters: bool = True) -> Run:
    b = json.loads((d / "benchmark.json").read_text())
    games = {gr["game_id"].split("-")[0]: gr for gr in b["game_runs"]}
    counters = {}
    cf = d / "deadprobe_counters.jsonl"
    if cf.exists():
        for line in cf.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                counters[str(rec["game"]).split("-")[0]] = rec
    run = Run(d.name, games, counters)
    missing = sorted(set(expected) - set(games))
    if missing:
        run.problems.append(f"missing games {missing}")
    for k, g in games.items():
        if g.get("state") == "crashed":
            run.problems.append(f"{k} crashed")
        if str(g.get("solver_note") or "").startswith("error:"):
            run.problems.append(f"{k} {g['solver_note'][:80]}")
        if g.get("final_score") is None:
            run.problems.append(f"{k} has no final_score")
    if need_counters:
        nc = sorted(set(games) - set(counters))
        if nc:
            run.problems.append(f"no counters for {nc}")
    return run


def arm_stats(runs: list[Run]) -> dict:
    ok = [r for r in runs if r.valid]
    out = {"n_valid": len(ok), "runs": [r.name for r in ok], "invalid": {r.name: r.problems for r in runs if not r.valid}}
    if not ok:
        return out
    means = [r.mean for r in ok]
    gr = [(k, g, r.counters.get(k, {})) for r in ok for k, g in r.games.items()]
    actions = sum(len(g.get("history") or []) for _, g, _ in gr)
    calls = sum(int(c.get("calls", 0)) for _, _, c in gr)
    turns = sum(int(c.get("turns", 0)) for _, _, c in gr)
    not_won = [(g, c) for _, g, c in gr if g.get("state") != "won"]
    cap = next((c.get("time_cap_s") for _, _, c in gr if c.get("time_cap_s")), 7920.0) or 7920.0
    at_cap = sum(1 for g, _ in not_won if float(g.get("final_wallclock_seconds") or 0) >= float(cap) - CAP_SLACK_S)
    s5 = [r.subset(SUBSET5) for r in ok]
    out.update(
        mean=statistics.fmean(means),
        run_means=means,
        spread=max(means) - min(means),
        sd=statistics.stdev(means) if len(means) > 1 else 0.0,
        levels=sum(r.levels for r in ok),
        run_levels=[r.levels for r in ok],
        game_runs=len(gr),
        actions=actions,
        turns=turns,
        calls=calls,
        skipped=sum(int(c.get("skipped", 0)) for _, _, c in gr),
        calls_per_action=calls / actions if actions else 0.0,
        wall_s_per_action=sum(float(g.get("final_wallclock_seconds") or 0) for _, g, _ in gr) / actions if actions else 0.0,
        model_s=sum(float(c.get("model_s", 0)) for _, _, c in gr),
        not_won=len(not_won),
        not_won_at_cap=at_cap,
        time_cap_s=float(cap),
        subset5_mean=statistics.fmean(m for m, _ in s5),
        subset5_levels=sum(lv for _, lv in s5),
    )
    return out


def verdict(c: dict, a: dict, n_required: int = N_REQUIRED) -> dict:
    v = {"n_required": n_required}
    if c.get("n_valid", 0) < n_required or a.get("n_valid", 0) < n_required:
        v["rule"] = f"PENDING (valid runs: control {c.get('n_valid', 0)}, answer {a.get('n_valid', 0)}, need {n_required} each)"
        v["ledger"] = "pending"
        return v
    win = a["mean"] > c["mean"] and a["levels"] >= c["levels"]
    v["rule"] = "WIN" if win else "LOSE"
    mech_ok = a["skipped"] > 0 and (c["calls_per_action"] == 0 or a["calls_per_action"] <= CALLS_TOLERANCE * c["calls_per_action"])
    v["mechanism_ok"] = mech_ok
    d = a["mean"] - c["mean"]
    if d > c["spread"] and a["levels"] >= c["levels"] and mech_ok:
        v["ledger"] = "keep"
    elif -d > c["spread"] or not mech_ok:
        v["ledger"] = "kill"
    else:
        v["ledger"] = "pending"
    v["diff"] = d
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--games", default="public25", choices=["public25", "subset5"])
    ap.add_argument("--n-required", type=int, default=N_REQUIRED)
    ap.add_argument("--no-counters", action="store_true", help="fixtures from before the counters existed")
    args = ap.parse_args(argv)
    expected = PUBLIC25 if args.games == "public25" else SUBSET5
    stats = {}
    for arm in ARMS:
        runs = [load_run(d, expected, not args.no_counters) for d in sorted(args.dir.glob(f"{arm}-s*")) if (d / "benchmark.json").exists()]
        stats[arm] = arm_stats(runs)
    v = verdict(stats["control"], stats["answer"], args.n_required)
    for arm in ARMS:
        s = stats[arm]
        if not s.get("n_valid"):
            print(f"{arm:8s} valid runs 0  invalid {s['invalid']}")
            continue
        print(
            f"{arm:8s} n={s['n_valid']} mean={r2(s['mean'])} runs=[{', '.join(r2(m) for m in s['run_means'])}] "
            f"spread={r2(s['spread'])} levels={s['levels']} {s['run_levels']} "
            f"calls/action={r2(s['calls_per_action'])} wall_s/action={r2(s['wall_s_per_action'])} skipped={s['skipped']} "
            f"at_cap={s['not_won_at_cap']}/{s['not_won']} subset5 mean={r2(s['subset5_mean'])} levels={s['subset5_levels']}"
        )
        if s["invalid"]:
            print(f"         invalid: {s['invalid']}")
    print(f"RULE: {v['rule']}")
    print(f"LEDGER: {v['ledger']}" + (f" (diff {r2(v['diff'])}, control spread {r2(stats['control']['spread'])}, mechanism_ok {v['mechanism_ok']})" if "diff" in v else ""))
    if args.json:
        (args.dir / "verdict.json").write_text(json.dumps({"stats": stats, "verdict": v}, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
