"""Verdict for runs/RULE-lane4-seqs16.md, from the run files. No number is typed by hand.

    python l4_verdict.py            # reads runs/ab-2026-09-23/control-* and runs/l4-2026-09-23/seqs16-s*
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ab_verdict as av

ROOT = Path(__file__).resolve().parent
CONTROL = [ROOT / "runs/ab-2026-09-23" / d for d in ("control-s1", "control-s2r", "control-s3r")]
TREAT_DIR = ROOT / "runs/l4-2026-09-23"
TREAT = "seqs16"
PROOF = "max_num_seqs=16"
N_REQUIRED = 2


def main() -> int:
    global TREAT_DIR, TREAT, PROOF
    if "--treat" in sys.argv and sys.argv[sys.argv.index("--treat") + 1] == "to1200":
        TREAT_DIR, TREAT, PROOF = ROOT / "runs/to-2026-09-23", "to1200", "ab=to1200"
    c = av.arm_stats([av.load_run(d, av.PUBLIC25) for d in CONTROL])
    runs = [av.load_run(d, av.PUBLIC25) for d in sorted(TREAT_DIR.glob(f"{TREAT}-s*")) if (d / "benchmark.json").exists()]
    for r in runs:
        proof = (r_dir := TREAT_DIR / r.name / "config_proof.txt").read_text() if (TREAT_DIR / r.name / "config_proof.txt").exists() else ""
        if PROOF not in proof:
            r.problems.append(f"proof line lacks {PROOF} ({r_dir})")
    t = av.arm_stats(runs)
    out = {"control": c, TREAT: t}
    for name, s in out.items():
        if not s.get("n_valid"):
            print(f"{name:7s} valid runs 0 {s.get('invalid')}")
            continue
        s["levels_per_run"] = s["levels"] / s["n_valid"]
        s["actions_per_game_run"] = s["actions"] / s["game_runs"]
        print(f"{name:7s} n={s['n_valid']} mean={av.r2(s['mean'])} runs=[{', '.join(av.r2(m) for m in s['run_means'])}] "
              f"levels/run={s['levels_per_run']:.1f} actions/game-run={s['actions_per_game_run']:.1f} "
              f"wall_s/action={av.r2(s['wall_s_per_action'])} at_cap={s['not_won_at_cap']}/{s['not_won']}")
    if t.get("n_valid", 0) < N_REQUIRED:
        v = {"rule": f"PENDING (valid {TREAT} runs {t.get('n_valid', 0)}, need {N_REQUIRED})", "ledger": "pending"}
    else:
        win = t["mean"] > c["mean"] and t["levels_per_run"] >= c["levels_per_run"]
        acted = t["actions_per_game_run"] > c["actions_per_game_run"] if TREAT == "seqs16" else True  # RULE-timeout1200: no acted test
        d = t["mean"] - c["mean"]
        if d > c["spread"] and t["levels_per_run"] >= c["levels_per_run"] and acted:
            ledger = "keep"
        elif -d > c["spread"] or not acted:
            ledger = "kill"
        else:
            ledger = "pending"
        v = {"rule": "WIN" if win else "LOSE", "ledger": ledger, "diff": d, "lever_acted": acted,
             "q1_actions_up_20pct": t["actions_per_game_run"] >= 1.2 * c["actions_per_game_run"],
             "q2_wall_per_action_down": t["wall_s_per_action"] < c["wall_s_per_action"],
             "q3_levels_up": t["levels_per_run"] > c["levels_per_run"]}
    print("RULE:", v["rule"])
    print("LEDGER:", v["ledger"], {k: v[k] for k in v if k not in ("rule", "ledger")})
    TREAT_DIR.mkdir(parents=True, exist_ok=True)
    (TREAT_DIR / "verdict.json").write_text(json.dumps({"stats": out, "verdict": v}, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
