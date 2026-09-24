"""Verdict for runs/RULE-kvcap.md (Loop A, KV capacity). Reads run files only. No number is typed by hand.

    python kvcap_verdict.py                       # arms under runs/kvcap-2026-09-26/, control = the 3 L1 control runs
    python kvcap_verdict.py --dir <dir> --out <verdict.json>

Per run (a directory pulled with `kaggle kernels output`):
  running      mean of every `Running: N reqs` match in vllm-openai-server.log (the vLLM periodic stats line)
  prompt/req   vllm:request_prompt_tokens_sum / _count in vllm-metrics-final.prom
  kv proof     `reserved X GiB memory for KV Cache` in vllm-openai-server.log
  ctx proof    `CTX16K_PROOF env=16384 tool_agent=16384` in the kernel log, and prompt/req <= 15,360
  score, levels, model calls: ab_verdict.load_run / arm_stats (benchmark.json and deadprobe_counters.jsonl)
Arm numbers: running = mean of the run means; calls per game-run = calls / game-runs; levels per run = levels / valid runs.

Rule (sealed in runs/RULE-kvcap.md before any run):
  MECH keep    iff n_valid >= 2 and running >= 5.0 and calls per game-run >= 68 and levels per run >= control (38.0)
  submit       the MECH-keep arm with the most levels per run (tie: higher running); its s1 kernel.
               No MECH keep -> the control notebook (ab-q38-control-s1 v1) for the board noise band.
  score LEDGER (LANES section 5, gates stacking only): keep iff diff > control spread and levels do not fall;
               kill iff -diff > control spread; pending otherwise.
  K1           both arms valid (n>=2) and both running < 4.0 -> the KV thesis is dead.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import ab_verdict as av

ROOT = Path(__file__).resolve().parent
CONTROL = [ROOT / "runs/ab-2026-09-23" / d for d in ("control-s1", "control-s2r", "control-s3r")]
DEFAULT_DIR = ROOT / "runs/kvcap-2026-09-26"
N_REQUIRED = 2
RUNNING_TARGET = 5.0
CALLS_TARGET = 68.0
K1_LINE = 4.0
CTX_PROMPT_BUDGET = 16384 - 512 - 512  # tool_agent: window - reply reserve (512) - safety margin (512)
# Frozen control baseline (computed 24 Sep from the 3 control dirs). The script recomputes and must agree.
FROZEN = {"running": 3.131, "calls_per_game_run": 56.72, "levels_per_run": 38.0, "prompt_per_req": 20034.6}
FROZEN_TOL = {"running": 0.001, "calls_per_game_run": 0.005, "levels_per_run": 0.0, "prompt_per_req": 0.1}
SUBMIT_CONTROL = {"slug": "oscarmorke/ab-q38-control-s1", "version": 1, "why": "no MECH keep: control again, board noise band"}
ARM_OF = {"kv8": "kv", "kv7": "kv", "ctx16k": "ctx16k"}

RUN_RE = re.compile(r"Running: (\d+) reqs")
KV_RE = re.compile(r"reserved ([0-9.]+) GiB memory for KV Cache")
KVTOK_RE = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")


def _prom(text: str, name: str) -> float | None:
    m = re.search(rf"^{re.escape(name)}\{{[^}}]*\}} (\S+)", text, re.M)
    return float(m.group(1)) if m else None


def vllm_stats(d: Path) -> dict:
    out: dict = {}
    lf = d / "vllm-openai-server.log"
    if lf.exists():
        t = lf.read_text(errors="replace")
        r = [int(x) for x in RUN_RE.findall(t)]
        out["running_n"] = len(r)
        out["running"] = statistics.fmean(r) if r else None
        kv = KV_RE.findall(t)
        out["kv_gib"] = float(kv[0]) if kv else None
        kt = KVTOK_RE.findall(t)
        out["kv_tokens"] = int(kt[0].replace(",", "")) if kt else None
    pf = d / "vllm-metrics-final.prom"
    if pf.exists():
        p = pf.read_text()
        s, c = _prom(p, "vllm:request_prompt_tokens_sum"), _prom(p, "vllm:request_prompt_tokens_count")
        out["prompt_per_req"] = s / c if s and c else None
        out["preemptions"] = _prom(p, "vllm:num_preemptions_total")
    return out


def proof_problems(d: Path, ab_id: str, v: dict) -> list[str]:
    probs = []
    cp = (d / "config_proof.txt").read_text() if (d / "config_proof.txt").exists() else ""
    if f"ab={ab_id} " not in cp:
        probs.append(f"config_proof.txt lacks ab={ab_id}")
    if v.get("running") is None:
        probs.append("no Running lines in vllm-openai-server.log")
    arm = ab_id.rsplit("-", 1)[0]
    want_kv = {"kv8": 8.0, "kv7": 7.0}.get(arm, 5.0)
    if v.get("kv_gib") != want_kv:
        probs.append(f"KV proof: reserved {v.get('kv_gib')} GiB, want {want_kv}")
    if arm == "ctx16k":
        logs = "".join(f.read_text(errors="replace") for f in d.glob("ab-*.log"))
        if "CTX16K_PROOF env=16384 tool_agent=16384" not in logs:
            probs.append("CTX16K_PROOF line missing")
        ppr = v.get("prompt_per_req")
        if ppr is None or ppr > CTX_PROMPT_BUDGET:
            probs.append(f"prompt tokens per request {ppr} above the 16k budget {CTX_PROMPT_BUDGET}")
    return probs


def load(d: Path, ab_id: str | None = None, check_proof: bool = True):
    r = av.load_run(d, av.PUBLIC25)
    v = vllm_stats(d)
    if check_proof:
        r.problems.extend(proof_problems(d, ab_id or d.name, v))
    return r, v


def arm_summary(pairs: list) -> dict:
    s = av.arm_stats([r for r, _ in pairs])
    ok = [(r, v) for r, v in pairs if r.valid]
    if not ok:
        return s
    def m(key):
        xs = [v[key] for _, v in ok if v.get(key) is not None]
        return statistics.fmean(xs) if xs else None
    s.update(
        running=m("running"), run_running=[v.get("running") for _, v in ok],
        prompt_per_req=m("prompt_per_req"), kv_tokens=m("kv_tokens"), preemptions=m("preemptions"),
        kv_gib=sorted({v.get("kv_gib") for _, v in ok}),
        calls_per_game_run=s["calls"] / s["game_runs"], levels_per_run=s["levels"] / s["n_valid"],
    )
    return s


def decide(c: dict, arms: dict) -> dict:
    out = {"arms": {}}
    for name, t in arms.items():
        a = {"n_valid": t.get("n_valid", 0)}
        if a["n_valid"] < N_REQUIRED:
            a.update(mech="PENDING", why=f"valid runs {a['n_valid']}, need {N_REQUIRED}", invalid=t.get("invalid"))
        elif len(t.get("kv_gib", [])) > 1:
            a.update(mech="PENDING", why=f"mixed KV sizes {t['kv_gib']} in one arm")
        else:
            checks = {
                "running_ge_5": t["running"] >= RUNNING_TARGET,
                "calls_per_game_run_ge_68": t["calls_per_game_run"] >= CALLS_TARGET,
                "levels_not_down": t["levels_per_run"] >= c["levels_per_run"],
            }
            d = t["mean"] - c["mean"]
            score = "keep" if d > c["spread"] and checks["levels_not_down"] else "kill" if -d > c["spread"] else "pending"
            a.update(mech="KEEP" if all(checks.values()) else "LOSE", checks=checks, diff=d, score_ledger=score,
                     running=t["running"], calls_per_game_run=t["calls_per_game_run"], levels_per_run=t["levels_per_run"])
        out["arms"][name] = a
    valid = {k: v for k, v in out["arms"].items() if v.get("mech") in ("KEEP", "LOSE")}
    out["k1_fires"] = len(valid) == len(arms) and all(arms[k]["running"] < K1_LINE for k in valid)
    keeps = [k for k, v in out["arms"].items() if v.get("mech") == "KEEP"]
    if keeps:
        best = max(keeps, key=lambda k: (arms[k]["levels_per_run"], arms[k]["running"]))
        s1 = next(n for n in arms[best]["runs"] if n.endswith("-s1")) if any(n.endswith("-s1") for n in arms[best]["runs"]) else arms[best]["runs"][0]
        out["submit"] = {"ab_id": s1, "slug": f"oscarmorke/ab-{s1}", "version": None, "why": f"MECH keep, arm {best}"}
    else:
        out["submit"] = dict(SUBMIT_CONTROL)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--control", type=Path, nargs="*", default=CONTROL)
    ap.add_argument("--no-freeze-check", action="store_true", help="fixtures only")
    args = ap.parse_args(argv)
    c = arm_summary([load(d, check_proof=False) for d in args.control])
    if not args.no_freeze_check:
        got = {k: c[k] for k in FROZEN}
        bad = {k: (got[k], FROZEN[k]) for k in FROZEN if abs(got[k] - FROZEN[k]) > FROZEN_TOL[k]}
        if bad:
            print("STOP: control baseline moved from the frozen values:", bad)
            return 4
    arms: dict[str, list] = {}
    for d in sorted(args.dir.glob("*-s[0-9]")):
        if not (d / "benchmark.json").exists():
            continue
        arm = ARM_OF.get(d.name.rsplit("-", 1)[0])
        if arm:
            arms.setdefault(arm, []).append(load(d, d.name))
    stats = {"control": c, **{k: arm_summary(v) for k, v in arms.items()}}
    for name, s in stats.items():
        if not s.get("n_valid"):
            print(f"{name:7s} valid runs 0 {s.get('invalid')}")
            continue
        print(f"{name:7s} n={s['n_valid']} runs={s['runs']} mean={av.r2(s['mean'])} levels/run={s['levels_per_run']:.1f} "
              f"calls/game-run={s['calls_per_game_run']:.2f} running={s['running']:.3f} prompt/req={s['prompt_per_req']:.0f} "
              f"kv_gib={s['kv_gib']} kv_tokens={s['kv_tokens']} preempt={s['preemptions']}")
    v = decide(c, {k: stats[k] for k in arms}) if arms else {"arms": {}, "k1_fires": False, "submit": dict(SUBMIT_CONTROL)}
    for name, a in v["arms"].items():
        print(f"MECH {name}: {a['mech']} " + json.dumps({k: a[k] for k in a if k not in ('mech', 'n_valid')}, default=str))
    print("K1:", "FIRES (KV thesis dead)" if v["k1_fires"] else "no")
    print("SUBMIT:", json.dumps(v["submit"]))
    out = args.out or args.dir / "verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stats": stats, "verdict": v}, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
