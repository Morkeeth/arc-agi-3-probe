"""Verdict for runs/RULE-compaction.md (compaction on top of the 26 Sep base). Reads run files only.

    python compaction_verdict.py --base kv8 --dir runs/compact-2026-09-27     # arm runs <dir>/kv8-cmp-s1, -s2

Base runs (the comparison, never re-run):
  control  the 3 L1 control runs, runs/ab-2026-09-23/control-s1, -s2r, -s3r (frozen: 38.0 levels per run)
  kv8 kv7 ctx16k   the 26 Sep Loop A runs, runs/kvcap-2026-09-26/<base>-s1, -s2
Both live in the main checkout (ARC_MAIN_REPO, default ~/CODE/arc-agi-3-probe); runs/ is gitignored.

Per arm run, proof (a run failing any of these is invalid, except the digest count, which is the kill line):
  config_proof.txt has `ab=<base>-cmp-s<k> ` and `'compaction.py': '<sha12 of the committed compaction.py>'`
  the kernel log has `COMPACT_PROOF max_chars=<MAX_CHARS>`
  the vLLM log reserves the base's KV size (8.0, 7.0 or 5.0 GiB); ctx16k base: `CTX16K_PROOF env=16384 tool_agent=16384`
  compaction_events.jsonl: digest events in at least 20 of the 25 games (else: proof missing in more than 5 games)

Rule (sealed in runs/RULE-compaction.md before any run):
  KEEP     n_valid >= 2 and levels per run >= base + 2.0 and mean score > base mean
  KILL     levels per run < base, or the digest proof is missing in more than 5 of 25 games in any valid run
  PENDING  otherwise
  C2 (reported, not gating): model calls per game-run within 5% of the base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import ab_verdict as av
import kvcap_verdict as kv

ROOT = Path(__file__).resolve().parent
MAIN = Path(os.environ.get("ARC_MAIN_REPO", Path.home() / "CODE/arc-agi-3-probe"))
KVCAP = Path(os.environ.get("COMPACT_KVCAP_DIR", MAIN / "runs/kvcap-2026-09-26"))
PATCH = ROOT / "kaggle/ab-compact/compaction.py"
MAX_CHARS = 3000
BASES = ("control", "kv8", "kv7", "ctx16k")
KV_GIB = {"kv8": 8.0, "kv7": 7.0}
N_REQUIRED = 2
KEEP_LEVELS = 2.0
PROOF_MIN_GAMES = 20
C2_TOL = 0.05
FROZEN_CONTROL_LEVELS = 38.0
GAME_RE = re.compile(r"[a-z0-9]{4}-[0-9a-f]{8}")


def base_dirs(base: str) -> list[Path]:
    if base == "control":
        return [MAIN / "runs/ab-2026-09-23" / d for d in ("control-s1", "control-s2r", "control-s3r")]
    return [KVCAP / f"{base}-s{k}" for k in (1, 2)]


def patch_sha() -> str:
    return hashlib.sha256(PATCH.read_bytes()).hexdigest()[:12]


def digest_games(d: Path) -> int:
    f = d / "compaction_events.jsonl"
    if not f.exists():
        return 0
    games = set()
    for line in f.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("kind") == "digest":
            m = GAME_RE.search(e.get("runtime_dir", ""))
            games.add(m.group(0) if m else e.get("runtime_dir", ""))
    return len(games)


SHA_RE = re.compile(r"'([\w.]+\.py)': '([0-9a-f]{12})'")


def dataset_shas(d: Path) -> dict[str, str]:
    """The deadprobe package file hashes from the DEADPROBE_LIVE proof line, compaction.py excluded."""
    cp = (d / "config_proof.txt").read_text() if (d / "config_proof.txt").exists() else ""
    return {k: v for k, v in SHA_RE.findall(cp) if k != "compaction.py"}


def arm_proof(d: Path, base: str, ab_id: str, v: dict, base_shas: dict[str, str] | None = None) -> list[str]:
    probs = []
    cp = (d / "config_proof.txt").read_text() if (d / "config_proof.txt").exists() else ""
    if base_shas and dataset_shas(d) != base_shas:
        probs.append(f"dataset files differ from the base runs: {dataset_shas(d)} vs {base_shas}")
    if f"ab={ab_id} " not in cp:
        probs.append(f"config_proof.txt lacks ab={ab_id}")
    if f"'compaction.py': '{patch_sha()}'" not in cp:
        probs.append(f"config_proof.txt lacks compaction.py sha {patch_sha()}")
    logs = "".join(f.read_text(errors="replace") for f in d.glob("ab-*.log"))
    if f"COMPACT_PROOF max_chars={MAX_CHARS}" not in logs:
        probs.append("COMPACT_PROOF line missing")
    if v.get("kv_gib") != KV_GIB.get(base, 5.0):
        probs.append(f"KV proof: reserved {v.get('kv_gib')} GiB, want {KV_GIB.get(base, 5.0)}")
    if base == "ctx16k" and "CTX16K_PROOF env=16384 tool_agent=16384" not in logs:
        probs.append("CTX16K_PROOF line missing")
    return probs


def decide(b: dict, a: dict, proof_games: dict[str, int]) -> dict:
    out: dict = {"n_valid": a.get("n_valid", 0), "digest_games": proof_games}
    if b.get("n_valid", 0) < 2:
        return dict(out, rule="PENDING", why=f"base has {b.get('n_valid', 0)} valid runs")
    low = {k: n for k, n in proof_games.items() if k in a.get("runs", []) and n < PROOF_MIN_GAMES}
    if low:
        return dict(out, rule="KILL", why=f"digest proof in fewer than {PROOF_MIN_GAMES} of 25 games: {low}")
    if out["n_valid"] < N_REQUIRED:
        return dict(out, rule="PENDING", why=f"valid runs {out['n_valid']}, need {N_REQUIRED}", invalid=a.get("invalid"))
    dl = a["levels_per_run"] - b["levels_per_run"]
    dm = a["mean"] - b["mean"]
    c2 = a["calls_per_game_run"] / b["calls_per_game_run"] - 1
    out.update(levels_diff=dl, mean_diff=dm, calls_ratio_minus_1=c2, c2_holds=abs(c2) <= C2_TOL,
               levels_per_run=a["levels_per_run"], base_levels_per_run=b["levels_per_run"],
               mean=a["mean"], base_mean=b["mean"])
    if dl >= KEEP_LEVELS and dm > 0:
        return dict(out, rule="KEEP", why=f"levels per run +{dl:.1f} (>= +{KEEP_LEVELS}) and mean +{dm:.2f}")
    if dl < 0:
        return dict(out, rule="KILL", why=f"levels per run {dl:+.1f} below the base")
    return dict(out, rule="PENDING", why=f"levels per run {dl:+.1f}, mean {dm:+.2f}: inside the band")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=BASES)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--base-dirs", type=Path, nargs="*", help="fixtures only")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    bdirs = args.base_dirs or base_dirs(args.base)
    b = kv.arm_summary([kv.load(d, d.name, check_proof=args.base != "control" and not args.base_dirs) for d in bdirs])
    if args.base == "control" and not args.base_dirs and b.get("levels_per_run") != FROZEN_CONTROL_LEVELS:
        print(f"STOP: control levels per run {b.get('levels_per_run')} != frozen {FROZEN_CONTROL_LEVELS}")
        return 4
    base_shas = dataset_shas(bdirs[0]) if bdirs else {}
    pairs, proof_games = [], {}
    for k in (1, 2):
        ab_id = f"{args.base}-cmp-s{k}"
        d = args.dir / ab_id
        if not (d / "benchmark.json").exists():
            continue
        r, v = kv.load(d, ab_id, check_proof=False)
        r.problems.extend(arm_proof(d, args.base, ab_id, v, base_shas))
        proof_games[r.name] = digest_games(d)
        pairs.append((r, v))
    a = kv.arm_summary(pairs) if pairs else {"n_valid": 0}
    for name, s in (("base", b), ("arm", a)):
        if not s.get("n_valid"):
            print(f"{name:4s} valid runs 0 {s.get('invalid')}")
            continue
        print(f"{name:4s} n={s['n_valid']} runs={s['runs']} mean={av.r2(s['mean'])} levels/run={s['levels_per_run']:.1f} "
              f"calls/game-run={s['calls_per_game_run']:.2f} running={s['running']:.3f} prompt/req={s['prompt_per_req']:.0f}")
    v = decide(b, a, proof_games)
    print(f"RULE {v['rule']}: {v['why']}")
    print("digest games per run:", json.dumps(proof_games))
    out = args.out or args.dir / "verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"base": args.base, "stats": {"base": b, "arm": a}, "verdict": v}, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
