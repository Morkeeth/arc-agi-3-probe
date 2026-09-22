"""Recompute every number in the paper draft from the run files. Fail on any mismatch.

    python paper/verify_numbers.py                 # offline checks
    python paper/verify_numbers.py --tests         # also run pytest (needs DUCK_SRC)
    python paper/verify_numbers.py --kaggle        # also read the submission row (read-only CLI)
    python paper/verify_numbers.py --draft X.md    # check another copy of the draft

Sources, all relative to the repo root:
  runs/duck-control/summary.txt, runs/duck-deadprobe/summary.txt   per-game score, levels, actions, tok/s
  runs/duck-*/log.txt   "[finished] ... per-level=a/b,..." lines, "deadprobe: game=... memo={...}" lines,
                        the GPU check line, max_runtime_s_per_game
  runs/census.json, runs/walk.json, runs/ab.json                   the offline census and random A/B
  deadprobe/score.py, deadprobe/duck_patch.py, tests/test_duck_patch.py

How it works. Each claim is a regex over the draft body with capture groups, and a function that
computes the expected text for those groups from the sources. A claim must match at least once and
every match must equal the computed text. Then every number left in the body that no claim covered,
and that no structural rule explains (a section number, a list item number, "Figure 1"), is an error:
a number the script does not know is a number nobody checked.

Rounding. Kaggle prints two decimals. We compare text to text, and where we derive a value ourselves
we round with Decimal ROUND_HALF_UP, never Python's round(), which rounds half to even.

Word count. Body = title line to end of Sources. Drop HTML comments and table separator rows, turn
each "|" into a space, count whitespace-separated tokens holding at least one letter or digit.
Limit 1,450 (Kaggle's is 1,500).
"""

from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import io
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
WORD_LIMIT = 1450
GAMES = ["tu93", "sc25", "cd82", "ls20", "g50t"]
COMP = "arc-prize-2026-arc-agi-3"


# --------------------------------------------------------------------------- sources


@dataclass
class GameRes:
    score: float
    score_text: str
    levels: int
    total_levels: int
    actions: int
    tokens: int


@dataclass
class Finished:
    state: str
    levels: int
    total_levels: int
    score_text: str
    actions: int
    per_level: list[tuple[int, int]]  # (agent actions, human baseline)


@dataclass
class Arm:
    mean: float
    mean_text: str
    median_text: str
    tok_s_text: str
    total_actions: int
    games: dict[str, GameRes] = field(default_factory=dict)
    finished: dict[str, Finished] = field(default_factory=dict)
    gpu: str = ""
    time_cap_s: float = 0.0


@dataclass
class Runs:
    control: Arm
    guard: Arm
    memo: dict[str, dict[str, int]]
    census: list[dict]
    walk: list[dict]
    ab: dict


def _summary(path: Path) -> Arm:
    t = path.read_text()
    g = lambda k: re.search(rf"^{k}:\s+(\S+)", t, re.M).group(1)  # noqa: E731
    arm = Arm(
        mean=float(g("mean score")),
        mean_text=g("mean score"),
        median_text=g("median score"),
        tok_s_text=g("generated tokens/sec"),
        total_actions=int(g("total actions")),
    )
    for m in re.finditer(r"^\s+(\w{4})-\w+: score=([\d.]+), levels=([\d.]+)/(\d+), actions=(\d+), tokens=(\d+)", t, re.M):
        arm.games[m[1]] = GameRes(float(m[2]), m[2], int(float(m[3])), int(m[4]), int(m[5]), int(m[6]))
    return arm


def _log(path: Path, arm: Arm) -> dict[str, dict[str, int]]:
    t = path.read_text()
    for m in re.finditer(
        r"\[finished\] (\w{4})-\w+ state=(\w+) level=(\d+)/(\d+) score=([\d.]+) actions=(\d+) tokens=\d+ per-level=([\d/,]+)", t
    ):
        pl = [tuple(int(x) for x in p.split("/")) for p in m[7].split(",")]
        arm.finished[m[1]] = Finished(m[2], int(m[3]), int(m[4]), m[5], int(m[6]), pl)
    gpu = re.search(r"CUDA GPU check passed for \S+ x1: \['([^']+)'\]", t)
    arm.gpu = gpu[1] if gpu else ""
    cap = re.search(r"max_runtime_s_per_game=([\d.]+)", t)
    arm.time_cap_s = float(cap[1]) if cap else 0.0
    memo = {}
    for m in re.finditer(r"deadprobe: game=(\w{4})-\w+ memo=(\{[^}]*\})", t):
        memo[m[1]] = ast.literal_eval(m[2])
    return memo


def load_runs(root: Path = ROOT) -> Runs:
    import json

    runs = root / "runs"
    control = _summary(runs / "duck-control" / "summary.txt")
    guard = _summary(runs / "duck-deadprobe" / "summary.txt")
    _log(runs / "duck-control" / "log.txt", control)
    memo = _log(runs / "duck-deadprobe" / "log.txt", guard)
    return Runs(
        control,
        guard,
        memo,
        json.loads((runs / "census.json").read_text())["census"],
        json.loads((runs / "walk.json").read_text())["walk"],
        json.loads((runs / "ab.json").read_text()),
    )


# --------------------------------------------------------------------------- helpers


def d2(x: float) -> str:
    return str(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def n(x: int) -> str:
    return f"{x:,}"


WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


def body_of(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    start = text.index("\n# ") + 1 if not text.startswith("# ") else 0
    return text[start:]


def word_count(body: str) -> int:
    lines = [ln for ln in body.splitlines() if not re.fullmatch(r"\s*\|?[\s|:-]*-{3,}[\s|:-]*", ln)]
    toks = " ".join(lines).replace("|", " ").split()
    return sum(1 for t in toks if re.search(r"[A-Za-z0-9]", t))


# --------------------------------------------------------------------------- claims


@dataclass
class Claim:
    name: str
    pattern: str
    expect: Callable[[Runs], tuple[str, ...]]
    source: str


def claims() -> list[Claim]:
    simple = lambda a: not a.startswith("MOUSE")  # noqa: E731

    def census_simple(R):
        dead = sum(1 for g in R.census for a in g["dead_at_start"] if simple(a))
        live = sum(1 for g in R.census for a in g["live_at_start"] if simple(a))
        games = sum(1 for g in R.census if any(simple(a) for a in g["dead_at_start"]))
        return str(dead), str(dead + live), str(games), str(len(R.census))

    def census_click(R):
        dead = sum(1 for g in R.census for a in g["dead_at_start"] if not simple(a))
        live = sum(1 for g in R.census for a in g["live_at_start"] if not simple(a))
        return str(dead), str(dead + live)

    def sc25_all_dead(R):
        g = next(x for x in R.census if x["game"] == "sc25")
        ok = {"UP", "DOWN", "LEFT", "RIGHT"} <= set(g["dead_at_start"])
        return ("UP, DOWN, LEFT and RIGHT were all dead" if ok else "MISMATCH",)

    def walk_steps(R):
        # harness.py walk --steps N: a GAME_OVER on the last step adds one RESET, so N or N+1
        steps = min(w["actions"] for w in R.walk)
        over = [w["game"] for w in R.walk if w["actions"] > steps + 1]
        return (str(steps) if not over else f"MISMATCH {over}",)

    def walk_counts(R):
        return (
            n(sum(w["noop"] for w in R.walk)),
            n(sum(w["observed"] for w in R.walk)),
            n(sum(w["repeat_noop"] for w in R.walk)),
        )

    def zero_noop(R):
        z = sorted(w["game"] for w in R.walk if w["noop"] == 0)
        word = WORDS.get(len(z), str(len(z))).capitalize()
        return word, ", ".join(z[:-1]) + " and " + z[-1]

    def ab_means(R):
        return d2(R.ab["mean_plain"]), d2(R.ab["mean_guard"]), str(len(R.ab["ab"]))

    def ab_only(R):
        won = sorted({r["game"] for r in R.ab["ab"] if r["plain"]["levels"] > 0 or r["guard"]["levels"] > 0})
        both = all(r["plain"]["levels"] > 0 and r["guard"]["levels"] > 0 for r in R.ab["ab"] if r["game"] in won)
        return (won[0] if len(won) == 1 and both else f"MISMATCH {won}",)

    def five_games(R):
        assert set(R.control.games) == set(R.guard.games)
        return str(len(R.control.games)), str(len(R.census))

    def game_split(R):
        # sc25, g50t, cd82, tu93 had dead actions in the random walk; ls20 had none
        walk = {w["game"]: w["noop"] for w in R.walk}
        many = [g for g in ("sc25", "g50t", "cd82", "tu93") if walk[g] > 0 and g in R.control.games]
        none = [g for g in ("ls20",) if walk[g] == 0 and g in R.control.games]
        return (", ".join(many[:-1]) + " and " + many[-1], none[0] if none else "MISMATCH")

    def row(game):
        def f(R):
            c, g = R.control.games[game], R.guard.games[game]
            cl = f"{c.levels} of {c.total_levels}"
            gl = f"{g.levels} of {g.total_levels}"
            if game == "tu93":
                cl += f" ({c.actions} actions)"
                gl += f" ({g.actions} actions)"
            return cl, c.score_text, gl, g.score_text, str(R.memo[game]["skipped"])

        return f

    def mean_row(R):
        cl = sum(g.levels for g in R.control.games.values())
        gl = sum(g.levels for g in R.guard.games.values())
        return str(cl), R.control.mean_text, str(gl), R.guard.mean_text, str(sum(m["skipped"] for m in R.memo.values()))

    def medians(R):
        return R.control.median_text, R.guard.median_text

    def tu93_ctrl(R):
        c = R.control.games["tu93"]
        return str(c.levels), str(c.actions)

    def skips_total(R):
        return (str(sum(m["skipped"] for m in R.memo.values())),)

    def tu93_skips(R):
        return (str(R.memo["tu93"]["skipped"]),)

    def repeats_through(R):
        return (str(sum(m["repeat_noop"] for m in R.memo.values())),)

    def toks(R):
        return R.control.tok_s_text, R.guard.tok_s_text

    def gave_up(R):
        fin = list(R.control.finished.values()) + list(R.guard.finished.values())
        return (str(len(fin)) if all(f.state == "gave_up" for f in fin) else f"MISMATCH {[f.state for f in fin]}",)

    def cap(R):
        assert R.control.time_cap_s == R.guard.time_cap_s
        return (n(int(R.control.time_cap_s)),)

    def gpu(R):
        assert R.control.gpu == R.guard.gpu
        return ("RTX PRO 6000" if "RTX PRO 6000" in R.control.gpu else f"MISMATCH {R.control.gpu}",)

    def limits_games(R):
        return str(len(R.control.games)), str(len(R.census))

    def bound(R):
        sys.path.insert(0, str(ROOT))
        from deadprobe.duck_patch import ProbeConfig

        k = ProbeConfig().max_consecutive_skips
        return (WORDS[k],)

    def bound2(R):
        return bound(R)

    def quarter(R):
        sys.path.insert(0, str(ROOT))
        from deadprobe.score import level_score

        return ("a quarter" if abs(level_score(20, 40) - 25.0) < 1e-9 else "MISMATCH",)

    def pr01_actions(R):
        t = (ROOT / "tests" / "test_duck_patch.py").read_text()
        ctrl = re.search(r"def test_control_arm_spends_every_wasted_action.*?actions_per_level\[0\] == (\d+)", t, re.S)[1]
        grd = re.search(r"def test_guard_skips_repeat_noops_for_free.*?actions_per_level\[0\] == (\d+)", t, re.S)[1]
        return ctrl, grd

    def hud_sightings(R):
        sys.path.insert(0, str(ROOT))
        from deadprobe.memo import NoOpMemo

        return (WORDS[NoOpMemo().promote_after],)

    def next_25(R):
        return (str(len(R.census)),)

    return [
        Claim("gpu", r"on one (RTX PRO 6000) \(the GPU check line", gpu, "runs/duck-*/log.txt GPU check"),
        Claim("quarter", r"Twice the human action count keeps (a quarter) of the level score", quarter, "deadprobe/score.py"),
        Claim("hud_sightings", r"becomes HUD only after (\w+) sightings", hud_sightings, "deadprobe/memo.py promote_after"),
        Claim("bound", r"After (\w+) refusals in a row", bound, "duck_patch.ProbeConfig.max_consecutive_skips"),
        Claim("bound2", r"The (\w+)-refusal bound lets some through", bound2, "duck_patch.ProbeConfig.max_consecutive_skips"),
        Claim("pr01", r"Control spends (\d+) actions on it and guard spends (\d+)\.", pr01_actions, "tests/test_duck_patch.py"),
        Claim("census_n", r"All (\d+) public games, offline", lambda R: (str(len(R.census)),), "runs/census.json"),
        Claim("census_simple", r"(\d+) of (\d+) simple actions changed nothing\. (\d+) of (\d+) games had at least one",
              census_simple, "runs/census.json"),
        Claim("sc25", r"In sc25, (UP, DOWN, LEFT and RIGHT were all dead)", sc25_all_dead, "runs/census.json"),
        Claim("census_click", r"(\d+) of (\d+) click targets", census_click, "runs/census.json"),
        Claim("walk_steps", r"A random policy, (\d+) steps per game", walk_steps, "runs/walk.json"),
        Claim("walk_counts", r"([\d,]+) of ([\d,]+) observed transitions were no-ops\. ([\d,]+) repeated", walk_counts, "runs/walk.json"),
        Claim("zero_noop", r"(\w+) games had zero no-ops: ([\w, ]+ and \w+)\.", zero_noop, "runs/walk.json"),
        Claim("ab_means", r"means of ([\d.]+) \(plain\) and ([\d.]+) \(guard\) over (\d+) games", ab_means, "runs/ab.json"),
        Claim("ab_only", r"Only (\w+) cleared a level, in both arms", ab_only, "runs/ab.json"),
        Claim("rule_games", r"mean score over the (\d+) games", lambda R: (str(len(R.control.games)),), "summary.txt"),
        Claim("game_split", r"The games were ([\w, ]+ and \w+), which had many dead actions, and (\w+), which had none",
              game_split, "summary.txt + runs/walk.json"),
        *[Claim(f"row_{g}", rf"^\| {g} \| ([^|]+?) \| ([\d.]+) \| ([^|]+?) \| ([\d.]+) \| (\d+) \|$", row(g), "summary.txt + memo lines")
          for g in GAMES],
        Claim("row_mean", r"^\| Mean \| (\d+) levels \| ([\d.]+) \| (\d+) levels \| ([\d.]+) \| (\d+) total \|$", mean_row,
              "summary.txt + memo lines"),
        Claim("medians", r"The median points the other way: ([\d.]+) for control, ([\d.]+) for guard", medians, "summary.txt"),
        Claim("tu93_ctrl", r"where it cleared (\d+) levels in (\d+) actions", tu93_ctrl, "summary.txt"),
        Claim("skips_total", r"The guard refused (\d+) repeated no-ops", skips_total, "memo lines"),
        Claim("tu93_skips", r"(\d+) refusals in tu93", tu93_skips, "memo lines"),
        Claim("repeats_through", r"A further (\d+) known repeats ran", repeats_through, "memo lines repeat_noop"),
        Claim("tok_s", r"control ([\d.]+), guard ([\d.]+) generated tokens per second", toks, "summary.txt"),
        Claim("gave_up", r"All (\d+) games ended `gave_up`", gave_up, "log.txt [finished] lines"),
        Claim("cap", r"time cap of ([\d,]+) s", cap, "log.txt max_runtime_s_per_game"),
        Claim("limits", r"One run per arm, on (\d+) of (\d+) public games", limits_games, "summary.txt + census"),
        Claim("next_25", r"on all (\d+) public games before claiming", next_25, "runs/census.json"),
    ]


# External facts: checked only with a flag, or not by this script at all. Listed so the
# number sweep knows about them and the report says out loud what was not recomputed.
EXTERNAL = [
    ("kaggle_ref", r"submission ref (\d+)", "--kaggle"),
    ("kaggle_time", r"on (2026-09-22) at (\d\d:\d\d) UTC", "--kaggle"),
    ("tests", r"(\d+) pass with the Duck source present", "--tests"),
    ("notebook_version", r"`oscarmorke/duck-control`, version (\d+)", "not checked: Kaggle notebook version"),
    ("score_range", r"Game scores run from (0) to (100)%", "not checked: Kaggle ARC-AGI-3 Evaluation page"),
    ("counter_games", r"and (two) games flip a counter cell", "not recomputed: deadprobe/memo.py docstring names vc33, su15"),
]

# Numbers that are structure, not data.
STRUCTURAL = [
    r"^#{1,3} \d+\. ",  # section numbers
    r"^\d+\. ",  # list item numbers
    r"Figure 1\b",
    r"at least (3) seeds per arm",  # a plan, not a result
]

NUM = re.compile(r"(?<![\w.,/-])\d[\d,]*(?:\.\d+)?(?![\w-])")


# --------------------------------------------------------------------------- checks


class Report:
    def __init__(self):
        self.fails = 0
        self.skips = 0
        self.passes = 0

    def ok(self, name, msg=""):
        self.passes += 1
        print(f"PASS  {name:18s} {msg}")

    def fail(self, name, msg):
        self.fails += 1
        print(f"FAIL  {name:18s} {msg}")

    def skip(self, name, msg):
        self.skips += 1
        print(f"SKIP  {name:18s} {msg}")


def check_claims(body: str, R: Runs, rep: Report) -> list[tuple[int, int]]:
    covered: list[tuple[int, int]] = []
    for c in claims():
        ms = list(re.finditer(c.pattern, body, re.M))
        if not ms:
            rep.fail(c.name, f"pattern not found in draft: {c.pattern!r}")
            continue
        try:
            want = tuple(s.strip() for s in c.expect(R))
        except Exception as exc:  # a source that cannot be read is a failure, not a pass
            rep.fail(c.name, f"cannot compute from {c.source}: {type(exc).__name__}: {exc}")
            continue
        bad = [tuple(g.strip() for g in m.groups()) for m in ms if tuple(g.strip() for g in m.groups()) != want]
        if bad:
            rep.fail(c.name, f"draft says {bad[0]}, {c.source} says {want}")
        else:
            rep.ok(c.name, f"{want} ({c.source})")
        covered += [m.span() for m in ms]
    return covered


def check_score_recompute(R: Runs, rep: Report) -> None:
    """Recompute each game score from the per-level action counts with deadprobe/score.py."""
    sys.path.insert(0, str(ROOT))
    from deadprobe.score import game_score

    for label, arm in (("control", R.control), ("guard", R.guard)):
        scores = []
        for g, f in arm.finished.items():
            s = game_score([b for _, b in f.per_level], [a for a, _ in f.per_level], f.levels)
            scores.append(s)
            if d2(s) != f.score_text or f.score_text != arm.games[g].score_text:
                rep.fail(f"score_{label}_{g}", f"recomputed {d2(s)}, log {f.score_text}, summary {arm.games[g].score_text}")
            elif sum(a for a, _ in f.per_level) != arm.games[g].actions:
                rep.fail(f"actions_{label}_{g}", "per-level actions do not sum to the summary's actions")
        if len(scores) != len(arm.games):
            rep.fail(f"scores_{label}", f"{len(scores)} [finished] lines for {len(arm.games)} games")
            continue
        mean, med = d2(statistics.mean(scores)), d2(statistics.median(scores))
        if mean != arm.mean_text or med != arm.median_text:
            rep.fail(f"scores_{label}", f"recomputed mean {mean} median {med}, summary {arm.mean_text} {arm.median_text}")
        else:
            rep.ok(f"scores_{label}", f"5 game scores, mean {mean}, median {med} recomputed from per-level actions")


def check_sweep(body: str, covered: list[tuple[int, int]], rep: Report) -> None:
    spans = list(covered)
    for pat in STRUCTURAL + [p for _, p, _ in EXTERNAL]:
        spans += [m.span() for m in re.finditer(pat, body, re.M)]
    # inline code, URLs and repo/Kaggle slugs are names, not numbers
    spans += [m.span() for m in re.finditer(r"`[^`]*`|\b[\w.-]+\.(?:com|org|py|json|txt|md)\S*|\b\w+/[\w./*-]+", body)]
    loose = []
    for m in NUM.finditer(body):
        if not any(a <= m.start() and m.end() <= b for a, b in spans):
            line = body.count("\n", 0, m.start()) + 1
            loose.append(f"'{m.group()}' (body line {line})")
    if loose:
        rep.fail("number_sweep", "numbers no claim checks: " + ", ".join(loose))
    else:
        rep.ok("number_sweep", "every number in the body is a checked claim, a listed external fact, or structure")


def check_words(body: str, rep: Report) -> None:
    body = body[: body.index("## Sources")] + body[body.index("## Sources"):]  # body runs to the end of Sources
    wc = word_count(body)
    (rep.ok if wc <= WORD_LIMIT else lambda n_, m: rep.fail(n_, m))("words", f"{wc} words (limit {WORD_LIMIT}, method in docstring)")


def check_placeholders(text: str, rep: Report) -> None:
    pats = [r"\bTODO\b", r"\bTBD\b", r"\bXXX\b", r"\bFIXME\b", r"\?\?", r"<[A-Z_ ]{3,}>", r"\[(?:TK|INSERT|PLACEHOLDER|CITE)[^\]]*\]",
            r"\blorem\b", r"\{\{[^}]*\}\}"]
    hits = [m.group() for p in pats for m in re.finditer(p, text, re.I)]
    if hits:
        rep.fail("placeholders", f"found {hits}")
    else:
        rep.ok("placeholders", "none of TODO, TBD, XXX, FIXME, ??, <FILL>, [TK], {{x}}")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def check_citations(body: str, rep: Report) -> None:
    """Every repo path cited in Sources must exist at the commit that line names."""
    src = body[body.index("## Sources"):]
    found_any = False
    for line in src.splitlines():
        m = re.search(r"commit `([0-9a-f]{7,40})`", line)
        if not m:
            continue
        found_any = True
        commit = m[1]
        ls = _git("ls-tree", "-r", "--name-only", commit)
        if ls.returncode != 0:
            rep.fail("cite_commit", f"commit {commit} not in this repo")
            continue
        files = ls.stdout.split()
        for p in re.findall(r"`([^`]+)`", line.split(":", 1)[1] if ":" in line else ""):
            if "/" not in p and not p.endswith((".py", ".json", ".txt", ".md")):
                continue
            pat = p.rstrip("/") + ("/*" if p.endswith("/") else "")
            if not any(fnmatch.fnmatch(f, pat) for f in files):
                rep.fail("cite_path", f"`{p}` is not in commit {commit}")
            else:
                rep.ok("cite_path", f"`{p}` in {commit}")
    if not found_any:
        rep.fail("cite_commit", "Sources names no commit")
    # every repo path in the body must exist at HEAD
    head = set(_git("ls-files").stdout.split())
    for p in sorted(set(re.findall(r"`((?:deadprobe|runs|kaggle|paper|tests)/[^`]+|harness\.py|kaggle_hook\.py)`", body))):
        if not re.search(r"\.\w+$|/$|\*", p):
            continue  # a branch name such as paper/arc-writeup, not a file
        pat = p.rstrip("/") + ("/*" if p.endswith("/") else "")
        if not any(fnmatch.fnmatch(f, pat) for f in head):
            rep.fail("body_path", f"`{p}` is not tracked at HEAD")


def check_figure(R: Runs, rep: Report) -> None:
    png = ROOT / "paper" / "ARC-PAPER-FIG1-AB.png"
    script = (ROOT / "paper" / "ARC-PAPER-FIG1-AB.py").read_text()
    scores = {g.score_text for a in (R.control, R.guard) for g in a.games.values()} | {R.control.mean_text, R.guard.mean_text}
    literals = sorted(s for s in scores - {"0.00"} if re.search(rf"(?<![\d.]){re.escape(s)}(?![\d])", script))
    if not png.exists():
        rep.fail("figure", "paper/ARC-PAPER-FIG1-AB.png missing")
    elif literals:
        rep.fail("figure", f"figure script hardcodes scores {literals}; it must read summary.txt")
    else:
        rep.ok("figure", "script reads the run files; PNG present (regenerate after any run change)")


def check_tests(rep: Report, want: str | None) -> None:
    if not os.environ.get("DUCK_SRC"):
        rep.fail("tests", "--tests needs DUCK_SRC set to the unzipped taaf-kaggle-source-share/src")
        return
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT)
    tail = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-300:]
    m = re.search(r"(\d+) passed", tail)
    got = m[1] if m else "0"
    if out.returncode != 0 or "skipped" in tail or got != want:
        rep.fail("tests", f"draft says {want} pass; pytest says: {tail}")
    else:
        rep.ok("tests", f"pytest: {tail}")


def check_kaggle(body: str, rep: Report) -> None:
    ref = re.search(r"submission ref (\d+)", body)[1]
    day, hhmm = re.search(r"on (2026-\d\d-\d\d) at (\d\d:\d\d) UTC", body).groups()
    out = subprocess.run(["kaggle", "competitions", "submissions", COMP, "-v"], capture_output=True, text=True)
    rows = [r for r in csv.DictReader(io.StringIO(out.stdout)) if r.get("ref") == ref]
    if not rows:
        rep.fail("kaggle", f"ref {ref} not in `kaggle competitions submissions {COMP}`: {out.stdout[:200]}{out.stderr[:200]}")
        return
    r = rows[0]
    if not r["date"].startswith(f"{day} {hhmm}"):
        rep.fail("kaggle", f"draft says {day} {hhmm} UTC, Kaggle says {r['date']}")
    else:
        rep.ok("kaggle", f"ref {ref} date {r['date']} (Kaggle API, UTC) status {r['status']} public={r['publicScore'] or '-'}")
    if "PENDING" not in r["status"]:
        print(f"WARN  kaggle             status is now {r['status']}: update section 1 and Limits")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--draft", default=str(ROOT / "paper" / "ARC-PAPER-WRITEUP-DRAFT.md"))
    ap.add_argument("--tests", action="store_true", help="run pytest and check the pass count (needs DUCK_SRC)")
    ap.add_argument("--kaggle", action="store_true", help="read the submission row with the read-only Kaggle CLI")
    a = ap.parse_args(argv)

    text = Path(a.draft).read_text()
    body = body_of(text)
    R = load_runs(ROOT)
    rep = Report()
    covered = check_claims(body, R, rep)
    check_score_recompute(R, rep)
    check_sweep(body, covered, rep)
    check_words(body, rep)
    check_placeholders(text, rep)
    check_citations(body, rep)
    check_figure(R, rep)
    tests_claim = re.search(r"(\d+) pass with the Duck source present", body)
    if a.tests:
        check_tests(rep, tests_claim[1] if tests_claim else None)
    else:
        rep.skip("tests", f"draft says {tests_claim[1] if tests_claim else '?'} pass; rerun with --tests and DUCK_SRC")
    if a.kaggle:
        check_kaggle(body, rep)
    else:
        rep.skip("kaggle", "submission ref, date and time; rerun with --kaggle (read-only CLI)")
    for name, _, how in EXTERNAL:
        if how.startswith("not"):
            rep.skip(name, how)
    print(f"\n{rep.passes} pass, {rep.fails} fail, {rep.skips} skip")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
