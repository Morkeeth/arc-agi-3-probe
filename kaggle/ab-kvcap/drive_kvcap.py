"""One-shot night driver for runs/RULE-kvcap.md (ARC KAGGLE Loop A). launchd label com.oscar.arc-kvcap.

    kaggle/ab-kvcap/drive-kvcap.sh                    # live, started by launchd (RunAtLoad), waits for the times below
    KVCAP_DRY=1 KVCAP_NOW=2026-09-25T23:50:00Z KVCAP_SCENARIO=lose .venv/bin/python kaggle/ab-kvcap/drive_kvcap.py

Timeline, 26 Sep 2026, UTC (Paris is UTC+2):
  00:05  if `kaggle quota` shows >= 9 h GPU left: push kv8-s1 and ctx16k-s1 (the 2-session cap)
  s1 done (about 02:20 to 02:40): push that arm's s2
  kv8 ERROR with an OOM line in its log: the kv arm switches to kv7 (7 GiB) for every seed not yet pushed,
         and the failed seed is pushed as kv7. Any other ERROR: one retry of the same notebook, then give up.
  05:30  run kvcap_verdict.py on what is pulled; again after each later pull
  07:23  submit: the verdict's `submit` (a MECH keep arm's s1, else ab-q38-control-s1 v1). Only if Kaggle shows no
         submission dated 26 Sep UTC, only once (marker written before the command), only before 23:30 UTC.
  quota  every push needs `kaggle quota` >= 2.5 h left (the first s1 push needs 9 h: proof the refresh landed).
         Quota is logged right before the submit and again when the submission turns COMPLETE.
  end    all four runs final, submit handled and the submission final, or 23:50 UTC at the latest: final verdict,
         remove the plist,
         `launchctl bootout` this job. A relaunch after DONE only removes the job again.
Never pushes an id twice (marker written before the push command; a crash mid-push is adopted, not re-pushed),
except the single documented ERROR retry.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "kaggle" / "ab-kvcap"
LABEL = "com.oscar.arc-kvcap"
PLIST = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
COMP = "arc-prize-2026-arc-agi-3"
DRY = os.environ.get("KVCAP_DRY") == "1"
SCEN = os.environ.get("KVCAP_SCENARIO", "lose")
OUT = Path(os.environ.get("KVCAP_OUT", REPO / "runs/kvcap-2026-09-26"))
STATE = OUT / "state"
LOG = OUT / "drive.log"
PY = str(REPO / ".venv/bin/python")


def T(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


T_PUSH = T("2026-09-26T00:05:00Z")
T_NO_NEW_S1 = T("2026-09-26T12:00:00Z")
T_VERDICT = T("2026-09-26T05:30:00Z")
T_SUBMIT = T("2026-09-26T07:23:00Z")
T_SUBMIT_LAST = T("2026-09-26T23:30:00Z")
T_END = T("2026-09-26T23:50:00Z")  # late enough to see the submission turn COMPLETE
SUBMIT_DAY = "2026-09-26"
QUOTA_MIN_H = 9.0  # first s1 push: proves the refresh landed (4 runs need about 8.8 h)
QUOTA_PER_RUN_H = 2.5  # floor before EVERY push: s1, s2, kv7 fallback, retry
POLL_S = 300
RUN_S = 2.2 * 3600  # dry-run only: fake run length
OOM_RE = re.compile(r"out of memory|OutOfMemoryError|No available memory for the cache blocks", re.I)

# ---------------------------------------------------------------- clock and state

_fake = {"now": T(os.environ["KVCAP_NOW"]) if os.environ.get("KVCAP_NOW") else None}


def now() -> float:
    return _fake["now"] if DRY else time.time()


def iso(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(now() if t is None else t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wait(seconds: float) -> None:
    if DRY:
        _fake["now"] += seconds
    else:
        time.sleep(seconds)


def say(msg: str) -> None:
    line = f"{iso()} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def mark(name: str, text: str = "") -> None:
    (STATE / name).write_text(text)


def marked(name: str) -> bool:
    return (STATE / name).exists()


def read(name: str) -> str:
    p = STATE / name
    return p.read_text() if p.exists() else ""


# ---------------------------------------------------------------- kaggle (real or fake)

def sh(cmd: list[str], timeout: int = 600) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # network hiccup: the loop polls again
        return f"EXC {e!r}"


def fake_push_time(ab: str) -> float | None:
    s = read(f"fake-push-{ab}")
    return float(s) if s else None


def k_quota_left() -> float | None:
    if DRY:
        if now() < T("2026-09-26T00:00:00Z"):
            return 1.32
        used = sum(min(now() - float(p.read_text()), RUN_S) for p in STATE.glob("fake-push-*")) / 3600
        left = round(30.0 - used, 2)
        return min(left, 2.0) if SCEN == "lowq" and now() >= T("2026-09-26T02:00:00Z") else left
    out = sh(["kaggle", "quota"])
    m = re.search(r"^GPU\s+\S+\s+([0-9.]+)h", out, re.M)
    return float(m.group(1)) if m else None


def k_push(ab: str) -> str:
    cmd = ["kaggle", "kernels", "push", "-p", str(HERE / ab)]
    if DRY:
        mark(f"fake-push-{ab}", str(now()))
        n = int(read(f"fake-ver-{ab}") or 0) + 1
        mark(f"fake-ver-{ab}", str(n))
        say(f"DRY would run: {' '.join(cmd)}")
        return f"Kernel version {n} successfully pushed.  Please check progress at https://www.kaggle.com/code/oscarmorke/ab-{ab}"
    out = sh(cmd)  # exactly one push command per call
    return out.splitlines()[-1] if out else ""


def k_status(ab: str) -> str:
    if DRY:
        t0 = fake_push_time(ab)
        if t0 is None:
            return "NOTFOUND"
        el = now() - t0
        if SCEN == "oom" and ab == "kv8-s1" and el >= 900:
            return "ERROR"
        return "COMPLETE" if el >= RUN_S else "RUNNING"
    out = sh(["kaggle", "kernels", "status", f"oscarmorke/ab-{ab}"], 120)
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", out)
    return m.group(1) if m else ("NOTFOUND" if "denied" in out or "404" in out else "UNKNOWN")


PULL_TRIES = 5  # a COMPLETE run whose download keeps failing is set aside after this many polls
NET_FAIL_RE = re.compile(r"Max retries exceeded|ConnectTimeoutError|ReadTimeoutError|Connection (?:reset|aborted)", re.I)


VERDICT_FILES = ("benchmark.json", "deadprobe_counters.jsonl", "vllm-openai-server.log", "vllm-metrics-final.prom")
VERDICT_RE = "^(" + "|".join(re.escape(n) for n in VERDICT_FILES) + ")$"
BULK_RE = "^(?!(" + "|".join(re.escape(n) for n in VERDICT_FILES) + ")$).*"


def _kaggle_output(ab: str, dest: Path, pattern: str) -> tuple[bool, str]:
    try:
        r = subprocess.run(["kaggle", "kernels", "output", f"oscarmorke/ab-{ab}", "-p", str(dest), "-o",
                            "--file-pattern", pattern], capture_output=True, text=True, timeout=3600)
        text, rc = (r.stdout or "") + (r.stderr or ""), r.returncode
    except Exception as e:  # a hung download: the next poll tries again
        text, rc = f"EXC {e!r}", -1
    with LOG.open("a") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    if rc != 0:
        return False, f"kaggle exit {rc}"
    if NET_FAIL_RE.search(text):
        return False, "a download timed out"
    return True, "ok"


def k_output(ab: str, dest: Path) -> tuple[bool, str]:
    """Download a kernel's output in two stages. True only when the verdict's files arrived.
    26 Sep: one run writes 300+ files, each its own request; kaggleusercontent.com timed out
    mid-download, and the kernel log (fetched last) and the vLLM log never arrived. Stage 1 pulls
    only the files the verdict reads, plus the kernel log. Stage 2 pulls the rest, best effort,
    and never re-downloads (so never truncates) a stage 1 file."""
    dest.mkdir(parents=True, exist_ok=True)
    if DRY:
        fixture(ab, dest)
        return True, "dry fixture"
    ok, why = _kaggle_output(ab, dest, VERDICT_RE)
    if not ok:
        return False, f"verdict files: {why}"
    bulk, bwhy = _kaggle_output(ab, dest, BULK_RE)
    if not bulk:
        say(f"{ab}: verdict files pulled; the bulk artifacts pull failed ({bwhy}), not needed for the verdict")
    return True, "ok"


def pull_complete(ab: str, dest: Path) -> tuple[bool, str]:
    """The files a verdict needs, all present: the kernel log (proof lines), the vLLM log (KV and Running
    lines) and benchmark.json. A folder without them is a failed download, not an invalid run."""
    need = [f"ab-{ab}.log", *VERDICT_FILES]
    missing = [n for n in need if not (dest / n).is_file() or (dest / n).stat().st_size == 0]
    return (False, "missing " + ", ".join(missing)) if missing else (True, "complete")


def k_submissions() -> str:
    if DRY:
        return read("fake-subs") or ("ref,fileName,date,description,status,publicScore,privateScore\n"
                                     "56515552,submission.parquet,2026-09-24 07:39:57.623000,L2 repeat,SubmissionStatus.COMPLETE,2.2,\n")
    return sh(["kaggle", "competitions", "submissions", COMP, "-v"], 120)


def k_sub_status(ref: str) -> str:
    """Status and public score of one submission, from Kaggle's list (dry: COMPLETE 3 h after submit)."""
    if DRY:
        t0 = float(read("fake-submit-time") or now())
        return "COMPLETE,2.4" if now() - t0 >= 3 * 3600 else "PENDING,"
    for line in k_submissions().splitlines():
        if line.startswith(f"{ref},"):
            m = re.search(r"SubmissionStatus\.([A-Z]+),([0-9.]*)", line)
            return f"{m.group(1)},{m.group(2)}" if m else "UNKNOWN,"
    return "NOTLISTED,"


def k_submit(slug: str, version: int | None, msg: str) -> str:
    cmd = ["kaggle", "competitions", "submit", COMP, "-k", slug] + (["-v", str(version)] if version else []) + [
        "-f", "submission.parquet", "-m", msg]
    if DRY:
        say(f"DRY would run: {' '.join(cmd)}")
        mark("fake-submit-time", str(now()))
        mark("fake-subs", k_submissions() + f"99999999,submission.parquet,{iso().replace('T', ' ')[:19]}.000000,{msg},SubmissionStatus.PENDING,,\n")
        return "Successfully submitted (DRY)"
    return sh(cmd)


# ---------------------------------------------------------------- dry-run fixtures

def fixture(ab: str, dest: Path) -> None:
    """Copy a real control run and edit it into the shape the arm's run would have."""
    src = {"1": "control-s1", "2": "control-s2r"}[ab[-1]]
    if ab.startswith("ctx16k"):
        src = {"1": "control-s3r", "2": "control-s2r"}[ab[-1]]
    shutil.copytree(REPO / "runs/ab-2026-09-23" / src, dest, dirs_exist_ok=True)
    arm = ab.rsplit("-", 1)[0]
    for lf in list(dest.glob("ab-*.log")):  # the kernel log as Kaggle would name it, with this arm's proof line
        txt = re.sub(r"DEADPROBE_LIVE ab=control-s\d", f"DEADPROBE_LIVE ab={ab}", lf.read_text(errors="replace"))
        lf.unlink()
        (dest / f"ab-{ab}.log").write_text(txt)
    (dest / "config_proof.txt").unlink(missing_ok=True)
    vl = dest / "vllm-openai-server.log"
    t = vl.read_text(errors="replace")
    if SCEN == "oom" and ab == "kv8-s1":
        vl.write_text(t[:2000] + "\ntorch.OutOfMemoryError: CUDA out of memory. (fixture)\n")
        return
    gib = {"kv8": "8.0", "kv7": "7.0"}.get(arm, "5.0")
    t = t.replace("reserved 5.0 GiB memory for KV Cache", f"reserved {gib} GiB memory for KV Cache")
    if SCEN == "keep":
        t = re.sub(r"Running: (\d+) reqs", lambda m: f"Running: {int(m.group(1)) * 2} reqs", t)
        cf = dest / "deadprobe_counters.jsonl"
        recs = [json.loads(x) for x in cf.read_text().splitlines() if x.strip()]
        for r in recs:
            r["calls"] = int(r["calls"] * 1.25)
        cf.write_text("".join(json.dumps(r) + "\n" for r in recs))
    vl.write_text(t)
    if arm == "ctx16k":
        with (dest / f"ab-{ab}.log").open("a") as f:
            f.write("CTX16K_PROOF env=16384 tool_agent=16384 (fixture)\n")
        pf = dest / "vllm-metrics-final.prom"
        p = pf.read_text()
        p = re.sub(r"^(vllm:request_prompt_tokens_sum\{[^}]*\}) (\S+)", lambda m: f"{m.group(1)} {float(m.group(2)) / 2}", p, flags=re.M)
        pf.write_text(p)


# ---------------------------------------------------------------- the run

SLOTS = {"kv": ("kv8-s1", "kv8-s2"), "ctx16k": ("ctx16k-s1", "ctx16k-s2")}


def kv_size() -> str:
    return "kv7" if marked("kv-fallback-7") else "kv8"


def slot_id(arm: str, seed: int) -> str:
    return f"{kv_size()}-s{seed}" if arm == "kv" else f"ctx16k-s{seed}"


FLEET = Path.home() / "CODE/fleet-ops"


def sealed() -> tuple[bool, str]:
    """Oscar's rule: the prediction is committed before the run. Checked against git, not a note."""
    checks = [
        (["git", "-C", str(REPO), "ls-files", "--error-unmatch", "runs/RULE-kvcap.md", "kaggle/ab-kvcap/kv8-s1/ab-kv8-s1.ipynb",
          "kaggle/ab-kvcap/ctx16k-s1/ab-ctx16k-s1.ipynb", "kaggle/ab-kvcap/kv7-s1/ab-kv7-s1.ipynb"], "RULE or notebooks not committed"),
        (["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", "runs/RULE-kvcap.md", "kaggle/ab-kvcap", "kvcap_verdict.py"],
         "RULE, notebooks or verdict changed after the commit"),
        (["git", "-C", str(FLEET), "grep", "-q", "id: LA-kvcap", "HEAD", "--", "reviews/ARC-DAILY-LOG.md"],
         "the sealed log entry is not in fleet-ops HEAD"),
    ]
    for cmd, why in checks:
        if DRY:
            continue
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return False, why
    return True, "sealed (dry: not checked)" if DRY else "sealed"


def push_once(ab: str, retry: bool = False) -> None:
    key = f"push-{ab}" + ("-retry" if retry else "")
    if marked(key):
        say(f"SKIP push {ab}: already pushed ({read(key).strip() or 'marker only'})")
        return
    if not retry and not DRY and k_status(ab) not in ("NOTFOUND", "UNKNOWN"):
        mark(key, "adopted: kernel existed before this push")
        say(f"ADOPT {ab}: kernel already exists on Kaggle, not pushing")
        return
    q = k_quota_left()
    if q is None or q < QUOTA_PER_RUN_H:
        if not marked(f"lowquota-{ab}"):
            mark(f"lowquota-{ab}", str(q))
            say(f"NO PUSH {ab}: GPU quota left {q} h < {QUOTA_PER_RUN_H} h per run. Checking again every 5 min")
        return
    say(f"quota before push {ab}: {q} h left")
    mark(key, "pushing")  # before the command: a crash here must not lead to a second push
    r = k_push(ab)
    m = re.search(r"Kernel version (\d+) successfully", r)
    mark(key, f"version {m.group(1)}" if m else f"FAILED {r}")
    if m:
        mark(f"version-{ab}", m.group(1))
    say(f"push {ab}: {r}")
    if not m:
        (STATE / key).unlink()  # the push did not happen, so a later loop may try again


def pulled(ab: str) -> bool:
    return marked(f"final-{ab}")


def handle(ab: str) -> None:
    """Poll one pushed id; pull it when final."""
    if pulled(ab) or not marked(f"push-{ab}"):
        return
    st = k_status(ab)
    if st == "NOTFOUND" and not marked(f"notfound-{ab}"):
        mark(f"notfound-{ab}")
        say(f"{ab}: marker says pushed ({read(f'push-{ab}').strip()}) but Kaggle has no kernel. Not re-pushing; this seed is lost")
    if st == "COMPLETE":
        # Pull into a staging dir. The verdict globs "*-s[0-9]" and must never see a half pull.
        stage = OUT / f"{ab}.pulling"
        ok, why = k_output(ab, stage)
        full, what = pull_complete(ab, stage)
        if ok and full:
            if (OUT / ab).exists():
                shutil.rmtree(OUT / ab)
            stage.rename(OUT / ab)
        else:
            n = int(read(f"pulltries-{ab}") or 0) + 1
            mark(f"pulltries-{ab}", str(n))
            if n < PULL_TRIES:
                say(f"{ab} COMPLETE but pull incomplete ({why}; {what}), try {n}/{PULL_TRIES}: pulling again next poll")
                return
            aside = OUT / f"{ab}-pullincomplete-{iso()[11:16].replace(':', '')}"
            stage.rename(aside)
            mark(f"final-{ab}", "PULL_INCOMPLETE")
            say(f"{ab} COMPLETE but pull incomplete after {n} tries ({why}; {what}): set aside as {aside.name}, "
                "not judged. The run's output is still on Kaggle")
            return
        proof = ""
        for lf in (OUT / ab).glob("ab-*.log"):
            m = re.search(r'DEADPROBE_LIVE[^"\\]*', lf.read_text(errors="replace"))
            proof = proof or (m.group(0) if m else "")
        if proof:
            (OUT / ab / "config_proof.txt").write_text(proof + "\n")
        proof = (OUT / ab / "config_proof.txt").read_text().strip() if (OUT / ab / "config_proof.txt").exists() else ""
        mark(f"final-{ab}", "COMPLETE")
        say(f"{ab} COMPLETE, pulled; proof: {proof or 'MISSING'}")
    elif st in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
        if not read(f"errdir-{ab}"):
            mark(f"errdir-{ab}", f"{ab}-invalid-{iso()[11:16].replace(':', '')}")
        bad = OUT / read(f"errdir-{ab}").strip()
        ok, why = k_output(ab, bad)
        if not ok:
            n = int(read(f"pulltries-{ab}") or 0) + 1
            mark(f"pulltries-{ab}", str(n))
            if n < PULL_TRIES:
                # The kernel log arrives last and holds the OOM line: deciding on a half pull
                # would retry an OOM run instead of falling back to kv7.
                say(f"{ab} {st} but pull failed ({why}), try {n}/{PULL_TRIES}: deciding after the next poll")
                return
            # Deciding retry or kv7 fallback without the kernel log could spend 2 h of quota blind.
            mark(f"final-{ab}", f"{st} PULL_FAILED")
            say(f"{ab} {st}: pull still failing after {n} tries ({why}). No retry and no fallback: "
                "the OOM line cannot be read. Left for Oscar")
            return
        (STATE / f"errdir-{ab}").unlink(missing_ok=True)
        (STATE / f"pulltries-{ab}").unlink(missing_ok=True)
        text = " ".join(f.read_text(errors="replace") for f in bad.glob("*.log"))
        mark(f"final-{ab}", st)
        if ab.startswith("kv8") and OOM_RE.search(text):
            mark("kv-fallback-7", f"OOM in {ab}")
            say(f"{ab} {st} with OOM: kv arm falls back to 7 GiB; pushing {ab.replace('kv8', 'kv7')}")
            push_once(ab.replace("kv8", "kv7"))
        elif not marked(f"push-{ab}-retry"):
            say(f"{ab} {st} (no OOM line): one retry of the same notebook")
            (STATE / f"final-{ab}").unlink()
            push_once(ab, retry=True)
            mark(f"push-{ab}", read(f"push-{ab}-retry"))  # keep polling the same id
        else:
            say(f"{ab} {st} after its retry: given up")


def s2_ready(s1: str) -> tuple[bool, str]:
    """Seed 2 starts only after seed 1 is final COMPLETE with a complete pull.
    26 Sep: kv7-s1 was given up at 05:57 and kv7-s2 was pushed at 06:02, because `pulled` only
    checked that a final marker existed, whatever it said."""
    st = read(f"final-{s1}").strip()
    if not st:
        return False, "s1 not final"
    if st != "COMPLETE":
        return False, f"s1 final {st}"
    return True, "s1 COMPLETE"


def all_final() -> bool:
    ids = []
    for arm in SLOTS:
        for seed in (1, 2):
            ids.append(slot_id(arm, seed))
    return all(pulled(i) or marked(f"gaveup-{i}") for i in ids)


def verdict() -> dict:
    r = subprocess.run([PY, str(REPO / "kvcap_verdict.py"), "--dir", str(OUT)], capture_output=True, text=True, cwd=REPO)
    for line in (r.stdout + r.stderr).strip().splitlines():
        say(f"verdict | {line}")
    vf = OUT / "verdict.json"
    return json.loads(vf.read_text())["verdict"] if r.returncode == 0 and vf.exists() else {}


def submit() -> None:
    if marked("submit"):
        say(f"SKIP submit: marker exists ({read('submit').strip()})")
        return
    ok, why = sealed()
    if not ok:
        mark("submit", f"not sealed: {why}")
        say(f"NO SUBMIT: {why}. Oscar's rule needs the prediction committed before the run")
        return
    if now() > T_SUBMIT_LAST:
        mark("submit", "window closed, not submitted")
        say("NO SUBMIT: past 23:30 UTC. Left for Oscar.")
        return
    subs = k_submissions()
    today = [l for l in subs.splitlines() if f",{SUBMIT_DAY} " in l]
    if today:
        mark("submit", "Kaggle already shows a 26 Sep submission")
        say(f"NO SUBMIT: Kaggle already has a submission dated {SUBMIT_DAY}: {today[0][:80]}")
        return
    v = verdict()
    pick = v.get("submit") or {"slug": "oscarmorke/ab-q38-control-s1", "version": 1, "why": "verdict unreadable: control"}
    ab = pick["slug"].split("/ab-", 1)[1]
    ver = pick.get("version") or (int(read(f"version-{ab}")) if read(f"version-{ab}") else None)
    msg = (f"RULE-kvcap: {ab} ({pick['why']})" if "control" not in ab
           else f"RULE-kvcap fallback: control again, board noise band n=3 ({pick['why']})")
    say(f"quota right before submit: {k_quota_left()} h left (kaggle quota)")
    mark("quota-before-submit", str(k_quota_left()))
    mark("submit", f"attempting {pick['slug']} v{ver}")  # before the command: never a second attempt
    r = k_submit(pick["slug"], ver, msg)
    mark("submit", f"{pick['slug']} v{ver}: {r[-200:]}")
    say(f"SUBMIT {pick['slug']} v{ver}: {r[-300:]}")
    rows = [l for l in k_submissions().splitlines() if f",{SUBMIT_DAY} " in l]
    if rows:
        mark("submit-ref", rows[0].split(",")[0])
        say(f"submission ref {rows[0].split(',')[0]}")


def board_check() -> bool:
    """True when the submission is final. On COMPLETE, log quota again: the diff shows whether the rerun draws quota."""
    ref = read("submit-ref").strip()
    if not ref:
        return True  # nothing submitted, nothing to wait for
    if marked("board-final"):
        return True
    st = k_sub_status(ref)
    if st.split(",")[0] in ("COMPLETE", "ERROR"):
        q = k_quota_left()
        before = read("quota-before-submit").strip()
        mark("board-final", f"{st} quota_after={q}")
        say(f"board {ref}: {st}. quota now {q} h, before submit {before} h (a drop with no push in between = the scoring rerun draws quota)")
        return True
    return False


def remove_job() -> None:
    cmds = [["rm", "-f", str(PLIST)], ["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"]]
    for c in cmds:
        if DRY:
            say(f"DRY would run: {' '.join(c)}")
        else:
            say(f"run: {' '.join(c)}")
            subprocess.run(c)  # bootout ends this process: it is the last call


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    if marked("DONE"):
        say("relaunched after DONE: removing the job, nothing else")
        remove_job()
        return 0
    lock = STATE / "lock"
    if lock.exists() and not DRY:
        pid = int(lock.read_text() or 0)
        try:
            os.kill(pid, 0)
            say(f"another driver is alive (pid {pid}): exit")
            return 0
        except OSError:
            say(f"stale lock from pid {pid}: taking over")
    lock.write_text(str(os.getpid()))
    say(f"driver start (dry={DRY} scenario={SCEN if DRY else '-'}) out={OUT} gpu_quota_left={k_quota_left()} h; {sealed()[1]}")
    last_verdict_pulls = -1
    while True:
        t = now()
        if t >= T_END:
            say("23:50 UTC hard end")
            break
        if t < T_PUSH:
            wait(min(T_PUSH - t, 1800))
            continue
        # s1 pushes (after the quota refresh), then s2 when that arm's s1 is final
        for arm in SLOTS:
            s1, s2 = slot_id(arm, 1), slot_id(arm, 2)
            if not marked(f"push-{s1}") and not pulled(s1):
                if t >= T_NO_NEW_S1:
                    mark(f"gaveup-{s1}", "too late"); mark(f"gaveup-{s2}", "too late")
                    continue
                ok, why = sealed()
                if not ok:
                    say(f"NO PUSH: {why}. Commit per the log entry; the driver checks again in 5 min")
                    continue
                q = k_quota_left()
                if q is None or q < QUOTA_MIN_H:
                    say(f"wait: GPU quota left {q} h < {QUOTA_MIN_H} h (refresh not seen yet)")
                    continue
                push_once(s1)
            elif pulled(s1) and not marked(f"push-{s2}") and not marked(f"gaveup-{s2}"):
                ready, why = s2_ready(s1)
                if ready:
                    push_once(s2)
                else:
                    mark(f"gaveup-{s2}", why)
                    say(f"NO PUSH {s2}: {why}. A second seed only follows a valid first seed")
        for arm in SLOTS:
            for seed in (1, 2):
                handle(slot_id(arm, seed))
        n_pulled = sum(1 for p in STATE.glob("final-*") if p.read_text() == "COMPLETE")
        if t >= T_VERDICT and n_pulled != last_verdict_pulls:
            say(f"verdict run ({n_pulled} runs pulled)")
            verdict()
            last_verdict_pulls = n_pulled
        if t >= T_SUBMIT and not marked("submit"):
            submit()
        if marked("submit") and all_final() and board_check():
            break
        t = now()
        nxt = [e - t for e in (T_VERDICT, T_SUBMIT) if e > t]
        wait(min([POLL_S] + nxt))
    say("final verdict")
    verdict()
    if not marked("submit") and now() >= T_SUBMIT:
        submit()
    mark("DONE", iso())
    say("DONE: removing the launchd job")
    lock.unlink(missing_ok=True)
    remove_job()
    return 0


if __name__ == "__main__":
    sys.exit(main())
