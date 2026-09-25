"""One-shot driver for runs/RULE-compaction.md (ARC KAGGLE, compaction on the 26 Sep base). launchd com.oscar.arc-compact.

    kaggle/ab-compact/drive-compact.sh           # live: launchd fires it 27 Sep 02:05 Paris = 00:05 UTC (no RunAtLoad)
    COMPACT_DRY=1 COMPACT_SCENARIO=kv8 COMPACT_NOW=2026-09-27T00:05:00Z .venv/bin/python kaggle/ab-compact/drive_compact.py

Timeline, 27 Sep 2026, UTC:
  before 00:05  a manual early start only sleeps; nothing is read or pushed
  00:05  base: wait for the 26 Sep driver's DONE marker (runs/kvcap-2026-09-26/state/DONE), at most until 01:00.
         Then read runs/kvcap-2026-09-26/verdict.json once: `submit.ab_id` of a MECH KEEP arm -> that arm
         (kv8, kv7 or ctx16k); anything else (no keep, no file, unreadable) -> control. Written to state/base, final.
  then   push <base>-cmp-s1 and <base>-cmp-s2 together (the 2-session cap), each only if:
           the rule and the notebooks are committed and unchanged (git, this checkout),
           the Kaggle dataset oscarmorke/arc-agi-3-probe is unchanged since the base runs (lastUpdated),
           `kaggle quota` shows >= 5 h GPU left for the first push and >= 2.5 h for any push,
           and it is before 12:00 UTC.
  runs   COMPLETE: pull. ERROR or CANCELLED: pull the log, one retry of the same notebook, then give up.
  end    both runs final (or given up), or 23:50 UTC: compaction_verdict.py, then remove the plist and the job.
NEVER submits to the competition. There is no submit command in this file.
Never pushes an id twice (marker before the push command), except the single documented ERROR retry.
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
HERE = REPO / "kaggle" / "ab-compact"
MAIN = Path(os.environ.get("ARC_MAIN_REPO", Path.home() / "CODE/arc-agi-3-probe"))
LABEL = "com.oscar.arc-compact"
PLIST = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
DATASET = "oscarmorke/arc-agi-3-probe"
DATASET_STAMP = "2026-09-23 07:52:00"  # lastUpdated of the version every base run mounted (kaggle datasets list, 25 Sep)
DRY = os.environ.get("COMPACT_DRY") == "1"
SCEN = os.environ.get("COMPACT_SCENARIO", "control")
KVCAP_DIR = Path(os.environ.get("COMPACT_KVCAP_DIR", MAIN / "runs/kvcap-2026-09-26"))
OUT = Path(os.environ.get("COMPACT_OUT", MAIN / "runs/compact-2026-09-27"))
STATE = OUT / "state"
LOG = OUT / "drive.log"
PY = str(MAIN / ".venv/bin/python")
ARM_OF = {"kv8": "kv", "kv7": "kv", "ctx16k": "ctx16k"}


def T(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


T_START = T("2026-09-27T00:05:00Z")
T_BASE_WAIT = T("2026-09-27T01:00:00Z")
T_NO_PUSH = T("2026-09-27T12:00:00Z")
T_END = T("2026-09-27T23:50:00Z")
QUOTA_FIRST_H = 5.0  # two runs need about 4.4 h
QUOTA_PER_RUN_H = 2.5
POLL_S = 300
RUN_S = 2.2 * 3600  # dry-run only

# ---------------------------------------------------------------- clock and state

_fake = {"now": T(os.environ["COMPACT_NOW"]) if os.environ.get("COMPACT_NOW") else None}


def now() -> float:
    return _fake["now"] if DRY else time.time()


def iso(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(now() if t is None else t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wait(seconds: float) -> None:
    if DRY:
        _fake["now"] += max(seconds, 1)
    else:
        time.sleep(max(seconds, 1))


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


def sh(cmd: list[str], timeout: int = 600) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # network hiccup: the loop polls again
        return f"EXC {e!r}"


# ---------------------------------------------------------------- kaggle (real or fake)

def k_quota_left() -> float | None:
    if DRY:
        if SCEN == "lowq":
            return 3.0
        used = sum(min(now() - float(p.read_text()), RUN_S) for p in STATE.glob("fake-push-*")) / 3600
        return round(21.2 - used, 2)
    m = re.search(r"^GPU\s+\S+\s+([0-9.]+)h", sh(["kaggle", "quota"]), re.M)
    return float(m.group(1)) if m else None


def k_dataset_stamp() -> str | None:
    if DRY:
        return "2026-09-27 00:01:00" if SCEN == "dataset" else DATASET_STAMP
    for line in sh(["kaggle", "datasets", "list", "--mine"], 120).splitlines():
        if line.startswith(DATASET + " "):
            m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)
            return m.group(0) if m else None
    return None


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
        t0 = read(f"fake-push-{ab}")
        if not t0:
            return "NOTFOUND"
        el = now() - float(t0)
        if SCEN == "error" and ab.endswith("-s1") and not marked(f"push-{ab}-retry") and el >= 900:
            return "ERROR"
        return "COMPLETE" if el >= RUN_S else "RUNNING"
    out = sh(["kaggle", "kernels", "status", f"oscarmorke/ab-{ab}"], 120)
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", out)
    return m.group(1) if m else ("NOTFOUND" if "denied" in out or "404" in out else "UNKNOWN")


def k_output(ab: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if DRY:
        fixture(ab, dest)
        return
    with LOG.open("a") as f:
        subprocess.run(["kaggle", "kernels", "output", f"oscarmorke/ab-{ab}", "-p", str(dest), "-o"],
                       stdout=f, stderr=subprocess.STDOUT, timeout=3600)


def fixture(ab: str, dest: Path) -> None:
    """Dry run: a real control run dressed as this arm's run (proof lines added), so the verdict has files to read."""
    import hashlib
    src = MAIN / "runs/ab-2026-09-23" / {"1": "control-s2r", "2": "control-s3r"}[ab[-1]]
    shutil.copytree(src, dest, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("transcripts", "prompts", "solver_analysis", "artifacts"))
    sha = hashlib.sha256((HERE / "compaction.py").read_bytes()).hexdigest()[:12]
    for lf in list(dest.glob("ab-*.log")):
        txt = re.sub(r"DEADPROBE_LIVE ab=control-s\d", f"DEADPROBE_LIVE ab={ab}", lf.read_text(errors="replace"))
        txt = txt.replace("sha={", f"sha={{'compaction.py': '{sha}', ", 1) + "\nCOMPACT_PROOF max_chars=3000 (fixture)\n"
        if ab.startswith("ctx16k"):
            txt += "CTX16K_PROOF env=16384 tool_agent=16384 (fixture)\n"
        lf.unlink()
        (dest / f"ab-{ab}.log").write_text(txt)
    (dest / "config_proof.txt").unlink(missing_ok=True)
    gib = {"kv8": "8.0", "kv7": "7.0"}.get(ab.split("-")[0], "5.0")
    vl = dest / "vllm-openai-server.log"
    vl.write_text(vl.read_text(errors="replace").replace("reserved 5.0 GiB memory for KV Cache", f"reserved {gib} GiB memory for KV Cache"))
    games = [json.loads(x)["game"] for x in (dest / "deadprobe_counters.jsonl").read_text().splitlines() if x.strip()]
    (dest / "compaction_events.jsonl").write_text("".join(
        json.dumps({"runtime_dir": f"/kaggle/working/runs/{g}_p0", "kind": "digest"}) + "\n" for g in games))


# ---------------------------------------------------------------- base, gates

def pick_base() -> tuple[str, str]:
    vf = KVCAP_DIR / "verdict.json"
    try:
        v = json.loads(vf.read_text())["verdict"]
    except Exception as e:
        return "control", f"no readable {vf} ({type(e).__name__}): control"
    ab = str((v.get("submit") or {}).get("ab_id") or "")
    prefix = ab.rsplit("-", 1)[0] if ab else ""
    arm = ARM_OF.get(prefix)
    if not arm:
        return "control", f"verdict submit is {v.get('submit', {}).get('slug')}: no MECH keep, control"
    if (v.get("arms", {}).get(arm) or {}).get("mech") != "KEEP":
        return "control", f"verdict names {ab} but arm {arm} is not MECH KEEP: control"
    if not (HERE / f"{prefix}-cmp-s1").is_dir():
        return "control", f"no notebooks for base {prefix}: control"
    return prefix, f"MECH keep arm {arm}, submit {ab}"


def sealed(base: str) -> tuple[bool, str]:
    """The prediction is committed before the run. Checked against git in this checkout, not a note."""
    nbs = [f"kaggle/ab-compact/{base}-cmp-s{k}/ab-{base}-cmp-s{k}.ipynb" for k in (1, 2)]
    checks = [
        (["git", "-C", str(REPO), "ls-files", "--error-unmatch", "runs/RULE-compaction.md", "compaction_verdict.py", *nbs],
         "RULE, verdict or notebooks not committed"),
        (["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", "runs/RULE-compaction.md", "kaggle/ab-compact",
          "compaction_verdict.py"], "RULE, notebooks or verdict changed after the commit"),
        (["git", "-C", str(REPO), "grep", "-q", "Status: SEALED", "HEAD", "--", "runs/RULE-compaction.md"],
         "RULE-compaction.md in HEAD is not marked SEALED"),
    ]
    for cmd, why in checks:
        if DRY:
            continue
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return False, why
    return True, "sealed (dry: not checked)" if DRY else "sealed"


def arms() -> list[str]:
    b = read("base").split()[0]
    return [f"{b}-cmp-s1", f"{b}-cmp-s2"]


def push_once(ab: str, retry: bool = False) -> None:
    key = f"push-{ab}" + ("-retry" if retry else "")
    if marked(key):
        return
    if not retry and not DRY and k_status(ab) not in ("NOTFOUND", "UNKNOWN"):
        mark(key, "adopted: kernel existed before this push")
        say(f"ADOPT {ab}: kernel already exists on Kaggle, not pushing")
        return
    q = k_quota_left()
    floor = QUOTA_FIRST_H if not any(STATE.glob("push-*")) else QUOTA_PER_RUN_H
    if q is None or q < floor:
        if not marked(f"lowquota-{ab}"):
            mark(f"lowquota-{ab}", str(q))
            say(f"NO PUSH {ab}: GPU quota left {q} h < {floor} h. Checking again every 5 min")
        return
    say(f"quota before push {ab}: {q} h left")
    mark(key, "pushing")  # before the command: a crash here must not lead to a second push
    r = k_push(ab)
    m = re.search(r"Kernel version (\d+) successfully", r)
    say(f"push {ab}: {r}")
    if m:
        mark(key, f"version {m.group(1)}")
    else:
        (STATE / key).unlink()  # the push did not happen (e.g. the 2-session cap), so a later loop may try again


def handle(ab: str) -> None:
    if marked(f"final-{ab}") or not marked(f"push-{ab}"):
        return
    st = k_status(ab)
    if st == "COMPLETE":
        k_output(ab, OUT / ab)
        proof = ""
        for lf in (OUT / ab).glob("ab-*.log"):
            m = re.search(r'DEADPROBE_LIVE[^"\\]*', lf.read_text(errors="replace"))
            proof = proof or (m.group(0) if m else "")
        if proof:
            (OUT / ab / "config_proof.txt").write_text(proof + "\n")
        mark(f"final-{ab}", "COMPLETE")
        say(f"{ab} COMPLETE, pulled; proof: {proof or 'MISSING'}")
    elif st in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
        k_output(ab, OUT / f"{ab}-invalid-{iso()[11:16].replace(':', '')}")
        if not marked(f"push-{ab}-retry"):
            say(f"{ab} {st}: one retry of the same notebook")
            push_once(ab, retry=True)
            if marked(f"push-{ab}-retry"):
                mark(f"push-{ab}", read(f"push-{ab}-retry"))
        else:
            mark(f"final-{ab}", st)
            say(f"{ab} {st} after its retry: given up")


def verdict() -> None:
    b = read("base").split()[0]
    r = subprocess.run([PY, str(REPO / "compaction_verdict.py"), "--base", b, "--dir", str(OUT)],
                       capture_output=True, text=True, cwd=REPO)
    for line in (r.stdout + r.stderr).strip().splitlines():
        say(f"verdict | {line}")


def remove_job() -> None:
    for c in (["rm", "-f", str(PLIST)], ["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"]):
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
    say(f"driver start (dry={DRY} scenario={SCEN if DRY else '-'}) repo={REPO} out={OUT}")
    while True:
        t = now()
        if t >= T_END:
            say("23:50 UTC hard end")
            break
        if t < T_START:
            say(f"before 27 Sep 00:05 UTC: sleeping {int(T_START - t)} s, nothing done")
            wait(min(T_START - t, 1800))
            continue
        if not marked("base"):
            if (KVCAP_DIR / "state/DONE").exists() or t >= T_BASE_WAIT:
                b, why = pick_base()
                mark("base", f"{b} {why}")
                say(f"BASE {b}: {why} (kvcap DONE marker {'present' if (KVCAP_DIR / 'state/DONE').exists() else 'absent at 01:00'})")
            else:
                say("waiting for the 26 Sep driver's DONE marker (until 01:00 UTC)")
                wait(POLL_S)
                continue
        base = read("base").split()[0]
        todo = [ab for ab in arms() if not marked(f"push-{ab}") and not marked(f"final-{ab}") and not marked(f"gaveup-{ab}")]
        if todo:
            if t >= T_NO_PUSH:
                for ab in todo:
                    mark(f"gaveup-{ab}", "too late")
                    say(f"GAVE UP {ab}: not pushed by 12:00 UTC")
            else:
                ok, why = sealed(base)
                stamp = k_dataset_stamp()
                if not ok:
                    say(f"NO PUSH: {why}. Checking again in 5 min")
                elif stamp != DATASET_STAMP:
                    say(f"NO PUSH: dataset {DATASET} lastUpdated {stamp}, base runs used {DATASET_STAMP}. "
                        "One knob would break. Left for Oscar")
                else:
                    for ab in todo:
                        push_once(ab)
        for ab in arms():
            handle(ab)
        if all(marked(f"final-{ab}") or marked(f"gaveup-{ab}") for ab in arms()):
            break
        wait(POLL_S)
    say("final verdict")
    if marked("base"):
        verdict()
    mark("DONE", iso())
    say("DONE: removing the launchd job. No submission was made or scheduled by this driver.")
    lock.unlink(missing_ok=True)
    remove_job()
    return 0


if __name__ == "__main__":
    sys.exit(main())
