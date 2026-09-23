#!/bin/bash
# Guard: launchctl submit restarts this job on exit. If lane T was already pushed, remove the job and stop.
LOGF=/Users/morkeeth/CODE/arc-agi-3-probe/runs/to-2026-09-23/drive.log
if grep -q "push to1200-s2: Kernel version 1 successfully" "$LOGF" 2>/dev/null; then launchctl remove com.oscar.arc-to-night; exit 0; fi
# Night run for RULE-timeout1200.md: push to1200-s1 and -s2 (2 GPU sessions, the cap), wait,
# pull outputs into runs/to-2026-09-23/, check the proof, compute the verdict. Also logs the board
# score of submission 56490516 when it lands. Never submits anything.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$REPO/runs/to-2026-09-23"; LOG="$OUT/drive.log"; mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
status() { kaggle kernels status "oscarmorke/ab-$1" 2>&1 | sed -E 's/.*KernelWorkerStatus\.([A-Z_]+).*/\1/'; }
push() { local r; r=$(kaggle kernels push -p "$REPO/kaggle/ab-to/$1" 2>&1 | tail -1); say "push $1: $r"; case "$r" in *successfully*) return 0;; *) return 1;; esac; }
board() { kaggle competitions submissions arc-prize-2026-arc-agi-3 -v 2>&1 | grep '^56490516,' | grep -o 'SubmissionStatus\.[A-Z]*,[0-9.]*'; }
pull() {
  local d="$OUT/$1"; mkdir -p "$d"
  kaggle kernels output "oscarmorke/ab-$1" -p "$d" -o >>"$LOG" 2>&1
  local lf; lf=$(ls "$d"/*.log 2>/dev/null | head -1); local proof=""
  [ -n "$lf" ] && proof=$(grep -o 'DEADPROBE_LIVE[^"\\]*' "$lf" | head -1)
  echo "$proof" >"$d/config_proof.txt"; say "$1 pulled; proof: ${proof:-MISSING}"
}
SEEN_BOARD=""
for s in to1200-s1 to1200-s2; do tries=0; until push "$s"; do tries=$((tries+1)); [ $tries -ge 12 ] && { say "STOP: cannot push $s"; exit 2; }; sleep 300; done; done
while :; do
  sleep 600
  b=$(board); if [ -n "$b" ] && [ "$b" != "$SEEN_BOARD" ]; then say "board 56490516: $b"; SEEN_BOARD="$b"; fi
  a=$(status to1200-s1); c=$(status to1200-s2); say "to1200-s1=$a to1200-s2=$c"
  for pair in "to1200-s1:$a" "to1200-s2:$c"; do
    id=${pair%%:*}; st=${pair#*:}
    if [ "$st" = ERROR ]; then
      pull "$id"; mv "$OUT/$id" "$OUT/$id-invalid-$(date -u +%H%M)"
      n=$(ls -d "$OUT/$id-invalid-"* 2>/dev/null | wc -l | tr -d ' ')
      [ "$n" -ge 3 ] && { say "STOP: $id failed $n times"; exit 3; }
      push "$id" || true
    fi
  done
  [ "$a" = COMPLETE ] && [ "$c" = COMPLETE ] && break
done
pull to1200-s1; pull to1200-s2
(cd "$REPO" && .venv/bin/python l4_verdict.py --treat to1200 | tee -a "$LOG")
until [ -n "$(board | grep COMPLETE)" ]; do sleep 900; done; say "board 56490516: $(board)"
say "DONE"
