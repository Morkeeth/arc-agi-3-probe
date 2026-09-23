#!/bin/bash
# Evening run for RULE-lane4-seqs16.md: push seqs16-s1 and -s2 (2 GPU sessions, the cap), wait,
# pull outputs into runs/l4-2026-09-23/, check the proof, compute the verdict. Also logs the board
# score of submission 56490516 when it lands. Never submits anything.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$REPO/runs/l4-2026-09-23"; LOG="$OUT/drive.log"; mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
status() { kaggle kernels status "oscarmorke/ab-l4-$1" 2>&1 | sed -E 's/.*KernelWorkerStatus\.([A-Z_]+).*/\1/'; }
push() { local r; r=$(kaggle kernels push -p "$REPO/kaggle/ab-l4/$1" 2>&1 | tail -1); say "push $1: $r"; case "$r" in *successfully*) return 0;; *) return 1;; esac; }
board() { kaggle competitions submissions arc-prize-2026-arc-agi-3 -v 2>&1 | grep '^56490516,' | grep -o 'SubmissionStatus\.[A-Z]*,[0-9.]*'; }
pull() {
  local d="$OUT/$1"; mkdir -p "$d"
  kaggle kernels output "oscarmorke/ab-l4-$1" -p "$d" -o >>"$LOG" 2>&1
  local lf; lf=$(ls "$d"/*.log 2>/dev/null | head -1); local proof=""
  [ -n "$lf" ] && proof=$(grep -o 'DEADPROBE_LIVE[^"\\]*' "$lf" | head -1)
  echo "$proof" >"$d/config_proof.txt"; say "$1 pulled; proof: ${proof:-MISSING}"
}
SEEN_BOARD=""
for s in seqs16-s1 seqs16-s2; do tries=0; until push "$s"; do tries=$((tries+1)); [ $tries -ge 12 ] && { say "STOP: cannot push $s"; exit 2; }; sleep 300; done; done
while :; do
  sleep 600
  b=$(board); if [ -n "$b" ] && [ "$b" != "$SEEN_BOARD" ]; then say "board 56490516: $b"; SEEN_BOARD="$b"; fi
  a=$(status seqs16-s1); c=$(status seqs16-s2); say "seqs16-s1=$a seqs16-s2=$c"
  for pair in "seqs16-s1:$a" "seqs16-s2:$c"; do
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
pull seqs16-s1; pull seqs16-s2
(cd "$REPO" && .venv/bin/python l4_verdict.py | tee -a "$LOG")
until [ -n "$(board | grep COMPLETE)" ]; do sleep 900; done; say "board 56490516: $(board)"
say "DONE"
