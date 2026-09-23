#!/bin/bash
# Drive the RULE-guard-answer.md A/B: one pair at a time (Kaggle caps batch GPU sessions at 2).
# For each pair: wait for both kernels to finish, pull every output file and the log into
# runs/ab-2026-09-23/<id>/, check the config proof, then push the next pair if quota allows.
# An ERROR run is re-pushed once (the rule's rerun), a second ERROR stops the driver.
#   bash kaggle/ab-q38/drive.sh 1     # start by waiting on pair 1 (already pushed)
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$REPO/runs/ab-2026-09-23"
LOG="$OUT/drive.log"
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
status() { kaggle kernels status "oscarmorke/ab-q38-$1" 2>&1 | sed -E 's/.*status "KernelWorkerStatus\.([A-Z_]+)".*/\1/'; }
quota_left() { python3 -c "
from kaggle.api.kaggle_api_extended import KaggleApi as K
a=K(); a.authenticate(); g=a.quota_view().gpu_quota
print(round((g.total_time_allowed.total_seconds()-g.time_used.total_seconds())/3600,2))"; }

pull() {  # $1 = id
  local d="$OUT/$1"
  mkdir -p "$d"
  kaggle kernels output "oscarmorke/ab-q38-$1" -p "$d" -o >>"$LOG" 2>&1
  local lf; lf=$(ls "$d"/*.log 2>/dev/null | head -1)
  local proof=""
  [ -n "$lf" ] && proof=$(grep -o 'DEADPROBE_LIVE[^"\\]*' "$lf" | head -1)
  say "$1 pulled: $(ls "$d" | wc -l | tr -d ' ') files; proof: ${proof:-MISSING}"
  echo "$proof" >"$d/config_proof.txt"
}

push() {  # $1 = id
  local r; r=$(kaggle kernels push -p "$REPO/kaggle/ab-q38/$1" 2>&1 | tail -1)
  say "push $1: $r"
  case "$r" in *successfully*) return 0;; *) return 1;; esac
}

run_pair() {  # $1 = k, $2 = already pushed (1/0)
  local k=$1 a="control-s$1" b="answer-s$1"
  if [ "$2" = 0 ]; then
    local q; q=$(quota_left); say "quota left ${q} h before pair $k"
    if python3 -c "import sys; sys.exit(0 if float('$q') >= 5.0 else 1)"; then :; else say "STOP: quota below 5 h"; exit 2; fi
    until push "$a"; do sleep 300; done
    until push "$b"; do sleep 300; done
  fi
  while :; do
    sleep 600
    local sa sb; sa=$(status "$a"); sb=$(status "$b")
    say "pair $k: $a=$sa $b=$sb"
    for pair in "$a:$sa" "$b:$sb"; do
      local id=${pair%%:*} st=${pair#*:}
      if [ "$st" = ERROR ] || [ "$st" = CANCEL_ACKNOWLEDGED ]; then
        pull "$id"
        mv "$OUT/$id" "$OUT/$id-invalid-$(date -u +%H%M)"
        if [ -e "$OUT/.retried-$id" ]; then say "STOP: $id failed twice"; exit 3; fi
        touch "$OUT/.retried-$id"
        until push "$id"; do sleep 300; done
      fi
    done
    [ "$(status "$a")" = COMPLETE ] && [ "$(status "$b")" = COMPLETE ] && break
  done
  pull "$a"; pull "$b"
  if ! grep -q "step_env=deadprobe.duck_patch" "$OUT/$b/config_proof.txt"; then say "STOP: answer arm proof missing, lever not live"; exit 4; fi
  if ! grep -q "mode=off" "$OUT/$a/config_proof.txt"; then say "STOP: control proof missing"; exit 4; fi
  (cd "$REPO" && .venv/bin/python ab_verdict.py "$OUT" | tee -a "$LOG")
}

START=${1:-1}
for k in 1 2 3; do
  [ "$k" -lt "$START" ] && continue
  if [ "$k" = "$START" ] && [ "${2:-pushed}" = pushed ]; then run_pair "$k" 1; else run_pair "$k" 0; fi
done
(cd "$REPO" && .venv/bin/python ab_verdict.py "$OUT" --json | tee -a "$LOG")
say "DONE: all pairs"
