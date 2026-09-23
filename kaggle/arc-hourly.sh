#!/bin/bash
# Hourly ARC watch (launchd com.oscar.arc-hourly). Read-only on Kaggle: it never pushes, never submits.
# Writes one status block to ~/.local/state/fleet/ARC-GLANCE.md and appends a line to runs/hourly.log.
export PATH=/Library/Frameworks/Python.framework/Versions/3.12/bin:/usr/local/bin:/usr/bin:/bin
REPO=/Users/morkeeth/CODE/arc-agi-3-probe
G=/Users/morkeeth/.local/state/fleet/ARC-GLANCE.md
now=$(date -u +%FT%H:%MZ)
st() { kaggle kernels status "oscarmorke/$1" 2>&1 | grep -o 'Status\.[A-Z_]*' | sed 's/Status\.//'; }
quota=$(kaggle quota 2>&1 | awk '/^GPU/{print $3" left of "$4}')
subs=$(kaggle competitions submissions arc-prize-2026-arc-agi-3 -v 2>&1 | awk -F, 'NR>1 && NR<5 {print $1" "$(NF-2)" "$(NF-1)}' | sed 's/SubmissionStatus\.//')
k1=$(st ab-l4-seqs16-s1); k2=$(st ab-l4-seqs16-s2)
drv=$(launchctl list | grep -q com.oscar.arc-l4-evening && echo alive || echo stopped)
verdict=$(grep -E '^(RULE|LEDGER):' "$REPO/runs/l4-2026-09-23/drive.log" 2>/dev/null | tail -2 | tr '\n' ' ')
echo "$now quota=$quota seqs16-s1=$k1 seqs16-s2=$k2 driver=$drv ${verdict}" >> "$REPO/runs/hourly.log"
cat > "$G" <<EOF
# ARC GLANCE (auto, hourly, read-only)
as of $now
- GPU quota: $quota (kaggle quota)
- Lane 4 runs: seqs16-s1 $k1, seqs16-s2 $k2. Driver $drv.
- Lane 4 verdict: ${verdict:-pending}
- Latest submissions (ref status public):
$(echo "$subs" | sed 's/^/  - /')
- Rule: runs/RULE-lane4-seqs16.md. Log: fleet-ops/reviews/ARC-DAILY-LOG.md
EOF
