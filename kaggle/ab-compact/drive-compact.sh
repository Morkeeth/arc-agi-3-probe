#!/bin/bash
# launchd com.oscar.arc-compact (plist: compact.plist in this folder, installed to ~/Library/LaunchAgents).
# One-shot driver for runs/RULE-compaction.md. Fires 27 Sep 2026 02:05 Paris (00:05 UTC); no RunAtLoad.
# Runs from the lane-compaction worktree, so the main checkout's HEAD (which the 26 Sep driver's git gates read) never moves.
# caffeinate keeps the Mac awake while the runs go. The driver removes its own plist and job when done. It never submits.
# Stop it: launchctl bootout gui/$(id -u)/com.oscar.arc-compact && rm -f ~/Library/LaunchAgents/com.oscar.arc-compact.plist
export PATH=/Library/Frameworks/Python.framework/Versions/3.12/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export HOME=/Users/morkeeth
MAIN=/Users/morkeeth/CODE/arc-agi-3-probe
WT=/Users/morkeeth/CODE/arc-agi-3-probe-compact
exec /usr/bin/caffeinate -i -s "$MAIN/.venv/bin/python" "$WT/kaggle/ab-compact/drive_compact.py"
