#!/bin/bash
# launchd com.oscar.arc-kvcap (plist: kvcap.plist in this folder, installed to ~/Library/LaunchAgents).
# One-shot night driver for runs/RULE-kvcap.md. caffeinate keeps the Mac awake while it waits and runs.
# The driver removes its own plist and job when done; a relaunch after DONE only removes the job again.
# Stop it: launchctl bootout gui/$(id -u)/com.oscar.arc-kvcap && rm -f ~/Library/LaunchAgents/com.oscar.arc-kvcap.plist
export PATH=/Library/Frameworks/Python.framework/Versions/3.12/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export HOME=/Users/morkeeth
REPO=/Users/morkeeth/CODE/arc-agi-3-probe
exec /usr/bin/caffeinate -i -s "$REPO/.venv/bin/python" "$REPO/kaggle/ab-kvcap/drive_kvcap.py"
