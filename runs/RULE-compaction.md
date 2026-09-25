# RULE: compaction lever, 25 public games

ARC KAGGLE. Status: SEALED. Drafted 2026-09-24 about 14:10 UTC. Sealed by the commit that carries this line, on branch `lane-compaction` of `arc-agi-3-probe`, on 2026-09-25 (Oscar's go, 25 Sep about 20:30 Paris: "yes build it lets go"). The commit time is the seal time. No compaction run exists before it. Nothing in this file may change after the commit: the 27 Sep driver refuses to push if it does.

## Why this lever

OpenAI's post on its ARC-AGI-3 settings names two: retained reasoning and compaction. Our base has retained reasoning (`kaggle.py:331`, `tool_agent.py:1892`). It does not have compaction. When a prompt passes the budget, the base drops the oldest messages (`_trim_messages_for_context` -> `_drop_oldest_history_block`), and it keeps at most 30 assistant turns (line 151). What it drops is gone. The model's own world-model note is carried every turn; nothing digests what was dropped.

## The knob

One knob: the compaction digest, `COMPACT_MAX_CHARS` = 3000 (off in the base). Patch: `kaggle/ab-compact/compaction.py`, embedded verbatim in the hook cell of each arm notebook (sha256 prefix printed in `DEADPROBE_LIVE`).

When messages leave the persistent history for good, a digest of them goes into the carried world-model notes as one "Compacted earlier turns" line: the visible reply of each dropped assistant turn (first 400 characters, not its reasoning), up to 3 `action(...)` calls from its python code (80 characters each), and the start of each tool result (200 characters). Frames are skipped. Only turns from the current level are kept, and the digest is cleared on a level change, run complete or game over, like the base's own notes. The oldest text falls off first at the cap. No extra model call.

## Amendments to the 24 Sep draft, made before sealing, from CPU evidence

1. **Drop path.** The draft wrapped only `_persistent_history_messages`. `analyze` also trims at the turn start (tool_agent.py:1756), on every request (1790) and on a context-overflow retry (1839), on a local list the draft never saw. Replayed on the 75 recorded control games (3 runs of 23 Sep, 4,081 turns), the draft saw 796 of 10,450 dropped messages (7.6%). The sealed patch compares the history a turn starts with, plus its final messages, against the history it keeps, so all four paths count: it saw 10,920 of 10,920 (`tests/test_compaction_replay.py`, which goes red if the patch looks only at the turn end).
2. **Level filter.** The base clears its notes mid-turn on a level change; old-level history is dropped just after. Without a filter the draft digest refilled with the old level at once. The sealed patch keeps only current-level turns.
3. **Actions.** About 40% of assistant turns have no visible reply, only tool code (transcripts: 1,404 tool calls, 820 replies in control-s1). The digest now carries their `action(...)` calls.
4. **Cap 3000, not 6000.** The base bakes its world-model block into every user prompt, and history keeps those prompts, so the digest is repeated in every kept frame. Replay: at 6000 the repeated digest costs 3,337 tokens per request (10.5% of the 31,744-token budget) and cuts raw history per request by 19%; at 3000 it costs 2,502 tokens (7.9%) and cuts raw history by 14%, holding about 6 dropped turns (505 characters added per update, median 470). 3000 is the chosen trade. The arm is therefore "digest in place of about 14% of raw history", and the verdict reads it as that.
5. **No dataset push.** The patch is embedded in the notebook, so `oscarmorke/arc-agi-3-probe` stays at the version every base run mounted. The driver refuses to push if its lastUpdated moved from 2026-09-23 07:52:00, and the verdict checks the package hashes against the base runs.
6. **Proof from a counter.** `prompts/*.log` holds only each game's last request (6 of 25 control games show no notes block there), so the proof is `compaction_events.jsonl`, one line per digest update, instead.

## Base and arms

- Base, decided once by the 27 Sep driver (`kaggle/ab-compact/drive_compact.py`) at 00:05 UTC, after the 26 Sep driver's DONE marker (or at 01:00 UTC without it): if `runs/kvcap-2026-09-26/verdict.json` names a MECH KEEP arm in `submit.ab_id`, that arm (kv8, kv7 or ctx16k). In every other case (no keep, no file, unreadable), control.
- Comparison: the base's own runs, never re-run. Control: the 3 L1 control runs of 23 Sep (38.0 levels per run, mean 8.59). A Loop A arm: its two 26 Sep runs.
- Arm: the base notebook plus the digest, n=2: `ab-<base>-cmp-s1` and `-s2`, all 8 built and committed now (`kaggle/ab-compact/build.py`, diff test `tests/test_compaction_notebooks.py`). About 4.4 GPU hours, pushed together at 27 Sep 00:05 UTC (quota at least 5 h).
- Config proof per run: `COMPACT_PROOF max_chars=3000` in the kernel log; `'compaction.py': '<sha>'` and `ab=<base>-cmp-s<k>` in `DEADPROBE_LIVE`; the base's KV size (and `CTX16K_PROOF` for a ctx16k base); digest events in at least 20 of 25 games. CPU replay gives 25 of 25 in every recorded run.
- Caveat written before the run: base runs and arm runs are on different days. The board noise band is about 1 point and the local band (3 identical control runs) is 6.43 to 11.40.

## Sealed predictions

- C1: levels per run up. Point +1.5 over the base, band -2 to +5.
- C2: model calls per game-run within 5% of the base. The prompt stays capped by the same budget; the digest displaces raw history, it does not add prompt length at the cap.
- C3: if the base is ctx16k, the effect is larger. Point +2.5 levels per run: more is dropped, so more is saved.
- C4 (mechanism): digest events in at least 20 of 25 games in each run.
- Chance of a keep: about 30%.

## Decision rule (`compaction_verdict.py`, no number typed by hand)

- KEEP: n=2 valid runs, levels per run at least the base + 2.0, and mean score above the base.
- KILL: levels per run below the base, or digest proof missing in more than 5 of 25 games in a valid run.
- PENDING otherwise. C2 and C3 are reported, not gating.
- Submission: none by this lane on 27 Sep. The driver has no submit command. A KEEP makes the arm's s1 a candidate for a later day's one submission, under Oscar's 24 Sep ruling, sealed separately before that submit.
