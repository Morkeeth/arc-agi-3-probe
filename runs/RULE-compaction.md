# RULE (DRAFT): compaction lever, 25 public games

ARC KAGGLE. DRAFT written 2026-09-24 about 14:10 UTC. Not sealed. It becomes sealed when it is committed before its first push, with the base named below. Nothing is pushed for it now.

## Why this lever

OpenAI's post on its ARC-AGI-3 settings names two: retained reasoning and compaction. Our base has retained reasoning (`kaggle.py:331`, `tool_agent.py:1892`). It does not have compaction. When a prompt passes the budget, `_trim_messages_for_context` (`tool_agent.py:1672-1689`) drops the oldest messages, and at most 30 assistant turns are kept (line 151). That is rolling truncation, the setting OpenAI replaced. The model's own world-model note is carried every turn (lines 1105-1111, 1236). That note is written by the model; nothing digests what was dropped.

## The knob

One knob: `ARC_COMPACT_MAX_CHARS`, 0 (off, the base) against 6000 (about 1,500 tokens). Patch: `kaggle/ab-compact/compaction.py`. When `_persistent_history_messages` drops messages for good, a digest of them goes into the carried notes as a "Compacted earlier turns" line: the visible reply of each dropped assistant turn (first 400 characters, not its reasoning) and the start of each tool result (200 characters). Frames are skipped. The oldest digest text falls off first. It is cleared on a level change, like the other notes. No extra model call.

CPU check: `tests/test_compaction_draft.py` runs the patch on the real fork `ToolAgent` history code. Knob off: nothing is patched. Knob on: the kept history is the same, and the dropped turns appear in the notes.

Not done yet, needed before a run: copy `compaction.py` into `deadprobe/` and `kaggle/dataset/` and push a dataset version; a builder (`kaggle/ab-compact/build.py`) that adds `install(6000)` to the hook cell of the base notebook, and a diff test; a verdict script (reuse `kvcap_verdict.py` stats).

## Base and arms

- Base: the Loop A winner if it has a MECH keep, else control-s1. The base's own n=2 or n=3 runs are the comparison.
- Arm: the base plus `install(6000)`. n=2. 2 runs, about 4.4 GPU hours.
- Config proof: `COMPACT_PROOF max_chars=6000` in the kernel log, and the "Compacted earlier turns" line in the prompts of at least 20 of 25 games (`prompts/`).

## Sealed-to-be predictions

- C1: levels per run up. Point +1.5 over the base, band -2 to +5.
- C2: model calls per game-run within 5% of the base. The digest adds up to about 1,500 prompt tokens, which cuts raw history by the same amount, so KV use and Running barely move.
- C3: if the base is ctx16k, the effect is larger. Point +2.5 levels per run: more is dropped, so more is saved.
- Chance of a keep: about 30%.

## Decision rule

- Keep: levels per run at least the base + 2.0, and mean score above the base, with 2 valid runs.
- Kill: levels per run below the base, or the proof is missing in more than 5 of 25 games.
- Pending otherwise.
- Submission: a keep makes the arm's s1 the candidate for the next day's submit, under Oscar's 24 Sep ruling.

## Amendments

None yet.
