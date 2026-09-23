# RULE: guard+answer against control, 3 repeats each, 25 public games

Written 2026-09-23, before any run in this rule started. Nothing below may change after the first run starts. A change before that goes in a dated amendment at the end. The verdict is computed by `ab_verdict.py` at the commit that adds this file. Nobody types a verdict number by hand.

## What is compared

- Base notebook: public fork `wuliao0/duck-qwen3-8-anim-base`, pulled 2026-09-23, stored as `kaggle/ab-q38/base.ipynb` (sha256 `72d6f35147b5`). Model `keithtyser/qwen3-8-flash-next-nvfp4` (Qwen3.8-Flash-Next NVFP4) on vLLM, RTX Pro 6000, docker image and datasets as in `kaggle/ab-q38/base-kernel-metadata.json`. Its Duck source files `inference/framework/solver.py` and `inference/agent/tool_agent.py` are byte-identical to `jeroencottaar/taaf-kaggle-source-share` (`diff -q`, 2026-09-23). So the patch targets are the same as in the 22 Sep runs.
- Both arms append one block to the hook cell (`kaggle/ab-q38/build.py`). It loads `deadprobe` from private dataset `oscarmorke/arc-agi-3-probe`, calls `duck_patch.install(MODE, ProbeConfig(budget_level1=6, budget_later=3, max_clicks=3))`, then `counters.install()`. The arms differ only in `MODE`.
  - `control`: `MODE = "off"`. Nothing is patched except the read-only counters.
  - `answer`: `MODE = "guard+answer"` (commit `de2bd8f`).
- Code: instrument commit `45a2f8e`. Notebooks sha256 (first 12): control-s1 `9ebb4350729c`, control-s2 `1f0c4f0a68c5`, control-s3 `2c80d0005ad8`, answer-s1 `75b8a576d470`, answer-s2 `803e24171f23`, answer-s3 `55ede6ff07fb`.
- Kaggle kernels, all private: `oscarmorke/ab-q38-control-s1` to `-s3`, `oscarmorke/ab-q38-answer-s1` to `-s3`.

## Games and repeats

- Games: the 25 public games, in the fork's fixed order (`PUBLIC_GAME_IDS`, the fork asserts all 25). A 5-game subset (tu93, sc25, cd82, ls20, g50t, the 22 Sep games) is reported as a secondary number. It does not decide anything.
- Repeats: 3 per arm, named s1, s2, s3. A repeat is an independent run of the same notebook. `LOCAL_ANALYZER_SEED` stays at Duck's default `-1`, the value a submission runs with. A fixed seed would be sent on every request, so a retried request with the same prompt would return the same text. That changes the agent. "Seed" in this file means repeat.
- Pairing: control-sK and answer-sK are pushed within the same few minutes, so both see the same queue and the same machine pool.

## Metrics

Per run, from `benchmark.json` and `deadprobe_counters.jsonl` in the kernel output:
- score: the mean of `final_score` over the 25 games (float).
- levels: the sum of `levels_completed` over the 25 games.
- per game: actions (`len(history)`), turns, model calls, model seconds, wall seconds, skipped no-ops.

Per arm: mean score = the mean of its 3 run scores. Levels = the sum over its 3 runs (75 game-runs). Noise band = the range (max minus min) and the standard deviation of the 3 run scores.

## Valid run

A run is valid if `benchmark.json` holds all 25 games, no game is `crashed`, no `solver_note` starts with `error:`, and every game has a counters record. An invalid run is kept in `runs/` and in the ledger. Its slot is rerun once as a new version of the same kernel. If the rerun is also invalid, that arm has no verdict and nothing is submitted.

## Decision rule

- RULE (Oscar's rule): the answer arm WINS if its mean score is strictly higher than control's AND its levels are at least control's. Anything else, a tie on mean included, is a LOSE. It is read only when both arms have 3 valid runs.
- LEDGER (kill-ledger rule, `ARC-RESEARCH-LANES-2026-09-23.md` section 5, with 3 repeats in place of 2): keep if the answer mean beats control by more than control's range and levels do not fall and the mechanism holds. Kill if it is below control by more than control's range, or the mechanism fails. Pending otherwise.
- Mechanism holds if the answer arm skipped at least 1 no-op and its model calls per action are at most 1.10 times control's.
- Submission: only if RULE says WIN. Then the candidate is the `ab-q38-answer-s1` version that ran. Submitting is Oscar's click (`~/.claude/rules/OUTWARD.md`). A LOSE means no submission of this arm, and the null result is written up.

## Sealed predictions

- P1, time binds: in the control arm, at least 90% of game-runs that do not end `won` stop at the per-game cap (wall seconds of at least 7,800 of 7,920).
- P2, mechanism: answer skipped is above 0 in every answer run, and answer model calls per action are at most 1.10 times control's.
- P3, the paper's claim: answer levels are higher than control levels.
- P4, my guess at the outcome: the difference in mean score is smaller than control's range, so LEDGER reads pending. RULE is a coin flip.

## Budget

Quota before the first push (`kaggle quota`, parsed object, 2026-09-23): 4.87 h used of 30 h, 25.13 h left, refresh 2026-09-26 00:00 UTC. The raw JSON string shows `totalTimeAllowed: 21600s` and a malformed `timeUsed`; the parsed durations are 108,000 s and 17,516 s, so the string serializer is wrong, not the quota. Not checked against the web page. Each run is about 2.2 to 2.5 GPU hours, so 6 runs are about 14 h. Before each push, `kaggle quota` must show at least 2.5 h left per run pushed.

## Amendments

None yet.
