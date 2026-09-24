# RULE: Loop A, KV capacity (kv8 and ctx16k), 25 public games

ARC KAGGLE. Written 2026-09-24 about 13:45 UTC, CPU only, before any Loop A run. Nothing below changes after the first push. The verdict comes from `kvcap_verdict.py`. Nobody types a verdict number by hand. Plan: `~/.local/state/day-run/2026-09-24/plans/C-arc-kaggle.md`, Loop A.

## Why this lever

The control-s1 vLLM log shows the server is bound by KV cache: `GPU KV cache size: 105,202 tokens, Maximum concurrency for 32,768 tokens per request: 3.21x`. The mean `Running` is 3.13 against a sequence cap of 8, with 21.55 requests waiting. Every game hits the 7,920 s cap, so a game gets more model calls only if more requests run at the same time. Two ways to fit more requests into the cache: a bigger cache (kv8), or shorter prompts (ctx16k).

## What is compared

Both arms are built by `kaggle/ab-kvcap/build.py` from `kaggle/ab-q38/control-s1/ab-q38-control-s1.ipynb`. Each changes one knob and the run label `AB_ID`. `tests/test_kvcap_notebooks.py` checks this.

| Arm | Knob | Where | Notebooks, sha256 first 12 |
|---|---|---|---|
| kv8 | `TAAF_VLLM_KV_CACHE_MEMORY_BYTES` 5368709120 to 8589934592 | cell 3, read by `serving_setup.py` | s1 `dbfb8311d1f8`, s2 `53641f852a3f` |
| kv7 (OOM fallback only) | same knob, 7516192768 | cell 3 | s1 `cfb183176257`, s2 `55a68c37158f` |
| ctx16k | `LOCAL_ANALYZER_CONTEXT_WINDOW` 32768 to 16384 | first statement of cell 11 | s1 `79c6f326cb29`, s2 `c04efa828960` |

- kv8 is byte-identical to `kaggle/ab-kv8` on branch `lane-mtp4` (`f37a155`).
- ctx16k placement. Cell 3 does not work. `serving_setup.py` (`persist_analyzer_environment`, line 2935) writes `"32768"` into `taaf_setup_env.json`, and cell 9 copies that file back into `os.environ` after each setup command. `tool_agent.py:138` reads the value once, at import. The first harness import is the `pickle.load` of `benchmark_initial.pkl` in cell 11: the pickle names `inference.framework.solver`, which imports `ToolAgent` at `solver.py:37`. Checked on CPU against the local fork bundle: after that `pickle.load`, `inference.agent.tool_agent` is in `sys.modules`. So the override is the first statement of cell 11. The cell also asserts that `tool_agent` is not yet imported, writes 16384 into the setup env file, and prints `CTX16K_PROOF`.
- The ctx16k prompt budget is 16,384 minus 512 reply reserve minus 512 safety margin, so 15,360 tokens. The control budget is 31,744 (`tool_agent.py` lines 937 and 945 to 948). The vLLM server keeps `max_model_len` 32,768. So its "Maximum concurrency for 32,768 tokens" line is the same in both arms and does not prove anything.
- Control: the same 3 control runs as lanes 4 and T (`control-s1`, `control-s2r`, `control-s3r`).

## Frozen control baseline

Computed 24 Sep by `kvcap_verdict.py` from the 3 control directories. The script recomputes these values and stops if they move.

| Number | Control mean | control-s1 | Source |
|---|---|---|---|
| vLLM Running, mean of `Running: N reqs` lines | 3.131 | 3.129 | `vllm-openai-server.log` |
| Model calls per game-run | 56.72 | 57.16 | `deadprobe_counters.jsonl`, 4,254 calls over 75 game-runs |
| Levels per run | 38.0 | 35 | `benchmark.json` |
| Prompt tokens per request | 20,035 | 20,146 | `vllm-metrics-final.prom` |
| KV reserved | 5.0 GiB | 5.0 GiB | `reserved 5.0 GiB memory for KV Cache` |
| Score, mean of run means | 8.59, spread 4.96 | 6.43 | `ab_verdict.py` |

## Config proofs, checked in each run's own files

A run without its proof is invalid and does not count.

- Every arm: `config_proof.txt` (the `DEADPROBE_LIVE` line from the kernel log) carries `ab=<id>`.
- kv8: `vllm-openai-server.log` has `reserved 8.0 GiB memory for KV Cache`. kv7: `reserved 7.0 GiB`. ctx16k: `reserved 5.0 GiB`.
- ctx16k: the kernel log has `CTX16K_PROOF env=16384 tool_agent=16384`, and mean prompt tokens per request is at most 15,360. The predicted value is near 10k, against 20,146.

## Sealed mechanism predictions

- K-A (kv8): mean Running 5.0 or more. Point 5.5, band 4.3 to 7.0. The cache grows 1.6 times, so the pool is about 168k tokens if it scales with bytes (`unverified`: the model is hybrid GDN, and the state share is not measured). The sequence cap of 8 is unchanged, so Running cannot pass 8.
- K-B (kv8): model calls per game-run 68 or more against 57. Point 66, band 58 to 76.
- K-C (ctx16k): mean prompt tokens per request near 10k. Point 10,500, band 8,000 to 14,000.
- K-D (ctx16k): mean Running 5.0 or more. Point 5.8, band 4.5 to 7.5. Calls per game-run 68 or more. Point 70, band 60 to 82.
- K-E (both): levels per run do not fall below 38.0. kv8 point 39, band 34 to 44. ctx16k point 37, band 31 to 43. ctx16k keeps less history, so it is the arm more likely to lose levels.
- Expected risk for ctx16k, written before any run. OpenAI's post on its ARC-AGI-3 settings names two: retained reasoning and compaction. Our base has retained reasoning (vLLM `--default-chat-template-kwargs {"preserve_thinking": true}`, `kaggle.py:331`; reasoning kept on the assistant message, `tool_agent.py:1892`). It does not have compaction. It drops the oldest messages when a prompt passes the budget (`tool_agent.py:1672-1689`) and keeps at most 30 assistant turns (line 151). The model's own world-model note is carried (lines 1105-1111 and 1236), but nothing digests the dropped turns. ctx16k halves the budget, so it drops more history. That is the direction OpenAI found harmful. A ctx16k level loss is an expected risk, not a surprise. The compaction lever (`runs/RULE-compaction.md`, draft) is the counter-test.
- Chance that an arm meets all three targets (MECH keep): kv8 about 40%, ctx16k about 35%, at least one about 55%.

## Verdict rule

Read only with 2 valid runs per arm. If kv8 falls back to kv7, the arm is read as kv7. An arm that mixes KV sizes reads PENDING.

- MECH keep: mean Running 5.0 or more, and calls per game-run 68 or more, and levels per run 38.0 or more. All three. Otherwise MECH LOSE.
- Score LEDGER (LANES section 5): keep if the score gap is above the control spread (4.96) and levels do not fall. Kill if the arm is below control by more than that spread. Pending otherwise. This clause gates stacking only, not the submit.
- K1: both arms valid and both mean Running under 4.0. Then the KV thesis is dead at this verdict.

## Submit rule, 26 Sep, one submission

Authority: Oscar's ruling of 24 Sep about 15:40 Paris. The machine may submit a lane winner that meets its pre-registered mechanism targets. One per day, with the prediction sealed and committed before the run.

- If one or more arms read MECH keep, submit the s1 kernel of the keep arm with the most levels per run. Tie: higher Running. The kernel version is the one the driver pushed. The command is `kaggle competitions submit arc-prize-2026-arc-agi-3 -k oscarmorke/ab-<arm>-s1 -v <N> -f submission.parquet -m "..."`.
- Otherwise, submit the control again (`oscarmorke/ab-q38-control-s1` version 1). This is the third board repeat for the noise band (56490516 and 56515552 are the first two).
- The driver submits at 07:23 UTC, and only if Kaggle's submission list shows no row dated 26 Sep UTC.

## OOM fallback

If a kv8 run ends in ERROR and its log has an out-of-memory line (`out of memory`, `OutOfMemoryError`, or `No available memory for the cache blocks`), the kv arm switches to kv7 (7 GiB). kv7 is used for that seed and every kv seed not yet pushed. The kv7 notebooks are built and tested now, so the driver never writes code at night. Any other ERROR gets one retry of the same notebook, logged as a deviation (as RULE-guard-answer A1 and A2). After that the seed is given up.

## Driver and budget

- Driver: `kaggle/ab-kvcap/drive_kvcap.py`, run by `drive-kvcap.sh` under `caffeinate -i -s`. launchd label `com.oscar.arc-kvcap`, plist `kaggle/ab-kvcap/kvcap.plist`. Outputs go to `runs/kvcap-2026-09-26/`.
- Times, 26 Sep UTC. 00:05: push kv8-s1 and ctx16k-s1, but only when `kaggle quota` shows 9 h or more left, and only when this file, the notebooks and the log entry are committed (checked in git). When an arm's s1 finishes, push its s2. 05:30: verdict. 07:23: submit. At the end, the driver removes its plist and runs `launchctl bootout` on itself. Hard end 23:50 UTC.
- Quota: every push (s1, s2, kv7 fallback, retry) needs `kaggle quota` 2.5 h or more left. The driver logs `kaggle quota` right before the submit and again when the submission turns COMPLETE. A drop between the two with no push in between means the scoring rerun draws on the quota (now `unverified`).
- No submit either unless this file, the notebooks and the log entry are committed.
- Budget: 4 runs at about 2.2 h, so 8.8 h of the 30 h weekly quota. `kaggle quota` 24 Sep 13:32 UTC: 1.32 h left, refresh 2026-09-26 00:00 UTC.

## Amendments

None yet.
