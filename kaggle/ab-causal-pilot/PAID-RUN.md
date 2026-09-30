# ARC KAGGLE causal-memory matched pilot v2

This is a command sheet, not authorization. It must not be run until Oscar explicitly approves the
GPU pair and the live dataset/model snapshots have been re-read against `seal.json`.

## Exact denominator

One contemporaneous control and one treatment, paired over the same 25 public games in the exact
`seal.json` order. Each game contributes one treatment-minus-control score delta. Historical control
mean 8.589519100442978 is context only and is not the denominator.

## Cheap local proof

```sh
python3 kaggle/ab-causal-pilot/dry_run.py \
  --output /Users/morkeeth/Downloads/NIGHT-REVIEW-2026-09-30/arc-kaggle-causal-pilot/dry-run
```

Expected: both arms valid, 25 prompt-bound games per arm, policy-only notebook diff true, and a result
at `.../dry-run/dry-run-result.json`. Its scores are explicitly synthetic.

## Exact future paid-GPU command sequence

Run from the approved isolated checkout after snapshot revalidation and explicit approval:

```sh
python3 kaggle/ab-causal-pilot/build.py
python3 -m pytest -q tests/test_causal_pilot_pair.py tests/test_causal_memory_cpu.py
kaggle kernels push -p kaggle/ab-causal-pilot/control
kaggle kernels push -p kaggle/ab-causal-pilot/treatment
kaggle kernels status oscarmorke/arc-causal-control-v2
kaggle kernels status oscarmorke/arc-causal-treatment-v2
mkdir -p runs/causal-pilot-v2/control runs/causal-pilot-v2/treatment
kaggle kernels output oscarmorke/arc-causal-control-v2 -p runs/causal-pilot-v2/control -o
kaggle kernels output oscarmorke/arc-causal-treatment-v2 -p runs/causal-pilot-v2/treatment -o
python3 kaggle/ab-causal-pilot/verdict.py \
  --control runs/causal-pilot-v2/control \
  --treatment runs/causal-pilot-v2/treatment \
  --output runs/causal-pilot-v2/verdict.json
```

Do not submit either notebook output to the competition.

## Expected evidence

- `causal_pilot_config.json` from each arm with observed settings equal to the seal.
- `benchmark.json` with exactly 25 terminal games, numeric scores and levels, in frozen order.
- `compaction_events.jsonl` for the control and `causal_memory_events.jsonl` for treatment, bound to all 25 games.
- `pilot_prompt_records.jsonl` and each game's retained `*_requests.jsonl`, proving the exact memory record reached the model input.
- No record above 3,000 characters.
- `verdict.json` with INVALID, FAILED, PROMISING, or INCONCLUSIVE under the prospective rule.

PROMISING permits only a request for a second separately approved pair. It is not submission evidence.
