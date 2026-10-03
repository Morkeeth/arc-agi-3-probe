# Read a saved ARC KAGGLE pair

Use this after the experiment owner has obtained the two approved runs. This command reads local files only. It cannot run a model, download output, rebuild a notebook or submit a result.

```sh
python3 kaggle/ab-causal-pilot/review.py \
  --control runs/causal-pilot-v2/control \
  --treatment runs/causal-pilot-v2/treatment \
  --output runs/causal-pilot-v2/review-1
```

The output must be a new directory outside both input directories. Open `REVIEW.md` for the short per-game view. `review.json` retains every record, source hash, file and line reference.

The review calls the existing frozen inspector and decision rule. It keeps all 25 games in the sealed order. It adds strict parsing and ambiguity checks to the review; errors make this review INVALID. It does not edit the original inspector or retroactively change a saved trial result. Exit 2 means INVALID or invalid command arguments. Exit 0 means the review was produced; it is not a performance or release pass.

For each game, read the outcome first, then the exact memory that reached the model. The short view chooses one example with a recorded continuation, otherwise one bound input, otherwise one emitted record. The JSON contains all examples, so the selected illustration is never the denominator.

A continuation is shown only when the next request retains the exact earlier message prefix in the same analysis step and adds an assistant message. Tool results must match that assistant's tool-call IDs. Duck's `event=response` entry repeats input and is deliberately not treated as model behavior. Context trimming, a turn change, a final call with no later request, or missing step metadata leaves the next action unknown. This is conservative: it can miss a real continuation. A transcript viewer can resolve those cases later, but this reader must not invent the link.

Memory followed by a different action is an observation, not proof that memory caused that action. The causal comparison remains the frozen matched game outcomes. The command authenticates neither the runtime nor the uploader; it reads the supplied artifacts and gives their hashes. Existing snapshot revalidation and explicit GPU approval still apply.

Synthetic flags in either arm make the whole pair `SYNTHETIC_REVIEW`; `behavioral_result` is null. The rule exercised by the fixture is stored only in `synthetic_rule_path`. Removing labels cannot turn fabricated data into authenticated execution, and this offline tool does not claim to detect malicious fabrication.

INVALID means repair evidence without filling missing outcomes. FAILED and INCONCLUSIVE remain those results. A real PROMISING pair permits only a request for a separately approved second pair. It does not permit submission. See the unchanged `PAID-RUN.md` for the original execution gate.

## Local verification

```sh
python3 -m pytest -q tests/test_causal_pair_review.py
```

Tests exercise real inspection and rendering over explicitly synthetic fixtures, including malformed input, missing games, duplicate streams, non-finite numbers and unsupported action bindings. They do not execute ARC games or estimate solver lift.
