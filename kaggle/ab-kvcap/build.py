"""Loop A (KV capacity) notebooks for runs/RULE-kvcap.md. Each arm is ab-q38-control-s1 with one knob changed.

    python kaggle/ab-kvcap/build.py      # writes kaggle/ab-kvcap/<arm>-s<k>/{notebook, kernel-metadata.json}

Arms:
  kv8     TAAF_VLLM_KV_CACHE_MEMORY_BYTES 5368709120 -> 8589934592 (cell 3, read by serving_setup.py).
          Same text edit as kaggle/ab-kv8/build.py on branch lane-mtp4 (f37a155); output is byte-identical.
  kv7     OOM fallback for kv8: 7516192768. Built now so the night driver never generates code.
  ctx16k  LOCAL_ANALYZER_CONTEXT_WINDOW 32768 -> 16384. It cannot go in cell 3: serving_setup.py
          (persist_analyzer_environment, line 2935) writes "32768" into taaf_setup_env.json and cell 9
          copies that back into os.environ. tool_agent.py:138 reads the value once, at import, and the
          first import is the pickle.load in cell 11 (benchmark_initial.pkl names inference.framework.solver,
          which imports tool_agent at solver.py:37). So the override is the first statement of cell 11.
          It also asserts tool_agent is not imported yet, rewrites the setup env file, and prints a proof line.
Every arm also changes AB_ID in the hook cell. Nothing else differs (tests/test_kvcap_notebooks.py).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "ab-q38" / "control-s1"
CONTROL_NB = SRC / "ab-q38-control-s1.ipynb"
SEEDS = (1, 2)


def _esc(s: str) -> str:
    """Python text -> the escaped form it has inside a JSON string in the notebook file."""
    return json.dumps(s)[1:-1]


KV_OLD = _esc('"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120"')
CELL11_HEAD = "# Restore the deployment target and record the real submission state on it.\n"
CELL11_TAIL = "    bm = pickle.load(file)\nbm.job_dir = WORKING_DIR"

CTX_PRE = (
    "# ---- arc-agi-3-probe lever ctx16k (runs/RULE-kvcap.md): the only knob. serving_setup.py put\n"
    "# LOCAL_ANALYZER_CONTEXT_WINDOW=32768 in the setup env (cell 9). tool_agent.py:138 reads it at import,\n"
    "# and the pickle.load below is the first harness import. So the override goes here. ----\n"
    'assert "inference.agent.tool_agent" not in sys.modules, "ctx16k: tool_agent imported before the override"\n'
    'os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = "16384"\n'
    "_ctx_env = json.loads(SETUP_ENV_PATH.read_text())\n"
    '_ctx_env["LOCAL_ANALYZER_CONTEXT_WINDOW"] = "16384"\n'
    'SETUP_ENV_PATH.write_text(json.dumps(_ctx_env, indent=2, sort_keys=True) + "\\n")\n'
)
CTX_POST = (
    "\nimport inference.agent.tool_agent as _ctx_ta\n"
    "print(f\"CTX16K_PROOF env={os.environ['LOCAL_ANALYZER_CONTEXT_WINDOW']} \"\n"
    '      f"tool_agent={_ctx_ta._LOCAL_ANALYZER_CONTEXT_WINDOW}", flush=True)\n'
    'assert _ctx_ta._LOCAL_ANALYZER_CONTEXT_WINDOW == 16384, "ctx16k: the knob did not reach tool_agent"'
)


def _kv(text: str, value: str) -> str:
    assert text.count(KV_OLD) == 1, "KV_CACHE_MEMORY_BYTES line not found exactly once"
    return text.replace(KV_OLD, _esc(f'"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "{value}"'))


def _ctx16k(text: str) -> str:
    head, tail = _esc(CELL11_HEAD), _esc(CELL11_TAIL)
    assert text.count(head) == 1 and text.count(tail) == 1, "cell 11 anchors not found exactly once"
    return text.replace(head, _esc(CTX_PRE) + head).replace(tail, tail + _esc(CTX_POST))


ARMS = {
    "kv8": lambda t: _kv(t, "8589934592"),
    "kv7": lambda t: _kv(t, "7516192768"),
    "ctx16k": _ctx16k,
}


def build() -> list[Path]:
    control = CONTROL_NB.read_text()
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    out = []
    for arm, edit in ARMS.items():
        for k in SEEDS:
            ab_id = f"{arm}-s{k}"
            slug = f"ab-{ab_id}"
            old_id = _esc('AB_ID = "control-s1"')
            assert control.count(old_id) == 1, "AB_ID line not found exactly once"
            text = edit(control).replace(old_id, _esc(f'AB_ID = "{ab_id}"'))
            json.loads(text)  # still a valid notebook
            d = HERE / ab_id
            d.mkdir(exist_ok=True)
            (d / f"{slug}.ipynb").write_text(text)
            m = dict(meta, id=f"oscarmorke/{slug}", title=slug, code_file=f"{slug}.ipynb", is_private=True)
            (d / "kernel-metadata.json").write_text(json.dumps(m, indent=2) + "\n")
            out.append(d)
    return out


if __name__ == "__main__":
    for d in build():
        print(d)
