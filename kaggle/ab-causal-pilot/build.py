"""Build the sealed one-control/one-treatment causal-memory pilot.

This script only writes local artifacts. It cannot push, run, submit, or spend.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "kaggle/ab-compact/control-cmp-s1/ab-control-cmp-s1.ipynb"
ENV = ROOT / "runs/ab-2026-09-23/control-s1/taaf_setup_env.json"
CAUSAL = ROOT / "kaggle/ab-compact/causal_memory.py"
AUDIT = ROOT / "tests/fixtures/arc-causal-audit-v1.json"
RULE = ROOT / "runs/RULE-causal-memory-pilot-v2.md"
HOOK_CELL = 13
MAX_CHARS = 3000

ARMS = {
    "control": "arc-causal-control-v2",
    "treatment": "arc-causal-treatment-v2",
}

EXPECTED_INPUT_HASHES = {
    "base_notebook": "a8221da6e3b2b289836d570a61ca39bb410cdb25a877449b6c5aa08673215f5b",
    "environment": "0dc9dfa1208e337f1f005cc9fad6e43faf4c4eadc2f158f3984a636bb78c3c22",
    "causal_source": "46a350b5f4405ffe53bbd6ad4172ea086c414124b97cb323095a83eff145223e",
    "audit_labels": "1bb1bdea680c2308d8a8b6b4cbc1e35b57b6064fa968d26c074c9f54d00e79ac",
    "cpu_population_manifest": "2817e712ab382b094fb7836a1a6cdfa3096957348554e4831072643d296c1c2a",
}

DATASETS = {
    "oscarmorke/arc-agi-3-probe": {
        "observed_last_updated": "2026-09-23 07:52:00.380000",
        "file_count": 7,
        "file_manifest_sha256": "e51954f3c8f0dc06eb8e91c22a4ff1f8099f8ded1a4c41447b5ee3041f1513b8",
    },
    "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1": {
        "observed_last_updated": "2026-08-31 23:47:17.740000",
        "file_count": 4408,
        "file_manifest_sha256": "ce36c1ae78a05bb59f4886292747adabc096ca420d9a499668c6dcfbc2ac2a05",
    },
    "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1": {
        "observed_last_updated": "2026-08-30 20:20:06.003000",
        "file_count": 7,
        "file_manifest_sha256": "9e912bca576f7452afe1a512cff637a6b9a450dfd05bf6d38e9a463cf554394c",
    },
    "competition:arc-prize-2026-arc-agi-3": {
        "file_count": 148,
        "file_manifest_sha256": "0a7c19f62424b042ccbc2ce1a356f03e62493316ec7de07c3670ff69c1feed15",
    },
}

MODEL = {
    "ref": "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1",
    "version_id": 977939,
    "version_number": 1,
    "external_revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    "metadata_sha256": "3004fd04633a12c9228a834df4b98f872705eee98796183c5d1bdea88b03c3a5",
}

GAMES = [
    "tn36-ef4dde99", "lf52-271a04aa", "cn04-2fe56bfb", "bp35-0a0ad940",
    "wa30-ee6fef47", "lp85-305b61c3", "r11l-495a7899", "tu93-0768757b",
    "sp80-589a99af", "m0r0-492f87ba", "vc33-5430563c", "ar25-0c556536",
    "ka59-38d34dbb", "sc25-635fd71a", "sk48-d8078629", "dc22-fdcac232",
    "cd82-fb555c5d", "ft09-0d8bbf25", "g50t-5849a774", "ls20-9607627b",
    "re86-8af5384d", "s5i5-18d95033", "sb26-7fbdac44", "su15-1944f8ab",
    "tr87-cd924810",
]

SETTINGS = {
    "games": GAMES,
    "game_order_sha256": hashlib.sha256((json.dumps(GAMES, separators=(",", ":")) + "\n").encode()).hexdigest(),
    "passes": 1,
    "model": "Qwen/Qwen3.8-Flash-Next-NVFP4",
    "context_window": 32768,
    "seed": -1,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_output_tokens": 0,
    "tool_output_tokens": 1024,
    "tool_steps": 0,
    "tool_timeout_seconds": 30,
    "yield_seconds": 60,
    "max_runtime_per_game_seconds": 7920.0,
    "notebook_budget_seconds": 32400.0,
    "analyzer_timeout_seconds": 900.0,
    "solver_concurrency": 28,
    "max_actions_per_game": None,
    "save_request_logs": True,
    "memory_record_max_chars": MAX_CHARS,
}

RAW_START = "# ---- arc-agi-3-probe lever compaction"
RAW_END = "# ---- end lever compaction ----"
CAUSAL_END = "# ---- end causal memory treatment ----"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _set_cell_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {text.count(old)}")
    return text.replace(old, new, 1)


def _causal_block(source: str) -> str:
    short = hashlib.sha256(source.encode()).hexdigest()[:12]
    return (
        "# ---- arc-agi-3-probe causal memory treatment "
        "(runs/RULE-causal-memory-pilot-v2.md). ----\n"
        f"CAUSAL_MAX_CHARS = {MAX_CHARS}\n"
        f"(_pkg / \"causal_memory.py\").write_text({source!r})\n"
        "import importlib; importlib.invalidate_caches()\n"
        "from deadprobe import causal_memory as _causal_memory\n"
        "_causal_events = Path(\"/kaggle/working/causal_memory_events.jsonl\")\n"
        "_causal_memory.install(CAUSAL_MAX_CHARS, events_path=_causal_events)\n"
        "import inference.agent.tool_agent as _causal_tool_agent\n"
        "print(\n"
        f"    \"CAUSAL_PROOF max_chars={MAX_CHARS} source_sha256={short} \"\n"
        "    f\"persist={_causal_tool_agent.ToolAgent._persistent_history_messages.__module__} \"\n"
        "    f\"ensure={_causal_tool_agent.ToolAgent._ensure_session.__module__} events={_causal_events}\",\n"
        "    flush=True,\n"
        ")\n"
        "# ---- end causal memory treatment ----"
    )


def _shared_runtime_block(arm: str) -> str:
    common = {
        "schema": "arc-causal-pilot-v2",
        "settings": SETTINGS,
        "datasets": DATASETS,
        "model": MODEL,
        "frozen_input_hashes": EXPECTED_INPUT_HASHES,
    }
    return (
        f"CAUSAL_PILOT_ARM = {arm!r}\n"
        f"CAUSAL_PILOT_COMMON = {common!r}\n"
        "_pilot_config = dict(CAUSAL_PILOT_COMMON, arm=CAUSAL_PILOT_ARM)\n"
        "(Path(\"/kaggle/working/causal_pilot_config.json\")).write_text(\n"
        "    json.dumps(_pilot_config, indent=2, sort_keys=True) + \"\\n\"\n"
        ")\n"
        "print(f\"CAUSAL_PILOT_CONFIG arm={CAUSAL_PILOT_ARM} request_logs=True\", flush=True)\n"
    )


def _replace_memory_block(text: str, replacement: str) -> str:
    start = text.find(RAW_START)
    end = text.find(RAW_END)
    if start < 0 or end < 0 or end < start:
        raise RuntimeError("frozen raw compaction block markers not found")
    end += len(RAW_END)
    return text[:start] + replacement + text[end:]


def _notebook(arm: str) -> dict:
    notebook = json.loads(BASE.read_text())
    hook = _cell_source(notebook["cells"][HOOK_CELL])
    hook = _replace_once(hook, "bm.solver.save_request_logs = False", "bm.solver.save_request_logs = True", "logs")
    hook = _replace_once(hook, 'AB_ID = "control-cmp-s1"', f'AB_ID = "{ARMS[arm]}"', "AB_ID")
    anchor = "bm.solver.save_request_logs = True\n"
    hook = _replace_once(hook, anchor, anchor + _shared_runtime_block(arm), "runtime seal")
    if arm == "treatment":
        hook = _replace_memory_block(hook, _causal_block(CAUSAL.read_text()))
    _set_cell_source(notebook["cells"][HOOK_CELL], hook)
    return notebook


def _metadata(arm: str) -> dict:
    meta = json.loads((BASE.parent / "kernel-metadata.json").read_text())
    slug = ARMS[arm]
    return dict(meta, id=f"oscarmorke/{slug}", title=slug, code_file=f"{slug}.ipynb", is_private=True)


def _normalized(notebook: dict) -> dict:
    out = copy.deepcopy(notebook)
    hook = _cell_source(out["cells"][HOOK_CELL])
    for arm, slug in ARMS.items():
        hook = hook.replace(slug, "<ARM_ID>").replace(repr(arm), repr("<ARM>"))
    _set_cell_source(out["cells"][HOOK_CELL], hook)
    return out


def _pair_diff(control: dict, treatment: dict) -> dict:
    c = _normalized(control)
    t = _normalized(treatment)
    changed = [i for i, (left, right) in enumerate(zip(c["cells"], t["cells"])) if left != right]
    c_hook = _cell_source(c["cells"][HOOK_CELL])
    t_hook = _cell_source(t["cells"][HOOK_CELL])
    c_start, c_end = c_hook.index(RAW_START), c_hook.index(RAW_END) + len(RAW_END)
    t_start = t_hook.index("# ---- arc-agi-3-probe causal memory treatment")
    t_end = t_hook.index(CAUSAL_END) + len(CAUSAL_END)
    prefix_equal = c_hook[:c_start] == t_hook[:t_start]
    suffix_equal = c_hook[c_end:] == t_hook[t_end:]
    top_equal = {k: v for k, v in c.items() if k != "cells"} == {k: v for k, v in t.items() if k != "cells"}
    outside_equal = all(c["cells"][i] == t["cells"][i] for i in range(len(c["cells"])) if i != HOOK_CELL)
    unexpected = []
    if changed != [HOOK_CELL]:
        unexpected.append(f"normalized changed cells {changed}, expected [{HOOK_CELL}]")
    if not prefix_equal:
        unexpected.append("hook-cell prefix differs outside policy block")
    if not suffix_equal:
        unexpected.append("hook-cell suffix differs outside policy block")
    if not top_equal or not outside_equal:
        unexpected.append("notebook differs outside hook cell")
    return {
        "schema": "arc-causal-pair-diff-v1",
        "normalized_changed_cells": changed,
        "expected_policy_cell": HOOK_CELL,
        "outside_policy_equal": top_equal and outside_equal and prefix_equal and suffix_equal,
        "neutral_differences_normalized": ["AB_ID", "CAUSAL_PILOT_ARM", "kernel id", "title", "code_file"],
        "control_policy": "raw-compaction-3000",
        "treatment_policy": "causal-record-3000",
        "unexpected": unexpected,
    }


def build() -> dict:
    actual = {
        "base_notebook": sha256(BASE),
        "environment": sha256(ENV),
        "causal_source": sha256(CAUSAL),
        "audit_labels": sha256(AUDIT),
    }
    for key, value in actual.items():
        if value != EXPECTED_INPUT_HASHES[key]:
            raise RuntimeError(f"frozen {key} moved: {value} != {EXPECTED_INPUT_HASHES[key]}")

    notebooks = {arm: _notebook(arm) for arm in ARMS}
    metadata = {arm: _metadata(arm) for arm in ARMS}
    for arm in ARMS:
        dest = HERE / arm
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{ARMS[arm]}.ipynb").write_text(json.dumps(notebooks[arm], indent=1, ensure_ascii=False) + "\n")
        (dest / "kernel-metadata.json").write_text(json.dumps(metadata[arm], indent=2, sort_keys=True) + "\n")

    diff = _pair_diff(notebooks["control"], notebooks["treatment"])
    if diff["unexpected"]:
        raise RuntimeError("pair diff failed: " + "; ".join(diff["unexpected"]))
    (HERE / "pair-diff.json").write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n")

    generated = {}
    for arm in ARMS:
        generated[arm] = {
            "notebook": sha256(HERE / arm / f"{ARMS[arm]}.ipynb"),
            "kernel_metadata": sha256(HERE / arm / "kernel-metadata.json"),
        }
    seal = {
        "schema": "arc-causal-pilot-seal-v2",
        "authorization": "LOCAL_PREPARATION_ONLY_NO_GPU_PUSH_UPLOAD_SUBMISSION_OR_SPEND",
        "rule_sha256": sha256(RULE),
        "frozen_input_hashes": EXPECTED_INPUT_HASHES,
        "datasets": DATASETS,
        "model": MODEL,
        "settings": SETTINGS,
        "generated": generated,
        "pair_diff_sha256": sha256(HERE / "pair-diff.json"),
    }
    (HERE / "seal.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    return seal


if __name__ == "__main__":
    result = build()
    print(json.dumps({"generated": result["generated"], "authorization": result["authorization"]}, indent=2))
