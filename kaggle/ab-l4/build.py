"""Build the four lane-4 A/B notebooks from the Qwen3.8 public fork.

    python kaggle/ab-l4/build.py        # writes kaggle/ab-l4/<arm>-s<k>/{notebook, kernel-metadata.json}

Base: same as kaggle/ab-q38/base.ipynb (wuliao0/duck-qwen3-8-anim-base).
Both arms keep MODE = "off" (read-only counters stay installed). The only knob that
differs is TAAF_VLLM_MAX_NUM_SEQS in PUBLIC25_VLLM_PROFILE_ENV: "8" vs "16".
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = {"seqs8": "8", "seqs16": "16"}
SEEDS = (1, 2)
MODE = "off"
HOOK_CELL_MARKER = "# Exact public-25 and competition settings."
MAX_NUM_SEQS_LINE = '"TAAF_VLLM_MAX_NUM_SEQS": "8"'

HOOK = '''
# ---- arc-agi-3-probe lane 4 A/B hook: only TAAF_VLLM_MAX_NUM_SEQS differs between arms. ----
MODE = "{mode}"  # both arms: off (read-only counters)
AB_ID = "{ab_id}"
import hashlib
import shutil

_probe_dirs = [Path("/kaggle/input/arc-agi-3-probe"), Path("/kaggle/input/datasets/oscarmorke/arc-agi-3-probe")]
_hit = next((d / "duck_patch.py" for d in _probe_dirs if (d / "duck_patch.py").exists()), None)
if _hit is None:
    _hit = next(Path("/kaggle/input").rglob("duck_patch.py"), None)
if _hit is None or not (_hit.parent / "counters.py").exists():
    raise RuntimeError("deadprobe dataset with counters.py not mounted: attach oscarmorke/arc-agi-3-probe")
_pkg = Path("/kaggle/working/deadprobe_pkg/deadprobe")
_pkg.mkdir(parents=True, exist_ok=True)
for _f in _hit.parent.glob("*.py"):
    shutil.copy(_f, _pkg / _f.name)
sys.path.insert(0, str(_pkg.parent))

from deadprobe import counters, duck_patch
import inference.framework.solver as _solver

_cfg = duck_patch.install(MODE, duck_patch.ProbeConfig(budget_level1=6, budget_later=3, max_clicks=3))
counters.install()
_S = _solver._HarnessGameSession
_sha = {{f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:12] for f in sorted(_pkg.glob("*.py"))}}
print(
    f"DEADPROBE_LIVE ab={{AB_ID}} mode={{_cfg.mode}} guard={{_cfg.guard}} answer={{_cfg.answer}} honest={{_cfg.honest}} "
    f"max_num_seqs={{os.environ['TAAF_VLLM_MAX_NUM_SEQS']}} "
    f"play={{_S.play.__module__}} step_env={{_S.step_env.__module__}} execute={{_S._execute_action.__module__}} sha={{_sha}}",
    flush=True,
)
bm.label = f"{{bm.label}}-ab-{{AB_ID}}"
'''


def _set_max_num_seqs(nb: dict, value: str) -> None:
    needle = MAX_NUM_SEQS_LINE
    replacement = f'"TAAF_VLLM_MAX_NUM_SEQS": "{value}"'
    hits = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if needle not in src and replacement not in src:
            continue
        if needle in src:
            src = src.replace(needle, replacement, 1)
            hits += 1
        elif value != "8" and f'"TAAF_VLLM_MAX_NUM_SEQS": "{value}"' in src:
            hits += 1
        cell["source"] = src.splitlines(keepends=True)
    assert hits == 1, f"expected exactly one MAX_NUM_SEQS assignment, got {hits}"


def build() -> list[Path]:
    base = json.loads((HERE / "base.ipynb").read_text())
    meta = json.loads((HERE / "base-kernel-metadata.json").read_text())
    out = []
    for arm, max_num_seqs in ARMS.items():
        for k in SEEDS:
            ab_id = f"{arm}-s{k}"
            slug = f"ab-l4-{arm}-s{k}"
            nb = copy.deepcopy(base)
            _set_max_num_seqs(nb, max_num_seqs)
            cells = [c for c in nb["cells"] if c["cell_type"] == "code" and HOOK_CELL_MARKER in "".join(c["source"])]
            assert len(cells) == 1, "hook cell not found exactly once"
            src = "".join(cells[0]["source"]).rstrip("\n") + "\n" + HOOK.format(mode=MODE, ab_id=ab_id)
            cells[0]["source"] = src.splitlines(keepends=True)
            d = HERE / ab_id
            d.mkdir(exist_ok=True)
            (d / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
            m = {k2: v for k2, v in meta.items() if k2 != "id_no"}
            m.update(
                id=f"oscarmorke/{slug}",
                title=slug,
                code_file=f"{slug}.ipynb",
                is_private=True,
                enable_internet=False,
                dataset_sources=meta["dataset_sources"] + ["oscarmorke/arc-agi-3-probe"],
            )
            (d / "kernel-metadata.json").write_text(json.dumps(m, indent=2) + "\n")
            out.append(d)
    return out


if __name__ == "__main__":
    for d in build():
        print(d)
