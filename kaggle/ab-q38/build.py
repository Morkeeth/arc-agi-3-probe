"""Build the six A/B notebooks from the public fork, changing only the section 6 hook cell.

    python kaggle/ab-q38/build.py        # writes kaggle/ab-q38/<arm>-s<k>/{notebook, kernel-metadata.json}

Base: wuliao0/duck-qwen3-8-anim-base, pulled 2026-09-23 as base.ipynb (sha256 in RULE-guard-answer.md).
Both arms append the same block to the hook cell. It loads deadprobe from the attached private dataset
oscarmorke/arc-agi-3-probe, installs duck_patch in MODE, then the read-only counters, and prints one
DEADPROBE_LIVE line that proves which code is live. The only difference between arms is MODE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = {"control": "off", "answer": "guard+answer"}
SEEDS = (1, 2, 3)
HOOK_CELL_MARKER = "# Exact public-25 and competition settings."

HOOK = '''
# ---- arc-agi-3-probe A/B hook: runs/RULE-guard-answer.md. Only MODE differs between arms. ----
MODE = "{mode}"  # control: off | treatment: guard+answer
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
    f"play={{_S.play.__module__}} step_env={{_S.step_env.__module__}} execute={{_S._execute_action.__module__}} sha={{_sha}}",
    flush=True,
)
bm.label = f"{{bm.label}}-ab-{{AB_ID}}"
'''


def build() -> list[Path]:
    base = json.loads((HERE / "base.ipynb").read_text())
    meta = json.loads((HERE / "base-kernel-metadata.json").read_text())
    out = []
    for arm, mode in ARMS.items():
        for k in SEEDS:
            ab_id = f"{arm}-s{k}"
            slug = f"ab-q38-{arm}-s{k}"
            nb = copy.deepcopy(base)
            cells = [c for c in nb["cells"] if c["cell_type"] == "code" and HOOK_CELL_MARKER in "".join(c["source"])]
            assert len(cells) == 1, "hook cell not found exactly once"
            src = "".join(cells[0]["source"]).rstrip("\n") + "\n" + HOOK.format(mode=mode, ab_id=ab_id)
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
