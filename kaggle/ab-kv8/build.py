"""Lane KV8 notebooks: the ab-q38 control notebook with KV cache 8 GiB in place of 5 GiB. Nothing else.

    python kaggle/ab-kv8/build.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "ab-q38"
OLD = '\\"TAAF_VLLM_KV_CACHE_MEMORY_BYTES\\": \\"5368709120\\"'
NEW = '\\"TAAF_VLLM_KV_CACHE_MEMORY_BYTES\\": \\"8589934592\\"'


def build() -> list[Path]:
    out = []
    for k in (1, 2):
        base = SRC / "control-s1" / "ab-q38-control-s1.ipynb"
        text = base.read_text()
        assert text.count(OLD) == 1, "KV_CACHE_MEMORY_BYTES line not found exactly once"
        text = text.replace(OLD, NEW).replace('AB_ID = \\"control-s1\\"', f'AB_ID = \\"kv8-s{k}\\"')
        assert f'kv8-s{k}' in text, "AB_ID not replaced"
        slug = f"ab-kv8-s{k}"
        d = HERE / f"kv8-s{k}"
        d.mkdir(exist_ok=True)
        (d / f"{slug}.ipynb").write_text(text)
        m = json.loads((SRC / "control-s1" / "kernel-metadata.json").read_text())
        m.update(id=f"oscarmorke/{slug}", title=slug, code_file=f"{slug}.ipynb", is_private=True)
        (d / "kernel-metadata.json").write_text(json.dumps(m, indent=2) + "\n")
        out.append(d)
    return out


if __name__ == "__main__":
    for d in build():
        print(d)
