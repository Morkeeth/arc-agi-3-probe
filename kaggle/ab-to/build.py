"""Lane T notebooks: the ab-q38 control notebook with analyzer_timeout 1200 s in place of 900 s. Nothing else.

    python kaggle/ab-to/build.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "ab-q38"
OLD = "bm.solver.analyzer_timeout = 900.0"
NEW = "bm.solver.analyzer_timeout = 1200.0"


def build() -> list[Path]:
    out = []
    for k in (1, 2):
        base = SRC / "control-s1" / "ab-q38-control-s1.ipynb"
        text = base.read_text()
        assert text.count(OLD) == 1, "analyzer_timeout line not found exactly once"
        text = text.replace(OLD, NEW).replace('AB_ID = \\"control-s1\\"', f'AB_ID = \\"to1200-s{k}\\"')
        assert f'to1200-s{k}' in text, "AB_ID not replaced"
        slug = f"ab-to1200-s{k}"
        d = HERE / f"to1200-s{k}"
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
