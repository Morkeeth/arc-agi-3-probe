"""Compaction arm notebooks for runs/RULE-compaction.md. Each is a base notebook plus one knob: the digest.

    python kaggle/ab-compact/build.py                    # every base: control kv8 kv7 ctx16k, seeds 1 and 2
    python kaggle/ab-compact/build.py --base kv8         # one base

The base is a parameter because the rule applies compaction on top of the 26 Sep Loop A winner, and the
winner is not known until the 26 Sep verdict. Every possible base is built now and committed before any
run, so the 27 Sep driver only picks a directory and never generates code (same pattern as kv7 in
kaggle/ab-kvcap/build.py).

  base     base notebook (its s1; s1 and s2 differ only in AB_ID)
  control  kaggle/ab-q38/control-s1/ab-q38-control-s1.ipynb
  kv8      kaggle/ab-kvcap/kv8-s1/ab-kv8-s1.ipynb
  kv7      kaggle/ab-kvcap/kv7-s1/ab-kv7-s1.ipynb
  ctx16k   kaggle/ab-kvcap/ctx16k-s1/ab-ctx16k-s1.ipynb

Output: kaggle/ab-compact/<base>-cmp-s<k>/{ab-<base>-cmp-s<k>.ipynb, kernel-metadata.json}.

The edit, all inside the hook cell (cell 13):
  1. AB_ID "<base>-s1" -> "<base>-cmp-s<k>"
  2. right after `counters.install()`: write compaction.py (embedded verbatim, so no Kaggle dataset
     version is needed and the mounted oscarmorke/arc-agi-3-probe files stay byte-identical to the base
     runs) into the deadprobe package, import it, install(MAX_CHARS). The DEADPROBE_LIVE sha line that
     follows then includes compaction.py's sha, which the verdict checks against the committed file.
Nothing else differs (tests/test_compaction_notebooks.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parent
PATCH = HERE / "compaction.py"
MAX_CHARS = 3000  # runs/RULE-compaction.md, "The knob": chosen from the CPU replay, see the rule
SEEDS = (1, 2)
BASES = {
    "control": KAGGLE / "ab-q38" / "control-s1",
    "kv8": KAGGLE / "ab-kvcap" / "kv8-s1",
    "kv7": KAGGLE / "ab-kvcap" / "kv7-s1",
    "ctx16k": KAGGLE / "ab-kvcap" / "ctx16k-s1",
}
ANCHOR = "counters.install()\n"


def _esc(s: str) -> str:
    """Python text -> the escaped form it has inside a JSON string in the notebook file."""
    return json.dumps(s)[1:-1]


def base_notebook(base: str) -> Path:
    d = BASES[base]
    (nb,) = d.glob("*.ipynb")
    return nb


def base_ab_id(base: str) -> str:
    return f"{base}-s1"


def arm_id(base: str, seed: int) -> str:
    return f"{base}-cmp-s{seed}"


def block(src: str, max_chars: int = MAX_CHARS) -> str:
    sha = hashlib.sha256(src.encode()).hexdigest()[:12]
    return (
        "# ---- arc-agi-3-probe lever compaction (runs/RULE-compaction.md): the only knob. When history is\n"
        "# dropped, a digest of it is kept in the world-model notes. Source embedded verbatim, sha256 "
        f"{sha}. ----\n"
        f"COMPACT_MAX_CHARS = {max_chars}\n"
        f"(_pkg / \"compaction.py\").write_text({src!r})\n"
        "import importlib; importlib.invalidate_caches()  # the package was imported above: see the new file\n"
        "from deadprobe import compaction as _compaction\n"
        "_compaction.install(COMPACT_MAX_CHARS)\n"
        "# ---- end lever compaction ----\n"
    )


def edit(text: str, base: str, seed: int, src: str) -> str:
    old_id = _esc(f'AB_ID = "{base_ab_id(base)}"')
    anchor = _esc(ANCHOR)
    assert text.count(old_id) == 1, f"{base}: AB_ID line not found exactly once"
    assert text.count(anchor) == 1, f"{base}: counters.install() anchor not found exactly once"
    text = text.replace(old_id, _esc(f'AB_ID = "{arm_id(base, seed)}"'))
    return text.replace(anchor, anchor + _esc(block(src)))


def build(bases=None) -> list[Path]:
    src = PATCH.read_text()
    out = []
    for base in bases or BASES:
        nb = base_notebook(base)
        text = nb.read_text()
        meta = json.loads((nb.parent / "kernel-metadata.json").read_text())
        for k in SEEDS:
            ab = arm_id(base, k)
            slug = f"ab-{ab}"
            new = edit(text, base, k, src)
            json.loads(new)  # still a valid notebook
            d = HERE / ab
            d.mkdir(exist_ok=True)
            (d / f"{slug}.ipynb").write_text(new)
            m = dict(meta, id=f"oscarmorke/{slug}", title=slug, code_file=f"{slug}.ipynb", is_private=True)
            (d / "kernel-metadata.json").write_text(json.dumps(m, indent=2) + "\n")
            out.append(d)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="*", choices=sorted(BASES), help="default: every base")
    for d in build(ap.parse_args().base):
        print(d)
