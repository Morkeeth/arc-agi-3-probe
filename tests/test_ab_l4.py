"""Lane 4 notebooks: seqs8 vs seqs16 differ only in MAX_NUM_SEQS and AB_ID/slug strings."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
L4 = ROOT / "kaggle" / "ab-l4"


def _load_build():
    spec = importlib.util.spec_from_file_location("ab_l4_build", L4 / "build.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _notebook_text(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "".join("".join(c.get("source", [])) for c in nb["cells"])


def _normalize(text: str, arm: str, seed: int) -> str:
    max_seqs = "8" if arm == "seqs8" else "16"
    ab_id = f"{arm}-s{seed}"
    slug = f"ab-l4-{arm}-s{seed}"
    text = text.replace(f'"TAAF_VLLM_MAX_NUM_SEQS": "{max_seqs}"', '"TAAF_VLLM_MAX_NUM_SEQS": "X"')
    text = text.replace(ab_id, "ARM_SEED")
    text = text.replace(slug, "SLUG")
    return text


def test_arms_differ_only_in_max_num_seqs_and_ids():
    build = _load_build()
    dirs = build.build()
    assert {d.name for d in dirs} == {"seqs8-s1", "seqs8-s2", "seqs16-s1", "seqs16-s2"}

    for seed in (1, 2):
        p8 = L4 / f"seqs8-s{seed}" / f"ab-l4-seqs8-s{seed}.ipynb"
        p16 = L4 / f"seqs16-s{seed}" / f"ab-l4-seqs16-s{seed}.ipynb"
        s8 = _notebook_text(p8)
        s16 = _notebook_text(p16)

        assert '"TAAF_VLLM_MAX_NUM_SEQS": "8"' in s8
        assert '"TAAF_VLLM_MAX_NUM_SEQS": "16"' in s16
        assert '"TAAF_VLLM_MAX_NUM_SEQS": "16"' not in s8
        assert 'MODE = "off"' in s8
        assert 'MODE = "off"' in s16
        assert "os.environ['TAAF_VLLM_MAX_NUM_SEQS']" in s8
        assert "os.environ['TAAF_VLLM_MAX_NUM_SEQS']" in s16

        assert _normalize(s8, "seqs8", seed) == _normalize(s16, "seqs16", seed)

        m8 = json.loads((L4 / f"seqs8-s{seed}" / "kernel-metadata.json").read_text())
        m16 = json.loads((L4 / f"seqs16-s{seed}" / "kernel-metadata.json").read_text())
        assert m8["id"] == f"oscarmorke/ab-l4-seqs8-s{seed}"
        assert m16["id"] == f"oscarmorke/ab-l4-seqs16-s{seed}"
        assert m8["title"] == f"ab-l4-seqs8-s{seed}"
        assert m16["title"] == f"ab-l4-seqs16-s{seed}"
        assert m8["is_private"] is True and m16["is_private"] is True

        def _meta_norm(m: dict, arm: str) -> dict:
            out = {k: v for k, v in m.items() if k not in ("id", "title", "code_file")}
            return out

        assert _meta_norm(m8, "seqs8") == _meta_norm(m16, "seqs16")
