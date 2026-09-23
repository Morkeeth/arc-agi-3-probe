"""MTP4 and KV8 lever notebooks: each differs from control-s1 in exactly one env line and AB_ID."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTROL = ROOT / "kaggle" / "ab-q38" / "control-s1" / "ab-q38-control-s1.ipynb"
LEVERS = (
    (
        "ab-mtp4",
        "mtp4",
        '    "TAAF_VLLM_MTP_TOKENS": "3",',
        '    "TAAF_VLLM_MTP_TOKENS": "4",',
    ),
    (
        "ab-kv8",
        "kv8",
        '    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120",',
        '    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "8589934592",',
    ),
)


def _load_build(name: str):
    path = ROOT / "kaggle" / name / "build.py"
    spec = importlib.util.spec_from_file_location(f"{name}_build", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _notebook_lines(path: Path) -> list[str]:
    nb = json.loads(path.read_text())
    lines: list[str] = []
    for cell in nb["cells"]:
        src = cell.get("source", [])
        if isinstance(src, str):
            lines.extend(src.splitlines())
        else:
            lines.extend("".join(src).splitlines())
    return lines


def _diff_lines(a: list[str], b: list[str]) -> list[tuple[str, str]]:
    assert len(a) == len(b), f"line count mismatch: {len(a)} vs {len(b)}"
    return [(x, y) for x, y in zip(a, b) if x != y]


def test_levers_differ_from_control_in_exactly_one_env_line_and_ab_id():
    control = _notebook_lines(CONTROL)
    assert 'AB_ID = "control-s1"' in control

    for folder, arm, old_env, new_env in LEVERS:
        build = _load_build(folder)
        dirs = build.build()
        assert {d.name for d in dirs} == {f"{arm}-s1", f"{arm}-s2"}

        for k in (1, 2):
            ab_id = f"{arm}-s{k}"
            slug = f"ab-{arm}-s{k}"
            nb_path = ROOT / "kaggle" / folder / ab_id / f"{slug}.ipynb"
            got = _notebook_lines(nb_path)
            diffs = _diff_lines(control, got)
            assert len(diffs) == 2, f"{ab_id}: expected 2 differing lines, got {len(diffs)}: {diffs}"

            env_diffs = [(a, b) for a, b in diffs if "TAAF_VLLM_" in a or "TAAF_VLLM_" in b]
            id_diffs = [(a, b) for a, b in diffs if "AB_ID" in a or "AB_ID" in b]
            assert len(env_diffs) == 1, f"{ab_id}: env diffs={env_diffs}"
            assert len(id_diffs) == 1, f"{ab_id}: id diffs={id_diffs}"
            assert env_diffs[0] == (old_env, new_env), f"{ab_id}: {env_diffs[0]}"
            assert id_diffs[0] == ('AB_ID = "control-s1"', f'AB_ID = "{ab_id}"'), f"{ab_id}: {id_diffs[0]}"

            meta = json.loads((ROOT / "kaggle" / folder / ab_id / "kernel-metadata.json").read_text())
            assert meta["id"] == f"oscarmorke/{slug}"
            assert meta["title"] == slug
            assert meta["code_file"] == f"{slug}.ipynb"
            assert meta["is_private"] is True
