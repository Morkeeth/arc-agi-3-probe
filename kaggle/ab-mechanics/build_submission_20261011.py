"""Build the 11 Oct competition-rerun variant of the completed Intel control.

Only remove two experiment-only TRUE_SUBMISSION assertions. All solver, model,
dataset, budget and gateway code remains byte-identical within notebook cells.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "control/experiment.ipynb"
OUT = HERE / "submit-20261011/experiment.ipynb"
META = HERE / "control/kernel-metadata.json"
OUT_META = HERE / "submit-20261011/kernel-metadata.json"
SEAL = HERE / "submit-20261011/build-seal.json"


def main():
    original = json.loads(SOURCE.read_text())
    candidate = json.loads(SOURCE.read_text())
    changed = []
    for i, cell in enumerate(candidate["cells"]):
        source = cell.get("source", "")
        joined = "".join(source) if isinstance(source, list) else source
        count = joined.count("assert not TRUE_SUBMISSION\n")
        if count:
            if count != 1 or i not in (2, 8):
                raise RuntimeError(f"Unexpected submission guard at cell {i}")
            changed.append(i)
            joined = joined.replace("assert not TRUE_SUBMISSION\n", "", 1)
            cell["source"] = joined
    if changed != [2, 8]:
        raise RuntimeError(f"Expected exactly cells 2 and 8, got {changed}")
    assert "TRUE_SUBMISSION = os.environ.get(\"KAGGLE_IS_COMPETITION_RERUN\"" in str(candidate["cells"][2]["source"])
    assert "bm.games = _competition_games()" in str(candidate["cells"][10]["source"])
    for i, (before, after) in enumerate(zip(original["cells"], candidate["cells"])):
        if i not in changed and before != after:
            raise RuntimeError(f"Undeclared notebook change at cell {i}")
    meta = json.loads(META.read_text())
    meta.update({"id": "oscarmorke/arc3-intel-control-submission-20261011",
                 "title": "ARC3 Intel control submission 20261011",
                 "code_file": "experiment.ipynb", "is_private": True})
    if meta.get("competition_sources") != ["arc-prize-2026-arc-agi-3"]:
        raise RuntimeError("Wrong competition attachment")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(candidate, ensure_ascii=False, indent=1) + "\n")
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n")
    SEAL.write_text(json.dumps({"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                                "candidate_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
                                "changed_cells": changed,
                                "change": "remove two experiment-only submission assertions"}, indent=2) + "\n")
    print(SEAL.read_text())


if __name__ == "__main__":
    main()
