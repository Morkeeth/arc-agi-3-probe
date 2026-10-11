"""Build the 11 Oct competition-rerun variant of the completed Intel control.

Remove two experiment-only TRUE_SUBMISSION assertions and write Kaggle's
submission-file handle after offline validation. Solver, model, dataset,
budget and gateway code remain unchanged.
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
    final_cell = candidate["cells"][10]
    final_source = final_cell["source"]
    final_source = "".join(final_source) if isinstance(final_source, list) else final_source
    anchor = "await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)\n"
    if final_source.count(anchor) != 1:
        raise RuntimeError("Expected one benchmark call in final cell")
    final_source = final_source.replace(anchor, anchor + """
if not TRUE_SUBMISSION:
    # The competition API attaches this completed notebook by output filename.
    # The real competition rerun reports gameplay through its gateway instead.
    import pandas as pd
    pd.DataFrame(
        [["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    ).to_parquet(WORKING_DIR / "submission.parquet", index=False)
""", 1)
    final_cell["source"] = final_source
    changed.append(10)
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
                                "change": "remove two experiment-only submission assertions and write offline Kaggle attachment file"}, indent=2) + "\n")
    print(SEAL.read_text())


if __name__ == "__main__":
    main()
