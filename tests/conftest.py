import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENVS = ROOT / "tests" / "envs"
sys.path.insert(0, str(ROOT))

# Duck source: set DUCK_SRC to the unzipped jeroencottaar/taaf-kaggle-source-share/src
DUCK_SRC = os.environ.get("DUCK_SRC")
if DUCK_SRC:
    for sub in ("ARC3-Inference", "tufa-arc-agi-framework/src"):
        sys.path.insert(0, str(Path(DUCK_SRC) / sub))


@pytest.fixture
def pr01_env():
    arc_agi = pytest.importorskip("arc_agi")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENVS))
    env = arc.make("pr01")
    assert env is not None
    return env
