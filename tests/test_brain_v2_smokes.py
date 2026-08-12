from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str) -> dict:
    env = os.environ.copy()
    env.pop("GPU_LAB_BRAIN_V2_RUN_REAL", None)
    env.pop("GPU_LAB_BRAIN_V2_RUN_LLM", None)
    env.pop("GPU_LAB_BRAIN_V2_RUN_BRANCHES", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


def test_science_smoke_is_nonexecuting_without_explicit_gate():
    result = _run("brain_v2_science_smoke.py")
    assert result["verification"] == "SCIENTIFIC_RESULT_NOT_EXECUTED"
    assert result["reason"] == "EXPLICIT_REAL_RUN_GATE_NOT_SET"


def test_llm_smoke_is_unverified_without_explicit_gate():
    result = _run("brain_v2_llm_smoke.py")
    assert result["verification"] == "IMPLEMENTED_UNVERIFIED"
    assert result["reason"] == "EXPLICIT_MODEL_RUN_GATE_NOT_SET"


def test_branch_smoke_is_nonexecuting_without_explicit_gate():
    result = _run("brain_v2_branch_science.py")
    assert result["verification"] == "SCIENTIFIC_RESULT_NOT_EXECUTED"
    assert result["reason"] == "EXPLICIT_BRANCH_RUN_GATE_NOT_SET"
