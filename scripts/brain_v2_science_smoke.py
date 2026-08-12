"""Gated Brain v2 science entrypoint.

This script performs preflight only by default. Set ``GPU_LAB_BRAIN_V2_RUN_REAL=1``
to delegate to the existing, approval-gated Brain v1 real CUDA smoke. A delegated
result is reported as inherited evidence; no synthetic result is manufactured here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from brain_v2_preflight import gate, report, rpc, wait_for_gateway


def main() -> int:
    gateway = wait_for_gateway()
    if gateway["status"] != "ready":
        report("brain_v2_science_smoke", "BLOCKED", reason="MCP_GATEWAY_UNAVAILABLE", **gateway)
        return 0
    required = {
        "brain_step",
        "experiment_plan_register",
        "research_experiment_execute",
        "brain_result_assess",
        "experiment_branch_create",
    }
    listed = rpc("tools/list", {})["tools"]
    missing = sorted(required - {item["name"] for item in listed})
    if missing:
        report("brain_v2_science_smoke", "BLOCKED", reason="MISSING_MCP_TOOLS", missing=missing)
        return 0
    if not gate("GPU_LAB_BRAIN_V2_RUN_REAL"):
        report(
            "brain_v2_science_smoke",
            "SCIENTIFIC_RESULT_NOT_EXECUTED",
            reason="EXPLICIT_REAL_RUN_GATE_NOT_SET",
            prerequisite="Set GPU_LAB_BRAIN_V2_RUN_REAL=1 after reviewing approval and runtime requirements.",
            inherited_real_smoke="scripts/brain_e2e_smoke.py",
        )
        return 0
    command = [sys.executable, str(Path(__file__).with_name("brain_e2e_smoke.py"))]
    result = subprocess.run(command, check=False, text=True)
    report(
        "brain_v2_science_smoke",
        "VERIFIED_REAL" if result.returncode == 0 else "BLOCKED",
        delegated_script="scripts/brain_e2e_smoke.py",
        exit_code=result.returncode,
        note="Only the delegated inspected result is evidence; this wrapper adds no scientific interpretation.",
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
