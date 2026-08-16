"""Gated real branch-science entrypoint.

The default path verifies branch-tool availability only. It deliberately does not
invent branch results or mark a ComparativeLesson from unexecuted nodes.
"""

from __future__ import annotations

from brain_v2_preflight import gate, report, rpc, wait_for_gateway


def main() -> int:
    gateway = wait_for_gateway()
    if gateway["status"] != "ready":
        report("brain_v2_branch_science", "BLOCKED", reason="MCP_GATEWAY_UNAVAILABLE", **gateway)
        return 0
    required = {
        "experiment_branch_create",
        "experiment_branch_node_add",
        "experiment_branch_result_record",
        "experiment_branch_compare",
    }
    listed = rpc("tools/list", {})["tools"]
    missing = sorted(required - {item["name"] for item in listed})
    if missing:
        report("brain_v2_branch_science", "BLOCKED", reason="MISSING_MCP_TOOLS", missing=missing)
        return 0
    if not gate("GPU_LAB_BRAIN_V2_RUN_BRANCHES"):
        report(
            "brain_v2_branch_science",
            "SCIENTIFIC_RESULT_NOT_EXECUTED",
            reason="EXPLICIT_BRANCH_RUN_GATE_NOT_SET",
            prerequisite="Set GPU_LAB_BRAIN_V2_RUN_BRANCHES=1 only after preregistration and approval.",
            existing_integration_smoke="scripts/branch_e2e_smoke.py",
        )
        return 0
    report(
        "brain_v2_branch_science",
        "BLOCKED",
        reason="NO_PROJECT_OR_PREREGISTERED_BRANCH_SPECIFIED",
        note="Refusing to fabricate branches, results, EvidenceFamilies, or ComparativeLessons.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
