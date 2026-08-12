"""Gated model-backed operator smoke; never fabricates operator output."""

from __future__ import annotations

from brain_v2_preflight import call, gate, report, wait_for_gateway


def main() -> int:
    gateway = wait_for_gateway()
    if gateway["status"] != "ready":
        report("brain_v2_llm_smoke", "BLOCKED", reason="MCP_GATEWAY_UNAVAILABLE", **gateway)
        return 0
    if not gate("GPU_LAB_BRAIN_V2_RUN_LLM"):
        report(
            "brain_v2_llm_smoke",
            "IMPLEMENTED_UNVERIFIED",
            reason="EXPLICIT_MODEL_RUN_GATE_NOT_SET",
            prerequisite="Set GPU_LAB_BRAIN_V2_RUN_LLM=1 only with task-scoped provider credentials.",
        )
        return 0
    status = call("research_operator_status", {})
    if status.get("status") != "ready":
        report("brain_v2_llm_smoke", "BLOCKED", reason="OPERATOR_PROVIDER_UNAVAILABLE", provider=status)
        return 0
    report(
        "brain_v2_llm_smoke",
        "IMPLEMENTED_UNVERIFIED",
        reason="Provider is reachable, but no bounded agenda/project was supplied for generation.",
        provider=status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
