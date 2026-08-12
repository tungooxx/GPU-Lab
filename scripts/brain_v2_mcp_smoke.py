"""Live MCP smoke for the Brain v2 strategic memory surface."""

import json
import os
import time
import urllib.error
import urllib.request
import uuid

MCP_URL = os.environ.get("GPU_LAB_MCP_URL", "http://127.0.0.1:8000/mcp")


def rpc(method: str, params: dict, timeout: int = 90):
    request = urllib.request.Request(
        MCP_URL,
        data=json.dumps(
            {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "gpu-lab-brain-v2-smoke/1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        envelope = json.loads(response.read())
    if "error" in envelope:
        raise RuntimeError(envelope["error"])
    result = envelope["result"]
    if "tools" in result:
        return result
    value = result.get("structuredContent", {}).get("result")
    return value if value is not None else json.loads(result["content"][0]["text"])


def wait_for_gateway() -> None:
    """Avoid treating the short post-Compose startup window as an MCP failure."""
    deadline = time.monotonic() + 45
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            rpc("tools/list", {}, timeout=5)
            return
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"GPU-Lab MCP did not become ready: {last_error}")


def call(name: str, arguments: dict, allow_error: bool = False):
    value = rpc("tools/call", {"name": name, "arguments": arguments})
    if isinstance(value, dict) and value.get("error") and not allow_error:
        raise RuntimeError({"tool": name, "error": value["error"]})
    return value


def main() -> None:
    wait_for_gateway()
    tools = {item["name"]: item for item in rpc("tools/list", {})["tools"]}
    required = {
        "research_null_model_create",
        "research_null_model_test",
        "research_decision_outcome_assess",
        "research_strategy_list",
        "research_strategy_dataset_export",
        "research_benchmark_compare",
    }
    assert required <= set(tools)
    assert tools["research_decision_outcome_assess"]["annotations"]["readOnlyHint"] is False
    assert tools["research_strategy_list"]["annotations"]["readOnlyHint"] is True

    nonce = uuid.uuid4().hex[:10]
    project = call(
        "research_project_create",
        {
            "name": f"Brain v2 MCP smoke {nonce}",
            "question": "Does a frozen diagnostic distinguish the active mechanisms?",
        },
    )
    project_id = project["project_id"]
    model = call(
        "world_model_create",
        {"project_id": project_id, "name": "v2 smoke model", "scope": "fixture"},
    )
    agenda = call("research_agenda_create", {"project_id": project_id, "name": "v2 smoke agenda"})
    agenda_item = call(
        "research_agenda_item_create",
        {
            "agenda_id": agenda["id"],
            "question": "Does a frozen diagnostic distinguish the active mechanisms?",
            "importance": 5,
            "uncertainty": 5,
            "scientific_scope": "fixture",
            "candidate_experiments": [
                {
                    "action_type": "FROZEN_DIAGNOSTIC",
                    "score": {
                        "scientific_importance": 5,
                        "expected_discrimination": 5,
                        "expected_information_gain": 5,
                        "feasibility": 5,
                        "compute_cost": 1,
                        "engineering_cost": 1,
                        "execution_risk": 1,
                    },
                }
            ],
        },
    )
    hypothesis = call(
        "hypothesis_create",
        {
            "project_id": project_id,
            "mechanism": "The intermediate state carries the failure signal.",
            "prediction": "Frozen state substitution changes the carrier.",
            "kill_condition": "The carrier is unchanged.",
        },
    )
    null_model = call(
        "research_null_model_create",
        {
            "project_id": project_id,
            "null_model": {
                "target_entity_id": hypothesis["id"],
                "name": "Magnitude-matched random substitution",
                "mechanism": "Perturbation magnitude alone explains the output shift.",
                "why_plausible": "The target intervention changes norm and state identity.",
                "discriminating_control": "Match perturbation norm while randomizing identity.",
                "expected_outcome": "The random control mimics the target if magnitude is causal.",
                "estimated_cost": 0.5,
                "strength": "STRONG",
            },
        },
    )
    decision = call("brain_step", {"project_id": project_id})
    assert decision["research_situation"]
    assert decision["selected_action"]["action_type"] in {
        "NULL_MODEL_TEST",
        "MAGNITUDE_MATCHED_CONTROL",
        "ARTIFACT_ANALYSIS",
        "REPRODUCTION",
    }
    null_status = call(
        "research_null_model_test",
        {
            "null_model_id": null_model["id"],
            "outcome": "ELIMINATED",
            "evidence_family_ids": ["00000000-0000-0000-0000-000000000001"],
            "rationale": "Fixture endpoint intentionally verifies validation before evidence.",
        },
        allow_error=True,
    )
    assert null_status["error"]["type"] == "INVALID_EVIDENCE_FAMILY"
    outcome = call(
        "research_decision_outcome_assess",
        {
            "decision_id": decision["decision_id"],
            "domain": "mcp-contract-smoke",
            "assessment": {
                "label": "UNKNOWN",
                "observed_result": {
                    "mcp_contract_verified": True,
                    "scientific_result": "NOT_EXECUTED",
                },
                "realized_information_gain": "UNKNOWN",
                "hindsight_assessment": (
                    "The MCP planning contract was verified; no scientific experiment was run."
                ),
            },
        },
    )
    assert outcome["strategy_patterns"] == []
    export = call("research_strategy_dataset_export", {"project_id": project_id})
    assert export["dataset_version"] == "brain-v2-strategy-dataset-v1"
    assert export["record_count"] == 1
    comparison = call("research_benchmark_compare", {})
    assert comparison["episode_count"] >= 3
    print(
        {
            "verification": "VERIFIED_INTEGRATION",
            "project_id": project_id,
            "world_model_id": model["world_model"]["id"],
            "agenda_item_id": agenda_item["id"],
            "decision_id": decision["decision_id"],
            "dataset_records": export["record_count"],
            "benchmark_episodes": comparison["episode_count"],
            "null_validation_error": null_status["error"]["type"],
        }
    )


if __name__ == "__main__":
    main()
