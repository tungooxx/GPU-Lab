"""Real Brain v1 smoke: PostgreSQL state -> real GPU evidence -> changed decision."""

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

MCP_URL = os.environ.get("GPU_LAB_MCP_URL", "http://127.0.0.1:8000/mcp")


def wait_for_server() -> None:
    health_url = MCP_URL.removesuffix("/mcp") + "/health"
    deadline = time.monotonic() + 45
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"GPU-Lab did not become healthy: {last_error}")


def call_tool(name: str, arguments: dict[str, Any], timeout: int = 90) -> Any:
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    request = urllib.request.Request(
        MCP_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "gpu-lab-brain-e2e/1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        envelope = json.loads(response.read())
    if "error" in envelope:
        raise RuntimeError({"tool": name, "transport_error": envelope["error"]})
    result = envelope["result"].get("structuredContent", {}).get("result")
    if result is None:
        result = json.loads(envelope["result"]["content"][0]["text"])
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError({"tool": name, "tool_error": result["error"]})
    return result


def wait_for_reproduction(reproduction_id: str) -> dict:
    for _ in range(40):
        result = call_tool("reproduction_sync", {"reproduction_id": reproduction_id})
        if result["status"] in {"PARTIAL", "FAILED", "REPRODUCED"}:
            return result
        time.sleep(1)
    raise TimeoutError("Reproduction did not reach a terminal execution state")


def wait_for_experiment(job_id: str) -> dict:
    for _ in range(60):
        result = call_tool("research_experiment_sync", {"job_id": job_id})
        if result["status"] in {"completed", "failed", "cancelled"}:
            return result
        time.sleep(1)
    raise TimeoutError("Experiment did not reach a terminal execution state")


def main() -> None:
    wait_for_server()
    nonce = uuid.uuid4().hex[:10]
    project = call_tool(
        "research_project_create",
        {
            "name": f"Brain v1 real smoke {nonce}",
            "question": "Does anchor-state substitution causally alter the failure carrier?",
        },
    )
    project_id = project["project_id"]

    paper = call_tool(
        "paper_ingest",
        {
            "project_id": project_id,
            "title": "Canonical VRCNet baseline",
            "url": "https://github.com/paul007pl/VRCNet",
            "version": "smoke-fixture",
            "card": {"purpose": "Executable baseline gate for Brain v1"},
        },
    )
    reproduction = call_tool(
        "reproduction_prepare",
        {
            "project_id": project_id,
            "paper_id": paper["id"],
            "repository": "/workspace/local-vlm/external/VRCNet",
            "commit": "smoke-fixture",
            "dataset": "saved prediction fixture",
            "checkpoint": "canonical VRC runtime",
            "evaluation_command": "CUDA availability and deterministic metric smoke",
            "reported_metric": {"name": "baseline_metric", "value": 1.0},
            "tolerance": 0.0,
        },
    )
    call_tool(
        "reproduction_run",
        {
            "reproduction_id": reproduction["id"],
            "python_env": "vrc-py313-torch260-cu124",
            "command": (
                "python -c \"import torch; assert torch.cuda.is_available(); "
                "x=torch.ones(1, device='cuda'); print('baseline_metric=', float(x.item()))\""
            ),
        },
    )
    reproduction_run = wait_for_reproduction(reproduction["id"])
    if reproduction_run["status"] != "PARTIAL":
        raise AssertionError(reproduction_run)
    reproduced = call_tool(
        "reproduction_compare",
        {"reproduction_id": reproduction["id"], "observed_metric": 1.0},
    )
    if reproduced["status"] != "REPRODUCED":
        raise AssertionError(reproduced)

    dead = call_tool(
        "negative_result_create",
        {
            "project_id": project_id,
            "proposal": "Static output correlation identifies the anchor state carrier mechanism",
            "prediction": "Output correlations isolate the causal carrier",
            "result": "Correlations failed to distinguish causal mechanisms",
            "failed_assumption": "Static output association is equivalent to internal causality",
            "revisit_condition": "Only revisit with an internal intervention",
        },
    )
    hypothesis = call_tool(
        "hypothesis_create",
        {
            "project_id": project_id,
            "mechanism": "Static output correlation anchor state carrier mechanism tested by internal intervention",
            "prediction": "Replacing the anchor state changes the downstream carrier",
            "kill_condition": "State replacement leaves the carrier unchanged",
            "scientific_difference": "Uses internal state substitution rather than static association",
        },
    )
    alternative = call_tool(
        "hypothesis_create",
        {
            "project_id": project_id,
            "mechanism": "Decoder amplification independently creates the failure carrier",
            "prediction": "Anchor replacement has no effect but decoder ablation does",
            "kill_condition": "Anchor replacement changes the carrier while decoder state is fixed",
        },
    )
    call_tool("research_embedding_store", {"object_id": dead["id"], "embedding": [1, 0, 0]})
    call_tool(
        "research_embedding_store",
        {"object_id": hypothesis["id"], "embedding": [1, 0.01, 0]},
    )
    semantic_dead = call_tool(
        "research_semantic_search",
        {
            "project_id": project_id,
            "embedding": [1, 0.01, 0],
            "kind": "NegativeResult",
            "limit": 5,
        },
    )
    if not semantic_dead or semantic_dead[0]["id"] != dead["id"]:
        raise AssertionError("pgvector did not retrieve the related dead idea")

    anomaly = call_tool(
        "anomaly_create",
        {
            "project_id": project_id,
            "expected": "One architecture-general output carrier",
            "observed": "Heterogeneous carriers across views",
            "scope": "VRCNet saved prediction fixture",
            "priority": "high",
            "possible_explanations": [
                "anchor-state propagation",
                "decoder amplification",
            ],
        },
    )
    model_result = call_tool(
        "world_model_create",
        {
            "project_id": project_id,
            "name": "VRC failure carrier model",
            "scope": "VRCNet internal intervention fixture",
        },
    )
    world_model_id = model_result["world_model"]["id"]
    anchor = call_tool(
        "world_entity_create",
        {
            "world_model_id": world_model_id,
            "kind": "MechanismState",
            "name": "anchor_state",
            "description": "Intermediate anchor representation",
        },
    )["entity"]
    carrier = call_tool(
        "world_entity_create",
        {
            "world_model_id": world_model_id,
            "kind": "Phenomenon",
            "name": "failure_carrier",
            "description": "Observed heterogeneous failure carrier",
        },
    )["entity"]
    edge = call_tool(
        "causal_edge_create",
        {
            "world_model_id": world_model_id,
            "source_id": anchor["id"],
            "target_id": carrier["id"],
            "relation": "CAUSES",
            "status": "HYPOTHESIZED_CAUSAL",
            "supporting_ids": [anomaly["id"]],
        },
    )["edge"]

    agenda = call_tool(
        "research_agenda_create",
        {"project_id": project_id, "name": "VRC causal agenda"},
    )
    causal_item = call_tool(
        "research_agenda_item_create",
        {
            "agenda_id": agenda["id"],
            "question": "Does anchor_state causally change failure_carrier?",
            "importance": 5,
            "uncertainty": 5,
            "scientific_scope": "VRCNet saved prediction fixture",
            "blocking_hypothesis_ids": [hypothesis["id"], alternative["id"]],
            "related_anomaly_ids": [anomaly["id"]],
            "reproduction_required": True,
            "candidate_experiments": [
                {
                    "action_type": "CAUSAL_INTERVENTION",
                    "predicted_outcomes": ["State substitution changes the carrier"],
                    "required_resources": ["canonical VRC runtime", "GTX 1650"],
                    "payload": {"intervention": "anchor-state substitution"},
                    "score": {
                        "scientific_importance": 5,
                        "expected_discrimination": 5,
                        "expected_information_gain": 5,
                        "feasibility": 5,
                        "compute_cost": 0.5,
                        "engineering_cost": 0.5,
                        "execution_risk": 0.5,
                    },
                },
                {
                    "action_type": "TRAINING_RUN",
                    "predicted_outcomes": ["Training may improve the benchmark"],
                    "required_resources": ["GPU hours"],
                    "score": {
                        "scientific_importance": 3,
                        "expected_discrimination": 2,
                        "expected_information_gain": 2,
                        "feasibility": 2,
                        "compute_cost": 5,
                        "engineering_cost": 4,
                        "execution_risk": 3,
                    },
                },
            ],
        },
    )
    call_tool(
        "research_agenda_item_create",
        {
            "agenda_id": agenda["id"],
            "question": "Does the supported edge generalize beyond the fixture?",
            "importance": 3,
            "uncertainty": 4,
            "scientific_scope": "unseen VRCNet samples",
            "candidate_experiments": [
                {
                    "action_type": "GENERALIZATION",
                    "predicted_outcomes": ["The intervention effect survives on unseen samples"],
                    "required_resources": ["held-out samples"],
                    "score": {
                        "scientific_importance": 3,
                        "expected_discrimination": 3,
                        "expected_information_gain": 3,
                        "feasibility": 4,
                        "compute_cost": 1,
                        "engineering_cost": 1,
                        "execution_risk": 1,
                    },
                }
            ],
        },
    )

    before = call_tool("brain_step", {"project_id": project_id})
    if before["selected_action"]["action_type"] != "CAUSAL_INTERVENTION":
        raise AssertionError(before)
    if dead["id"] not in {item["id"] for item in before["dead_ideas_retrieved"]}:
        raise AssertionError("brain_step did not retrieve negative memory")

    plan = call_tool(
        "experiment_plan_register",
        {
            "project_id": project_id,
            "hypothesis_id": hypothesis["id"],
            "plan": {
                "research_question": causal_item["data"]["question"],
                "prediction": "CUDA state substitution fixture produces the preregistered effect",
                "alternative_hypotheses": [alternative["id"]],
                "intervention": "Run the frozen CUDA substitution fixture",
                "control": "Unmodified CUDA tensor",
                "primary_metric": "effect",
                "secondary_metrics": ["CUDA availability", "runtime identity"],
                "expected_direction": "positive",
                "pass_condition": "effect == 1.0 and CUDA is available",
                "fail_condition": "effect != 1.0 or CUDA is unavailable",
                "interpretation_if_pass": "Supports the scoped causal edge in this fixture only",
                "interpretation_if_fail": "Weakens the scoped causal edge",
                "estimated_runtime_minutes": 1,
                "estimated_gpu_cost_usd": 0,
            },
        },
    )
    attempt = str(uuid.uuid4())
    execution = call_tool(
        "research_experiment_execute",
        {
            "experiment_id": plan["id"],
            "execution_attempt_uuid": attempt,
            "python_env": "vrc-py313-torch260-cu124",
            "command": (
                "mkdir -p \"$GPU_LAB_JOB_DIR/artifacts\"\n"
                "python - <<'PY'\n"
                "import json, os, torch\n"
                "assert torch.cuda.is_available()\n"
                "control = torch.zeros(1, device='cuda')\n"
                "intervention = torch.ones(1, device='cuda')\n"
                "effect = float((intervention - control).item())\n"
                "result = {'effect': effect, 'gpu': torch.cuda.get_device_name(0)}\n"
                "path = os.path.join(os.environ['GPU_LAB_JOB_DIR'], 'artifacts', 'result.json')\n"
                "with open(path, 'w', encoding='utf-8') as handle: json.dump(result, handle)\n"
                "print(json.dumps(result))\n"
                "PY"
            ),
        },
    )
    repeated = call_tool(
        "research_experiment_execute",
        {
            "experiment_id": plan["id"],
            "execution_attempt_uuid": attempt,
            "python_env": "vrc-py313-torch260-cu124",
            "command": (
                "mkdir -p \"$GPU_LAB_JOB_DIR/artifacts\"\n"
                "python - <<'PY'\n"
                "import json, os, torch\n"
                "assert torch.cuda.is_available()\n"
                "control = torch.zeros(1, device='cuda')\n"
                "intervention = torch.ones(1, device='cuda')\n"
                "effect = float((intervention - control).item())\n"
                "result = {'effect': effect, 'gpu': torch.cuda.get_device_name(0)}\n"
                "path = os.path.join(os.environ['GPU_LAB_JOB_DIR'], 'artifacts', 'result.json')\n"
                "with open(path, 'w', encoding='utf-8') as handle: json.dump(result, handle)\n"
                "print(json.dumps(result))\n"
                "PY"
            ),
        },
    )
    if (execution["run_id"], execution["job_id"]) != (
        repeated["run_id"],
        repeated["job_id"],
    ):
        raise AssertionError("Retry did not preserve canonical execution mapping")
    finished = wait_for_experiment(execution["job_id"])
    if finished["status"] != "completed":
        raise AssertionError(finished)

    inspect_decision = call_tool("brain_step", {"project_id": project_id})
    if inspect_decision["selected_action"]["payload"].get("mode") != "INSPECT_RESULT":
        raise AssertionError("Brain did not prioritize the uninspected result")
    assessment = call_tool(
        "brain_result_assess",
        {
            "run_id": execution["run_id"],
            "decision_id": before["decision_id"],
            "hypothesis_id": hypothesis["id"],
            "agenda_item_id": causal_item["id"],
            "prediction_outcome": "effect == 1.0 on the preregistered CUDA fixture",
            "guard_condition_outcome": "PASSED: CUDA available; canonical runtime used",
            "evidence_supporting": ["result.json effect equals 1.0"],
            "evidence_against": [],
            "unexpected_observations": [],
            "alternative_explanations": ["The fixture does not prove dataset generalization"],
            "scope": "VRCNet CUDA intervention smoke fixture",
            "hypothesis_transition": "SURVIVES_INITIAL_TEST",
            "rationale": "The preregistered scoped prediction passed on real CUDA execution.",
            "causal_edge_id": edge["id"],
            "causal_edge_status": "INTERVENTION_SUPPORTED",
            "actual_information_gain": "HIGH",
        },
    )
    after = call_tool("brain_step", {"project_id": project_id})
    if after["selected_action"]["action_type"] != "GENERALIZATION":
        raise AssertionError(after)
    if before["selected_action"]["action_type"] == after["selected_action"]["action_type"]:
        raise AssertionError("Evidence did not change the next Brain decision")

    state = call_tool("research_state_get", {"project_id": project_id})
    events = call_tool("research_events", {"project_id": project_id, "limit": 100})
    model = call_tool("world_model_get", {"world_model_id": world_model_id})
    event_types = {event["event_type"] for event in events}
    required_events = {
        "RESEARCH_DECISION_SELECTED",
        "EXPERIMENT_RESULT_INSPECTED",
        "CAUSAL_EDGE_STATUS_CHANGED",
        "WORLD_MODEL_VERSION_CREATED",
        "RESEARCH_STATE_UPDATED",
    }
    if not required_events.issubset(event_types):
        raise AssertionError({"missing_events": sorted(required_events - event_types)})
    if model["edges"][0]["data"]["edge_status"] != "INTERVENTION_SUPPORTED":
        raise AssertionError("WorldModel edge did not update from evidence")

    print(
        json.dumps(
            {
                "project_id": project_id,
                "reproduction_status": reproduced["status"],
                "negative_memory_id": dead["id"],
                "semantic_negative_memory_retrieved": True,
                "decision_before": before["selected_action"]["action_type"],
                "uninspected_result_recovered": True,
                "run_id": execution["run_id"],
                "job_id": execution["job_id"],
                "gpu_logs": finished["run"]["data"].get("logs_tail"),
                "assessment_status": assessment["verification_status"],
                "world_model_edge_after": model["edges"][0]["data"]["edge_status"],
                "decision_after": after["selected_action"]["action_type"],
                "state_fact_count": len(state["canonical_state"]["established_facts"]),
                "verification_status": "VERIFIED_REAL",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
