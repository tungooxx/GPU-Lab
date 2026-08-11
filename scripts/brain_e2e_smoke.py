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


def read_json_artifact(job_id: str, path: str) -> dict:
    artifact = call_tool("local_artifact_read", {"job_id": job_id, "path": path})
    if artifact.get("truncated"):
        raise AssertionError(f"Artifact was truncated: {path}")
    return json.loads(artifact["content"])


def main() -> None:
    wait_for_server()
    nonce = uuid.uuid4().hex[:10]
    vrc_working_directory = "/workspace/local-vlm/external/VRCNet"
    baseline_command = (
        "set -e\n"
        "mkdir -p \"$GPU_LAB_JOB_DIR/artifacts\"\n"
        "python - <<'PY'\n"
        "import json, os, sys, numpy as np, torch\n"
        "sys.path.insert(0, '../../scripts')\n"
        "from run_hasi1_hierarchical_splices import state, out\n"
        "from run_stage6k_decoder_entry_mediation import ARRAYS, load_model\n"
        "assert torch.cuda.is_available()\n"
        "model=load_model(); uid='novel_0148'; view=5; root=ARRAYS/uid\n"
        "inp=np.load(root/f'view_{view:02d}_input.npy').astype('float32')\n"
        "saved=np.load(root/f'view_{view:02d}_pred.npy').astype('float32')\n"
        "current=out(state(model, inp)[0].fine).astype('float32')\n"
        "result={'native_reconstruction_maxabs': float(np.abs(current-saved).max()), "
        "'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}\n"
        "with open(os.path.join(os.environ['GPU_LAB_JOB_DIR'],'artifacts','baseline.json'),'w') as h: json.dump(result,h)\n"
        "print(json.dumps(result))\n"
        "PY"
    )
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
            "repository": vrc_working_directory,
            "commit": "smoke-fixture",
            "dataset": "saved prediction fixture",
            "checkpoint": "canonical VRC runtime",
            "evaluation_command": baseline_command,
            "reported_metric": {"name": "native_reconstruction_maxabs", "value": 0.0},
            "tolerance": 0.0,
        },
    )
    call_tool(
        "reproduction_run",
        {
            "reproduction_id": reproduction["id"],
            "python_env": "vrc-py313-torch260-cu124",
            "working_directory": vrc_working_directory,
            "env": {"PYTHONPATH": "/opt/gpu-lab/envs/vrc-analysis-deps"},
            "command": baseline_command,
        },
    )
    reproduction_run = wait_for_reproduction(reproduction["id"])
    if reproduction_run["status"] != "PARTIAL":
        raise AssertionError(reproduction_run)
    baseline = read_json_artifact(
        reproduction_run["data"]["job_id"], "artifacts/baseline.json"
    )
    if baseline["gpu"] != "NVIDIA GeForce GTX 1650":
        raise AssertionError(baseline)
    reproduced = call_tool(
        "reproduction_compare",
        {
            "reproduction_id": reproduction["id"],
            "observed_metric": baseline["native_reconstruction_maxabs"],
        },
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
                "prediction": "Frozen VRCNet hierarchical state substitution changes the recipient output",
                "alternative_hypotheses": [alternative["id"]],
                "intervention": "Run the frozen HASI hierarchical-state splices",
                "control": "Native saved VRCNet predictions for the same fixture views",
                "primary_metric": "max_recipient_effect_R",
                "secondary_metrics": ["native_reconstruction_maxabs", "GPU identity"],
                "expected_direction": "positive",
                "pass_condition": "native_reconstruction_maxabs == 0 and max_recipient_effect_R > 0 on NVIDIA GeForce GTX 1650",
                "fail_condition": "native reconstruction differs, no recipient effect is measured, or GPU identity differs",
                "interpretation_if_pass": "Supports the scoped causal edge in this fixture only",
                "interpretation_if_fail": "Weakens the scoped causal edge",
                "estimated_runtime_minutes": 1,
                "estimated_gpu_cost_usd": 0,
            },
        },
    )
    attempt = str(uuid.uuid4())
    experiment_command = (
        "set -e\n"
        "mkdir -p \"$GPU_LAB_JOB_DIR/artifacts\"\n"
        "python - <<'PY'\n"
        "import json, os, sys, torch\n"
        "sys.path.insert(0, '../../scripts')\n"
        "from run_hasi1_hierarchical_splices import load_model, runpair\n"
        "assert torch.cuda.is_available()\n"
        "rows, states, _ = runpair(load_model(), 'novel_0148', 5, 18, 'strict_residual')\n"
        "result={'status':'completed','fixture':'novel_0148:5->18',"
        "'native_reconstruction_maxabs':max(float(row['native_reconstruction_maxabs']) for row in rows),"
        "'max_recipient_effect_R':max(float(row['R']) for row in rows),"
        "'rows':len(rows),'state_rows':len(states),'gpu':torch.cuda.get_device_name(0)}\n"
        "with open(os.path.join(os.environ['GPU_LAB_JOB_DIR'],'artifacts','result.json'),'w') as h: json.dump(result,h)\n"
        "print(json.dumps(result))\n"
        "PY"
    )
    execution_decision = call_tool(
        "research_decision_create",
        {
            "project_id": project_id,
            "experiment_id": plan["id"],
            "command": experiment_command,
            "working_directory": vrc_working_directory,
            "env": {"PYTHONPATH": "/opt/gpu-lab/envs/vrc-analysis-deps"},
            "python_env": "vrc-py313-torch260-cu124",
        },
    )
    if execution_decision.get("execution_binding_error"):
        raise AssertionError(execution_decision)
    execution_arguments = {
        "experiment_id": plan["id"],
        "decision_id": execution_decision["decision_id"],
        "execution_attempt_uuid": attempt,
        "python_env": "vrc-py313-torch260-cu124",
        "working_directory": vrc_working_directory,
        "env": {"PYTHONPATH": "/opt/gpu-lab/envs/vrc-analysis-deps"},
        "command": experiment_command,
    }
    rejected = call_tool("research_experiment_execute", execution_arguments)
    if rejected.get("authorization_error", {}).get("type") != "RESEARCH_APPROVAL_REQUIRED":
        raise AssertionError("High-impact execution was not blocked before human approval")
    call_tool(
        "brain_decision_approve",
        {
            "decision_id": execution_decision["decision_id"],
            "approver": "brain-e2e-smoke",
            "rationale": "Authorize the preregistered low-cost frozen VRCNet intervention.",
        },
    )
    execution = call_tool(
        "research_experiment_execute",
        execution_arguments,
    )
    repeated = call_tool(
        "research_experiment_execute",
        execution_arguments,
    )
    if (execution["run_id"], execution["job_id"]) != (
        repeated["run_id"],
        repeated["job_id"],
    ):
        raise AssertionError("Retry did not preserve canonical execution mapping")
    finished = wait_for_experiment(execution["job_id"])
    if finished["status"] != "completed":
        raise AssertionError(finished)
    experiment_result = read_json_artifact(execution["job_id"], "artifacts/result.json")
    if experiment_result["gpu"] != "NVIDIA GeForce GTX 1650":
        raise AssertionError(experiment_result)
    if experiment_result["native_reconstruction_maxabs"] != 0.0:
        raise AssertionError(experiment_result)
    if experiment_result["max_recipient_effect_R"] <= 0:
        raise AssertionError(experiment_result)

    inspect_decision = call_tool("brain_step", {"project_id": project_id})
    if inspect_decision["selected_action"]["payload"].get("mode") != "INSPECT_RESULT":
        raise AssertionError("Brain did not prioritize the uninspected result")
    assessment = call_tool(
        "brain_result_assess",
        {
            "run_id": execution["run_id"],
            "decision_id": execution_decision["decision_id"],
            "hypothesis_id": hypothesis["id"],
            "agenda_item_id": causal_item["id"],
            "prediction_outcome": "Frozen HASI state substitution changed the recipient output",
            "guard_condition_outcome": "Measured from the persisted result.json artifact",
            "condition_evaluations": {
                "native_reconstruction_maxabs == 0 and max_recipient_effect_R > 0 on NVIDIA GeForce GTX 1650": True
            },
            "evidence_supporting": [
                f"result.json max_recipient_effect_R={experiment_result['max_recipient_effect_R']}"
            ],
            "evidence_against": [],
            "unexpected_observations": [],
            "alternative_explanations": ["The fixture does not prove dataset generalization"],
            "scope": {
                "description": "VRCNet CUDA intervention smoke fixture",
                "models": ["VRCNet"],
                "architectures": ["VRCNet"],
                "checkpoints": ["canonical VRC checkpoint"],
                "datasets": ["Brain v1 smoke fixture"],
                "objects": ["synthetic recipient fixture"],
                "interventions": ["frozen anchor-state substitution"],
                "metrics": ["max_recipient_effect_R"],
            },
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
                "approval_gate_verified": True,
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
