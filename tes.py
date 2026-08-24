"""Three-branch GPU test scaffold.

Prereqs:
- GPU_LAB_MCP_URL points at the live MCP endpoint
- real execution is approved before the experiment is launched
- you already have a project, hypothesis, and preregistered question
"""

from __future__ import annotations

import json
import os
import urllib.request
import uuid
from typing import Any

MCP_URL = os.environ.get("GPU_LAB_MCP_URL", "http://127.0.0.1:8000/mcp")


def rpc(method: str, params: dict[str, Any], timeout: int = 30) -> Any:
    req = urllib.request.Request(
        MCP_URL,
        data=json.dumps(
            {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}
        ).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        env = json.loads(resp.read())
    if "error" in env:
        raise RuntimeError(env["error"])
    result = env.get("result", {})
    structured = result.get("structuredContent", {})
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]
    content = result.get("content", [])
    text = content[0].get("text", "") if content and isinstance(content[0], dict) else ""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            {
                "type": "MCP_NON_JSON_RESPONSE",
                "message": text or "MCP returned no readable content",
                "raw_result": result,
            }
        ) from exc


def call(name: str, arguments: dict[str, Any]) -> Any:
    print(f"CALL {name}", flush=True)
    result = rpc("tools/call", {"name": name, "arguments": arguments})
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(result["error"])
    return result


def main() -> None:
    print(f"MCP endpoint: {MCP_URL}", flush=True)
    project = call(
        "research_project_create",
        {
            "name": f"three-branch-gpu-{uuid.uuid4().hex[:8]}",
            "question": "Which branch best disambiguates the carrier mechanism?",
        },
    )
    project_id = project["project_id"]

    hypothesis = call(
        "hypothesis_create",
        {
            "project_id": project_id,
            "mechanism": "The intermediate state carries the failure signal.",
            "prediction": "A frozen state substitution changes the carrier metric.",
            "kill_condition": "The carrier metric is unchanged under substitution.",
        },
    )
    hypothesis_id = hypothesis["id"]

    # 1) Make the comparison branch
    branch = call(
        "experiment_branch_create",
        {
            "project_id": project_id,
            "hypothesis_id": hypothesis_id,
            "objective": "Choose the cheapest discriminating causal test",
            "budget": {"gpu_hours": 0.25, "max_experiments": 3},
        },
    )
    branch_id = branch["id"]

    # 2) Register three candidate experiments
    exps = [
        {
            "name": "branch_a",
            "action": "STATIC_CORRELATION",
            "metric": "correlation_delta",
            "discrimination": 2,
            "info": 2,
            "cost": 1,
        },
        {
            "name": "branch_b",
            "action": "STATE_SUBSTITUTION",
            "metric": "carrier_delta",
            "discrimination": 5,
            "info": 5,
            "cost": 1,
        },
        {
            "name": "branch_c",
            "action": "NULL_MODEL_TEST",
            "metric": "null_delta",
            "discrimination": 4,
            "info": 4,
            "cost": 1,
        },
    ]

    nodes = []
    for exp in exps:
        plan = call(
            "experiment_plan_register",
            {
                "project_id": project_id,
                "hypothesis_id": hypothesis_id,
                "plan": {
                    "research_question": "Which intervention best distinguishes the mechanisms?",
                    "prediction": f"{exp['action']} changes {exp['metric']}",
                    "alternative_hypotheses": ["Correlation is non-causal"],
                    "intervention": exp["action"],
                    "control": "Frozen baseline",
                    "primary_metric": exp["metric"],
                    "secondary_metrics": ["guard_maxabs"],
                    "expected_direction": "increase",
                    "pass_condition": f"{exp['metric']} > 0",
                    "fail_condition": f"{exp['metric']} == 0",
                    "interpretation_if_pass": "Supports the scoped mechanism",
                    "interpretation_if_fail": "Weakens the scoped mechanism",
                    "estimated_runtime_minutes": 1,
                    "estimated_gpu_cost_usd": 0,
                },
            },
        )

        node = call(
            "experiment_branch_node_add",
            {
                "branch_id": branch_id,
                "draft": {
                    "branch_action": exp["action"],
                    "description": f"Candidate {exp['name']} intervention",
                    "predicted_outcomes": {"pass": "metric changes", "fail": "metric unchanged"},
                    "scientific_importance": 5,
                    "expected_discrimination": exp["discrimination"],
                    "expected_information_gain": exp["info"],
                    "feasibility": 5,
                    "compute_cost": exp["cost"],
                    "engineering_cost": 1,
                    "execution_risk": 1,
                    "experiment_id": plan["id"],
                },
            },
        )
        nodes.append(node)

    # 3) Ask the branch policy which one to execute first
    choice = call("experiment_branch_next", {"branch_id": branch_id})
    print("NEXT_BRANCH:", json.dumps(choice, indent=2), flush=True)

    # 4) Do not record placeholder results. After real preregistration, approval,
    # execution, and inspection, call experiment_branch_result_record with the
    # canonical run_id and measured result. Then compare two inspected nodes with
    # experiment_branch_compare.

    print(
        json.dumps(
            {"project_id": project_id, "branch_id": branch_id, "nodes": [n["id"] for n in nodes]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
