"""Live branch smoke: MCP/PostgreSQL scoring, typed relations, and restart durability."""

import json
import subprocess
import uuid

from qd_e2e_smoke import REPO_ROOT, call, rpc, wait_for_gateway


def main() -> None:
    wait_for_gateway()
    tools = {item["name"] for item in rpc("tools/list", {})["tools"]}
    required = {
        "experiment_branch_create",
        "experiment_branch_node_add",
        "experiment_branch_get",
        "experiment_branch_next",
        "experiment_branch_result_record",
        "experiment_branch_compare",
    }
    assert required <= tools
    nonce = uuid.uuid4().hex[:10]
    project = call(
        "research_project_create",
        {"name": f"Branch live smoke {nonce}", "question": "Which causal test comes first?"},
    )
    project_id = project["project_id"]
    niche = call(
        "hypothesis_niche_create",
        {
            "project_id": project_id,
            "name": "state propagation",
            "description": "Intermediate state transmission mechanisms",
            "diversity_signature": {"family": "state propagation"},
        },
    )
    hypothesis = call(
        "hypothesis_qd_create",
        {
            "project_id": project_id,
            "draft": {
                "mechanism": "Anchor state causally transmits evidence into the output carrier",
                "prediction": "A frozen anchor-state substitution changes the carrier",
                "kill_condition": "Substitution leaves the carrier unchanged",
                "niche_id": niche["id"],
                "assumptions": ["saved baseline is exactly reproduced"],
                "variables": ["anchor_state", "carrier"],
                "information_path": ["evidence", "anchor_state", "carrier"],
                "scope": "VRCNet frozen inference",
            },
        },
    )

    def experiment(intervention: str, metric: str):
        return call(
            "experiment_plan_register",
            {
                "project_id": project_id,
                "hypothesis_id": hypothesis["id"],
                "plan": {
                    "research_question": "Does the intervention change the carrier?",
                    "prediction": f"{intervention} changes {metric}",
                    "alternative_hypotheses": ["Observed correlation is non-causal"],
                    "intervention": intervention,
                    "control": "Frozen baseline inference",
                    "primary_metric": metric,
                    "secondary_metrics": ["guard_maxabs"],
                    "expected_direction": "increase",
                    "pass_condition": f"{metric} > 0",
                    "fail_condition": f"{metric} == 0",
                    "interpretation_if_pass": "Prediction survives this scoped test",
                    "interpretation_if_fail": "Weaken the scoped mechanism",
                    "estimated_runtime_minutes": 1,
                    "estimated_gpu_cost_usd": 0,
                },
            },
        )

    correlation = experiment("Static correlation measurement", "correlation_delta")
    substitution = experiment("Frozen anchor-state substitution", "carrier_delta")
    branch = call(
        "experiment_branch_create",
        {
            "project_id": project_id,
            "hypothesis_id": hypothesis["id"],
            "objective": "Choose the cheapest discriminating causal test",
            "budget": {"gpu_hours": 0.1, "max_experiments": 2},
        },
    )

    def draft(experiment_id: str, action: str, discrimination: float, info: float, cost: float):
        return {
            "branch_action": action,
            "description": f"Run {action} under the frozen baseline",
            "predicted_outcomes": {"pass": "metric changes", "fail": "metric unchanged"},
            "scientific_importance": 5,
            "expected_discrimination": discrimination,
            "expected_information_gain": info,
            "feasibility": 5,
            "compute_cost": cost,
            "engineering_cost": 1,
            "execution_risk": 1,
            "experiment_id": experiment_id,
        }

    weak = call(
        "experiment_branch_node_add",
        {
            "branch_id": branch["id"],
            "draft": draft(correlation["id"], "STATIC_CORRELATION", 2, 2, 2),
        },
    )
    strong = call(
        "experiment_branch_node_add",
        {
            "branch_id": branch["id"],
            "parent_node_id": weak["id"],
            "draft": draft(substitution["id"], "STATE_SUBSTITUTION", 5, 5, 1),
        },
    )
    selected = call("experiment_branch_next", {"branch_id": branch["id"]})
    assert selected["node_id"] == strong["id"]
    assert selected["action"] == "EXECUTE_BRANCH_NODE"

    subprocess.run(
        ["docker", "compose", "restart", "postgres", "gpu-lab"],
        check=True,
        cwd=REPO_ROOT,
    )
    wait_for_gateway()
    recovered = call("experiment_branch_get", {"branch_id": branch["id"]})
    assert len(recovered["nodes"]) == 2
    assert len(recovered["relations"]) == 1
    print(
        json.dumps(
            {
                "project_id": project_id,
                "branch_id": branch["id"],
                "selected_node_id": strong["id"],
                "selected_action": "STATE_SUBSTITUTION",
                "restart_recovered": True,
                "scientific_result": "NOT_EXECUTED",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
