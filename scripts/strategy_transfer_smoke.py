"""Demonstrate scoped Brain v2 strategy transfer on a PostgreSQL Research OS database.

This is deliberately a policy-memory smoke, not a scientific experiment. It creates
isolated fixture projects: A/D teach a domain-scoped diagnostic strategy; B retrieves
it under matching conditions; C is lexically similar but lacks internal-state access
and must reject transfer.
"""

import os
import time

from gpu_lab.research import ResearchStore
from gpu_lab.strategy import ResearchStrategyService


def _situation(domain: str, internal_state_access: bool = True) -> dict:
    return {
        "domain": domain,
        "research_stage": "MECHANISM_TESTING",
        "phenomenon_type": "representation failure",
        "uncertainty_type": "MECHANISM",
        "mechanism_status": "HYPOTHESIZED_CAUSAL",
        "baseline_reproduced": True,
        "active_hypothesis_count": 2,
        "dead_related_count": 1,
        "internal_state_access": internal_state_access,
        "strong_null_available": False,
        "uninspected_result_available": False,
        "contradiction_present": False,
        "anomaly_present": True,
        "scope_stage": "WITHIN_SCOPE",
        "available_action_types": ["FROZEN_DIAGNOSTIC", "LITERATURE_SEARCH"],
        "compute_budget_class": "LOW",
        "engineering_cost_class": "LOW",
        "world_model_signature": "smoke-world",
        "dominant_confounds": [],
        "prior_failed_strategies": [],
        "situation_signature": "smoke-situation",
    }


def _outcome(project_id: str, decision_id: str, domain: str) -> dict:
    conditions = {
        "research_stage": "MECHANISM_TESTING",
        "mechanism_status": "HYPOTHESIZED_CAUSAL",
        "baseline_reproduced": True,
        "internal_state_access": True,
        "strong_null_available": False,
        "scope_stage": "WITHIN_SCOPE",
    }
    return {
        "project_id": project_id,
        "decision_id": decision_id,
        "before_situation_id": "fixture-before",
        "domain": domain,
        "problem_signature": "smoke-reproduced-mechanism-testing",
        "research_stage": "MECHANISM_TESTING",
        "conditions": conditions,
        "applicability_conditions": conditions,
        "action_type": "FROZEN_DIAGNOSTIC",
        "action_parameters_pattern": {"mode": "state-substitution"},
        "label": "HIGH_VALUE",
        "realized_information_gain": "HIGH",
        "hindsight_assessment": "The frozen diagnostic eliminated a competing mechanism.",
        "observed_result": {"fixture": True},
        "experiment_run_ids": [],
        "evidence_family_ids": [],
        "failure_conditions": [],
        "retrieved_strategy_pattern_ids": [],
    }


def _teach(store: ResearchStore, domain: str, suffix: str) -> str:
    project = store.project_create(
        f"strategy-smoke-{suffix}-{time.time_ns()}",
        "Does a frozen state diagnostic discriminate the mechanism?",
    )
    decision = store.object_create(
        project["project_id"],
        "ResearchDecision",
        {
            "agenda_item_id": "fixture-agenda",
            "selected_action": {"action_type": "FROZEN_DIAGNOSTIC"},
        },
        "RESEARCH_DECISION_SELECTED",
        "SELECTED",
    )
    store.decision_outcome_apply(
        decision["id"],
        _outcome(project["project_id"], decision["id"], domain),
        _situation(domain),
    )
    return project["project_id"]


def main() -> None:
    database_url = os.environ.get("GPU_LAB_RESEARCH_DATABASE_URL")
    if not database_url:
        raise SystemExit("Set GPU_LAB_RESEARCH_DATABASE_URL before running this smoke.")
    store = ResearchStore(database_url)
    strategy = ResearchStrategyService(store)
    domain = "strategy-transfer-smoke-domain"
    project_a = _teach(store, domain, "a")
    project_d = _teach(store, domain, "d")

    matching = strategy.retrieve("project-b", _situation(domain))
    mismatch = strategy.retrieve("project-c", _situation(domain, internal_state_access=False))
    applied = [item for item in matching["applied"] if item["action_type"] == "FROZEN_DIAGNOSTIC"]
    rejected = [item for item in mismatch["rejected"] if item["action_type"] == "FROZEN_DIAGNOSTIC"]
    assert applied, "Project B did not retrieve the matching domain strategy"
    assert rejected, "Project C did not reject an inapplicable strategy"
    assert any(
        "INTERNAL_STATE_ACCESS_MISMATCH" in item["mismatch_reasons"] for item in rejected
    ), "Project C transfer was not blocked by structured applicability"
    print(
        {
            "verification": "VERIFIED_INTEGRATION",
            "teaching_projects": [project_a, project_d],
            "project_b_applied_strategy_ids": [item["id"] for item in applied],
            "project_c_rejected_strategy_ids": [item["id"] for item in rejected],
        }
    )


if __name__ == "__main__":
    main()
