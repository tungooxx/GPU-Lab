import os
import time

import pytest

from gpu_lab.research import ResearchStore
from gpu_lab.strategy import ResearchStrategyService

TEST_DATABASE_URL = os.getenv("GPU_LAB_TEST_DATABASE_URL")


class StrategyStore:
    def __init__(self, patterns=None, decisions=None, outcomes=None):
        self.patterns = patterns or []
        self.decisions = decisions or []
        self.outcomes = outcomes or []

    def objects_global_list(self, kind, *_args, **_kwargs):
        if kind == "ResearchStrategyPattern":
            return self.patterns
        if kind == "ResearchSituation":
            return []
        if kind == "ResearchDecision":
            return self.decisions
        if kind == "ResearchDecisionOutcome":
            return self.outcomes
        return []

    def objects_list(self, _project_id, kind, *_args, **_kwargs):
        if kind == "ResearchDecision":
            return self.decisions
        if kind == "ResearchDecisionOutcome":
            return self.outcomes
        return []


def _situation(**updates):
    return {
        "domain": "point-cloud completion",
        "research_stage": "MECHANISM_TESTING",
        "phenomenon_type": "representation failure",
        "uncertainty_type": "MECHANISM",
        "mechanism_status": "HYPOTHESIZED_CAUSAL",
        "baseline_reproduced": True,
        "active_hypothesis_count": 2,
        "dead_related_count": 1,
        "internal_state_access": True,
        "strong_null_available": False,
        "uninspected_result_available": False,
        "contradiction_present": False,
        "anomaly_present": True,
        "scope_stage": "WITHIN_SCOPE",
        "available_action_types": ["FROZEN_DIAGNOSTIC", "LITERATURE_SEARCH"],
        "compute_budget_class": "LOW",
        "engineering_cost_class": "LOW",
        "world_model_signature": "world",
        "dominant_confounds": [],
        "prior_failed_strategies": [],
        "situation_signature": "situation",
        **updates,
    }


def _pattern(scope_level="DOMAIN"):
    return {
        "id": "pattern-a",
        "project_id": "project-a",
        "kind": "ResearchStrategyPattern",
        "status": "ACTIVE",
        "data": {
            "scope_level": scope_level,
            "action_type": "FROZEN_DIAGNOSTIC",
            "support_level": "DOMAIN_REPEATED",
            "projects_observed": ["project-a", "project-b"],
            "domains_observed": ["point-cloud completion"],
            "problem_signature": "mechanism-testing",
            "research_stage": "MECHANISM_TESTING",
            "conditions": {},
            "applicability_conditions": {
                "research_stage": "MECHANISM_TESTING",
                "mechanism_status": "HYPOTHESIZED_CAUSAL",
                "baseline_reproduced": True,
                "internal_state_access": True,
                "strong_null_available": False,
                "scope_stage": "WITHIN_SCOPE",
            },
            "counterexamples": [],
            "historical_successes": 2,
            "historical_failures": 0,
            "decision_ids": ["decision-a"],
            "outcome_ids": ["outcome-a"],
        },
    }


def test_positive_transfer_retrieves_structurally_applicable_domain_pattern():
    service = ResearchStrategyService(StrategyStore(patterns=[_pattern()]))

    retrieved = service.retrieve("project-b", _situation())

    assert [item["id"] for item in retrieved["applied"]] == ["pattern-a"]
    assert retrieved["applied"][0]["applicability"] == "HIGH"


def test_negative_transfer_is_blocked_by_structured_access_mismatch_not_similarity():
    service = ResearchStrategyService(StrategyStore(patterns=[_pattern()]))

    retrieved = service.retrieve("project-c", _situation(internal_state_access=False))

    assert retrieved["applied"] == []
    assert retrieved["rejected"][0]["id"] == "pattern-a"
    assert "INTERNAL_STATE_ACCESS_MISMATCH" in retrieved["rejected"][0]["mismatch_reasons"]


def test_diminishing_returns_penalizes_repeated_cheap_action_and_rewards_decisive_one():
    service = ResearchStrategyService(StrategyStore())
    candidates = [
        {
            "action_type": "FROZEN_DIAGNOSTIC",
            "priority": 10.0,
            "score": {"compute_cost": 0.5, "expected_discrimination": 2},
        },
        {
            "action_type": "CAUSAL_INTERVENTION",
            "priority": 9.0,
            "score": {"compute_cost": 2, "expected_discrimination": 5},
        },
    ]
    retrieval = {
        "applied": [
            {
                "id": "pattern-a",
                "action_type": "FROZEN_DIAGNOSTIC",
                "applicability": "HIGH",
                "historical_successes": 1,
                "historical_failures": 0,
                "counterexamples": [],
            }
        ],
        "rejected": [],
    }

    adjusted = service.adjust_candidates(
        candidates,
        retrieval,
        {"flag": "DIMINISHING_RETURNS"},
        hard_gate=False,
    )

    assert adjusted[0]["diminishing_return_adjustment"] < 0
    assert adjusted[1]["diminishing_return_adjustment"] > 0
    assert adjusted[1]["final_priority"] > adjusted[0]["final_priority"]


def test_hard_gate_ignores_strategy_adjustments():
    service = ResearchStrategyService(StrategyStore())
    candidate = {
        "action_type": "REPRODUCTION",
        "priority": 5.0,
        "score": {"compute_cost": 1, "expected_discrimination": 5},
    }

    adjusted = service.adjust_candidates(
        [candidate],
        {
            "applied": [
                {
                    "id": "pattern",
                    "action_type": "REPRODUCTION",
                    "applicability": "HIGH",
                    "historical_successes": 4,
                    "historical_failures": 4,
                    "counterexamples": [],
                }
            ],
            "rejected": [],
        },
        {"flag": "DIMINISHING_RETURNS"},
        hard_gate=True,
    )

    assert adjusted[0]["priority"] == 5.0
    assert adjusted[0]["hard_gate_preserved"] is True


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_postgres_outcome_and_project_strategy_are_one_reassessable_transition():
    store = ResearchStore(TEST_DATABASE_URL)
    project = store.project_create(
        f"strategy-{time.time_ns()}", "Does a diagnostic improve research decisions?"
    )
    project_id = project["project_id"]
    decision = store.object_create(
        project_id,
        "ResearchDecision",
        {
            "agenda_item_id": "agenda-fixture",
            "selected_action": {"action_type": "FROZEN_DIAGNOSTIC"},
        },
        "RESEARCH_DECISION_SELECTED",
        "SELECTED",
    )
    outcome = {
        "project_id": project_id,
        "decision_id": decision["id"],
        "before_situation_id": "situation-before",
        "domain": "point-cloud completion",
        "problem_signature": "reproduced-internal-mechanism",
        "research_stage": "MECHANISM_TESTING",
        "conditions": {
            "research_stage": "MECHANISM_TESTING",
            "mechanism_status": "HYPOTHESIZED_CAUSAL",
            "baseline_reproduced": True,
            "internal_state_access": True,
            "strong_null_available": False,
            "scope_stage": "WITHIN_SCOPE",
        },
        "applicability_conditions": {
            "research_stage": "MECHANISM_TESTING",
            "mechanism_status": "HYPOTHESIZED_CAUSAL",
            "baseline_reproduced": True,
            "internal_state_access": True,
            "strong_null_available": False,
            "scope_stage": "WITHIN_SCOPE",
        },
        "action_type": "FROZEN_DIAGNOSTIC",
        "action_parameters_pattern": {"mode": "state-substitution"},
        "label": "HIGH_VALUE",
        "realized_information_gain": "HIGH",
        "hindsight_assessment": "The intervention eliminated a mechanism family.",
        "observed_result": {"guard_passed": True},
        "experiment_run_ids": [],
        "evidence_family_ids": [],
        "failure_conditions": [],
        "retrieved_strategy_pattern_ids": [],
    }
    after = _situation()

    first = store.decision_outcome_apply(decision["id"], outcome, after)
    revised = store.decision_outcome_apply(
        decision["id"], {**outcome, "label": "USEFUL", "realized_information_gain": "MEDIUM"}, after
    )

    persisted_outcomes = store.objects_list(
        project_id, "ResearchDecisionOutcome", limit=None
    )
    patterns = store.objects_list(project_id, "ResearchStrategyPattern", limit=None)
    assert first["outcome"]["id"] == revised["outcome"]["id"]
    assert len(persisted_outcomes) == 1
    assert persisted_outcomes[0]["data"]["assessment_history"]
    project_patterns = [
        item for item in patterns if item["data"]["scope_level"] == "PROJECT"
    ]
    assert len(project_patterns) == 1
    assert project_patterns[0]["data"]["support_level"] == "PROJECT_OBSERVATION"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_postgres_strategy_promotes_project_to_domain_then_global_only_with_scope_evidence():
    store = ResearchStore(TEST_DATABASE_URL)
    nonce = str(time.time_ns())
    signature = f"shared-cross-domain-signature-{nonce}"

    def record(domain: str, label: str = "HIGH_VALUE"):
        project = store.project_create(
            f"promotion-{domain}-{time.time_ns()}", "Which research action is useful?"
        )
        decision = store.object_create(
            project["project_id"],
            "ResearchDecision",
            {
                "agenda_item_id": "agenda-fixture",
                "selected_action": {"action_type": "FROZEN_DIAGNOSTIC"},
            },
            "RESEARCH_DECISION_SELECTED",
            "SELECTED",
        )
        outcome = {
            "project_id": project["project_id"],
            "decision_id": decision["id"],
            "before_situation_id": "situation-before",
            "domain": domain,
            "problem_signature": signature,
            "research_stage": "MECHANISM_TESTING",
            "conditions": {
                "research_stage": "MECHANISM_TESTING",
                "mechanism_status": "HYPOTHESIZED_CAUSAL",
                "baseline_reproduced": True,
                "internal_state_access": True,
                "strong_null_available": False,
                "scope_stage": "WITHIN_SCOPE",
            },
            "applicability_conditions": {
                "research_stage": "MECHANISM_TESTING",
                "mechanism_status": "HYPOTHESIZED_CAUSAL",
                "baseline_reproduced": True,
                "internal_state_access": True,
                "strong_null_available": False,
                "scope_stage": "WITHIN_SCOPE",
            },
            "action_type": "FROZEN_DIAGNOSTIC",
            "action_parameters_pattern": {},
            "label": label,
            "realized_information_gain": "HIGH" if label == "HIGH_VALUE" else "ZERO",
            "hindsight_assessment": "Fixture outcome with explicit scope.",
            "observed_result": {"fixture": True},
            "experiment_run_ids": [],
            "evidence_family_ids": [],
            "failure_conditions": ["internal state absent"] if label != "HIGH_VALUE" else [],
            "retrieved_strategy_pattern_ids": [],
        }
        return store.decision_outcome_apply(decision["id"], outcome, _situation(domain=domain))

    first = record(f"domain-a-{nonce}")
    assert {item["data"]["scope_level"] for item in first["strategy_patterns"]} == {"PROJECT"}

    second = record(f"domain-a-{nonce}")
    assert any(
        item["data"]["scope_level"] == "DOMAIN"
        and item["data"]["support_level"] == "DOMAIN_REPEATED"
        for item in second["strategy_patterns"]
    )

    record(f"domain-b-{nonce}")
    record(f"domain-b-{nonce}")
    fifth = record(f"domain-c-{nonce}")
    global_candidate = next(
        item for item in fifth["strategy_patterns"] if item["data"]["scope_level"] == "GLOBAL"
    )
    assert global_candidate["data"]["support_level"] == "GLOBAL_SUPPORTED"
    assert len(global_candidate["data"]["projects_observed"]) == 5
    assert len(global_candidate["data"]["domains_observed"]) == 3
