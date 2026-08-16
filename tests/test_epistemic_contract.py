from gpu_lab.research import classify_decision_data, strategy_learning_eligibility
from gpu_lab.strategy import ResearchStrategyService


def _outcome(**updates):
    return {
        "status": "RESULT_INSPECTED",
        "data": {
            "realized_information_gain": "HIGH",
            "hindsight_assessment": "The scoped intervention separated the hypotheses.",
            "information_gain_basis": ["UNCERTAINTY_RESOLVED"],
            **updates,
        },
    }


def test_real_gpu_system_smoke_is_not_scientific_strategy():
    decision = {
        "data": {
            "decision_role": "SYSTEM_VERIFICATION",
            "execution_verification": "REAL_GPU",
            "scientific_role": "SYSTEM_SMOKE",
            "cycle_status": "CLOSED",
        }
    }
    classification = classify_decision_data(decision["data"])
    eligibility = strategy_learning_eligibility(decision, _outcome())

    assert classification["execution_verification"] == "REAL_GPU"
    assert classification["scientific_role"] == "SYSTEM_SMOKE"
    assert eligibility["eligible"] is False
    assert "DECISION_ROLE_NOT_SCIENTIFIC_ACTION" in eligibility["exclusions"]


def test_legacy_backfill_and_incomplete_cycles_are_excluded():
    legacy = {
        "data": {
            "legacy_provenance": {"reconstructed": True},
            "cycle_status": "CLOSED",
        }
    }
    incomplete = {
        "data": {
            "decision_role": "SCIENTIFIC_ACTION",
            "scientific_role": "CAUSAL_TEST",
            "cycle_status": "SELECTED",
        }
    }

    assert classify_decision_data(legacy["data"])["decision_role"] == "LEGACY_BACKFILL"
    assert strategy_learning_eligibility(legacy, _outcome())["eligible"] is False
    assert strategy_learning_eligibility(incomplete, None)["eligible"] is False


def test_closed_scientific_cycle_with_basis_is_eligible():
    decision = {
        "data": {
            "decision_role": "SCIENTIFIC_ACTION",
            "scientific_role": "CAUSAL_TEST",
            "learning_namespace": "PRODUCTION_SCIENCE",
            "cycle_status": "CLOSED",
        }
    }

    result = strategy_learning_eligibility(decision, _outcome())

    assert result["eligible"] is True
    assert result["exclusions"] == []


def test_missing_decision_context_fails_closed():
    result = strategy_learning_eligibility({}, _outcome())

    assert result["eligible"] is False
    assert result["exclusions"] == ["DECISION_CONTEXT_MISSING"]


class _AuditStore:
    def object_get(self, object_id):
        assert object_id == "decision-1"
        return {
            "id": object_id,
            "project_id": "project-1",
            "kind": "ResearchDecision",
            "data": {
                "decision_role": "SCIENTIFIC_ACTION",
                "scientific_role": "CAUSAL_TEST",
                "execution_verification": "REAL_GPU",
                "scientific_verification": "RESULT_INSPECTED",
                "cycle_status": "CLOSED",
                "runner_up_candidate_index": 1,
                "selected_action": {"hypotheses_discriminated": ["h1", "h2"]},
                "hindsight_assessment": "The intervention narrowed the scoped claim.",
            },
        }

    def objects_list(self, project_id, kind, **kwargs):
        assert project_id == "project-1"
        assert kind == "ResearchDecisionOutcome"
        return [
            {
                "status": "RESULT_INSPECTED",
                "data": {
                    "decision_id": "decision-1",
                    "realized_information_gain": "HIGH",
                    "hindsight_assessment": "The intervention narrowed the scoped claim.",
                    "information_gain_basis": ["UNCERTAINTY_RESOLVED"],
                    "scope": {"models": ["fixture"]},
                },
            }
        ]


def test_decision_epistemic_audit_reports_scientific_contract():
    result = ResearchStrategyService(_AuditStore()).decision_epistemic_audit("decision-1")

    assert result["is_scientific"] is True
    assert result["runner_up_compared"] is True
    assert result["hypotheses_distinguished"] is True
    assert result["hindsight_present"] is True
    assert result["realized_information_basis"] == ["UNCERTAINTY_RESOLVED"]
    assert result["scope_present"] is True
