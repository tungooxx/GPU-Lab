import pytest

from gpu_lab.brain import ResearchBrain
from gpu_lab.errors import GPUError


class TechnicalAssessmentStore:
    def __init__(self):
        self.applied = None
        self.objects = {
            "run": {
                "id": "run",
                "project_id": "project",
                "kind": "ExperimentRun",
                "status": "failed",
                "data": {"decision_id": "decision", "experiment_id": "experiment"},
            },
            "decision": {
                "id": "decision",
                "project_id": "project",
                "kind": "ResearchDecision",
                "status": "COMPLETED",
                "data": {"agenda_item_id": "agenda", "hypotheses_affected": ["hypothesis"]},
            },
            "hypothesis": {
                "id": "hypothesis",
                "project_id": "project",
                "kind": "Hypothesis",
                "status": "ACTIVE",
                "data": {},
            },
            "agenda": {
                "id": "agenda",
                "project_id": "project",
                "kind": "AgendaItem",
                "status": "ACTIVE",
                "data": {},
            },
            "experiment": {
                "id": "experiment",
                "project_id": "project",
                "kind": "Experiment",
                "status": "PREREGISTERED",
                "data": {
                    "hypothesis_id": "hypothesis",
                    "plan": {"pass_condition": "metric > 0"},
                },
            },
        }

    def object_get(self, object_id):
        return self.objects[object_id]

    def objects_list(self, *_args, **_kwargs):
        return [{"data": {"pass_condition": "metric > 0"}}]

    def technical_result_inspection_apply(self, **kwargs):
        self.applied = kwargs
        return {"run": {"status": "failed"}, "technical_non_scientific": True}


def test_failed_technical_result_uses_non_scientific_inspection_with_basis():
    store = TechnicalAssessmentStore()

    result = ResearchBrain(store).result_assess(
        run_id="run",
        decision_id="decision",
        hypothesis_id="hypothesis",
        agenda_item_id="agenda",
        prediction_outcome="INVALID_TECHNICAL_VALIDITY_GUARD_POLARITY",
        guard_condition_outcome="Guard not reached",
        condition_evaluations={"metric > 0": False},
        evidence_supporting=[],
        evidence_against=[],
        unexpected_observations=[],
        alternative_explanations=[],
        scope="repair validation",
        hypothesis_transition="INCONCLUSIVE",
        rationale="The guard polarity was corrected before a canonical retry.",
        actual_information_gain="INVALID",
        information_gain_basis=["Prevented a technical bug from becoming scientific evidence."],
    )

    assert result["verification_status"] == "TECHNICAL_FAILURE_INSPECTED"
    assert store.applied["actual_information_gain"] == "INVALID"
    assert store.applied["information_gain_basis"] == [
        "Prevented a technical bug from becoming scientific evidence."
    ]


def test_completed_process_with_failed_reset_attestation_cannot_be_scientifically_assessed():
    store = TechnicalAssessmentStore()
    store.objects["run"]["status"] = "completed"
    store.objects["run"]["data"]["execution_attestation"] = {
        "technical_status": "FAIL",
        "stages": {
            "runtime_initialized": True,
            "environment_reset": False,
            "initial_observation_received": False,
            "planner_called": False,
            "plan_parsed": False,
            "executor_started": False,
            "environment_actions_executed": 0,
            "scientific_eligibility_evaluated": False,
            "scientific_metric_evaluated": False,
        },
        "technical_errors": [
            {"stage": "environment_reset", "type": "NameError", "message": "name 'r' is not defined"}
        ],
    }

    with pytest.raises(GPUError) as error:
        ResearchBrain(store).result_assess(
            run_id="run", decision_id="decision", hypothesis_id="hypothesis", agenda_item_id="agenda",
            prediction_outcome="INCONCLUSIVE_INSUFFICIENT_NATURAL_TRAJECTORIES",
            guard_condition_outcome="zero qualified", condition_evaluations={"metric > 0": False},
            evidence_supporting=[], evidence_against=[], unexpected_observations=[], alternative_explanations=[],
            scope="fixture", hypothesis_transition="INCONCLUSIVE", rationale="fixture",
        )
    assert error.value.error_type == "SCIENTIFIC_OUTCOME_INVALID"
