from gpu_lab.brain import ResearchBrain


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
