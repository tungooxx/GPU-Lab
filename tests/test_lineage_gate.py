from gpu_lab.qd import HypothesisQDService


class Store:
    def __init__(self):
        self.records = [
            {
                "id": "negative-1", "project_id": "project", "kind": "NegativeResult", "status": "REFUTED",
                "data": {"proposal": "whole task success requires twelve action gate", "failed_assumption": "twelve action gate is required"},
            },
            {
                "id": "run-1", "project_id": "project", "kind": "ExperimentRun", "status": "COMPLETED",
                "data": {"summary": "representation intervention changed action selection"},
            },
        ]

    def objects_list(self, project_id, kind, limit=None):
        return [item for item in self.records if item["project_id"] == project_id and item["kind"] == kind]


def test_lineage_audit_requires_response_to_related_failed_assumption():
    service = HypothesisQDService(Store())
    candidate = {
        "mechanism": "whole task success uses twelve action gate", "enabling_method": "variable slot control",
        "mechanistic_hypothesis": "state representation changes action selection", "parent_ids": [],
    }
    blocked = service.hypothesis_lineage_audit("project", candidate)
    assert blocked["passed"] is False
    assert "DISCOVERY_LINEAGE_INCOMPLETE" in blocked["blockers"]
    assert "DISCOVERY_DEAD_ASSUMPTION_UNDECLARED" in blocked["blockers"]

    passed = service.hypothesis_lineage_audit("project", {
        **candidate,
        "falsified_prerequisites": ["twelve action gate is required"],
        "lineage_responses": [
            {"record_id": "negative-1", "implication_for_candidate": "Do not retain the gate.", "candidate_already_addresses_it": True},
            {"record_id": "run-1", "implication_for_candidate": "Use the representation intervention as the control.", "candidate_addresses": True},
        ],
    })
    assert passed["passed"] is True
    assert passed["cross_lineage_synthesis"][0]["candidate_already_addresses_it"] is True
    assert passed["cross_lineage_synthesis"][0]["explicit_lineage"] is True


def test_lineage_audit_does_not_make_unrelated_history_a_prerequisite():
    service = HypothesisQDService(Store())
    candidate = {
        "mechanism": "London-session liquidity mechanism uses a distinct trigger.",
        "enabling_method": "session feature basis",
        "mechanistic_hypothesis": "London liquidity changes entry selection",
        "lineage_responses": [
            {"record_id": "negative-1", "implication_for_candidate": "Rules out the twelve-action gate.",
             "candidate_already_addresses_it": True, "how_addressed": "Uses a distinct trigger."},
            {"record_id": "run-1", "implication_for_candidate": "Retain representation controls.",
             "candidate_already_addresses_it": True},
        ],
    }
    audit = service.hypothesis_lineage_audit("project", candidate)
    assert audit["passed"] is True
    assert audit["falsified_prerequisites_still_relevant"] == []
