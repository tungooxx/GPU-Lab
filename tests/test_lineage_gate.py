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


def test_lineage_audit_accepts_reviewed_record_id_aliases():
    service = HypothesisQDService(Store())
    candidate = {
        "mechanism": "whole task success uses twelve action gate",
        "enabling_method": "variable slot control",
        "mechanistic_hypothesis": "state representation changes action selection",
        "falsified_prerequisites": ["twelve action gate is required"],
        "lineage_responses": [
            {"experiment_or_record": "negative-1", "implication_for_candidate": "Do not retain the gate.", "candidate_already_addresses_it": True},
            {"id": "run-1", "implication_for_candidate": "Use representation controls.", "candidate_already_addresses_it": True},
        ],
    }
    assert service.hypothesis_lineage_audit("project", candidate)["passed"] is True


class DecisionRootStore:
    def __init__(self, with_root=True, rootless=False):
        self.records = []
        if with_root:
            self.records.append({
                "id": "hypothesis-root", "project_id": "project", "kind": "Hypothesis", "status": "REFUTED",
                "data": {"mechanism": "triangular parity dislocation closes after observed mismatch"},
            })
        decision_data = {"question": "triangular parity dislocation closes after observed mismatch"}
        if not rootless:
            decision_data["hypotheses_affected"] = ["hypothesis-root"]
        self.records.append({
            "id": "decision-derived", "project_id": "project", "kind": "ResearchDecision", "status": "COMPLETED",
            "data": decision_data,
        })

    def objects_list(self, project_id, kind, limit=None):
        return [item for item in self.records if item["project_id"] == project_id and item["kind"] == kind]


def _parity_candidate(responses=None):
    return {
        "mechanism": "triangular parity dislocation closes after observed mismatch",
        "enabling_method": "synchronized direct versus synthetic quote comparison",
        "mechanistic_hypothesis": "temporary executable parity gaps close after fully observed quote mismatch",
        "lineage_responses": responses or [],
    }


def test_derived_research_decision_is_satisfied_by_explicit_scientific_root():
    service = HypothesisQDService(DecisionRootStore())
    audit = service.hypothesis_lineage_audit("project", _parity_candidate([
        {"record_id": "hypothesis-root", "implication_for_candidate": "Do not reuse the refuted formulation.", "candidate_already_addresses_it": True}
    ]))
    assert audit["passed"] is True
    assert [item["experiment_or_record"] for item in audit["cross_lineage_synthesis"]] == ["hypothesis-root"]


def test_omitted_material_scientific_root_still_blocks():
    audit = HypothesisQDService(DecisionRootStore()).hypothesis_lineage_audit("project", _parity_candidate())
    assert audit["passed"] is False
    assert "DISCOVERY_LINEAGE_INCOMPLETE" in audit["blockers"]
    assert [item["experiment_or_record"] for item in audit["cross_lineage_synthesis"]] == ["hypothesis-root"]


def test_rootless_research_decision_remains_mandatory():
    audit = HypothesisQDService(DecisionRootStore(with_root=False, rootless=True)).hypothesis_lineage_audit("project", _parity_candidate())
    assert audit["passed"] is False
    assert [item["experiment_or_record"] for item in audit["cross_lineage_synthesis"]] == ["decision-derived"]
