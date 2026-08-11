import uuid

import pytest

from gpu_lab.branches import ExperimentBranchService
from gpu_lab.errors import GPUError


class FakeStore:
    def __init__(self):
        self.items = []
        self.edges = []

    def object_create(self, project_id, kind, data, event_type, status="ACTIVE"):
        item = {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "kind": kind,
            "status": status,
            "data": data,
            "event_type": event_type,
        }
        self.items.append(item)
        return item

    def object_get(self, object_id):
        return next(item for item in self.items if item["id"] == object_id)

    def object_update(self, object_id, data_update, status, event_type):
        item = self.object_get(object_id)
        item["data"] = {**item["data"], **data_update}
        item["status"] = status
        item["event_type"] = event_type
        return item

    def objects_list(self, project_id, kind=None, statuses=None, limit=100, data_filters=None):
        rows = [
            item
            for item in self.items
            if item["project_id"] == project_id
            and (kind is None or item["kind"] == kind)
            and (not statuses or item["status"] in statuses)
            and all(item["data"].get(key) == value for key, value in (data_filters or {}).items())
        ]
        return rows if limit is None else rows[:limit]

    def edge_create(self, source, target, relation):
        self.edges.append((source, target, relation))

    def experiment_branch_node_create(
        self, project_id, branch_id, data, parent_node_id, experiment_id
    ):
        node = self.object_create(
            project_id, "ExperimentNode", data, "EXPERIMENT_NODE_CREATED", "PLANNED"
        )
        self.edges.append((branch_id, node["id"], "CONTAINS_NODE"))
        relation_id = None
        if parent_node_id:
            relation = self.object_create(
                project_id,
                "BranchRelation",
                {
                    "branch_id": branch_id,
                    "source_node_id": parent_node_id,
                    "target_node_id": node["id"],
                    "relation": "BRANCHES_TO",
                },
                "BRANCH_RELATION_CREATED",
            )
            relation_id = relation["id"]
            self.edges.append((parent_node_id, node["id"], "BRANCHES_TO"))
        return {**node, "relation_id": relation_id}

    def comparative_lesson_create(
        self, project_id, branch_id, node_a_id, node_b_id, data
    ):
        lesson = self.object_create(
            project_id,
            "ComparativeLesson",
            data,
            "COMPARATIVE_LESSON_CREATED",
            "RESULT_INSPECTED",
        )
        relation = self.object_create(
            project_id,
            "BranchRelation",
            {
                "branch_id": branch_id,
                "source_node_id": node_a_id,
                "target_node_id": node_b_id,
                "relation": "COMPARED_WITH",
                "comparative_lesson_id": lesson["id"],
            },
            "BRANCH_RELATION_CREATED",
        )
        return {**lesson, "relation_id": relation["id"]}


def node_draft(experiment_id, action, information, compute=1.0):
    return {
        "branch_action": action,
        "description": f"Run the discriminating {action} experiment",
        "predicted_outcomes": {"pass": "carrier changes", "fail": "carrier unchanged"},
        "scientific_importance": 5,
        "expected_discrimination": 5,
        "expected_information_gain": information,
        "feasibility": 5,
        "compute_cost": compute,
        "engineering_cost": 1,
        "execution_risk": 1,
        "experiment_id": experiment_id,
    }


def setup_branch():
    store = FakeStore()
    service = ExperimentBranchService(store)
    hypothesis = store.object_create(
        "project", "Hypothesis", {"mechanism": "state propagation"}, "HYPOTHESIS_CREATED"
    )
    branch = service.create(
        "project", hypothesis["id"], "Discriminate anchor causality", {"gpu_hours": 1}
    )
    experiment_a = store.object_create(
        "project", "Experiment", {"frozen": True}, "EXPERIMENT_PREREGISTERED"
    )
    experiment_b = store.object_create(
        "project", "Experiment", {"frozen": True}, "EXPERIMENT_PREREGISTERED"
    )
    return store, service, branch, experiment_a, experiment_b


def test_branch_selects_highest_information_per_cost_and_persists_relation():
    _store, service, branch, experiment_a, experiment_b = setup_branch()
    low = service.node_add(
        branch["id"], node_draft(experiment_a["id"], "large correlation study", 2, compute=4)
    )
    high = service.node_add(
        branch["id"],
        node_draft(experiment_b["id"], "state substitution", 5),
        parent_node_id=low["id"],
    )

    selected = service.next_action(branch["id"])
    recovered = service.get(branch["id"])

    assert selected["action"] == "EXECUTE_BRANCH_NODE"
    assert selected["node_id"] == high["id"]
    assert selected["priority_score"] > low["data"]["priority_score"]
    assert high["relation_id"] is not None
    assert recovered["relations"][0]["data"]["relation"] == "BRANCHES_TO"


def test_branch_prioritizes_inspection_and_recovery_before_new_execution():
    store, service, branch, experiment_a, experiment_b = setup_branch()
    first = service.node_add(branch["id"], node_draft(experiment_a["id"], "state substitution", 5))
    second = service.node_add(branch["id"], node_draft(experiment_b["id"], "path block", 4))
    running = store.object_create(
        "project",
        "ExperimentRun",
        {"experiment_id": experiment_a["id"]},
        "EXPERIMENT_STARTED",
        "RUNNING",
    )

    recover = service.next_action(branch["id"])
    inspectable = store.object_create(
        "project",
        "ExperimentRun",
        {"experiment_id": experiment_b["id"]},
        "EXPERIMENT_STARTED",
        "RESULT_NOT_INSPECTED",
    )
    inspect = service.next_action(branch["id"])

    assert recover["action"] == "RECOVER_UNFINISHED"
    assert recover["node_id"] == first["id"]
    assert recover["run_id"] == running["id"]
    assert inspect["action"] == "INSPECT_RESULT"
    assert inspect["node_id"] == second["id"]
    assert inspect["run_id"] == inspectable["id"]


def test_result_gate_and_comparative_lesson_preserve_confounded_interpretation():
    store, service, branch, experiment_a, experiment_b = setup_branch()
    node_a = service.node_add(branch["id"], node_draft(experiment_a["id"], "state swap", 5))
    node_b = service.node_add(branch["id"], node_draft(experiment_b["id"], "path block", 4))
    run_a = store.object_create(
        "project",
        "ExperimentRun",
        {"experiment_id": experiment_a["id"]},
        "EXPERIMENT_STARTED",
        "completed",
    )
    with pytest.raises(GPUError) as error:
        service.record_result(node_a["id"], run_a["id"], {"metric": 1}, "Changed", {}, "HIGH")
    assert error.value.error_type == "BRANCH_RESULT_NOT_INSPECTED"

    run_a["status"] = "RESULT_INSPECTED"
    run_b = store.object_create(
        "project",
        "ExperimentRun",
        {"experiment_id": experiment_b["id"]},
        "EXPERIMENT_STARTED",
        "RESULT_INSPECTED",
    )
    for node, run, metric in ((node_a, run_a, 1.0), (node_b, run_b, 0.0)):
        service.record_result(
            node["id"], run["id"], {"carrier_change": metric}, "Inspected result", {"gpu_hours": 0.1}, "HIGH"
        )

    assert service.next_action(branch["id"])["action"] == "COMPARE_RESULTS"
    lesson = service.compare(
        branch["id"],
        node_a["id"],
        node_b["id"],
        {
            "code_delta": {},
            "config_delta": {},
            "state_delta": {"intervention": ["swap", "block"]},
            "data_delta": {},
            "metric_delta": {"carrier_change": 1.0},
            "shared_conditions": ["same checkpoint", "same samples"],
            "candidate_causal_difference": "The intervention location changed the carrier response",
            "scope": "VRCNet frozen inference",
            "confidence": "MEDIUM",
            "remaining_confounds": ["intervention magnitude differs"],
        },
    )

    assert lesson["data"]["remaining_confounds"]
    assert lesson["data"]["warning"].endswith("it is not proof.")
    assert service.next_action(branch["id"])["action"] == "BRANCH_COMPLETE"


def test_branch_rejects_refuted_hypothesis_and_nonfinite_scores():
    store = FakeStore()
    service = ExperimentBranchService(store)
    dead = store.object_create(
        "project", "Hypothesis", {}, "HYPOTHESIS_REFUTED", "REFUTED"
    )
    with pytest.raises(GPUError) as error:
        service.create("project", dead["id"], "Test it again", {"gpu_hours": 1})
    assert error.value.error_type == "BRANCH_HYPOTHESIS_NOT_ACTIVE"

    live = store.object_create("project", "Hypothesis", {}, "HYPOTHESIS_CREATED")
    branch = service.create("project", live["id"], "Test it", {"gpu_hours": 1})
    experiment = store.object_create("project", "Experiment", {}, "EXPERIMENT_CREATED")
    with pytest.raises(GPUError) as error:
        service.node_add(
            branch["id"],
            node_draft(experiment["id"], "bad score", float("nan")),
        )
    assert error.value.error_type == "INVALID_EXPERIMENT_BRANCH_NODE"
