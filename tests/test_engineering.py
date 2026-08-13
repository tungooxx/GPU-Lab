import pytest

from gpu_lab.engineering import EngineeringService
from gpu_lab.errors import GPUError


class Store:
    def __init__(self):
        self.items = {}
        self.events = []

    def object_create(self, project_id, kind, data, event_type, status="ACTIVE"):
        ident = f"{kind}-{len(self.items)}"
        item = {"id": ident, "project_id": project_id, "kind": kind, "data": data, "status": status}
        self.items[ident] = item
        self.events.append(event_type)
        return item

    def object_get(self, ident):
        return self.items[ident]

    def object_update(self, ident, update, status, event_type):
        item = self.items[ident]
        item["data"] = {**item["data"], **update}
        item["status"] = status
        self.events.append(event_type)
        return item


def _task(service):
    return service.task_create(
        "project", "Implement frozen state substitution", "SCIENTIFIC_INTERVENTION_IMPLEMENTATION",
        scientific_variable_changed="anchor_state",
        scientific_variables_held_fixed=["seed", "checkpoint"],
        scientific_invariants={"must_change": ["anchor_state"], "held_fixed": ["seed", "checkpoint"]},
    )


def test_verified_implementation_is_not_scientific_evidence():
    store = Store()
    service = EngineeringService(store)
    task = _task(service)

    result = service.result_record(task["id"], {
        "implementation_verification": "VERIFIED_REAL_EXECUTION",
        "scientific_invariant_results": [
            {"name": "anchor_state", "passed": True}, {"name": "seed", "passed": True},
            {"name": "checkpoint", "passed": True},
        ],
        "implementation_guard_results": [{"name": "native-off", "passed": True}],
    })

    assert result["data"]["scientific_result"] == "NOT_ASSESSED"
    assert store.items[task["id"]]["status"] == "COMPLETED"
    assert "ENGINEERING_VERIFIED" in store.events


def test_missing_or_failed_invariant_blocks_implementation_not_hypothesis():
    service = EngineeringService(Store())
    task = _task(service)
    with pytest.raises(GPUError) as missing:
        service.result_record(task["id"], {"scientific_invariant_results": []})
    assert missing.value.error_type == "SCIENTIFIC_INVARIANT_GUARD_MISSING"

    result = service.result_record(task["id"], {
        "implementation_verification": "VERIFIED_TARGETED",
        "scientific_invariant_results": [
            {"name": "anchor_state", "passed": False}, {"name": "seed", "passed": True},
            {"name": "checkpoint", "passed": True},
        ],
    })
    assert result["data"]["implementation_verification"] == "INVALID_IMPLEMENTATION"
    assert service.task_get(task["id"])["status"] == "BLOCKED"


def test_engineering_result_cannot_assess_scientific_truth():
    service = EngineeringService(Store())
    task = _task(service)
    with pytest.raises(GPUError) as forbidden:
        service.result_record(task["id"], {"scientific_result": "SUPPORTED"})
    assert forbidden.value.error_type == "ENGINEERING_SCIENTIFIC_RESULT_FORBIDDEN"


def test_context_is_compact_and_preserves_frozen_invariants():
    service = EngineeringService(Store())
    task = _task(service)
    context = service.context_get(task["id"])
    assert context["scientific_variable_changed"] == "anchor_state"
    assert context["scientific_invariants"]["held_fixed"] == ["seed", "checkpoint"]
    assert "hypotheses" not in context


def test_task_verify_exposes_readiness_without_scientific_assessment():
    service = EngineeringService(Store())
    task = _task(service)
    result = service.result_record(task["id"], {
        "implementation_verification": "VERIFIED_TARGETED",
        "scientific_invariant_results": [
            {"name": "anchor_state", "passed": True}, {"name": "seed", "passed": True},
            {"name": "checkpoint", "passed": True},
        ],
    })
    verified = service.task_verify(task["id"])
    assert verified["result_id"] == result["id"]
    assert verified["ready_for_scientific_execution"] is True
    assert verified["scientific_result"] == "NOT_ASSESSED"
