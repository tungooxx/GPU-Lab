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
        implementation_guards=[{"name": "native-off"}],
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
    assert missing.value.error_type == "ENGINEERING_GUARD_MISSING"

    result = service.result_record(task["id"], {
        "implementation_verification": "VERIFIED_TARGETED",
        "implementation_guard_results": [{"name": "native-off", "passed": True}],
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


def test_declared_guard_must_be_present_and_well_formed():
    service = EngineeringService(Store())
    task = _task(service)
    with pytest.raises(GPUError) as missing:
        service.result_record(task["id"], {"scientific_invariant_results": [
            {"name": "anchor_state", "passed": True}, {"name": "seed", "passed": True},
            {"name": "checkpoint", "passed": True},
        ]})
    assert missing.value.error_type == "ENGINEERING_GUARD_MISSING"
    with pytest.raises(GPUError) as malformed:
        service.result_record(task["id"], {
            "implementation_guard_results": [{"name": "native-off"}],
            "scientific_invariant_results": [
                {"name": "anchor_state", "passed": True}, {"name": "seed", "passed": True},
                {"name": "checkpoint", "passed": True},
            ],
        })
    assert malformed.value.error_type == "ENGINEERING_GUARD_RESULT_INVALID"


def test_unverified_task_cannot_be_handed_to_scientific_execution():
    service = EngineeringService(Store())
    task = _task(service)
    with pytest.raises(GPUError) as blocked:
        service.assert_ready_for_experiment(task["id"], "experiment-1")
    assert blocked.value.error_type == "ENGINEERING_RESULT_REQUIRED"


def test_measurement_guards_detect_native_regression_noop_and_held_fixed_drift():
    service = EngineeringService(Store())
    task = service.task_create(
        "project", "Frozen intervention", "SCIENTIFIC_INTERVENTION_IMPLEMENTATION",
        scientific_invariants={"must_change": ["target"], "held_fixed": ["seed"]},
        implementation_guards=[
            {"name": "native-off", "type": "NATIVE_OFF", "tolerance": 0.0},
            {"name": "target-on", "type": "TARGET_CHANGED", "minimum_delta": 0.01},
            {"name": "seed-fixed", "type": "HELD_FIXED", "tolerance": 0.0},
        ],
    )
    result = service.result_record(task["id"], {
        "implementation_verification": "VERIFIED_REAL_EXECUTION",
        "guard_measurements": [
            {"name": "native-off", "before": 1.0, "after": 1.0},
            {"name": "target-on", "before": 0.0, "after": 0.0},
            {"name": "seed-fixed", "before": "a", "after": "b"},
        ],
        "scientific_invariant_results": [
            {"name": "target", "passed": True}, {"name": "seed", "passed": False},
        ],
    })
    guards = result["data"]["implementation_guard_results"]
    assert guards[0]["passed"] is True
    assert guards[1]["passed"] is False  # A no-op intervention is invalid.
    assert guards[2]["passed"] is False
    assert result["data"]["implementation_verification"] == "INVALID_IMPLEMENTATION"


def test_measurement_guard_requires_before_and_after_values():
    service = EngineeringService(Store())
    task = service.task_create(
        "project", "Check native path", "EXPERIMENT_INSTRUMENTATION",
        implementation_guards=[{"name": "native-off", "type": "NATIVE_OFF"}],
    )
    with pytest.raises(GPUError) as invalid:
        service.result_record(task["id"], {"guard_measurements": [{"name": "native-off", "before": 0.0}]})
    assert invalid.value.error_type == "ENGINEERING_GUARD_MEASUREMENT_INVALID"


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
        "implementation_guard_results": [{"name": "native-off", "passed": True}],
        "scientific_invariant_results": [
            {"name": "anchor_state", "passed": True}, {"name": "seed", "passed": True},
            {"name": "checkpoint", "passed": True},
        ],
    })
    verified = service.task_verify(task["id"])
    assert verified["result_id"] == result["id"]
    assert verified["ready_for_scientific_execution"] is True
    assert verified["scientific_result"] == "NOT_ASSESSED"
