import pytest

from gpu_lab.engineering import CodingExecutionPolicy, EngineeringService
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
    task = service.task_create(
        "project", "Implement frozen state substitution", "SCIENTIFIC_INTERVENTION_IMPLEMENTATION",
        scientific_variable_changed="anchor_state",
        scientific_variables_held_fixed=["seed", "checkpoint"],
        scientific_invariants={"must_change": ["anchor_state"], "held_fixed": ["seed", "checkpoint"]},
        implementation_guards=[{"name": "native-off"}],
    )
    service.task_start(task["id"], {"files_read": ["src/model.py"]}, {"commands_run": ["pytest tests/test_model.py"], "passed": True})
    service.diff_review(task["id"], {
        "files_changed": ["src/model.py"], "diff_summary": "Scoped intervention hook.",
        "unrelated_changes": False, "scientific_variable_drift": False,
    })
    return task


def test_verified_implementation_is_not_scientific_evidence():
    store = Store()
    service = EngineeringService(store)
    task = _task(service)

    result = service.result_record(task["id"], {
        "implementation_verification": "VERIFIED_REAL_EXECUTION",
        "resulting_commit_or_diff_identity": "abc123",
        "commands_run": ["pytest tests/test_model.py"],
        "tests_run": ["tests/test_model.py"],
        "tests_passed": ["tests/test_model.py"],
        "artifacts": [{"path": "artifacts/native-off.json"}],
        "scientific_invariant_results": [
            {"name": "anchor_state", "passed": True}, {"name": "seed", "passed": True},
            {"name": "checkpoint", "passed": True},
        ],
        "implementation_guard_results": [{"name": "native-off", "passed": True}],
    })

    assert result["data"]["scientific_result"] == "NOT_ASSESSED"
    assert result["data"]["files_read"] == ["src/model.py"]
    assert result["data"]["files_changed"] == ["src/model.py"]
    assert result["data"]["baseline_result"]["passed"] is True
    assert result["data"]["resulting_commit_or_diff_identity"] == "abc123"
    assert result["data"]["engineering_status"] == "COMPLETED"
    assert store.items[task["id"]]["status"] == "COMPLETED"
    assert "ENGINEERING_VERIFIED" in store.events


def test_task_parent_links_are_project_and_kind_checked():
    store = Store()
    store.items.update({
        "decision": {"id": "decision", "project_id": "project", "kind": "ResearchDecision", "data": {}},
        "experiment": {"id": "experiment", "project_id": "project", "kind": "Experiment", "data": {}},
        "other": {"id": "other", "project_id": "other-project", "kind": "Experiment", "data": {}},
    })
    service = EngineeringService(store)
    task = service.task_create("project", "Linked implementation", "BUG_FIX", research_decision_id="decision", experiment_id="experiment")
    assert task["data"]["research_decision_id"] == "decision"
    with pytest.raises(GPUError) as mismatch:
        service.task_create("project", "Wrong project", "BUG_FIX", experiment_id="other")
    assert mismatch.value.error_type == "RESEARCH_PROJECT_MISMATCH"
    with pytest.raises(GPUError) as wrong_kind:
        service.task_create("project", "Wrong kind", "BUG_FIX", research_decision_id="experiment")
    assert wrong_kind.value.error_type == "ENGINEERING_RESEARCH_DECISION_KIND_INVALID"


def test_coding_policy_enforces_explore_to_handback_order():
    state = CodingExecutionPolicy.initial()
    for phase in CodingExecutionPolicy.phases[1:]:
        state = CodingExecutionPolicy.advance(state, phase)
    assert state["phase"] == "HAND_BACK"
    assert state["completed"][0] == "RECEIVE"
    with pytest.raises(GPUError) as skipped:
        CodingExecutionPolicy.advance(CodingExecutionPolicy.initial(), "EDIT")
    assert skipped.value.error_type == "ENGINEERING_POLICY_ORDER_VIOLATION"


def test_coding_policy_is_provider_neutral_and_scientifically_fail_closed():
    contract = CodingExecutionPolicy.contract()
    assert contract["scientific_result"] == "NOT_ASSESSED"
    assert not any(name.lower() in str(contract).lower() for name in ("claude", "codex", "openai"))


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


def test_result_rejects_unknown_engineering_status():
    service = EngineeringService(Store())
    task = _task(service)
    with pytest.raises(GPUError) as invalid:
        service.result_record(task["id"], {
            "engineering_status": "HYPOTHESIS_SUPPORTED",
            "implementation_guard_results": [{"name": "native-off", "passed": True}],
            "scientific_invariant_results": [
                {"name": "anchor_state", "passed": True}, {"name": "seed", "passed": True},
                {"name": "checkpoint", "passed": True},
            ],
        })
    assert invalid.value.error_type == "ENGINEERING_STATUS_INVALID"


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
    task["data"]["experiment_id"] = "experiment-1"
    with pytest.raises(GPUError) as blocked:
        service.assert_ready_for_experiment(task["id"], "experiment-1")
    assert blocked.value.error_type == "ENGINEERING_RESULT_REQUIRED"


def test_result_cannot_be_recorded_twice_after_completion():
    service = EngineeringService(Store())
    task = _task(service)
    payload = {
        "implementation_guard_results": [{"name": "native-off", "passed": True}],
        "scientific_invariant_results": [
            {"name": "anchor_state", "passed": True}, {"name": "seed", "passed": True},
            {"name": "checkpoint", "passed": True},
        ],
    }
    service.result_record(task["id"], payload)
    with pytest.raises(GPUError) as duplicate:
        service.result_record(task["id"], payload)
    assert duplicate.value.error_type == "ENGINEERING_RESULT_ALREADY_RECORDED"


def test_measurement_guard_rejects_non_numeric_threshold():
    service = EngineeringService(Store())
    task = service.task_create(
        "project", "Check native path", "EXPERIMENT_INSTRUMENTATION",
        implementation_guards=[{"name": "native-off", "type": "NATIVE_OFF", "tolerance": "nope"}],
    )
    service.task_start(task["id"], {"files_read": ["src/model.py"]}, {"commands_run": ["pytest"], "passed": True})
    service.diff_review(task["id"], {"files_changed": ["src/model.py"], "diff_summary": "Guard path.", "unrelated_changes": False, "scientific_variable_drift": False})
    with pytest.raises(GPUError) as invalid:
        service.result_record(task["id"], {"guard_measurements": [{"name": "native-off", "before": 0, "after": 0}]})
    assert invalid.value.error_type == "ENGINEERING_GUARD_MEASUREMENT_INVALID"


def test_task_start_requires_inspection_and_passing_baseline():
    service = EngineeringService(Store())
    task = service.task_create("project", "Inspect", "REPOSITORY_INSPECTION")
    with pytest.raises(GPUError) as no_files:
        service.task_start(task["id"], {}, {"commands_run": ["pytest"], "passed": True})
    assert no_files.value.error_type == "ENGINEERING_INSPECTION_REQUIRED"
    blocked = service.task_start(
        task["id"], {"files_read": ["src/model.py"]}, {"commands_run": ["pytest"], "passed": False}
    )
    assert blocked["status"] == "BLOCKED"
    assert service.task_get(task["id"])["data"]["baseline_verified"] is False


def test_task_start_preserves_inspected_file_hash_records():
    service = EngineeringService(Store())
    task = service.task_create("project", "Inspect", "REPOSITORY_INSPECTION")

    started = service.task_start(
        task["id"],
        {"files_read": [{"path": "src/model.py", "sha256": "a" * 64}]},
        {"commands_run": ["pytest -q"], "passed": True},
    )

    assert started["data"]["inspection"]["files_read"][0]["sha256"] == "a" * 64


def test_result_requires_task_start_evidence():
    service = EngineeringService(Store())
    task = service.task_create(
        "project", "Implement", "BUG_FIX", implementation_guards=[{"name": "smoke"}]
    )
    with pytest.raises(GPUError) as blocked:
        service.result_record(task["id"], {"implementation_guard_results": [{"name": "smoke", "passed": True}]})
    assert blocked.value.error_type == "ENGINEERING_INSPECTION_REQUIRED"


def test_result_requires_passing_diff_review_and_protects_frozen_design():
    service = EngineeringService(Store())
    task = service.task_create("project", "Implement", "BUG_FIX", implementation_guards=[{"name": "smoke"}])
    service.task_start(task["id"], {"files_read": ["src/x.py"]}, {"commands_run": ["pytest"], "passed": True})
    with pytest.raises(GPUError) as no_review:
        service.result_record(task["id"], {"implementation_guard_results": [{"name": "smoke", "passed": True}]})
    assert no_review.value.error_type == "ENGINEERING_DIFF_REVIEW_REQUIRED"
    with pytest.raises(GPUError) as frozen:
        service.task_update(task["id"], "ACTIVE", {"scientific_invariants": {"must_change": ["other"]}})
    assert frozen.value.error_type == "SCIENTIFIC_DESIGN_CHANGE_REQUIRED"
    review = service.diff_review(task["id"], {
        "files_changed": ["src/x.py"], "diff_summary": "Changed wrong state too.",
        "unrelated_changes": False, "scientific_variable_drift": True,
    })
    assert review["status"] == "BLOCKED"


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
    service.task_start(task["id"], {"files_read": ["src/model.py"]}, {"commands_run": ["pytest"], "passed": True})
    service.diff_review(task["id"], {"files_changed": ["src/model.py"], "diff_summary": "Scoped hook.", "unrelated_changes": False, "scientific_variable_drift": False})
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
    service.task_start(task["id"], {"files_read": ["src/model.py"]}, {"commands_run": ["pytest"], "passed": True})
    service.diff_review(task["id"], {"files_changed": ["src/model.py"], "diff_summary": "Guard path.", "unrelated_changes": False, "scientific_variable_drift": False})
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
