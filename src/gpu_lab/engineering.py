"""Provider-neutral scientist-coder execution records.

Engineering records document implementation work.  They deliberately cannot
write hypothesis, claim, or causal-edge state; scientific interpretation stays
inside the existing assessment services.
"""

from __future__ import annotations

from typing import Any

from .errors import GPUError

TASK_TYPES = {
    "REPOSITORY_INSPECTION", "BUG_REPRODUCTION", "BUG_FIX", "BASELINE_REPAIR",
    "EXPERIMENT_INSTRUMENTATION", "SCIENTIFIC_INTERVENTION_IMPLEMENTATION",
    "CONTROL_IMPLEMENTATION", "METRIC_IMPLEMENTATION", "DATA_PIPELINE_CHANGE",
    "REPRODUCTION_IMPLEMENTATION", "EXPERIMENT_PROTOTYPE",
    "REFACTOR_REQUIRED_FOR_EXPERIMENT",
}
TASK_STATUSES = {"OPEN", "ACTIVE", "BLOCKED", "COMPLETED", "INCONCLUSIVE"}
VERIFICATIONS = {"UNVERIFIED", "PARTIALLY_VERIFIED", "VERIFIED_TARGETED",
                 "VERIFIED_INTEGRATION", "VERIFIED_REAL_EXECUTION", "INVALID_IMPLEMENTATION"}
ENGINEERING_RESULT_STATUSES = {
    "COMPLETED", "FAILED", "BLOCKED", "DESIGN_REQUIRES_REVISION", "INCONCLUSIVE",
}
_FROZEN_TASK_FIELDS = {
    "research_decision_id", "experiment_id", "scientific_variable_changed",
    "scientific_variables_held_fixed", "scientific_invariants", "engineering_invariants",
    "prohibited_changes", "implementation_guards", "base_commit",
}


def _numeric_delta(before: Any, after: Any) -> float | None:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return abs(float(after) - float(before))
    return None


def evaluate_guard_measurements(
    declarations: list[dict[str, Any]], measurements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Evaluate machine-readable guard measurements without interpreting science."""
    by_name = {
        str(item.get("name")): item for item in measurements
        if isinstance(item, dict) and item.get("name")
    }
    evaluated: list[dict[str, Any]] = []
    for declaration in declarations:
        if not isinstance(declaration, dict) or not declaration.get("name"):
            raise GPUError("ENGINEERING_GUARD_DECLARATION_INVALID", "Each guard needs a name")
        name = str(declaration["name"])
        kind = str(declaration.get("type", "BOOLEAN")).upper()
        observed = by_name.get(name)
        if observed is None:
            raise GPUError("ENGINEERING_GUARD_MEASUREMENT_MISSING", name)
        if kind == "BOOLEAN":
            if not isinstance(observed.get("passed"), bool):
                raise GPUError("ENGINEERING_GUARD_RESULT_INVALID", name)
            passed = observed["passed"]
        else:
            if "before" not in observed or "after" not in observed:
                raise GPUError("ENGINEERING_GUARD_MEASUREMENT_INVALID", name)
            before, after = observed["before"], observed["after"]
            delta = _numeric_delta(before, after)
            tolerance = float(declaration.get("tolerance", 0.0))
            minimum_delta = float(declaration.get("minimum_delta", 0.0))
            if kind in {"EQUAL", "NATIVE_OFF", "HELD_FIXED"}:
                if delta is not None:
                    passed = delta <= tolerance
                else:
                    passed = before == after
            elif kind in {"CHANGED", "TARGET_CHANGED", "DIFFERENT"}:
                if delta is not None:
                    passed = delta > minimum_delta
                else:
                    passed = before != after
            elif kind == "CHECKSUM_EQUAL":
                passed = before == after
            elif kind == "CHECKSUM_DIFFERENT":
                passed = before != after
            else:
                raise GPUError("ENGINEERING_GUARD_TYPE_INVALID", kind)
            observed = {**observed, "delta": delta, "tolerance": tolerance,
                        "minimum_delta": minimum_delta}
        evaluated.append({**observed, "name": name, "type": kind, "passed": bool(passed)})
    return evaluated


def _list(value: Any, field: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise GPUError("ENGINEERING_FIELD_INVALID", f"{field} must be a list")
    return value


def _bounded_strings(value: Any, field: str, limit: int = 200) -> list[str]:
    return [str(item)[:4000] for item in _list(value, field)[:limit]]


class EngineeringService:
    def __init__(self, store):
        self.store = store

    def task_create(self, project_id: str, purpose: str, task_type: str,
                    change_request: str = "", repository: str = "",
                    repository_root: str = "", base_commit: str | None = None,
                    relevant_files: list[str] | None = None,
                    relevant_symbols: list[str] | None = None,
                    scientific_variable_changed: str | None = None,
                    scientific_variables_held_fixed: list[str] | None = None,
                    scientific_invariants: dict[str, Any] | None = None,
                    engineering_invariants: dict[str, Any] | None = None,
                    prohibited_changes: list[str] | None = None,
                    acceptance_tests: list[str] | None = None,
                    baseline_commands: list[str] | None = None,
                    targeted_tests: list[str] | None = None,
                    broader_tests: list[str] | None = None,
                    expected_artifacts: list[str] | None = None,
                    implementation_guards: list[dict[str, Any]] | None = None,
                    research_decision_id: str | None = None,
                    experiment_id: str | None = None) -> dict:
        task_type = task_type.upper()
        if task_type not in TASK_TYPES:
            raise GPUError("ENGINEERING_TASK_TYPE_INVALID", task_type)
        if not purpose.strip():
            raise GPUError("ENGINEERING_PURPOSE_REQUIRED", "purpose is required")
        invariants = scientific_invariants or {}
        if not isinstance(invariants, dict):
            raise GPUError("ENGINEERING_FIELD_INVALID", "scientific_invariants must be an object")
        if "must_change" not in invariants and scientific_variable_changed:
            invariants = {**invariants, "must_change": [scientific_variable_changed]}
        data = {
            "purpose": purpose.strip(), "task_type": task_type, "change_request": change_request,
            "repository": repository, "repository_root": repository_root,
            "base_commit": base_commit, "relevant_files": _list(relevant_files, "relevant_files"),
            "relevant_symbols": _list(relevant_symbols, "relevant_symbols"),
            "scientific_variable_changed": scientific_variable_changed,
            "scientific_variables_held_fixed": _list(scientific_variables_held_fixed, "scientific_variables_held_fixed"),
            "scientific_invariants": invariants,
            "engineering_invariants": engineering_invariants or {},
            "prohibited_changes": _list(prohibited_changes, "prohibited_changes"),
            "acceptance_tests": _list(acceptance_tests, "acceptance_tests"),
            "baseline_commands": _list(baseline_commands, "baseline_commands"),
            "targeted_tests": _list(targeted_tests, "targeted_tests"),
            "broader_tests": _list(broader_tests, "broader_tests"),
            "expected_artifacts": _list(expected_artifacts, "expected_artifacts"),
            "implementation_guards": _list(implementation_guards, "implementation_guards"),
            "research_decision_id": research_decision_id, "experiment_id": experiment_id,
            "scientific_result": "NOT_ASSESSED",
        }
        return self.store.object_create(project_id, "EngineeringTask", data, "ENGINEERING_TASK_CREATED", "OPEN")

    def task_get(self, task_id: str) -> dict:
        item = self.store.object_get(task_id)
        if item["kind"] != "EngineeringTask":
            raise GPUError("ENGINEERING_TASK_REQUIRED", task_id)
        return item

    def task_start(self, task_id: str, inspection: dict[str, Any], baseline: dict[str, Any]) -> dict:
        """Record bounded repository inspection and baseline evidence before editing."""
        task = self.task_get(task_id)
        if task["status"] not in {"OPEN", "ACTIVE"}:
            raise GPUError("ENGINEERING_TASK_NOT_STARTABLE", task["status"])
        if not isinstance(inspection, dict) or not isinstance(baseline, dict):
            raise GPUError("ENGINEERING_START_INVALID", "inspection and baseline must be objects")
        files_read = _list(inspection.get("files_read"), "files_read")
        if not files_read:
            raise GPUError("ENGINEERING_INSPECTION_REQUIRED", "At least one repository file must be recorded")
        commands = _list(baseline.get("commands_run"), "commands_run")
        if not commands:
            raise GPUError("ENGINEERING_BASELINE_REQUIRED", "At least one baseline command must be recorded")
        passed = baseline.get("passed")
        if not isinstance(passed, bool):
            raise GPUError("ENGINEERING_BASELINE_INVALID", "baseline.passed must be boolean")
        evidence = {
            "inspection": {
                "files_read": files_read[:200],
                "symbols_checked": _list(inspection.get("symbols_checked"), "symbols_checked")[:200],
                "relevant_callers": _list(inspection.get("relevant_callers"), "relevant_callers")[:100],
                "notes": str(inspection.get("notes", ""))[:4000],
            },
            "baseline": {
                "commands_run": [str(command)[:1000] for command in commands[:100]],
                "passed": passed,
                "summary": str(baseline.get("summary", ""))[:4000],
                "artifacts": _list(baseline.get("artifacts"), "artifacts")[:100],
            },
        }
        if not passed:
            return self.store.object_update(
                task_id, {**evidence, "baseline_verified": False}, "BLOCKED", "BASELINE_FAILED"
            )
        return self.store.object_update(
            task_id, {**evidence, "inspection_completed": True, "baseline_verified": True},
            "ACTIVE", "BASELINE_VERIFIED"
        )

    def task_update(self, task_id: str, status: str, update: dict[str, Any] | None = None) -> dict:
        status = status.upper()
        if status not in TASK_STATUSES:
            raise GPUError("ENGINEERING_TASK_STATUS_INVALID", status)
        self.task_get(task_id)
        update = update or {}
        if not isinstance(update, dict):
            raise GPUError("ENGINEERING_FIELD_INVALID", "update must be an object")
        protected = sorted(_FROZEN_TASK_FIELDS & set(update))
        if protected:
            raise GPUError("SCIENTIFIC_DESIGN_CHANGE_REQUIRED", ", ".join(protected))
        return self.store.object_update(task_id, update, status, "ENGINEERING_TASK_UPDATED")

    def diff_review(self, task_id: str, review: dict[str, Any]) -> dict:
        """Persist a bounded review of the implementation diff before handoff."""
        task = self.task_get(task_id)
        if task["data"].get("baseline_verified") is not True:
            raise GPUError("ENGINEERING_BASELINE_REQUIRED", "A passing baseline is required first")
        if not isinstance(review, dict):
            raise GPUError("ENGINEERING_DIFF_REVIEW_INVALID", "review must be an object")
        required = ("files_changed", "diff_summary", "unrelated_changes", "scientific_variable_drift")
        if any(field not in review for field in required):
            raise GPUError("ENGINEERING_DIFF_REVIEW_INVALID", "Missing required review fields")
        files_changed = _list(review["files_changed"], "files_changed")
        unrelated = review["unrelated_changes"]
        drift = review["scientific_variable_drift"]
        if not isinstance(unrelated, bool) or not isinstance(drift, bool):
            raise GPUError("ENGINEERING_DIFF_REVIEW_INVALID", "Review flags must be boolean")
        passed = not unrelated and not drift
        payload = {
            "diff_review": {
                "files_changed": [str(item)[:1000] for item in files_changed[:500]],
                "diff_summary": str(review["diff_summary"])[:8000],
                "unrelated_changes": unrelated,
                "scientific_variable_drift": drift,
                "prohibited_changes_detected": _list(review.get("prohibited_changes_detected"), "prohibited_changes_detected")[:100],
                "passed": passed,
            }
        }
        return self.store.object_update(
            task_id, payload, "ACTIVE" if passed else "BLOCKED",
            "ENGINEERING_DIFF_REVIEWED" if passed else "SCIENTIFIC_INVARIANT_VIOLATION",
        )

    def result_record(self, task_id: str, result: dict[str, Any]) -> dict:
        task = self.task_get(task_id)
        if not isinstance(result, dict):
            raise GPUError("ENGINEERING_RESULT_INVALID", "result must be an object")
        scientific_result = result.get("scientific_result", "NOT_ASSESSED")
        if scientific_result != "NOT_ASSESSED":
            raise GPUError("ENGINEERING_SCIENTIFIC_RESULT_FORBIDDEN", "Scientific assessment belongs to Research OS")
        if task["data"].get("inspection_completed") is not True:
            raise GPUError("ENGINEERING_INSPECTION_REQUIRED", "Start the task with repository inspection first")
        if task["data"].get("baseline_verified") is not True:
            raise GPUError("ENGINEERING_BASELINE_REQUIRED", "A passing baseline is required before implementation")
        if task["data"].get("diff_review", {}).get("passed") is not True:
            raise GPUError("ENGINEERING_DIFF_REVIEW_REQUIRED", "A passing diff review is required before completion")
        guard_fields = ("implementation_guard_results", "scientific_invariant_results")
        guard_results = {field: _list(result.get(field), field) for field in guard_fields}
        declared_guards = task["data"].get("implementation_guards", [])
        declared_guard_names = {
            str(item.get("name")) for item in declared_guards if isinstance(item, dict) and item.get("name")
        }
        measurements = result.get("guard_measurements")
        if measurements is not None:
            measurements = _list(measurements, "guard_measurements")
            guard_results["implementation_guard_results"] = evaluate_guard_measurements(
                declared_guards, measurements
            )
        supplied_guard_names = {
            str(item.get("name")) for item in guard_results["implementation_guard_results"]
            if isinstance(item, dict) and item.get("name")
        }
        if any(not isinstance(item, dict) or not item.get("name") or "passed" not in item
               for items in guard_results.values() for item in items):
            raise GPUError("ENGINEERING_GUARD_RESULT_INVALID", "Each guard result needs name and passed")
        missing_guards = sorted(declared_guard_names - supplied_guard_names)
        if missing_guards:
            raise GPUError("ENGINEERING_GUARD_MISSING", ", ".join(missing_guards))
        declared = task["data"].get("scientific_invariants", {})
        required = set(declared.get("must_change", [])) | set(declared.get("held_fixed", []))
        required |= set(task["data"].get("scientific_variables_held_fixed", []))
        checked = {str(item.get("name")) for item in guard_results["scientific_invariant_results"] if isinstance(item, dict)}
        missing = sorted(required - checked)
        if missing:
            raise GPUError("SCIENTIFIC_INVARIANT_GUARD_MISSING", ", ".join(missing))
        failed = [field for field, items in guard_results.items() if any(item["passed"] is not True for item in items)]
        verification = result.get("implementation_verification", "UNVERIFIED")
        if verification not in VERIFICATIONS:
            raise GPUError("ENGINEERING_VERIFICATION_INVALID", str(verification))
        if failed:
            verification = "INVALID_IMPLEMENTATION"
        engineering_status = str(result.get("engineering_status", "COMPLETED")).upper()
        if engineering_status not in ENGINEERING_RESULT_STATUSES:
            raise GPUError("ENGINEERING_STATUS_INVALID", engineering_status)
        if verification == "INVALID_IMPLEMENTATION":
            engineering_status = "INCONCLUSIVE"
        inspection = task["data"]["inspection"]
        baseline = task["data"]["baseline"]
        diff_review = task["data"]["diff_review"]
        stored = {
            "engineering_task_id": task_id,
            "base_commit": task["data"].get("base_commit"),
            "resulting_commit_or_diff_identity": result.get("resulting_commit_or_diff_identity"),
            "files_read": inspection["files_read"],
            "files_changed": diff_review["files_changed"],
            "symbols_changed": _bounded_strings(result.get("symbols_changed"), "symbols_changed"),
            "diff_summary": diff_review["diff_summary"],
            "commands_run": _bounded_strings(result.get("commands_run"), "commands_run"),
            "tests_run": _bounded_strings(result.get("tests_run"), "tests_run"),
            "tests_passed": _bounded_strings(result.get("tests_passed"), "tests_passed"),
            "tests_failed": _bounded_strings(result.get("tests_failed"), "tests_failed"),
            "build_result": result.get("build_result"),
            "typecheck_result": result.get("typecheck_result"),
            "lint_result": result.get("lint_result"),
            "baseline_result": baseline,
            "diff_review": diff_review,
            **guard_results,
            "unexpected_changes": _bounded_strings(result.get("unexpected_changes"), "unexpected_changes"),
            "unresolved_failures": _bounded_strings(result.get("unresolved_failures"), "unresolved_failures"),
            "artifacts": _list(result.get("artifacts"), "artifacts")[:200],
            "implementation_verification": verification,
            "engineering_status": engineering_status,
            "scientific_result": "NOT_ASSESSED",
        }
        status = "INCONCLUSIVE" if verification == "INVALID_IMPLEMENTATION" else "COMPLETED"
        record = self.store.object_create(task["project_id"], "EngineeringResult", stored,
                                          "ENGINEERING_RESULT_RECORDED", status)
        self.store.object_update(task_id, {"latest_result_id": record["id"], "implementation_verification": verification},
                                 "BLOCKED" if verification == "INVALID_IMPLEMENTATION" else "COMPLETED",
                                 "ENGINEERING_IMPLEMENTATION_INVALID" if verification == "INVALID_IMPLEMENTATION" else "ENGINEERING_VERIFIED")
        return record

    def result_get(self, result_id: str) -> dict:
        item = self.store.object_get(result_id)
        if item["kind"] != "EngineeringResult":
            raise GPUError("ENGINEERING_RESULT_REQUIRED", result_id)
        return item

    def task_verify(self, task_id: str) -> dict:
        task = self.task_get(task_id)
        result_id = task["data"].get("latest_result_id")
        if not result_id:
            raise GPUError("ENGINEERING_RESULT_REQUIRED", "No EngineeringResult is recorded")
        result = self.result_get(result_id)
        verification = result["data"].get("implementation_verification", "UNVERIFIED")
        return {"task_id": task_id, "result_id": result_id, "implementation_verification": verification,
                "scientific_result": "NOT_ASSESSED", "ready_for_scientific_execution": verification in {
                "VERIFIED_TARGETED", "VERIFIED_INTEGRATION", "VERIFIED_REAL_EXECUTION"}}

    def assert_ready_for_experiment(self, task_id: str, experiment_id: str) -> dict:
        task = self.task_get(task_id)
        linked_experiment = task["data"].get("experiment_id")
        if linked_experiment and str(linked_experiment) != str(experiment_id):
            raise GPUError("ENGINEERING_EXPERIMENT_MISMATCH", str(experiment_id))
        readiness = self.task_verify(task_id)
        if not readiness["ready_for_scientific_execution"]:
            raise GPUError("ENGINEERING_IMPLEMENTATION_NOT_VERIFIED", readiness["implementation_verification"])
        return readiness

    def context_get(self, task_id: str) -> dict:
        task = self.task_get(task_id)
        data = task["data"]
        return {"task_id": task["id"], "project_id": task["project_id"], "purpose": data["purpose"],
                "repository": data["repository"], "repository_root": data["repository_root"],
                "base_commit": data["base_commit"], "relevant_files": data["relevant_files"],
                "scientific_variable_changed": data["scientific_variable_changed"],
                "scientific_variables_held_fixed": data["scientific_variables_held_fixed"],
                "scientific_invariants": data["scientific_invariants"],
                "prohibited_changes": data["prohibited_changes"], "tests": {
                    "acceptance": data["acceptance_tests"], "baseline": data["baseline_commands"],
                    "targeted": data["targeted_tests"], "broader": data["broader_tests"]}}
