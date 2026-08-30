"""Provider-neutral scientist-coder execution records.

Engineering records document implementation work.  They deliberately cannot
write hypothesis, claim, or causal-edge state; scientific interpretation stays
inside the existing assessment services.
"""

from __future__ import annotations

import hashlib
import json
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
ENGINEERING_PHASES = (
    "RECEIVE", "INSPECT", "UNDERSTAND", "REPRODUCE", "PLAN", "EDIT",
    "TEST", "VERIFY_INVARIANTS", "REVIEW_DIFF", "RECORD", "HAND_BACK",
)
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
            try:
                tolerance = float(declaration.get("tolerance", 0.0))
                minimum_delta = float(declaration.get("minimum_delta", 0.0))
            except (TypeError, ValueError) as exc:
                raise GPUError("ENGINEERING_GUARD_MEASUREMENT_INVALID", name) from exc
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


class CodingExecutionPolicy:
    """Provider-neutral deterministic phase contract for engineering work."""

    phases = ENGINEERING_PHASES

    @classmethod
    def initial(cls) -> dict[str, Any]:
        return {"phase": cls.phases[0], "completed": [], "policy_version": "v2.2"}

    @classmethod
    def advance(cls, state: dict[str, Any], phase: str) -> dict[str, Any]:
        if not isinstance(state, dict) or state.get("phase") not in cls.phases:
            raise GPUError("ENGINEERING_POLICY_STATE_INVALID", "Unknown current phase")
        phase = phase.upper()
        if phase not in cls.phases:
            raise GPUError("ENGINEERING_POLICY_PHASE_INVALID", phase)
        current_index = cls.phases.index(state["phase"])
        target_index = cls.phases.index(phase)
        if target_index != current_index + 1:
            raise GPUError(
                "ENGINEERING_POLICY_ORDER_VIOLATION",
                f"Expected {cls.phases[current_index + 1] if current_index + 1 < len(cls.phases) else 'END'}, got {phase}",
            )
        return {
            "phase": phase,
            "completed": [*state.get("completed", []), state["phase"]],
            "policy_version": "v2.2",
        }

    @classmethod
    def contract(cls) -> dict[str, Any]:
        return {"policy_version": "v2.2", "phases": list(cls.phases), "scientific_result": "NOT_ASSESSED"}


class EngineeringService:
    def __init__(self, store):
        self.store = store

    def _validate_parent(self, project_id: str, identifier: str | None, kind: str, field: str) -> dict | None:
        if identifier is None:
            return None
        try:
            item = self.store.object_get(identifier)
        except (KeyError, GPUError) as exc:
            if isinstance(exc, GPUError):
                raise
            raise GPUError("RESEARCH_OBJECT_NOT_FOUND", identifier) from exc
        if item.get("kind") != kind:
            raise GPUError(f"ENGINEERING_{field.upper()}_KIND_INVALID", identifier)
        if str(item.get("project_id")) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", identifier)
        return item

    @staticmethod
    def _canonical_hash(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()

    def scientific_spec_create(
        self, project_id: str, subject_id: str, protected_scientific_fields: dict[str, Any],
        lineage_snapshot_id: str, lineage_as_of: str, project_state_version: int | None = None,
        world_model_version: str | None = None,
    ) -> dict:
        """Freeze the scientific semantics that a semantic review may approve.

        Operational identifiers (worker, GPU, path, runtime timestamp) are
        deliberately not part of this fingerprint.  They belong in release
        candidates and deterministic preflight, not scientific identity.
        """
        if not subject_id or not lineage_snapshot_id or not lineage_as_of:
            raise GPUError("SCIENTIFIC_SPEC_IDENTITY_REQUIRED", "subject_id, lineage_snapshot_id, lineage_as_of")
        if not isinstance(protected_scientific_fields, dict) or not protected_scientific_fields:
            raise GPUError("SCIENTIFIC_SPEC_PROTECTED_FIELDS_REQUIRED", "protected_scientific_fields")
        semantic_hash = self._canonical_hash(protected_scientific_fields)
        payload = {
            "subject_id": subject_id,
            "protected_scientific_fields": protected_scientific_fields,
            "scientific_semantic_hash": semantic_hash,
            "lineage_snapshot_id": lineage_snapshot_id,
            "lineage_as_of": lineage_as_of,
            "project_state_version": project_state_version,
            "world_model_version": world_model_version,
            "frozen": True,
            "scientific_result": "NOT_ASSESSED",
        }
        for existing in self.store.objects_list(project_id, "ScientificSpecVersion", limit=None):
            if existing.get("data") == payload:
                return {**existing, "idempotent": True}
        return self.store.object_create(project_id, "ScientificSpecVersion", payload, "SCIENTIFIC_SPEC_FROZEN", "PREPARED")

    def scientific_spec_get(self, spec_id: str) -> dict:
        spec = self.store.object_get(spec_id)
        if spec["kind"] != "ScientificSpecVersion":
            raise GPUError("SCIENTIFIC_SPEC_REQUIRED", spec_id)
        return spec

    def semantic_review_record(
        self, scientific_spec_id: str, reviewer_id: str, verdict: str,
        rationale: str, latest_relevant_scientific_event_at: str,
        review_escalation_reason_id: str | None = None,
    ) -> dict:
        """Record review #1 for a frozen scientific specification.

        A second semantic review of unchanged semantics needs a durable
        escalation reason; this prevents technical work from silently
        restarting the scientific review ladder.
        """
        spec = self.scientific_spec_get(scientific_spec_id)
        verdict = verdict.upper()
        if verdict not in {"PASS", "FAIL", "ESCALATE_SCIENTIFIC_REVIEW"}:
            raise GPUError("SEMANTIC_REVIEW_VERDICT_INVALID", verdict)
        if not reviewer_id.strip() or not rationale.strip() or not latest_relevant_scientific_event_at:
            raise GPUError("SEMANTIC_REVIEW_FIELDS_REQUIRED", "reviewer_id, rationale, latest_relevant_scientific_event_at")
        prior = [
            item for item in self.store.objects_list(str(spec["project_id"]), "ScientificSemanticReview", limit=None)
            if str(item.get("data", {}).get("scientific_spec_id")) == str(scientific_spec_id)
            and item.get("data", {}).get("verdict") == "PASS"
        ]
        if prior and not review_escalation_reason_id:
            raise GPUError("SEMANTIC_REVIEW_LIMIT_REACHED", "An unchanged ScientificSpecVersion already has semantic PASS; use DeltaReview or provide ReviewEscalationReason")
        if review_escalation_reason_id:
            reason = self._validate_parent(str(spec["project_id"]), review_escalation_reason_id, "ReviewEscalationReason", "review_escalation_reason")
            if not str(reason.get("data", {}).get("reason", "")).strip():
                raise GPUError("REVIEW_ESCALATION_REASON_REQUIRED", review_escalation_reason_id)
        payload = {
            "scientific_spec_id": scientific_spec_id,
            "scientific_semantic_hash": spec["data"]["scientific_semantic_hash"],
            "lineage_snapshot_id": spec["data"]["lineage_snapshot_id"],
            "lineage_as_of": spec["data"]["lineage_as_of"],
            "project_state_version": spec["data"].get("project_state_version"),
            "world_model_version": spec["data"].get("world_model_version"),
            "latest_relevant_scientific_event_at": latest_relevant_scientific_event_at,
            "reviewer_id": reviewer_id,
            "verdict": verdict,
            "rationale": rationale,
            "review_escalation_reason_id": review_escalation_reason_id,
            "scientific_result": "NOT_ASSESSED",
        }
        status = "APPROVED" if verdict == "PASS" else "BLOCKED"
        review = self.store.object_create(str(spec["project_id"]), "ScientificSemanticReview", payload, "SCIENTIFIC_SEMANTIC_REVIEW_RECORDED", status)
        if verdict == "PASS":
            self.store.object_update(scientific_spec_id, {"semantic_review_id": review["id"]}, "APPROVED", "SCIENTIFIC_SPEC_SEMANTIC_REVIEW_PASSED")
        return review

    def review_escalation_reason_create(self, project_id: str, scientific_spec_id: str, reason: str, rationale: str) -> dict:
        spec = self.scientific_spec_get(scientific_spec_id)
        if str(spec["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", scientific_spec_id)
        allowed = {
            "PRIOR_REVIEW_INVALIDATED", "REVIEWER_RELIED_ON_FALSE_FACTS", "MATERIAL_SCIENTIFIC_CONTRADICTION",
            "MATERIAL_NEW_SCIENTIFIC_EVIDENCE", "CONFIRMATION_OR_PUBLICATION_POLICY", "UNKNOWN_DRIFT_CONSEQUENTIAL",
        }
        if reason.upper() not in allowed or not rationale.strip():
            raise GPUError("REVIEW_ESCALATION_REASON_INVALID", reason)
        return self.store.object_create(project_id, "ReviewEscalationReason", {
            "scientific_spec_id": scientific_spec_id, "reason": reason.upper(), "rationale": rationale,
            "scientific_result": "NOT_ASSESSED",
        }, "REVIEW_ESCALATION_REASON_RECORDED", "APPROVED")

    def delta_manifest_create(
        self, project_id: str, base_release_candidate_id: str, candidate_release_candidate_id: str,
        changed_files: list[str], changed_fields: list[str], changed_artifacts: list[str],
        changed_configs: list[str], previous_hashes: dict[str, str], current_hashes: dict[str, str],
        scientific_semantic_hash_before: str, scientific_semantic_hash_after: str,
        scientific_variable_drift: str, affected_engineering_invariants: list[str],
        affected_preflight_checks: list[str], unaffected_review_refs: list[str],
        required_verification: list[str], rationale: str,
    ) -> dict:
        """Create a mechanical release delta classification before any review."""
        base = self.pre_science_release_candidate_get(base_release_candidate_id)
        candidate = self.pre_science_release_candidate_get(candidate_release_candidate_id)
        if str(base["project_id"]) != project_id or str(candidate["project_id"]) != project_id:
            raise GPUError("RESEARCH_PROJECT_MISMATCH", "release candidates")
        drift = scientific_variable_drift.upper()
        if drift not in {"FALSE", "TRUE", "UNKNOWN"}:
            raise GPUError("SCIENTIFIC_VARIABLE_DRIFT_INVALID", scientific_variable_drift)
        if not rationale.strip() or not required_verification:
            raise GPUError("DELTA_MANIFEST_FIELDS_REQUIRED", "rationale and required_verification")
        if not isinstance(previous_hashes, dict) or not isinstance(current_hashes, dict):
            raise GPUError("DELTA_MANIFEST_HASHES_INVALID", "previous_hashes and current_hashes must be objects")
        if drift == "FALSE" and scientific_semantic_hash_before != scientific_semantic_hash_after:
            raise GPUError("SCIENTIFIC_VARIABLE_DRIFT_CONTRADICTION", "FALSE requires matching semantic hashes")
        payload = {
            "base_release_candidate_id": base_release_candidate_id,
            "candidate_release_candidate_id": candidate_release_candidate_id,
            "base_subject_id": base["data"].get("experiment_id"),
            "base_subject_version": base["data"].get("canonical_subject_version"),
            "changed_files": _bounded_strings(changed_files, "changed_files"),
            "changed_fields": _bounded_strings(changed_fields, "changed_fields"),
            "changed_artifacts": _bounded_strings(changed_artifacts, "changed_artifacts"),
            "changed_configs": _bounded_strings(changed_configs, "changed_configs"),
            "previous_hashes": dict(sorted(previous_hashes.items())),
            "current_hashes": dict(sorted(current_hashes.items())),
            "scientific_semantic_hash_before": scientific_semantic_hash_before,
            "scientific_semantic_hash_after": scientific_semantic_hash_after,
            "scientific_variable_drift": drift,
            "affected_engineering_invariants": _bounded_strings(affected_engineering_invariants, "affected_engineering_invariants"),
            "affected_preflight_checks": _bounded_strings(affected_preflight_checks, "affected_preflight_checks"),
            "unaffected_review_refs": _bounded_strings(unaffected_review_refs, "unaffected_review_refs"),
            "required_verification": _bounded_strings(required_verification, "required_verification"),
            "rationale": rationale,
            "scientific_result": "NOT_ASSESSED",
        }
        return self.store.object_create(project_id, "DeltaManifest", payload, "DELTA_MANIFEST_CREATED", "PREPARED")

    def delta_review_record(self, delta_manifest_id: str, reviewer_id: str, verdict: str, rationale: str, completed_checks: list[str]) -> dict:
        manifest = self.store.object_get(delta_manifest_id)
        if manifest["kind"] != "DeltaManifest":
            raise GPUError("ENGINEERING_DELTA_MANIFEST_KIND_INVALID", delta_manifest_id)
        verdict = verdict.upper()
        if verdict not in {"PASS", "FAIL", "ESCALATE_SCIENTIFIC_REVIEW"}:
            raise GPUError("DELTA_REVIEW_VERDICT_INVALID", verdict)
        drift = manifest["data"].get("scientific_variable_drift")
        if drift in {"TRUE", "UNKNOWN"} and verdict == "PASS":
            raise GPUError("DELTA_REVIEW_SCIENTIFIC_ESCALATION_REQUIRED", drift)
        required = set(manifest["data"].get("required_verification", []))
        if verdict == "PASS" and not required.issubset(set(completed_checks)):
            raise GPUError("DELTA_REVIEW_CHECKS_MISSING", ", ".join(sorted(required - set(completed_checks))))
        if not reviewer_id.strip() or not rationale.strip():
            raise GPUError("DELTA_REVIEW_FIELDS_REQUIRED", "reviewer_id and rationale")
        payload = {"delta_manifest_id": delta_manifest_id, "reviewer_id": reviewer_id, "verdict": verdict,
                   "rationale": rationale, "completed_checks": _bounded_strings(completed_checks, "completed_checks"),
                   "scientific_result": "NOT_ASSESSED"}
        return self.store.object_create(str(manifest["project_id"]), "DeltaReview", payload,
                                        "DELTA_REVIEW_RECORDED", "APPROVED" if verdict == "PASS" else "BLOCKED")

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
        decision = self._validate_parent(project_id, research_decision_id, "ResearchDecision", "research_decision")
        experiment = self._validate_parent(project_id, experiment_id, "Experiment", "experiment")
        if decision and experiment:
            decision_experiment = decision.get("data", {}).get("experiment_id")
            if decision_experiment and str(decision_experiment) != str(experiment_id):
                raise GPUError("ENGINEERING_PARENT_EXPERIMENT_MISMATCH", experiment_id)
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
        previous_result_id = task["data"].get("latest_result_id")
        if task["status"] == "COMPLETED":
            raise GPUError("ENGINEERING_RESULT_ALREADY_RECORDED", task_id)
        if previous_result_id:
            previous = self.result_get(previous_result_id)
            if (
                previous["data"].get("implementation_verification") != "INVALID_IMPLEMENTATION"
                or task["status"] != "ACTIVE"
                or task["data"].get("diff_review", {}).get("passed") is not True
            ):
                raise GPUError("ENGINEERING_RESULT_ALREADY_RECORDED", task_id)
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

    def pre_science_release_candidate_create(
        self, project_id: str, canonical_subject_version: str, experiment_id: str,
        research_decision_id: str, engineering_result_id: str,
        deterministic_preflight_id: str, runtime_fingerprint: str,
        scientific_hashes: dict[str, str], successor_work: dict[str, Any],
        semantic_review_required: bool = True,
        scientific_spec_id: str | None = None,
    ) -> dict:
        """Create one immutable bundle for a pre-science release.

        This is deliberately not an execution authorization.  It makes the
        exact design/decision, verified implementation, deterministic
        preflight, runtime binding, and known successor inspectable as one
        release candidate before the separately sealed confirmation path.
        """
        if not canonical_subject_version.strip() or not runtime_fingerprint.strip():
            raise GPUError("PRE_SCIENCE_RELEASE_IDENTITY_REQUIRED", "canonical_subject_version and runtime_fingerprint")
        if not isinstance(semantic_review_required, bool):
            raise GPUError("PRE_SCIENCE_RELEASE_REVIEW_POLICY_INVALID", "semantic_review_required")
        if not isinstance(scientific_hashes, dict) or not scientific_hashes:
            raise GPUError("PRE_SCIENCE_RELEASE_HASHES_REQUIRED", "scientific_hashes")
        hashes = {str(key): str(value) for key, value in scientific_hashes.items()}
        if any(not key.strip() or not value.strip() for key, value in hashes.items()):
            raise GPUError("PRE_SCIENCE_RELEASE_HASH_INVALID", "scientific_hashes must have nonempty names and values")
        if not isinstance(successor_work, dict):
            raise GPUError("PRE_SCIENCE_RELEASE_SUCCESSOR_REQUIRED", "successor_work")
        successor_required = ("kind", "title", "description", "scientific_role")
        if any(not str(successor_work.get(field, "")).strip() for field in successor_required):
            raise GPUError("PRE_SCIENCE_RELEASE_SUCCESSOR_REQUIRED", ", ".join(successor_required))

        spec = None
        if scientific_spec_id:
            spec = self._validate_parent(project_id, scientific_spec_id, "ScientificSpecVersion", "scientific_spec")
            if str(spec["data"].get("subject_id")) != str(experiment_id):
                raise GPUError("PRE_SCIENCE_RELEASE_SPEC_SUBJECT_MISMATCH", scientific_spec_id)
            supplied_hash = hashes.get("scientific_semantic_hash")
            if supplied_hash != spec["data"].get("scientific_semantic_hash"):
                raise GPUError("PRE_SCIENCE_RELEASE_SPEC_HASH_MISMATCH", scientific_spec_id)

        experiment = self._validate_parent(project_id, experiment_id, "Experiment", "experiment")
        decision = self._validate_parent(project_id, research_decision_id, "ResearchDecision", "research_decision")
        result = self._validate_parent(project_id, engineering_result_id, "EngineeringResult", "engineering_result")
        if str((decision or {}).get("data", {}).get("experiment_id") or "") != str(experiment_id):
            raise GPUError("PRE_SCIENCE_RELEASE_DECISION_MISMATCH", experiment_id)
        if str((result or {}).get("data", {}).get("engineering_task_id") or "") == "":
            raise GPUError("PRE_SCIENCE_RELEASE_ENGINEERING_RESULT_INVALID", engineering_result_id)
        if (result or {}).get("data", {}).get("implementation_verification") not in {
            "VERIFIED_TARGETED", "VERIFIED_INTEGRATION", "VERIFIED_REAL_EXECUTION",
        }:
            raise GPUError("PRE_SCIENCE_RELEASE_ENGINEERING_NOT_VERIFIED", engineering_result_id)

        # Preflight IDs are durable rows in the Research OS schema.  Verify the
        # exact successful row instead of accepting a client-provided PASS.
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT project_id,scientific_object_id,canonical_subject_version,status,result_hash "
                "FROM lab_deterministic_preflights WHERE id=%s",
                (deterministic_preflight_id,),
            )
            preflight = cur.fetchone()
        if not preflight or str(preflight["project_id"]) != str(project_id):
            raise GPUError("PRE_SCIENCE_RELEASE_PREFLIGHT_NOT_FOUND", deterministic_preflight_id)
        if preflight["status"] != "PASS":
            raise GPUError("PRE_SCIENCE_RELEASE_PREFLIGHT_NOT_PASS", deterministic_preflight_id)
        if str(preflight["scientific_object_id"]) != str(experiment_id) or str(preflight["canonical_subject_version"]) != canonical_subject_version:
            raise GPUError("PRE_SCIENCE_RELEASE_PREFLIGHT_MISMATCH", deterministic_preflight_id)

        payload = {
            "canonical_subject_version": canonical_subject_version,
            "experiment_id": experiment_id,
            "research_decision_id": research_decision_id,
            "engineering_result_id": engineering_result_id,
            "deterministic_preflight_id": deterministic_preflight_id,
            "deterministic_preflight_result_hash": str(preflight["result_hash"]),
            "runtime_fingerprint": runtime_fingerprint,
            "scientific_hashes": dict(sorted(hashes.items())),
            "scientific_spec_id": scientific_spec_id,
            "successor_work": {field: successor_work[field] for field in successor_required},
            "semantic_review_required": semantic_review_required,
            "execution_authorized": False,
            "scientific_result": "NOT_ASSESSED",
        }
        existing = self.store.objects_list(project_id, "PreScienceReleaseCandidate", limit=None)
        superseded_candidates = []
        for candidate in existing:
            if candidate.get("data") == payload:
                return {**candidate, "idempotent": True}
        # There is one current release assembly for a scientific version. The
        # immutable prior RC remains auditable but cannot survive as a second
        # executable/reviewable authority.
        for candidate in existing:
            data = candidate.get("data", {})
            same_spec = scientific_spec_id and str(data.get("scientific_spec_id")) == str(scientific_spec_id)
            legacy_same_subject = not scientific_spec_id and (
                str(data.get("experiment_id")) == str(experiment_id)
                and str(data.get("canonical_subject_version")) == str(canonical_subject_version)
            )
            if (same_spec or legacy_same_subject) and candidate.get("status") in {"PREPARED", "APPROVED"}:
                superseded_candidates.append(candidate)
        created = self.store.object_create(
            project_id, "PreScienceReleaseCandidate", payload,
            "PRE_SCIENCE_RELEASE_CANDIDATE_CREATED", "PREPARED",
        )
        for candidate in superseded_candidates:
            self.store.object_update(candidate["id"], {
                "superseded_by_release_candidate": created["id"],
                "supersession_reason": "A newer immutable PRE_SCIENCE release candidate was assembled.",
            }, "SUPERSEDED", "PRE_SCIENCE_RELEASE_CANDIDATE_SUPERSEDED")
        return created

    def pre_science_release_candidate_get(self, candidate_id: str) -> dict:
        candidate = self.store.object_get(candidate_id)
        if candidate["kind"] != "PreScienceReleaseCandidate":
            raise GPUError("PRE_SCIENCE_RELEASE_CANDIDATE_REQUIRED", candidate_id)
        return candidate

    def final_pre_science_release_review_record(
        self, release_candidate_id: str, reviewer_id: str, verdict: str,
        rationale: str, completed_checks: list[str],
        latest_relevant_scientific_event_at: str,
        review_escalation_reason_id: str | None = None,
    ) -> dict:
        """Record the second (and normally final) full review boundary.

        It binds review evidence to exact RC bytes. A technical repair must
        create a new RC and use DeltaReview; it cannot append arbitrary full
        reviews to an unchanged scientific specification.
        """
        candidate = self.pre_science_release_candidate_get(release_candidate_id)
        if candidate.get("status") == "SUPERSEDED":
            raise GPUError("PRE_SCIENCE_RELEASE_CANDIDATE_SUPERSEDED", release_candidate_id)
        verdict = verdict.upper()
        if verdict not in {"PASS", "FAIL", "ESCALATE_SCIENTIFIC_REVIEW"}:
            raise GPUError("FINAL_RELEASE_REVIEW_VERDICT_INVALID", verdict)
        if not reviewer_id.strip() or not rationale.strip() or not latest_relevant_scientific_event_at:
            raise GPUError("FINAL_RELEASE_REVIEW_FIELDS_REQUIRED", "reviewer_id, rationale, latest_relevant_scientific_event_at")
        if not isinstance(completed_checks, list) or not completed_checks:
            raise GPUError("FINAL_RELEASE_REVIEW_CHECKS_REQUIRED", "completed_checks")
        spec_id = candidate.get("data", {}).get("scientific_spec_id")
        if not spec_id:
            raise GPUError("FINAL_RELEASE_REVIEW_SPEC_REQUIRED", release_candidate_id)
        spec = self.scientific_spec_get(str(spec_id))
        project_id = str(candidate["project_id"])
        if str(spec["project_id"]) != project_id:
            raise GPUError("RESEARCH_PROJECT_MISMATCH", str(spec_id))
        semantic_reviews = [
            item for item in self.store.objects_list(project_id, "ScientificSemanticReview", limit=None)
            if str(item.get("data", {}).get("scientific_spec_id")) == str(spec_id)
            and item.get("data", {}).get("verdict") == "PASS"
            and item.get("status") == "APPROVED"
        ]
        if not semantic_reviews:
            raise GPUError("FINAL_RELEASE_REVIEW_SEMANTIC_REVIEW_REQUIRED", str(spec_id))
        semantic = semantic_reviews[-1]
        if str(semantic["data"].get("scientific_semantic_hash")) != str(spec["data"].get("scientific_semantic_hash")):
            raise GPUError("FINAL_RELEASE_REVIEW_SEMANTIC_HASH_MISMATCH", str(spec_id))
        if str(semantic["data"].get("lineage_snapshot_id")) != str(spec["data"].get("lineage_snapshot_id")):
            raise GPUError("FINAL_RELEASE_REVIEW_LINEAGE_MISMATCH", str(spec_id))
        if str(semantic["data"].get("reviewer_id")) == reviewer_id:
            raise GPUError("FINAL_RELEASE_REVIEW_INDEPENDENCE_REQUIRED", "semantic and final reviewer must differ")
        prior = [
            item for item in self.store.objects_list(project_id, "FinalPreScienceReleaseReview", limit=None)
            if str(item.get("data", {}).get("scientific_spec_id")) == str(spec_id)
            and item.get("data", {}).get("verdict") == "PASS"
        ]
        if prior and not review_escalation_reason_id:
            raise GPUError("FINAL_RELEASE_REVIEW_LIMIT_REACHED", "An unchanged ScientificSpecVersion already has final release PASS")
        if review_escalation_reason_id:
            self._validate_parent(project_id, review_escalation_reason_id, "ReviewEscalationReason", "review_escalation_reason")
        payload = {
            "release_candidate_id": release_candidate_id,
            "scientific_spec_id": str(spec_id),
            "scientific_semantic_hash": spec["data"]["scientific_semantic_hash"],
            "lineage_snapshot_id": spec["data"]["lineage_snapshot_id"],
            "lineage_as_of": spec["data"]["lineage_as_of"],
            "latest_relevant_scientific_event_at": latest_relevant_scientific_event_at,
            "semantic_review_id": semantic["id"],
            "reviewer_id": reviewer_id,
            "verdict": verdict,
            "rationale": rationale,
            "completed_checks": _bounded_strings(completed_checks, "completed_checks"),
            "review_escalation_reason_id": review_escalation_reason_id,
            "scientific_result": "NOT_ASSESSED",
        }
        review = self.store.object_create(project_id, "FinalPreScienceReleaseReview", payload,
                                          "FINAL_PRE_SCIENCE_RELEASE_REVIEW_RECORDED",
                                          "APPROVED" if verdict == "PASS" else "BLOCKED")
        if verdict == "PASS":
            self.store.object_update(release_candidate_id, {"final_release_review_id": review["id"], "frozen": True},
                                     "APPROVED", "PRE_SCIENCE_RELEASE_CANDIDATE_FROZEN")
        return review

    def confirmation_authorization_create(
        self, project_id: str, release_candidate_id: str, approver_id: str,
        verdict: str, rationale: str,
    ) -> dict:
        """Create the explicit, separate authorization required for confirmation."""
        candidate = self.pre_science_release_candidate_get(release_candidate_id)
        if str(candidate["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", release_candidate_id)
        verdict = verdict.upper()
        if verdict not in {"PASS", "FAIL"} or not approver_id.strip() or not rationale.strip():
            raise GPUError("CONFIRMATION_AUTHORIZATION_INVALID", "approver_id, verdict, rationale")
        if verdict == "PASS":
            final_id = candidate.get("data", {}).get("final_release_review_id")
            if candidate.get("status") != "APPROVED" or not final_id:
                raise GPUError("CONFIRMATION_FINAL_RELEASE_REVIEW_REQUIRED", release_candidate_id)
        return self.store.object_create(project_id, "ConfirmationAuthorization", {
            "release_candidate_id": release_candidate_id,
            "approver_id": approver_id,
            "verdict": verdict,
            "rationale": rationale,
            "scientific_result": "NOT_ASSESSED",
        }, "CONFIRMATION_AUTHORIZATION_RECORDED", "APPROVED" if verdict == "PASS" else "BLOCKED")

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
        if not linked_experiment or str(linked_experiment) != str(experiment_id):
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
