"""Prospective, isolated cross-project methodological strategy transfer (v3.5).

This module deliberately does not replace the v2 strategy-memory service.  The
older service remains observational process memory; v3.5 adds a falsifiable
transfer lifecycle with an explicit target-side hypothesis and outcome.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import GPUError
from .research import ResearchStore


class StrategyScope(StrEnum):
    PROJECT = "PROJECT"
    DOMAIN = "DOMAIN"
    GLOBAL = "GLOBAL"


class StrategyMaturity(StrEnum):
    OBSERVED = "OBSERVED"
    CANDIDATE = "CANDIDATE"
    PROSPECTIVE_TESTING = "PROSPECTIVE_TESTING"
    CROSS_PROJECT_SUPPORTED = "CROSS_PROJECT_SUPPORTED"
    CROSS_DOMAIN_SUPPORTED = "CROSS_DOMAIN_SUPPORTED"
    WEAKENED = "WEAKENED"
    REFUTED = "REFUTED"
    DEPRECATED = "DEPRECATED"
    CORE_INVARIANT = "CORE_INVARIANT"


class StrategyType(StrEnum):
    CAUSAL_DESIGN = "CAUSAL_DESIGN"
    EXPERIMENT_DESIGN = "EXPERIMENT_DESIGN"
    FALSIFICATION = "FALSIFICATION"
    MEASUREMENT = "MEASUREMENT"
    REPRODUCTION = "REPRODUCTION"
    DISCOVERY_SEARCH = "DISCOVERY_SEARCH"
    REPRESENTATION_SEARCH = "REPRESENTATION_SEARCH"
    NULL_MODEL = "NULL_MODEL"
    GENERALIZATION = "GENERALIZATION"
    ENGINEERING = "ENGINEERING"
    VALIDATION = "VALIDATION"
    LITERATURE = "LITERATURE"
    COORDINATION = "COORDINATION"
    CORRECTION = "CORRECTION"
    COMPUTE_ALLOCATION = "COMPUTE_ALLOCATION"
    DOMAIN_HEURISTIC = "DOMAIN_HEURISTIC"
    OPERATIONAL_HEURISTIC = "OPERATIONAL_HEURISTIC"


class ApplicabilityState(StrEnum):
    STRONG_MATCH = "STRONG_MATCH"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    WEAK_MATCH = "WEAK_MATCH"
    CONTRAINDICATED = "CONTRAINDICATED"
    UNKNOWN = "UNKNOWN"


class TransferStatus(StrEnum):
    PROPOSED = "PROPOSED"
    SCREENED_OUT = "SCREENED_OUT"
    ELIGIBLE = "ELIGIBLE"
    APPLIED = "APPLIED"
    NOT_APPLIED = "NOT_APPLIED"
    COMPLETED = "COMPLETED"
    INVALID = "INVALID"
    SUPERSEDED = "SUPERSEDED"


class TransferOutcomeKind(StrEnum):
    POSITIVE_TRANSFER = "POSITIVE_TRANSFER"
    NEGATIVE_TRANSFER = "NEGATIVE_TRANSFER"
    NEUTRAL_TRANSFER = "NEUTRAL_TRANSFER"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID_TRANSFER = "INVALID_TRANSFER"


class StrategyApplicabilityModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_conditions: dict[str, str | bool | int | float] = Field(default_factory=dict)
    contraindications: dict[str, str | bool | int | float] = Field(default_factory=dict)
    structural_features: list[str] = Field(default_factory=list, max_length=30)
    rationale: str = Field(min_length=1, max_length=10_000)


class StrategyPatternCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=300)
    strategy_type: StrategyType
    principle: str = Field(min_length=1, max_length=20_000)
    mechanism_of_value: str = Field(min_length=1, max_length=20_000)
    source_implementation_example: str = Field(min_length=1, max_length=20_000)
    applicability: StrategyApplicabilityModel
    source_decision_ids: list[str] = Field(default_factory=list)
    source_outcome_ids: list[str] = Field(default_factory=list)
    source_domains: list[str] = Field(default_factory=list)
    scope: StrategyScope = StrategyScope.PROJECT
    maturity: StrategyMaturity = StrategyMaturity.OBSERVED
    authority_class: str = Field(default="CONTEXTUAL", pattern="^(CORE|VALIDATED|CONTEXTUAL)$")
    retrospective_backfill: bool = False
    parent_strategy_id: str | None = None
    adaptation_note: str | None = Field(default=None, max_length=10_000)


class StrategyTransferPropose(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str
    target_context: dict[str, str | bool | int | float] = Field(default_factory=dict)
    target_problem_structure: list[str] = Field(default_factory=list, max_length=30)
    selection_mechanism: str = Field(min_length=1, max_length=10_000)
    predicted_benefit: str = Field(min_length=1, max_length=10_000)
    predicted_failure: str = Field(min_length=1, max_length=10_000)
    planned_use: str = Field(min_length=1, max_length=20_000)
    decision_id: str | None = None
    target_implementation_realization: str | None = Field(default=None, max_length=20_000)


class StrategyApplicabilityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: ApplicabilityState
    matched_conditions: list[str] = Field(default_factory=list)
    mismatch_conditions: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=10_000)
    reviewer_identity: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def screen_is_coherent(self) -> "StrategyApplicabilityAssessment":
        if self.state == ApplicabilityState.CONTRAINDICATED and not self.mismatch_conditions:
            raise ValueError("CONTRAINDICATED requires at least one mismatch condition")
        return self


class StrategyTransferApply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str | None = None
    applied_method: str = Field(min_length=1, max_length=20_000)
    target_implementation_realization: str = Field(min_length=1, max_length=20_000)
    execution_plan_ref: str | None = None


class StrategyTransferOutcomeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: TransferOutcomeKind
    rationale: str = Field(min_length=1, max_length=20_000)
    observed_process_effects: dict[str, str | int | float | bool] = Field(default_factory=dict)
    decision_outcome_id: str | None = None
    experiment_run_ids: list[str] = Field(default_factory=list)
    correction_case_ids: list[str] = Field(default_factory=list)
    independence_factors: dict[str, str] = Field(default_factory=dict)
    applicability_update: StrategyApplicabilityModel | None = None

    @field_validator("independence_factors")
    @classmethod
    def accepted_independence_factors(cls, value: dict[str, str]) -> dict[str, str]:
        allowed = {"domain", "benchmark", "codebase", "team", "model_family", "policy", "candidate_generator"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError("unsupported independence factors: " + ", ".join(unknown))
        return value


class StrategyTransferService:
    """Invariant-preserving service.  No API here copies source scientific data."""

    def __init__(self, store: ResearchStore):
        self.store = store

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _safe_pattern(item: dict[str, Any]) -> dict[str, Any]:
        data = item["data"]
        # Explicitly omit raw source results/evidence.  A target learns only the
        # method, applicability, and opaque provenance IDs.
        return {
            "id": str(item["id"]), "source_project_id": str(item["project_id"]),
            "name": data["name"], "strategy_type": data["strategy_type"],
            "principle": data["principle"], "mechanism_of_value": data["mechanism_of_value"],
            "applicability": data["applicability"], "scope": data["scope"],
            "maturity": data["maturity"], "authority_class": data["authority_class"],
            "transfer_counts": data.get("transfer_counts", {}),
            "source_reference_ids": {"decision_ids": data.get("source_decision_ids", []), "outcome_ids": data.get("source_outcome_ids", [])},
        }

    def pattern_create(self, project_id: str, draft: StrategyPatternCreate) -> dict[str, Any]:
        self.store.project_get(project_id)
        if draft.authority_class == "CORE" and draft.maturity != StrategyMaturity.CORE_INVARIANT:
            raise GPUError("STRATEGY_CORE_INVARIANT_MATURITY_REQUIRED", "CORE authority requires CORE_INVARIANT maturity")
        if draft.maturity == StrategyMaturity.CORE_INVARIANT and draft.authority_class != "CORE":
            raise GPUError("STRATEGY_CORE_INVARIANT_AUTHORITY_REQUIRED", "CORE_INVARIANT must be authority class CORE")
        if draft.parent_strategy_id:
            parent = self.store.object_get(draft.parent_strategy_id)
            if parent["kind"] != "ResearchStrategyPattern":
                raise GPUError("NOT_A_STRATEGY_PATTERN", draft.parent_strategy_id)
        data = {**draft.model_dump(mode="json"), "created_at": self._now(), "transfer_counts": {"retrieved": 0, "considered": 0, "applied": 0, "positive": 0, "negative": 0, "neutral": 0, "invalid": 0, "inconclusive": 0}, "genealogy": ([{"relation": "DERIVED_FROM", "strategy_id": draft.parent_strategy_id}] if draft.parent_strategy_id else [])}
        return self.store.object_create(project_id, "ResearchStrategyPattern", data, "STRATEGY_PATTERN_CREATED", "ACTIVE")

    def search(self, target_project_id: str, context: dict[str, str | bool | int | float], *, decision_id: str | None = None, discovery_mode: str = "STRATEGY_AUGMENTED_GENERATION", limit: int = 5) -> dict[str, Any]:
        self.store.project_get(target_project_id)
        if discovery_mode == "STATE_ONLY_GENERATION":
            return {"mode": discovery_mode, "strategies": [], "anchoring_protected": True, "note": "No cross-project strategy context was retrieved."}
        candidates = []
        for item in self.store.objects_global_list("ResearchStrategyPattern", {"ACTIVE", "WEAKENED"}, limit=None):
            if str(item["project_id"]) == str(target_project_id):
                continue
            data = item["data"]
            if data.get("authority_class") == "CORE":
                continue  # Core invariants are mandatory local policy, not transfer candidates.
            if data.get("maturity") in {StrategyMaturity.REFUTED, StrategyMaturity.DEPRECATED}:
                continue
            applicability = data.get("applicability", {})
            required = applicability.get("required_conditions", {})
            contraindications = applicability.get("contraindications", {})
            mismatches = [key for key, value in required.items() if key in context and context[key] != value]
            contraindicated = [key for key, value in contraindications.items() if context.get(key) == value]
            state = ApplicabilityState.CONTRAINDICATED if contraindicated else (ApplicabilityState.PARTIAL_MATCH if mismatches else ApplicabilityState.STRONG_MATCH)
            candidates.append({**self._safe_pattern(item), "applicability_state": state, "mismatch_conditions": mismatches + contraindicated})
        candidates.sort(key=lambda x: (x["applicability_state"] == ApplicabilityState.STRONG_MATCH, x["transfer_counts"].get("positive", 0) - x["transfer_counts"].get("negative", 0)), reverse=True)
        selected = candidates[:max(1, min(limit, 10))]
        retrieval = self.store.object_create(
            target_project_id,
            "StrategyRetrievalEvent",
            {
                "strategy_ids": [x["id"] for x in selected],
                "decision_id": decision_id,
                "mode": discovery_mode,
                "context": context,
                "recorded_at": self._now(),
                "scientific_evidence_transfer": "FORBIDDEN",
            },
            "STRATEGY_RETRIEVAL_RECORDED",
            "COMPLETED",
        )
        for selected_pattern in selected:
            source = self.store.object_get(selected_pattern["id"])
            counts = {**source["data"].get("transfer_counts", {})}
            counts["retrieved"] = int(counts.get("retrieved", 0)) + 1
            self.store.object_update(str(source["id"]), {"transfer_counts": counts}, source["status"], "STRATEGY_RETRIEVED")
        if decision_id:
            self._instrument_decision(decision_id, "retrieved_strategy_ids", [x["id"] for x in selected], target_project_id)
        return {"mode": discovery_mode, "strategies": selected, "retrieval_event_id": str(retrieval["id"]), "anchoring_protected": False, "note": "Methods only; source-project scientific results are not exposed as target evidence."}

    def propose(self, target_project_id: str, proposal: StrategyTransferPropose) -> dict[str, Any]:
        pattern = self.store.object_get(proposal.strategy_id)
        if pattern["kind"] != "ResearchStrategyPattern":
            raise GPUError("NOT_A_STRATEGY_PATTERN", proposal.strategy_id)
        if str(pattern["project_id"]) == str(target_project_id):
            raise GPUError("STRATEGY_TRANSFER_SOURCE_EQUALS_TARGET", "Use local strategy selection rather than cross-project transfer")
        self.store.project_get(target_project_id)
        if proposal.decision_id:
            self._instrument_decision(proposal.decision_id, "considered_strategy_ids", [proposal.strategy_id], target_project_id)
        data = {**proposal.model_dump(), "source_project_id": str(pattern["project_id"]), "source_strategy_snapshot": self._safe_pattern(pattern), "status": TransferStatus.PROPOSED, "prospective_frozen_at": self._now(), "scientific_evidence_transfer": "FORBIDDEN"}
        return self.store.object_create(target_project_id, "StrategyTransferCandidate", data, "STRATEGY_TRANSFER_PROPOSED", TransferStatus.PROPOSED)

    def applicability_assess(self, candidate_id: str, assessment: StrategyApplicabilityAssessment) -> dict[str, Any]:
        candidate = self._candidate(candidate_id, {TransferStatus.PROPOSED, TransferStatus.ELIGIBLE})
        status = TransferStatus.SCREENED_OUT if assessment.state == ApplicabilityState.CONTRAINDICATED else TransferStatus.ELIGIBLE
        return self.store.object_update(candidate_id, {"applicability_assessment": assessment.model_dump(mode="json"), "assessed_at": self._now()}, status, "STRATEGY_APPLICABILITY_UPDATED")

    def apply(self, candidate_id: str, application: StrategyTransferApply) -> dict[str, Any]:
        candidate = self._candidate(candidate_id, {TransferStatus.ELIGIBLE})
        if application.decision_id:
            self._instrument_decision(application.decision_id, "applied_strategy_ids", [candidate["data"]["strategy_id"]], str(candidate["project_id"]))
        hypothesis = self.store.object_create(str(candidate["project_id"]), "StrategyTransferHypothesis", {"transfer_candidate_id": candidate_id, "strategy_id": candidate["data"]["strategy_id"], "prediction": candidate["data"]["predicted_benefit"], "failure_prediction": candidate["data"]["predicted_failure"], "applicability_assumptions": candidate["data"].get("applicability_assessment"), "frozen_at": self._now()}, "STRATEGY_TRANSFER_HYPOTHESIS_FROZEN", "PROSPECTIVE_TESTING")
        updated = self.store.object_update(candidate_id, {"application": application.model_dump(), "transfer_hypothesis_id": str(hypothesis["id"]), "applied_at": self._now()}, TransferStatus.APPLIED, "STRATEGY_TRANSFER_APPLIED")
        return {"candidate": updated, "transfer_hypothesis": hypothesis}

    def outcome_record(self, candidate_id: str, record: StrategyTransferOutcomeRecord) -> dict[str, Any]:
        candidate = self._candidate(candidate_id, {TransferStatus.APPLIED})
        if record.kind == TransferOutcomeKind.NEGATIVE_TRANSFER and not record.applicability_update:
            raise GPUError("NEGATIVE_TRANSFER_APPLICABILITY_UPDATE_REQUIRED", "Negative transfer must refine applicability or contraindications")
        outcome = self.store.object_create(str(candidate["project_id"]), "StrategyTransferOutcome", {**record.model_dump(mode="json"), "transfer_candidate_id": candidate_id, "strategy_id": candidate["data"]["strategy_id"], "source_project_id": candidate["data"]["source_project_id"], "target_project_id": str(candidate["project_id"]), "prospective_frozen_at": candidate["data"]["prospective_frozen_at"], "recorded_at": self._now(), "scientific_claim_transfer": "FORBIDDEN"}, f"STRATEGY_TRANSFER_{record.kind}", record.kind)
        pattern = self.store.object_get(candidate["data"]["strategy_id"])
        counts = {**pattern["data"].get("transfer_counts", {})}
        key = {TransferOutcomeKind.POSITIVE_TRANSFER: "positive", TransferOutcomeKind.NEGATIVE_TRANSFER: "negative", TransferOutcomeKind.NEUTRAL_TRANSFER: "neutral", TransferOutcomeKind.INVALID_TRANSFER: "invalid", TransferOutcomeKind.INCONCLUSIVE: "inconclusive"}[record.kind]
        counts[key] = int(counts.get(key, 0)) + 1
        update: dict[str, Any] = {"transfer_counts": counts, "last_transfer_outcome_id": str(outcome["id"]), "last_transfer_outcome_kind": record.kind}
        if record.kind == TransferOutcomeKind.NEGATIVE_TRANSFER:
            update["applicability"] = record.applicability_update.model_dump(mode="json")
            update["maturity"] = StrategyMaturity.WEAKENED
        self.store.object_update(str(pattern["id"]), update, "WEAKENED" if record.kind == TransferOutcomeKind.NEGATIVE_TRANSFER else pattern["status"], "STRATEGY_APPLICABILITY_UPDATED")
        completed = self.store.object_update(candidate_id, {"outcome_id": str(outcome["id"]), "completed_at": self._now()}, TransferStatus.INVALID if record.kind == TransferOutcomeKind.INVALID_TRANSFER else TransferStatus.COMPLETED, f"STRATEGY_TRANSFER_{record.kind}")
        return {"candidate": completed, "outcome": outcome}

    def promotion_status(self, strategy_id: str) -> dict[str, Any]:
        pattern = self.store.object_get(strategy_id)
        if pattern["kind"] != "ResearchStrategyPattern":
            raise GPUError("NOT_A_STRATEGY_PATTERN", strategy_id)
        outcomes = [x for x in self.store.objects_global_list("StrategyTransferOutcome", limit=None) if x["data"].get("strategy_id") == strategy_id]
        valid = [x for x in outcomes if x["status"] != TransferOutcomeKind.INVALID_TRANSFER]
        targets = {str(x["data"].get("target_project_id")) for x in valid}
        domains = {str(x["data"].get("independence_factors", {}).get("domain")) for x in valid if x["data"].get("independence_factors", {}).get("domain")}
        positive = [x for x in valid if x["status"] == TransferOutcomeKind.POSITIVE_TRANSFER]
        factor_signatures = {tuple(sorted(x["data"].get("independence_factors", {}).items())) for x in positive}
        eligible_domain = len(targets) >= 2 and len(positive) >= 2 and len(factor_signatures) >= 2
        eligible_global = eligible_domain and len(domains) >= 2 and len({tuple(sorted(x["data"].get("independence_factors", {}).items())) for x in positive}) >= 3
        return {"strategy_id": strategy_id, "current_scope": pattern["data"].get("scope"), "current_maturity": pattern["data"].get("maturity"), "outcomes": {"positive": len(positive), "negative": sum(x["status"] == TransferOutcomeKind.NEGATIVE_TRANSFER for x in valid), "invalid": len(outcomes) - len(valid), "target_projects": len(targets), "domains": len(domains)}, "eligible_domain_promotion": eligible_domain, "eligible_global_promotion": eligible_global, "fail_closed_note": "Promotion is a separate reversible meta-scientific decision; counts alone never promote."}

    def promotion_decide(self, strategy_id: str, target_scope: StrategyScope, rationale: str, correction_case_ids: list[str] | None = None) -> dict[str, Any]:
        """Make a durable, reversible scope decision after an explicit evidence check."""
        pattern = self.store.object_get(strategy_id)
        if pattern["kind"] != "ResearchStrategyPattern":
            raise GPUError("NOT_A_STRATEGY_PATTERN", strategy_id)
        status = self.promotion_status(strategy_id)
        if target_scope == StrategyScope.DOMAIN and not status["eligible_domain_promotion"]:
            raise GPUError("STRATEGY_PROMOTION_EVIDENCE_INSUFFICIENT", "Cross-project prospective support with independent contexts is required")
        if target_scope == StrategyScope.GLOBAL and not status["eligible_global_promotion"]:
            raise GPUError("STRATEGY_PROMOTION_EVIDENCE_INSUFFICIENT", "Cross-domain prospective support with independent contexts is required")
        correction_case_ids = correction_case_ids or []
        for case_id in correction_case_ids:
            case = self.store.object_get(case_id)
            if case["kind"] != "CorrectionCase":
                raise GPUError("NOT_A_CORRECTION_CASE", case_id)
        prior_scope = pattern["data"].get("scope")
        prior_maturity = pattern["data"].get("maturity")
        decision = self.store.object_create(
            str(pattern["project_id"]), "StrategyPromotionDecision",
            {"strategy_id": strategy_id, "from_scope": prior_scope, "to_scope": target_scope, "rationale": rationale, "promotion_status": status, "correction_case_ids": correction_case_ids, "reversible": True, "decided_at": self._now()},
            "STRATEGY_SCOPE_PROMOTION_DECIDED", "COMPLETED",
        )
        maturity = StrategyMaturity.CROSS_DOMAIN_SUPPORTED if target_scope == StrategyScope.GLOBAL else StrategyMaturity.CROSS_PROJECT_SUPPORTED
        updated = self.store.object_update(strategy_id, {"scope": target_scope, "maturity": maturity, "promotion_history": [*pattern["data"].get("promotion_history", []), {"decision_id": str(decision["id"]), "from_scope": prior_scope, "from_maturity": prior_maturity, "to_scope": target_scope, "to_maturity": maturity, "at": self._now()}]}, "ACTIVE", "STRATEGY_SCOPE_PROMOTED")
        return {"promotion_decision": decision, "strategy": updated}

    def registry_summary(self, project_id: str | None = None) -> dict[str, Any]:
        patterns = self.store.objects_global_list("ResearchStrategyPattern", limit=None) if project_id is None else self.store.objects_list(project_id, "ResearchStrategyPattern", limit=None)
        rows = [self._safe_pattern(x) for x in patterns]
        return {"version": "strategy-transfer-v3.5", "patterns": rows, "summary": {"total": len(rows), "by_scope": {scope: sum(x["scope"] == scope for x in rows) for scope in StrategyScope}, "global_without_prospective_transfer": [x["id"] for x in rows if x["scope"] == StrategyScope.GLOBAL and not x["transfer_counts"].get("positive")]}}

    def _candidate(self, candidate_id: str, allowed: set[TransferStatus]) -> dict[str, Any]:
        candidate = self.store.object_get(candidate_id)
        if candidate["kind"] != "StrategyTransferCandidate":
            raise GPUError("NOT_A_STRATEGY_TRANSFER_CANDIDATE", candidate_id)
        if candidate["status"] not in allowed:
            raise GPUError("INVALID_STRATEGY_TRANSFER_STATE", f"{candidate['status']} is not one of {sorted(allowed)}")
        return candidate

    def _instrument_decision(self, decision_id: str, field: str, strategy_ids: list[str], project_id: str) -> None:
        decision = self.store.object_get(decision_id)
        if decision["kind"] != "ResearchDecision":
            raise GPUError("NOT_A_RESEARCHDECISION", decision_id)
        if str(decision["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", decision_id)
        values = list(dict.fromkeys([*decision["data"].get(field, []), *strategy_ids]))
        self.store.object_update(decision_id, {field: values, "strategy_transfer_instrumented_at": self._now()}, decision["status"], "STRATEGY_DECISION_INSTRUMENTED")
