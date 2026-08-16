import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .errors import GPUError
from .research import ResearchStore
from .research import strategy_learning_eligibility as assess_strategy_learning_eligibility

DECISION_OUTCOME_LABELS = {
    "HIGH_VALUE",
    "USEFUL",
    "LOW_VALUE",
    "ZERO_INFORMATION",
    "REDUNDANT",
    "PREMATURE",
    "INVALID",
    "BLOCKED",
    "UNKNOWN",
}
REALIZED_INFORMATION_LABELS = {"HIGH", "MEDIUM", "LOW", "ZERO", "INVALID", "UNKNOWN"}
POSITIVE_OUTCOMES = {"HIGH_VALUE", "USEFUL"}
NEGATIVE_OUTCOMES = {
    "LOW_VALUE",
    "ZERO_INFORMATION",
    "REDUNDANT",
    "PREMATURE",
    "INVALID",
}
STRATEGY_POLICY_VERSION = "brain-v2-strategy-v1"
SCORING_POLICY_VERSION = "brain-v2-transparent-scoring-v1"


class ResearchSituationData(BaseModel):
    domain: str = Field(min_length=1, max_length=200)
    research_stage: str = Field(min_length=1, max_length=100)
    phenomenon_type: str = Field(min_length=1, max_length=200)
    uncertainty_type: str = Field(min_length=1, max_length=200)
    mechanism_status: str = Field(min_length=1, max_length=100)
    baseline_reproduced: bool
    active_hypothesis_count: int = Field(ge=0)
    dead_related_count: int = Field(ge=0)
    internal_state_access: bool
    strong_null_available: bool
    uninspected_result_available: bool
    contradiction_present: bool
    anomaly_present: bool
    scope_stage: str = Field(min_length=1, max_length=100)
    available_action_types: list[str] = Field(default_factory=list)
    compute_budget_class: str = "UNKNOWN"
    engineering_cost_class: str = "UNKNOWN"
    world_model_signature: str
    dominant_confounds: list[str] = Field(default_factory=list)
    prior_failed_strategies: list[str] = Field(default_factory=list)
    situation_signature: str
    policy_version: str = STRATEGY_POLICY_VERSION


class DecisionOutcomeAssessment(BaseModel):
    label: str
    observed_result: dict[str, Any]
    realized_information_gain: str
    hindsight_assessment: str = Field(min_length=1, max_length=10_000)
    experiment_run_ids: list[str] = Field(default_factory=list)
    evidence_family_ids: list[str] = Field(default_factory=list)
    actual_compute_cost: float | None = Field(default=None, ge=0)
    actual_engineering_cost: float | None = Field(default=None, ge=0)
    uncertainties_resolved: list[str] = Field(default_factory=list)
    hypotheses_eliminated: list[str] = Field(default_factory=list)
    failure_conditions: list[str] = Field(default_factory=list)
    information_gain_basis: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_labels(self) -> "DecisionOutcomeAssessment":
        if self.label not in DECISION_OUTCOME_LABELS:
            raise ValueError(f"Unsupported decision outcome label: {self.label}")
        if self.realized_information_gain not in REALIZED_INFORMATION_LABELS:
            raise ValueError(
                f"Unsupported realized information label: {self.realized_information_gain}"
            )
        if self.realized_information_gain not in {"UNKNOWN", "ZERO", "INVALID"} and not (
            self.information_gain_basis
            or self.uncertainties_resolved
            or self.hypotheses_eliminated
            or self.evidence_family_ids
        ):
            raise ValueError(
                "A non-zero information label requires an information-gain basis"
            )
        return self


class NullModelDraft(BaseModel):
    target_entity_id: str
    name: str = Field(min_length=1, max_length=300)
    mechanism: str = Field(min_length=1, max_length=5000)
    why_plausible: str = Field(min_length=1, max_length=5000)
    discriminating_control: str = Field(min_length=1, max_length=5000)
    expected_outcome: str = Field(min_length=1, max_length=5000)
    estimated_cost: float = Field(ge=0, le=5)
    strength: Literal["WEAK", "MEDIUM", "STRONG"] = "MEDIUM"
    action_type: str = "NULL_MODEL_TEST"
    scope: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class ResearchStrategyService:
    """Explicit, scoped research-process memory; never a scientific truth source."""

    def __init__(self, store: ResearchStore):
        self.store = store

    def _decision_with_outcome(self, decision_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        decision = self.store.object_get(decision_id)
        if decision["kind"] != "ResearchDecision":
            raise GPUError("NOT_A_RESEARCHDECISION", decision_id)
        outcomes = self.store.objects_list(str(decision["project_id"]), "ResearchDecisionOutcome", limit=None, data_filters={"decision_id": str(decision_id)})
        return decision, outcomes[0] if outcomes else None

    def strategy_learning_eligibility(self, decision_id: str) -> dict[str, Any]:
        decision, outcome = self._decision_with_outcome(decision_id)
        return assess_strategy_learning_eligibility(decision, outcome)

    def decision_epistemic_audit(self, decision_id: str) -> dict[str, Any]:
        """Return a read-only epistemic contract audit for one decision."""
        decision, outcome = self._decision_with_outcome(decision_id)
        eligibility = assess_strategy_learning_eligibility(decision, outcome)
        data = decision.get("data", {})
        selected = data.get("selected_action", {}) if isinstance(data.get("selected_action"), dict) else {}
        comparisons = data.get("candidate_comparison", [])
        critic = data.get("critics", [])
        critic_text = json.dumps(critic, default=str).lower()
        return {
            "decision_id": str(decision_id),
            "is_scientific": eligibility["decision_role"] == "SCIENTIFIC_ACTION"
            and eligibility["scientific_role"] not in {"NOT_SCIENTIFIC", "SYSTEM_SMOKE", "CONTRACT_TEST"},
            "decision_role": eligibility["decision_role"],
            "scientific_role": eligibility["scientific_role"],
            "execution_verification": eligibility["execution_verification"],
            "scientific_verification": eligibility["scientific_verification"],
            "cycle_status": eligibility["cycle_status"],
            "strategy_learning_eligible": eligibility["eligible"],
            "classification_reasons": eligibility.get("reasons", []),
            "warnings": eligibility.get("exclusions", []),
            "runner_up_compared": data.get("runner_up_candidate_index") is not None
            or bool(data.get("runner_up_action_type")),
            "hypotheses_distinguished": bool(selected.get("hypotheses_discriminated"))
            or any(item.get("hypotheses_discriminated") for item in comparisons if isinstance(item, dict)),
            "strong_null_considered": bool(data.get("strongest_null")) or "strongest_null" in critic_text,
            "hindsight_present": bool(data.get("hindsight_assessment") or (outcome or {}).get("data", {}).get("hindsight_assessment")),
            "realized_information_basis": (outcome or {}).get("data", {}).get("information_gain_basis", []),
            "scope_present": bool((outcome or {}).get("data", {}).get("scope") or data.get("scope")),
        }

    @staticmethod
    def _hash(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()

    @staticmethod
    def _terms(value: Any) -> set[str]:
        text = json.dumps(value, sort_keys=True, default=str).lower()
        return {
            term
            for term in "".join(character if character.isalnum() else " " for character in text).split()
            if len(term) >= 4
        }

    def construct_situation_data(
        self,
        project_id: str,
        agenda_item: dict[str, Any],
        world_model: dict[str, Any],
        candidate_actions: list[dict[str, Any]],
        dead_related_count: int,
        as_of: str | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        temporal = {"as_of": as_of} if as_of is not None else {}
        reproductions = self.store.objects_list(
            project_id, "Reproduction", limit=None, **temporal
        )
        hypotheses = self.store.objects_list(
            project_id,
            "Hypothesis",
            {"ACTIVE", "SURVIVES_INITIAL_TEST"},
            limit=None,
            **temporal,
        )
        dead = self.store.objects_list(
            project_id, "Hypothesis", {"REFUTED"}, limit=None, **temporal
        )
        edges = self.store.objects_list(
            project_id, "CausalEdge", limit=None, **temporal
        )
        anomalies = self.store.objects_list(
            project_id, "Anomaly", {"ACTIVE"}, limit=None, **temporal
        )
        contradictions = self.store.objects_list(
            project_id, "Contradiction", {"ACTIVE"}, limit=None, **temporal
        )
        nulls = self.store.objects_list(
            project_id, "NullModel", {"ACTIVE", "OPEN", "PROPOSED"}, limit=None, **temporal
        )
        uninspected = self.store.experiment_run_first(
            project_id,
            {"completed", "RESULT_NOT_INSPECTED", "failed", "cancelled"},
            inspected=False,
            **temporal,
        )
        baseline_reproduced = any(item["status"] == "REPRODUCED" for item in reproductions)
        action_types = sorted(
            {
                str(item.get("action_type"))
                for item in candidate_actions
                if item.get("action_type") and item.get("available", True)
            }
        )
        causal_supported = any(
            item["status"] == "INTERVENTION_SUPPORTED" for item in edges
        )
        if uninspected:
            research_stage = "RESULT_INSPECTION"
        elif reproductions and not baseline_reproduced:
            research_stage = "REPRODUCTION"
        elif not hypotheses:
            research_stage = "HYPOTHESIS_GENERATION"
        elif causal_supported:
            research_stage = "GENERALIZATION"
        else:
            research_stage = "MECHANISM_TESTING"
        mechanism_status = (
            "INTERVENTION_SUPPORTED"
            if causal_supported
            else "HYPOTHESIZED_CAUSAL"
            if any(item["status"] == "HYPOTHESIZED_CAUSAL" for item in edges)
            else "UNKNOWN"
        )
        scope = agenda_item["data"].get("scientific_scope")
        inferred_domain = (
            domain
            or agenda_item["data"].get("domain")
            or world_model["data"].get("domain")
            or (scope if isinstance(scope, str) and scope.strip() else None)
            or "UNSPECIFIED"
        )
        # Scientific scope can be long provenance prose; ResearchSituation.domain
        # is a bounded indexing label. Preserve the full scope on the agenda and
        # experiment records, but never let it violate the situation schema.
        inferred_domain = str(inferred_domain).strip()[:200] or "UNSPECIFIED"
        compute_costs = [
            float(item.get("score", {}).get("compute_cost", 0))
            for item in candidate_actions
            if isinstance(item.get("score", {}).get("compute_cost"), (int, float))
        ]
        engineering_costs = [
            float(item.get("score", {}).get("engineering_cost", 0))
            for item in candidate_actions
            if isinstance(item.get("score", {}).get("engineering_cost"), (int, float))
        ]
        world_signature_data = sorted(
            (
                item["data"].get("relation"),
                item["status"],
                item["data"].get("support_level"),
                item["data"].get("scope", {}),
            )
            for item in edges
        )
        strong_nulls = [
            item
            for item in nulls
            if item["data"].get("strength") == "STRONG"
            and float(item["data"].get("estimated_cost", 5)) <= 2
        ]
        structural = {
            "domain": inferred_domain,
            "research_stage": research_stage,
            "phenomenon_type": agenda_item["data"].get(
                "phenomenon_type", "UNSPECIFIED"
            ),
            "uncertainty_type": agenda_item["data"].get(
                "uncertainty_type", "MECHANISM"
            ),
            "mechanism_status": mechanism_status,
            "baseline_reproduced": baseline_reproduced,
            "active_hypothesis_count": len(hypotheses),
            "dead_related_count": max(dead_related_count, len(dead)),
            "internal_state_access": any(
                item in action_types
                for item in ("FROZEN_DIAGNOSTIC", "CAUSAL_INTERVENTION", "ABLATION")
            ),
            "strong_null_available": bool(strong_nulls),
            "uninspected_result_available": bool(uninspected),
            "contradiction_present": bool(contradictions),
            "anomaly_present": bool(anomalies),
            "scope_stage": "GENERALIZATION" if causal_supported else "WITHIN_SCOPE",
            "available_action_types": action_types,
            "compute_budget_class": self._cost_class(compute_costs),
            "engineering_cost_class": self._cost_class(engineering_costs),
            "world_model_signature": self._hash(world_signature_data),
            "dominant_confounds": [str(item["id"]) for item in [*anomalies, *contradictions, *strong_nulls]],
            "prior_failed_strategies": [],
        }
        signature_basis = {
            key: structural[key]
            for key in (
                "research_stage",
                "phenomenon_type",
                "uncertainty_type",
                "mechanism_status",
                "baseline_reproduced",
                "active_hypothesis_count",
                "dead_related_count",
                "internal_state_access",
                "strong_null_available",
                "uninspected_result_available",
                "contradiction_present",
                "anomaly_present",
                "scope_stage",
                "available_action_types",
                "compute_budget_class",
                "engineering_cost_class",
                "world_model_signature",
            )
        }
        return ResearchSituationData(
            **structural, situation_signature=self._hash(signature_basis)
        ).model_dump()

    def retrieve(
        self,
        project_id: str,
        situation: dict[str, Any],
        as_of: str | None = None,
    ) -> dict[str, Any]:
        patterns = self.store.objects_global_list(
            "ResearchStrategyPattern",
            {"ACTIVE", "WEAKENED"},
            limit=None,
            as_of=as_of,
        )
        applied, rejected = [], []
        situation_terms = self._terms(
            {
                key: situation.get(key)
                for key in (
                    "research_stage",
                    "phenomenon_type",
                    "uncertainty_type",
                    "mechanism_status",
                    "scope_stage",
                    "dominant_confounds",
                )
            }
        )
        for pattern in patterns:
            data = pattern["data"]
            reasons = self._applicability_mismatches(project_id, situation, pattern)
            pattern_terms = self._terms(
                {
                    "problem_signature": data.get("problem_signature"),
                    "research_stage": data.get("research_stage"),
                    "conditions": data.get("conditions", {}),
                }
            )
            union = situation_terms | pattern_terms
            semantic_similarity = (
                len(situation_terms & pattern_terms) / len(union) if union else 0.0
            )
            record = {
                "id": str(pattern["id"]),
                "scope_level": data.get("scope_level"),
                "support_level": data.get("support_level"),
                "action_type": data.get("action_type"),
                "semantic_similarity": round(semantic_similarity, 4),
                "applicability_conditions": data.get("applicability_conditions", {}),
                "counterexamples": data.get("counterexamples", []),
                "historical_successes": data.get("historical_successes", 0),
                "historical_failures": data.get("historical_failures", 0),
                "provenance": {
                    "decision_ids": data.get("decision_ids", []),
                    "outcome_ids": data.get("outcome_ids", []),
                    "project_ids": data.get("projects_observed", []),
                    "domains": data.get("domains_observed", []),
                },
            }
            if reasons:
                rejected.append({**record, "applicable": False, "mismatch_reasons": reasons})
                continue
            structured_matches = self._structured_match_count(situation, data)
            applicability = "HIGH" if structured_matches >= 4 else "MEDIUM"
            applied.append(
                {
                    **record,
                    "applicable": True,
                    "applicability": applicability,
                    "structured_match_count": structured_matches,
                }
            )
        applied.sort(
            key=lambda item: (
                item["applicability"] == "HIGH",
                item["historical_successes"] - item["historical_failures"],
                item["semantic_similarity"],
            ),
            reverse=True,
        )
        return {
            "policy_version": STRATEGY_POLICY_VERSION,
            "applied": applied[:25],
            "rejected": rejected[:25],
            "semantic_note": "Structured applicability gates transfer; lexical situation similarity is secondary.",
        }

    def agenda_telemetry(
        self,
        project_id: str,
        agenda_item: dict[str, Any],
        as_of: str | None = None,
    ) -> dict[str, Any]:
        temporal = {"as_of": as_of} if as_of is not None else {}
        decisions = self.store.objects_list(
            project_id,
            "ResearchDecision",
            limit=None,
            data_filters={"agenda_item_id": str(agenda_item["id"])},
            **temporal,
        )
        outcomes = self.store.objects_list(
            project_id, "ResearchDecisionOutcome", limit=None, **temporal
        )
        outcome_by_decision = {
            str(item["data"].get("decision_id")): item for item in outcomes
        }
        action_types = [
            str(item["data"].get("selected_action", {}).get("action_type", "UNKNOWN"))
            for item in decisions
        ]
        compute = [
            self._finite_nonnegative(
                outcome_by_decision.get(str(item["id"]), {}).get("data", {}).get(
                    "actual_compute_cost"
                )
            )
            for item in decisions
        ]
        information = [
            outcome_by_decision.get(str(item["id"]), {}).get("data", {}).get(
                "realized_information_gain"
            )
            for item in decisions
        ]
        low_count = sum(value == "LOW" for value in information)
        zero_count = sum(value == "ZERO" for value in information)
        cheap_probes = sum(
            float(item["data"].get("selected_action", {}).get("score", {}).get("compute_cost", 5))
            <= 1
            for item in decisions
        )
        repeated = max((action_types.count(item) for item in set(action_types)), default=0)
        recent = information[:3]
        age_days = max(
            0.0,
            (datetime.now(UTC) - self._aware(agenda_item.get("created_at"))).total_seconds()
            / 86400,
        )
        diminishing = (low_count + zero_count >= 2 and repeated >= 2) or (
            cheap_probes >= 3 and not any(value in {"HIGH", "MEDIUM"} for value in recent)
        )
        return {
            "agenda_item_id": str(agenda_item["id"]),
            "age_days": round(age_days, 4),
            "number_of_actions": len(decisions),
            "number_of_cheap_probes": cheap_probes,
            "repeated_action_family_count": repeated,
            "cumulative_compute": round(sum(value for value in compute if value is not None), 6),
            "cumulative_information_gain": information,
            "recent_information_gain": recent,
            "number_low_information_actions": low_count,
            "number_zero_information_actions": zero_count,
            "unresolved_duration_days": round(age_days, 4),
            "flag": "DIMINISHING_RETURNS" if diminishing else None,
        }

    def adjust_candidates(
        self,
        candidates: list[dict[str, Any]],
        retrieval: dict[str, Any],
        telemetry: dict[str, Any],
        hard_gate: bool,
    ) -> list[dict[str, Any]]:
        action_history: dict[str, list[dict[str, Any]]] = {}
        for pattern in retrieval["applied"]:
            action_history.setdefault(str(pattern["action_type"]), []).append(pattern)
        result = []
        for candidate in candidates:
            base = float(candidate["priority"])
            positive_adjustment = 0.0
            negative_adjustment = 0.0
            pattern_ids = []
            counterexamples = []
            if not hard_gate:
                for pattern in action_history.get(candidate["action_type"], []):
                    pattern_ids.append(pattern["id"])
                    successes = int(pattern["historical_successes"])
                    failures = int(pattern["historical_failures"])
                    applicability_weight = 1.0 if pattern["applicability"] == "HIGH" else 0.5
                    positive_adjustment += min(successes, 4) * 0.08 * applicability_weight
                    negative_adjustment -= min(failures, 4) * 0.12 * applicability_weight
                    counterexamples.extend(pattern["counterexamples"])
            diminishing_adjustment = 0.0
            if not hard_gate and telemetry["flag"] == "DIMINISHING_RETURNS":
                repeated_type = any(
                    pattern.get("action_type") == candidate["action_type"]
                    for pattern in retrieval["applied"]
                )
                cheap = float(candidate["score"]["compute_cost"]) <= 1
                if cheap and repeated_type:
                    diminishing_adjustment = -0.35
                elif float(candidate["score"]["expected_discrimination"]) >= 4:
                    diminishing_adjustment = 0.2
            multiplier = max(
                0.1,
                1 + positive_adjustment + negative_adjustment + diminishing_adjustment,
            )
            final = base if hard_gate else round(base * multiplier, 6)
            result.append(
                {
                    **candidate,
                    "base_priority": base,
                    "strategy_pattern_ids": pattern_ids,
                    "positive_strategy_adjustment": round(positive_adjustment, 6),
                    "negative_strategy_adjustment": round(negative_adjustment, 6),
                    "strategy_counterexamples": counterexamples,
                    "diminishing_return_adjustment": diminishing_adjustment,
                    "priority": final,
                    "final_priority": final,
                    "scoring_policy_version": SCORING_POLICY_VERSION,
                    "hard_gate_preserved": hard_gate,
                }
            )
        return result

    def decision_outcome_assess(
        self,
        decision_id: str,
        assessment: dict[str, Any],
        domain: str | None = None,
    ) -> dict[str, Any]:
        parsed = DecisionOutcomeAssessment.model_validate(assessment)
        decision = self.store.object_get(decision_id)
        if decision["kind"] != "ResearchDecision":
            raise GPUError("NOT_A_RESEARCHDECISION", decision_id)
        project_id = str(decision["project_id"])
        selected = decision["data"].get("selected_action", {})
        if not selected.get("action_type"):
            raise GPUError("RESEARCH_DECISION_INCOMPLETE", decision_id)
        situation_id = decision["data"].get("research_situation_id")
        if situation_id:
            situation = self.store.object_get(str(situation_id))
            before = situation["data"]
        else:
            before = self._legacy_situation(decision, domain)
            reconstructed = self.store.object_create(
                project_id,
                "ResearchSituation",
                {
                    **before,
                    "created_from_decision_id": decision_id,
                    "legacy_reconstructed": True,
                },
                "RESEARCH_SITUATION_CREATED",
            )
            situation_id = reconstructed["id"]
        agenda_item_id = decision["data"].get("agenda_item_id")
        if not agenda_item_id:
            raise GPUError("RESEARCH_DECISION_INCOMPLETE", "Missing agenda_item_id")
        agenda_item = self.store.object_get(str(agenda_item_id))
        models = self.store.objects_list(project_id, "WorldModel", limit=1)
        if not models:
            raise GPUError("BRAIN_STATE_INCOMPLETE", "Missing: WorldModel")
        candidate_actions = decision["data"].get("candidate_actions", [selected])
        after = self.construct_situation_data(
            project_id,
            agenda_item,
            models[0],
            candidate_actions,
            dead_related_count=len(decision["data"].get("dead_ideas_retrieved", [])),
            domain=domain or before.get("domain"),
        )
        retrieved_ids = sorted(
            {
                str(pattern_id)
                for item in decision["data"].get("strategy_patterns_retrieved", [])
                for pattern_id in [item.get("id")]
                if pattern_id
            }
            | {
                str(pattern_id)
                for pattern_id in selected.get("strategy_pattern_ids", [])
            }
        )
        conditions = {
            key: before.get(key)
            for key in (
                "research_stage",
                "mechanism_status",
                "baseline_reproduced",
                "internal_state_access",
                "strong_null_available",
                "scope_stage",
                "compute_budget_class",
                "engineering_cost_class",
            )
        }
        outcome_data = {
            **parsed.model_dump(),
            "project_id": project_id,
            "decision_id": decision_id,
            "before_situation_id": str(situation_id),
            "domain": domain or before.get("domain", "UNSPECIFIED"),
            "problem_signature": before.get("situation_signature")
            or self._hash(self._legacy_signature_basis(before)),
            "research_stage": before.get("research_stage", "UNKNOWN"),
            "conditions": conditions,
            "applicability_conditions": conditions,
            "action_type": selected["action_type"],
            "action_parameters_pattern": self._action_parameters(selected),
            "retrieved_strategy_pattern_ids": retrieved_ids,
            "policy_version": decision["data"].get(
                "brain_policy_version", "brain-v1-legacy"
            ),
        }
        return self.store.decision_outcome_apply(decision_id, outcome_data, after)

    def null_model_create(self, project_id: str, data: dict[str, Any]) -> dict[str, Any]:
        draft = NullModelDraft.model_validate(data)
        target = self.store.object_get(draft.target_entity_id)
        if str(target["project_id"]) != project_id or target["kind"] not in {
            "Claim",
            "Hypothesis",
            "CausalEdge",
            "Mechanism",
        }:
            raise GPUError("INVALID_NULL_MODEL_TARGET", draft.target_entity_id)
        return self.store.object_create(
            project_id,
            "NullModel",
            {**draft.model_dump(), "tested": False},
            "NULL_MODEL_CREATED",
            "ACTIVE",
        )

    def null_model_test(
        self,
        null_model_id: str,
        outcome: str,
        evidence_family_ids: list[str],
        rationale: str,
    ) -> dict[str, Any]:
        item = self.store.object_get(null_model_id)
        if item["kind"] != "NullModel":
            raise GPUError("NOT_A_NULLMODEL", null_model_id)
        if outcome not in {"ELIMINATED", "MIMICS_TARGET", "INCONCLUSIVE"}:
            raise GPUError("INVALID_NULL_MODEL_OUTCOME", outcome)
        if not evidence_family_ids or not rationale.strip():
            raise GPUError(
                "NULL_MODEL_TEST_EVIDENCE_REQUIRED",
                "A tested null requires EvidenceFamily provenance and rationale",
            )
        canonical_family_ids = [
            self.store._canonical_uuid(family_id) for family_id in evidence_family_ids
        ]
        references = self.store.references_get(canonical_family_ids)
        for family_id in canonical_family_ids:
            family = references.get(str(family_id))
            if (
                not family
                or family["kind"] != "EvidenceFamily"
                or str(family["project_id"]) != str(item["project_id"])
            ):
                raise GPUError("INVALID_EVIDENCE_FAMILY", family_id)
        status = "REFUTED" if outcome == "ELIMINATED" else "SUPPORTED" if outcome == "MIMICS_TARGET" else "INCONCLUSIVE"
        return self.store.object_update(
            null_model_id,
            {
                "tested": True,
                "test_outcome": outcome,
                "evidence_family_ids": evidence_family_ids,
                "test_rationale": rationale.strip(),
                "tested_at": datetime.now(UTC).isoformat(),
            },
            status,
            "NULL_MODEL_TESTED",
        )

    def strategy_list(
        self, project_id: str | None = None, as_of: str | None = None
    ) -> list[dict[str, Any]]:
        if project_id:
            return self.store.objects_list(
                project_id, "ResearchStrategyPattern", limit=None, as_of=as_of
            )
        return self.store.objects_global_list(
            "ResearchStrategyPattern", limit=None, as_of=as_of
        )

    def dataset_export(self, project_id: str | None = None) -> dict[str, Any]:
        if project_id:
            situations = self.store.objects_list(
                project_id, "ResearchSituation", limit=None
            )
            decisions = self.store.objects_list(
                project_id, "ResearchDecision", limit=None
            )
            outcomes = self.store.objects_list(
                project_id, "ResearchDecisionOutcome", limit=None
            )
        else:
            situations = self.store.objects_global_list("ResearchSituation", limit=None)
            decisions = self.store.objects_global_list("ResearchDecision", limit=None)
            outcomes = self.store.objects_global_list(
                "ResearchDecisionOutcome", limit=None
            )
        situations_by_id = {str(item["id"]): item for item in situations}
        outcomes_by_decision = {
            str(item["data"].get("decision_id")): item for item in outcomes
        }
        records = []
        for decision in decisions:
            outcome = outcomes_by_decision.get(str(decision["id"]))
            if not outcome:
                continue
            before_id = outcome["data"].get("before_situation_id")
            after_id = outcome["data"].get("after_situation_id")
            records.append(
                {
                    "decision_id": str(decision["id"]),
                    "project_id": str(decision["project_id"]),
                    "S_t": situations_by_id.get(str(before_id)),
                    "candidate_actions": decision["data"].get("candidate_actions", []),
                    "A_t": decision["data"].get("selected_action"),
                    "strategy_patterns_retrieved": decision["data"].get(
                        "strategy_patterns_retrieved", []
                    ),
                    "operator_outputs": decision["data"].get("operator_outputs", []),
                    "O_t": outcome["data"].get("observed_result"),
                    "R_t": {
                        "label": outcome["data"].get("label"),
                        "realized_information_gain": outcome["data"].get(
                            "realized_information_gain"
                        ),
                    },
                    "evidence_family_ids": outcome["data"].get(
                        "evidence_family_ids", []
                    ),
                    "S_t_plus_1": situations_by_id.get(str(after_id)),
                    "policy_version": outcome["data"].get("policy_version"),
                    "strategy_learning_eligibility": assess_strategy_learning_eligibility(
                        decision, outcome
                    ),
                }
            )
        return {
            "dataset_version": "brain-v2-strategy-dataset-v1",
            "exported_at": datetime.now(UTC).isoformat(),
            "project_id": project_id,
            "records": records,
            "record_count": len(records),
            "warning": "This is observational policy data, not an unbiased causal training set.",
        }

    @staticmethod
    def _cost_class(values: list[float]) -> str:
        if not values:
            return "UNKNOWN"
        value = min(values)
        return "LOW" if value <= 1 else "MEDIUM" if value <= 3 else "HIGH"

    @staticmethod
    def _finite_nonnegative(value: Any) -> float | None:
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            return float(value)
        return None

    @staticmethod
    def _aware(value: Any) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                parsed = datetime.now(UTC)
        else:
            parsed = datetime.now(UTC)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @staticmethod
    def _structured_match_count(
        situation: dict[str, Any], pattern_data: dict[str, Any]
    ) -> int:
        conditions = pattern_data.get("applicability_conditions", {})
        return sum(
            conditions.get(key) == situation.get(key)
            for key in (
                "research_stage",
                "mechanism_status",
                "baseline_reproduced",
                "internal_state_access",
                "strong_null_available",
                "scope_stage",
            )
            if key in conditions
        )

    @staticmethod
    def _applicability_mismatches(
        project_id: str, situation: dict[str, Any], pattern: dict[str, Any]
    ) -> list[str]:
        data = pattern["data"]
        mismatches = []
        scope_level = data.get("scope_level")
        if scope_level == "PROJECT" and project_id not in {
            str(item) for item in data.get("projects_observed", [])
        }:
            mismatches.append("PROJECT_SCOPE_MISMATCH")
        if scope_level == "DOMAIN" and situation.get("domain") not in set(
            data.get("domains_observed", [])
        ):
            mismatches.append("DOMAIN_SCOPE_MISMATCH")
        conditions = data.get("applicability_conditions", {})
        hard_fields = (
            "research_stage",
            "mechanism_status",
            "baseline_reproduced",
            "internal_state_access",
            "strong_null_available",
            "scope_stage",
        )
        for field in hard_fields:
            expected = conditions.get(field)
            if expected is not None and expected != situation.get(field):
                mismatches.append(f"{field.upper()}_MISMATCH")
        available = set(situation.get("available_action_types", []))
        if data.get("action_type") not in available:
            mismatches.append("ACTION_UNAVAILABLE")
        if data.get("support_level") in {"REFUTED", "WEAKENED"}:
            mismatches.append("STRATEGY_NOT_SUPPORTED")
        return mismatches

    @staticmethod
    def _action_parameters(selected: dict[str, Any]) -> dict[str, Any]:
        payload = selected.get("payload", {})
        return {
            key: payload[key]
            for key in sorted(payload)
            if key
            in {
                "mode",
                "intervention_type",
                "control_type",
                "metric_type",
                "provider",
            }
        }

    def _legacy_situation(
        self, decision: dict[str, Any], domain: str | None
    ) -> dict[str, Any]:
        snapshot = decision["data"].get("state_snapshot", {}).get("research_state", {})
        selected = decision["data"].get("selected_action", {})
        basis = {
            "domain": domain or "UNSPECIFIED",
            "research_stage": "LEGACY_RECONSTRUCTED",
            "phenomenon_type": "UNSPECIFIED",
            "uncertainty_type": "MECHANISM",
            "mechanism_status": "UNKNOWN",
            "baseline_reproduced": bool(snapshot.get("reproduction_status")),
            "active_hypothesis_count": len(snapshot.get("active_hypotheses", [])),
            "dead_related_count": len(decision["data"].get("dead_ideas_retrieved", [])),
            "internal_state_access": selected.get("action_type")
            in {"FROZEN_DIAGNOSTIC", "CAUSAL_INTERVENTION", "ABLATION"},
            "strong_null_available": False,
            "uninspected_result_available": False,
            "contradiction_present": bool(snapshot.get("active_contradictions")),
            "anomaly_present": bool(snapshot.get("active_anomalies")),
            "scope_stage": "WITHIN_SCOPE",
            "available_action_types": [selected.get("action_type", "UNKNOWN")],
            "compute_budget_class": "UNKNOWN",
            "engineering_cost_class": "UNKNOWN",
            "world_model_signature": self._hash(
                decision["data"].get("state_snapshot", {}).get("world_model_version_id")
            ),
            "dominant_confounds": [],
            "prior_failed_strategies": [],
            "policy_version": "brain-v1-legacy-reconstruction",
        }
        return {
            **basis,
            "situation_signature": self._hash(self._legacy_signature_basis(basis)),
        }

    @staticmethod
    def _legacy_signature_basis(value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.get(key)
            for key in (
                "research_stage",
                "phenomenon_type",
                "uncertainty_type",
                "mechanism_status",
                "baseline_reproduced",
                "active_hypothesis_count",
                "internal_state_access",
                "strong_null_available",
                "scope_stage",
                "available_action_types",
                "compute_budget_class",
                "engineering_cost_class",
                "world_model_signature",
            )
        }
