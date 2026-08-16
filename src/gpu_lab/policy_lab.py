"""Bounded, evidence-first evaluation of changes to research policy.

This module intentionally operates on policy records only.  It never writes a
WorldModel, scientific hypothesis, agenda, or strategy-memory record.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .brain_bench import BenchmarkDecision, BenchmarkPolicy, BenchmarkSplit, ResearchBrainBench
from .errors import GPUError
from .prompt_compiler import CORE_EPISTEMIC_INVARIANTS, PROVIDERS, PromptCompiler
from .research import ResearchStore, strategy_learning_eligibility


class ResearchPolicyData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    parent_policy_id: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    scientific_policy: dict[str, Any] = Field(default_factory=dict)
    engineering_policy: dict[str, Any] = Field(default_factory=dict)
    decision_policy: dict[str, Any] = Field(default_factory=dict)
    critic_policy: dict[str, Any] = Field(default_factory=dict)
    retrieval_policy: dict[str, Any] = Field(default_factory=dict)
    strategy_policy: dict[str, Any] = Field(default_factory=dict)
    evidence_policy: dict[str, Any] = Field(default_factory=dict)
    literature_policy: dict[str, Any] = Field(default_factory=dict)
    generalization_policy: dict[str, Any] = Field(default_factory=dict)
    meta_review_policy: dict[str, Any] = Field(default_factory=dict)
    core_epistemic_invariants: list[str] = Field(default_factory=lambda: list(CORE_EPISTEMIC_INVARIANTS))
    provider_adapters: dict[str, Any] = Field(default_factory=dict)
    known_failure_modes: list[str] = Field(default_factory=list)
    applicability: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class PromotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_strong_action_gain: float = 0.0
    max_scope_error_increase: float = 0.0
    max_cost_multiplier: float = 1.25
    require_held_out: bool = True
    auto_promote_production: bool = False


class PolicyDelta(BaseModel):
    """Validated, data-only change used by the blinded policy runner."""

    model_config = ConfigDict(extra="forbid")

    decision_policy: dict[str, Any] = Field(default_factory=dict)
    critic_policy: dict[str, Any] = Field(default_factory=dict)


class PolicyLabService:
    """Create and assess finite policy proposals without autonomous deployment."""

    def __init__(
        self,
        store: ResearchStore,
        bench: ResearchBrainBench,
        *,
        auto_evaluate: bool = True,
        auto_reject: bool = True,
        auto_revise: bool = True,
        max_revisions: int = 1,
        auto_promote_production: bool = False,
    ):
        self.store = store
        self.bench = bench
        self.auto_evaluate = auto_evaluate
        self.auto_reject = auto_reject
        self.auto_revise = auto_revise
        self.max_revisions = max_revisions
        self.promotion_policy = PromotionPolicy(auto_promote_production=auto_promote_production)
        self.compiler = PromptCompiler()

    def _create(self, project_id: str, kind: str, data: dict[str, Any], event: str, status: str):
        return self.store.object_create(project_id, kind, data, event, status)

    def _objects(self, project_id: str, kind: str) -> list[dict[str, Any]]:
        return self.store.objects_list(project_id, kind, limit=None)

    def _compile(self, policy: dict[str, Any], provider: str, runtime: str = "default") -> dict[str, Any]:
        return self.compiler.compile(policy, provider, runtime)

    def compile_policy(self, policy_id: str, target_provider: str = "GENERIC", target_runtime: str = "default", *, persist: bool = True) -> dict[str, Any]:
        policy = self.store.object_get(policy_id)
        if policy["kind"] != "ResearchPolicy":
            raise GPUError("NOT_A_RESEARCH_POLICY", policy_id)
        artifact = self._compile(policy, target_provider, target_runtime)
        if not persist:
            return artifact
        existing = next((item for item in self._objects(str(policy["project_id"]), "ResearchPolicyArtifact") if item["data"].get("content_hash") == artifact["content_hash"]), None)
        if existing:
            return existing
        return self._create(str(policy["project_id"]), "ResearchPolicyArtifact", artifact, "RESEARCH_POLICY_PROMPT_COMPILED", "COMPLETED")

    def _compile_production_artifacts(self, policy: dict[str, Any]) -> None:
        for provider in PROVIDERS:
            self.compile_policy(str(policy["id"]), provider)

    def ensure_production_policy(self, project_id: str) -> dict[str, Any]:
        policies = self._objects(project_id, "ResearchPolicy")
        production = next((p for p in policies if p["status"] == "PRODUCTION"), None)
        if production:
            self._compile_production_artifacts(production)
            return production
        payload = ResearchPolicyData(
            version=1,
            provenance={"source_type": "V2_5_BOOTSTRAP", "created_at": datetime.now(UTC).isoformat()},
            scientific_policy={"closed_cycle_required": True, "learning_namespace": "PRODUCTION_SCIENCE"},
            decision_policy={"falsification_first": True, "reproduction_gate": True},
            critic_policy={"null_reasoning": True},
            strategy_policy={"eligibility": "v2.2 fail-closed"},
            evidence_policy={"independence_required": True},
            literature_policy={"external_content_untrusted": True},
            meta_review_policy={"bounded": True},
            applicability={"scope": "PROJECT"},
            notes="Canonical v2.5 baseline; deterministic v2.2 code remains the implementation source.",
        ).model_dump(mode="json")
        transactional_ensure = getattr(self.store, "production_policy_ensure", None)
        policy = transactional_ensure(project_id, payload) if callable(transactional_ensure) else self._create(project_id, "ResearchPolicy", payload, "RESEARCH_POLICY_CREATED", "PRODUCTION")
        self._compile_production_artifacts(policy)
        return policy

    def detect_weaknesses(self, project_id: str, component: str | None = None) -> list[dict[str, Any]]:
        outcomes = self._objects(project_id, "ResearchDecisionOutcome")
        decisions = {str(item["id"]): item for item in self._objects(project_id, "ResearchDecision")}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for outcome in outcomes:
            data = outcome["data"]
            if data.get("label") not in {"LOW_VALUE", "ZERO_INFORMATION", "REDUNDANT", "PREMATURE", "INVALID"}:
                continue
            decision = decisions.get(str(data.get("decision_id")))
            if not decision or not strategy_learning_eligibility(decision, outcome)["eligible"]:
                continue
            action = str(data.get("action_type") or decision["data"].get("selected_action", {}).get("action_type") or "UNKNOWN")
            grouped.setdefault(action, []).append(outcome)
        weaknesses = []
        for action, records in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0]))[:5]:
            severity = sum(r["data"].get("label") in {"PREMATURE", "INVALID"} for r in records) + len(records)
            if len(records) < 2 and severity < 2:
                # A single ordinary low-value decision is an anecdote, not a
                # policy weakness. Severe invalid/premature events remain visible.
                continue
            payload = {
                "component": component or "experiment_selection",
                "description": f"Repeated low-information or invalid {action} decisions.",
                "supporting_outcome_ids": [str(r["id"]) for r in records],
                "frequency": len(records), "severity": severity,
                "cost": sum(float(r["data"].get("actual_compute_cost") or 0) for r in records),
                "confidence": min(0.95, 0.3 + 0.15 * len(records)),
                "scope": "PROJECT", "counterexamples": [],
            }
            weaknesses.append(self._create(project_id, "ResearchPolicyWeakness", payload, "POLICY_WEAKNESS_DETECTED", "ACTIVE"))
        return weaknesses

    @staticmethod
    def _paper_principle(paper: str) -> str:
        """Extract a deliberately small method principle from untrusted paper text.

        This is content classification only: embedded commands and architecture
        instructions are never interpreted as executable authority.
        """
        lowered = paper.lower()
        if "failure" in lowered or "negative result" in lowered:
            return "Use relevant failed assumptions as explicit constraints during candidate generation."
        if "experiment" in lowered or "hypothesis" in lowered:
            return "Require an explicit hypothesis-outcome matrix before selecting an experiment."
        return "Compare a proposed action with a mechanistically distinct runner-up before selection."

    @staticmethod
    def _hypothesis_payload(source_type: str, observed_problem: str, change: str, component: str, source_ids: list[str] | None = None) -> dict[str, Any]:
        return {
            "source_type": source_type, "source_ids": source_ids or [], "observed_problem": observed_problem,
            "proposed_change": change, "mechanism_of_improvement": "Makes the decision constraint explicit before action selection.",
            "applicability_conditions": {"component": component},
            "expected_benefits": ["better scientific discrimination"],
            "possible_harms": ["additional reasoning cost", "unnecessary structure in simple cases"],
            "affected_components": [component], "benchmark_predictions": {"strong_next_action_accuracy": "non-decreasing"},
            "regression_risks": ["scope_error_rate", "expected_cost"], "cost_expectation": "LOW",
        }

    def _hypotheses_for(self, project_id: str, source_type: str, problem: str, component: str, *, limit: int = 3, source_ids: list[str] | None = None) -> list[dict[str, Any]]:
        changes = [
            "Require an explicit hypothesis-outcome matrix for competing mechanisms.",
            "Require a runner-up action comparison before selecting repeated diagnostics.",
            "Require a null-focused critic when a cheap falsifier is available.",
        ]
        return [self._create(project_id, "PolicyHypothesis", self._hypothesis_payload(source_type, problem, change, component, source_ids), "POLICY_HYPOTHESIS_CREATED", "CANDIDATE") for change in changes[:limit]]

    def _duplicate_negative(self, project_id: str, semantic_change: str) -> dict[str, Any] | None:
        fingerprint = self._fingerprint(semantic_change)
        return next((item for item in self._objects(project_id, "PolicyNegativeResult") if item["data"].get("semantic_fingerprint") == fingerprint), None)

    @staticmethod
    def _fingerprint(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

    def _patch(self, project_id: str, policy: dict[str, Any], hypothesis: dict[str, Any]) -> dict[str, Any]:
        proposed = hypothesis["data"]["proposed_change"]
        duplicate = self._duplicate_negative(project_id, proposed)
        data = {
            "base_policy_id": str(policy["id"]), "policy_hypothesis_id": str(hypothesis["id"]),
            "affected_policy_sections": ["decision_policy", "critic_policy"],
            "semantic_change": proposed, "semantic_fingerprint": self._fingerprint(proposed),
            "implementation_change": {"type": "POLICY_CONSTRAINT", "enabled": True},
            "patch_type": "POLICY_SEMANTIC", "semantic_before": {}, "semantic_after": {},
            "generated_prompt_diff": None, "expected_behavior_change": hypothesis["data"]["expected_benefits"],
            "unintended_behavior_risks": hypothesis["data"]["possible_harms"], "source": hypothesis["data"]["source_type"],
            "prompt_change": None, "code_change": None, "config_change": None,
            "expected_effect": hypothesis["data"]["expected_benefits"],
            "applicability": hypothesis["data"]["applicability_conditions"], "exclusions": [],
            "regression_risks": hypothesis["data"]["regression_risks"], "generated_by": "policy_lab", "revision_count": 0,
        }
        if duplicate:
            data["duplicate_of_policy_negative_result_id"] = str(duplicate["id"])
            return self._create(project_id, "ResearchPolicyPatch", data, "POLICY_PATCH_REJECTED_DUPLICATE", "REJECTED")
        return self._create(project_id, "ResearchPolicyPatch", data, "POLICY_PATCH_CREATED", "CANDIDATE")

    @staticmethod
    def _policy_delta(semantic_change: Any) -> PolicyDelta:
        if not isinstance(semantic_change, str) or not semantic_change.strip():
            raise GPUError("POLICY_PATCH_SEMANTIC_CHANGE_INVALID", "semantic_change must be non-empty text")
        normalized = semantic_change.lower()
        if "hypothesis-outcome matrix" in normalized:
            return PolicyDelta(decision_policy={"required_artifact": "hypothesis_outcome_matrix", "preferred_action_types": ["REPRODUCTION", "CAUSAL_INTERVENTION"]})
        if "runner-up action comparison" in normalized:
            return PolicyDelta(decision_policy={"required_artifact": "runner_up_comparison", "preferred_action_types": ["REPRODUCTION", "NULL_MODEL_TEST"]})
        if "null-focused critic" in normalized:
            return PolicyDelta(critic_policy={"null_reasoning": "required_when_cheap_falsifier_available"}, decision_policy={"preferred_action_types": ["NULL_MODEL_TEST"]})
        raise GPUError("POLICY_PATCH_SEMANTIC_CHANGE_INVALID", "unsupported semantic policy change")

    def _run_patch(self, patch: dict[str, Any], episode: Any) -> BenchmarkDecision:
        # Candidate behavior is constrained to the blinded payload. A policy patch
        # cannot access labels, costs, tags, or hidden future state.
        payload = episode.visible_payload()
        actions = [a for a in payload["candidate_actions"] if a.get("feasible", True)]
        delta = self._policy_delta(patch["data"].get("semantic_change"))
        preferred = delta.decision_policy.get("preferred_action_types", [])
        selected = next((a for action_type in preferred for a in actions if a["action_type"] == action_type), actions[0])
        return BenchmarkDecision(selected_action_id=selected["action_id"], considered_null_models=["policy-required-null"])

    def _candidate_prompt(self, policy: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        self._validate_core_patch(patch)
        delta = self._policy_delta(patch["data"].get("semantic_change")).model_dump(mode="json")
        candidate = {**policy, "data": {**policy["data"], **{name: {**policy["data"].get(name, {}), **value} for name, value in delta.items() if value}}}
        return self._compile(candidate, "GENERIC")

    @staticmethod
    def _validate_core_patch(patch: dict[str, Any]) -> None:
        attempted = patch["data"].get("core_epistemic_invariants")
        if attempted is not None and set(attempted) != set(CORE_EPISTEMIC_INVARIANTS):
            raise GPUError("CORE_POLICY_INVARIANT_VIOLATION", "ordinary patches cannot alter core epistemic invariants")
        if patch["data"].get("patch_type") == "CORE_POLICY_CHANGE":
            raise GPUError("CORE_POLICY_CHANGE_REQUIRES_STRONG_REVIEW", "core policy changes require a separate review path")
        forbidden = {"benchmark_labels", "benchmark_splits", "evaluation_metrics", "promotion_thresholds", "evaluator_change"}
        if forbidden & set(patch["data"]):
            raise GPUError("INVALID_POLICY_EXPERIMENT", "candidate patches cannot alter their evaluator")

    def evaluate(self, project_id: str, patch_id: str) -> dict[str, Any]:
        patch = self.store.object_get(patch_id)
        if patch["kind"] != "ResearchPolicyPatch":
            raise GPUError("NOT_A_RESEARCH_POLICY_PATCH", patch_id)
        if str(patch["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", patch_id)
        if patch["status"] == "REJECTED":
            return {"patch": patch, "decision": "REJECTED", "reason": "duplicate or invalid candidate"}
        implementation = patch["data"].get("implementation_change")
        try:
            self._policy_delta(patch["data"].get("semantic_change"))
            baseline_policy = self.store.object_get(patch["data"]["base_policy_id"])
            baseline_prompt = self._compile(baseline_policy, "GENERIC")
            candidate_prompt = self._candidate_prompt(baseline_policy, patch)
        except GPUError as exc:
            audit = self._create(
                project_id,
                "PolicyEvaluationAudit",
                {
                    "patch_id": patch_id,
                    "outcome": "REJECTED_BEFORE_EVALUATION",
                    "reason_type": exc.error_type,
                    "reason": exc.message,
                    "held_out_access": "NONE",
                    "evaluator_integrity": "PRESERVED",
                    "leakage_audit": "PASS",
                },
                "POLICY_EVALUATOR_FIREWALL_AUDITED",
                "COMPLETED",
            )
            updated = self.store.object_update(
                patch_id,
                {"invalid_evaluation_reason": exc.error_type, "evaluation_audit_id": str(audit["id"])},
                "INVALID_EVALUATION",
                "POLICY_PATCH_INVALID",
            )
            return {"patch": updated, "decision": "INVALID_EVALUATION", "reason": exc.error_type, "audit": audit}
        if not patch["data"].get("semantic_change") or not isinstance(implementation, dict) or not implementation.get("enabled"):
            updated = self.store.object_update(
                patch_id,
                {"invalid_evaluation_reason": "POLICY_PATCH_NO_OP"},
                "INVALID_EVALUATION",
                "POLICY_PATCH_INVALID",
            )
            return {"patch": updated, "decision": "INVALID_EVALUATION", "reason": "POLICY_PATCH_NO_OP"}
        episodes = self.bench.load_all()
        if not episodes:
            raise GPUError("POLICY_BENCHMARK_EMPTY", "No benchmark episodes are available")
        baseline_cards = [
            self.bench.score(
                episode,
                BenchmarkPolicy.BRAIN_V2_STRATEGY_AUGMENTED,
                self.bench.baseline_decision(episode, BenchmarkPolicy.BRAIN_V2_STRATEGY_AUGMENTED),
            )
            for episode in episodes
        ]
        cards = [
            self.bench.score(episode, BenchmarkPolicy.BRAIN_V2_STRATEGY_AUGMENTED, self._run_patch(patch, episode))
            for episode in episodes
        ]
        baseline = self.bench.aggregate(baseline_cards).model_dump(mode="json")
        aggregate = self.bench.aggregate(cards).model_dump(mode="json")
        split_cards = {
            split.value: self.bench.aggregate(
                [card for episode, card in zip(episodes, cards, strict=True) if episode.benchmark_split == split]
            ).model_dump(mode="json")
            for split in BenchmarkSplit
        }
        hard_rate_metrics = {
            "future_information_leakage_rate",
            "scope_violation_rate",
            "bad_action_selection_rate",
            "architecture_too_early_rate",
        }
        regressions = [
            name
            for name in hard_rate_metrics
            if aggregate["metrics"].get(name, {}).get("mean") not in (None, 0)
            and aggregate["metrics"][name]["mean"] > baseline["metrics"].get(name, {}).get("mean", 0)
        ]
        held_out = split_cards[BenchmarkSplit.HELD_OUT.value]
        held_out_has_data = held_out["scorecards"] > 0
        no_primary_gain = (
            aggregate["metrics"]["strong_next_action_recall"]["mean"]
            < baseline["metrics"]["strong_next_action_recall"]["mean"]
        )
        if not held_out_has_data:
            regressions.append("held_out_coverage_missing")
        if no_primary_gain:
            regressions.append("strong_next_action_recall")
        status = "REJECTED" if regressions and self.auto_reject else "CANDIDATE" if regressions else "SUPPORTED_ON_BENCHMARK"
        experiment = self._create(project_id, "PolicyExperiment", {
            "baseline_policy_id": patch["data"]["base_policy_id"], "candidate_patch_id": patch_id,
            "policy_hypothesis_id": patch["data"]["policy_hypothesis_id"], "benchmark_version": "brain-bench-v2.5",
            "splits": {
                "development": [e.episode_id for e in episodes if e.benchmark_split == BenchmarkSplit.DEVELOPMENT],
                "validation": [e.episode_id for e in episodes if e.benchmark_split == BenchmarkSplit.VALIDATION],
                "held_out": [e.episode_id for e in episodes if e.benchmark_split == BenchmarkSplit.HELD_OUT],
            },
            "models": ["deterministic-policy-runner"], "seeds": [], "results": {"baseline": baseline, "overall": aggregate, "by_split": split_cards},
            "compiled_prompts": {"baseline": {key: baseline_prompt[key] for key in ("content_hash", "compiled_prompt_tokens")}, "candidate": {key: candidate_prompt[key] for key in ("content_hash", "compiled_prompt_tokens")}},
            "regressions": regressions, "decision": status, "confidence": "LIMITED", "namespace": "BENCHMARK",
        }, "POLICY_EXPERIMENT_COMPLETED", status)
        updated = self.store.object_update(patch_id, {"experiment_id": str(experiment["id"]), "benchmark_results": aggregate, "regressions": regressions, "generated_prompt_diff": {"baseline_hash": baseline_prompt["content_hash"], "candidate_hash": candidate_prompt["content_hash"], "token_delta": candidate_prompt["compiled_prompt_tokens"] - baseline_prompt["compiled_prompt_tokens"]}}, status, "POLICY_PATCH_EVALUATED")
        if status == "REJECTED":
            self._create(project_id, "PolicyNegativeResult", {"proposal": patch_id, "source": "automatic benchmark", "expected_improvement": patch["data"]["expected_effect"], "observed_result": aggregate, "failure_mode": "hard benchmark regression", "regressions": regressions, "benchmark_scope": "development", "models_tested": ["deterministic-policy-runner"], "revisit_condition": "materially different mechanism", "related_policy_patches": [patch_id], "semantic_fingerprint": patch["data"]["semantic_fingerprint"]}, "POLICY_NEGATIVE_RESULT_CREATED", "REJECTED")
        return {"patch": updated, "experiment": experiment, "decision": status}

    def _revise(self, project_id: str, rejected_patch: dict[str, Any], *, max_revisions: int | None = None) -> dict[str, Any] | None:
        revision = int(rejected_patch["data"].get("revision_count", 0)) + 1
        if revision > (self.max_revisions if max_revisions is None else max_revisions):
            return None
        data = {
            **rejected_patch["data"],
            "semantic_change": rejected_patch["data"]["semantic_change"] + " (bounded revision: preserve hard regression gates).",
            "revision_count": revision,
            "revises_patch_id": str(rejected_patch["id"]),
        }
        data["semantic_fingerprint"] = self._fingerprint(data["semantic_change"])
        return self._create(project_id, "ResearchPolicyPatch", data, "POLICY_PATCH_REVISED", "CANDIDATE")

    def _tournament(self, project_id: str, supported: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Compare surviving candidates on the frozen benchmark; never combine by brute force."""
        if len(supported) < 2:
            return None
        participants = []
        for evaluation in supported:
            patch = evaluation["patch"]
            metrics = patch["data"].get("benchmark_results", {}).get("metrics", {})
            primary = metrics.get("strong_next_action_recall", {}).get("mean")
            participants.append({"patch_id": str(patch["id"]), "primary_score": primary if isinstance(primary, (int, float)) else -1.0})
        ranked = sorted(participants, key=lambda item: (-item["primary_score"], item["patch_id"]))
        return self._create(
            project_id,
            "PolicyTournament",
            {
                "candidate_patch_ids": [item["patch_id"] for item in ranked],
                "primary_metric": "strong_next_action_recall",
                "ranking": ranked,
                "winner_patch_id": ranked[0]["patch_id"],
                "combination_policy": "No combinations evaluated without an explicit complementarity hypothesis.",
            },
            "POLICY_TOURNAMENT_COMPLETED",
            "COMPLETED",
        )

    def improve(self, project_id: str, *, idea: str | None = None, paper: str | None = None, failure: str | None = None, component: str | None = None, search: bool = False, prompt: bool = False, candidate_budget: int | None = None, max_revisions: int | None = None, source_context: dict[str, list[str]] | None = None) -> dict[str, Any]:
        if candidate_budget is not None and candidate_budget < 1:
            raise GPUError("POLICY_CANDIDATE_BUDGET_EXHAUSTED", "candidate_budget must be at least one")
        if max_revisions is not None and max_revisions < 0:
            raise GPUError("POLICY_REVISION_BUDGET_INVALID", "max_revisions must be non-negative")
        policy = self.ensure_production_policy(project_id)
        source_type = "USER_IDEA" if idea else "PAPER" if paper else "BENCHMARK_FAILURE" if failure else "INTERNAL_META_REVIEW"
        problem = idea or failure or (
            self._paper_principle(paper)
            if paper
            else "Repeated closed-cycle low-value research decisions"
        )
        weaknesses = [] if (idea or paper or failure) else self.detect_weaknesses(project_id, component)
        source_ids = sorted({item for values in (source_context or {}).values() for item in values})
        hypotheses = self._hypotheses_for(project_id, source_type, problem, component or "experiment_selection", limit=candidate_budget or 3, source_ids=source_ids)
        patches = [self._patch(project_id, policy, h) for h in hypotheses]
        if prompt:
            for patch in patches:
                patch["data"].update({"patch_type": "PROMPT_PRESENTATION", "affected_policy_sections": ["decision_policy"], "prompt_mode": True})
        evaluations = [self.evaluate(project_id, str(p["id"])) for p in patches] if self.auto_evaluate else []
        if self.auto_revise:
            revisions = [
                self._revise(project_id, evaluation["patch"], max_revisions=max_revisions)
                for evaluation in evaluations
                if evaluation["decision"] == "REJECTED"
            ]
            revisions = [patch for patch in revisions if patch is not None]
            patches.extend(revisions)
            evaluations.extend(self.evaluate(project_id, str(patch["id"])) for patch in revisions)
        supported = [e for e in evaluations if e["decision"] == "SUPPORTED_ON_BENCHMARK"]
        tournament = self._tournament(project_id, supported)
        best_patch_id = tournament["data"]["winner_patch_id"] if tournament else str(supported[0]["patch"]["id"]) if supported else None
        run = self._create(project_id, "ImprovementRun", {"input": {"idea": idea, "paper": paper, "failure": failure, "component": component, "search": search, "prompt": prompt}, "source_context": source_context or {}, "budget": {"candidate_budget": candidate_budget or 3, "max_revisions": self.max_revisions if max_revisions is None else max_revisions}, "base_policy_id": str(policy["id"]), "weakness_ids": [str(w["id"]) for w in weaknesses], "hypothesis_ids": [str(h["id"]) for h in hypotheses], "patch_ids": [str(p["id"]) for p in patches], "evaluation_ids": [str(e["experiment"]["id"]) for e in evaluations if e.get("experiment")], "invalid_patch_ids": [str(e["patch"]["id"]) for e in evaluations if e["decision"] == "INVALID_EVALUATION"], "tournament_id": str(tournament["id"]) if tournament else None, "best_supported_patch_id": best_patch_id, "recommendation": "PROMOTE" if supported else "REJECT_OR_REVISE", "production_unchanged": True, "namespace": "META_RESEARCH"}, "IMPROVEMENT_RUN_COMPLETED", "COMPLETED")
        return {"improvement_run": run, "production_policy_id": str(policy["id"]), "weaknesses": weaknesses, "hypotheses": hypotheses, "patches": patches, "evaluations": evaluations, "tournament": tournament, "recommendation": run["data"]["recommendation"]}

    def promote(self, project_id: str, patch_id: str) -> dict[str, Any]:
        patch = self.store.object_get(patch_id)
        if patch["kind"] != "ResearchPolicyPatch" or str(patch["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", patch_id)
        if patch["status"] not in {"SUPPORTED_ON_BENCHMARK", "CROSS_PROJECT_SUPPORTED", "CROSS_MODEL_SUPPORTED", "RECOMMENDED_FOR_PROMOTION"}:
            raise GPUError("POLICY_PROMOTION_NOT_SUPPORTED", "Policy patch lacks required evidence")
        current = self.ensure_production_policy(project_id)
        next_version = max([int(p["data"].get("version", 0)) for p in self._objects(project_id, "ResearchPolicy")] or [0]) + 1
        delta = self._policy_delta(patch["data"]["semantic_change"]).model_dump(mode="json")
        self._validate_core_patch(patch)
        data = {**current["data"], **{section: {**current["data"].get(section, {}), **change} for section, change in delta.items() if change}, "version": next_version, "parent_policy_id": str(current["id"]), "provenance": {"source_type": "POLICY_PATCH", "patch_id": patch_id}, "notes": f"Promoted patch {patch_id}", "applied_patch_ids": [patch_id], "applied_policy_delta": delta}
        transactional_promote = getattr(self.store, "production_policy_promote", None)
        if callable(transactional_promote):
            promoted = transactional_promote(project_id, str(current["id"]), data)
            self._compile_production_artifacts(promoted)
            return promoted
        self.store.object_update(str(current["id"]), {}, "SUPERSEDED", "RESEARCH_POLICY_SUPERSEDED")
        promoted = self._create(project_id, "ResearchPolicy", data, "RESEARCH_POLICY_PROMOTED", "PRODUCTION")
        self._compile_production_artifacts(promoted)
        return promoted

    def classify_transfer(
        self,
        policy_experiment_id: str,
        project_results: dict[str, bool],
        model_results: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Attach explicit project/model transfer evidence to an isolated experiment."""
        experiment = self.store.object_get(policy_experiment_id)
        if experiment["kind"] != "PolicyExperiment":
            raise GPUError("NOT_A_POLICY_EXPERIMENT", policy_experiment_id)
        models = model_results or {}
        if len(project_results) <= 1:
            classification = "PROJECT_SPECIFIC"
        elif all(project_results.values()):
            classification = "CROSS_PROJECT_SUPPORTED"
        else:
            classification = "REJECTED"
        if models and len(set(models.values())) > 1:
            classification = "MODEL_SENSITIVE"
        patch_id = str(experiment["data"]["candidate_patch_id"])
        self.store.object_update(
            patch_id,
            {"transfer_classification": classification, "project_results": project_results, "model_results": models},
            classification,
            "POLICY_TRANSFER_CLASSIFIED",
        )
        return self.store.object_update(
            policy_experiment_id,
            {"transfer_classification": classification, "project_results": project_results, "model_results": models},
            classification,
            "POLICY_EXPERIMENT_TRANSFER_CLASSIFIED",
        )

    def evaluate_provider_compatibility(self, project_id: str, provider: str, model: str) -> dict[str, Any]:
        """Persist a compact adapter-compatibility check without claiming live-model transfer.

        Compiling the canonical policy checks that an adapter can express the
        invariants.  It is intentionally *not* evidence that the named model
        follows those instructions; such evidence requires an available model
        runner and is recorded as cross-model support separately.
        """
        policy = self.ensure_production_policy(project_id)
        normalized = provider.strip().upper()
        target_provider = {"OPENAI": "OPENAI_API", "ANTHROPIC": "CLAUDE_API"}.get(normalized, normalized)
        if target_provider not in PROVIDERS:
            target_provider = "GENERIC"
        artifact = self.compile_policy(str(policy["id"]), target_provider)
        return self._create(
            project_id,
            "PolicyExperiment",
            {
                "baseline_policy_id": str(policy["id"]),
                "candidate_patch_id": None,
                "policy_hypothesis_id": None,
                "benchmark_version": "provider-compatibility-v3",
                "provider": provider,
                "model": model,
                "adapter_provider": target_provider,
                "compiled_prompt": {key: artifact["data"].get(key) for key in ("content_hash", "compiled_prompt_tokens")},
                "results": {"adapter_compilation": "PASS", "live_model_evaluation": "UNAVAILABLE"},
                "transfer_classification": "CROSS_MODEL_UNVERIFIED",
                "namespace": "BENCHMARK",
            },
            "POLICY_COMPATIBILITY_EVALUATED",
            "CROSS_MODEL_UNVERIFIED",
        )

    def rollback(self, project_id: str, policy_id: str) -> dict[str, Any]:
        target = self.store.object_get(policy_id)
        if target["kind"] != "ResearchPolicy" or str(target["project_id"]) != project_id:
            raise GPUError("INVALID_POLICY_ROLLBACK_TARGET", policy_id)
        current = self.ensure_production_policy(project_id)
        if str(current["id"]) != policy_id:
            transactional_rollback = getattr(self.store, "production_policy_rollback", None)
            if callable(transactional_rollback):
                return transactional_rollback(project_id, str(current["id"]), policy_id)
            self.store.object_update(str(current["id"]), {}, "SUPERSEDED", "RESEARCH_POLICY_SUPERSEDED")
            return self.store.object_update(policy_id, {"rollback_from_policy_id": str(current["id"])}, "PRODUCTION", "RESEARCH_POLICY_ROLLED_BACK")
        return target

    def start_canary(self, project_id: str, candidate_policy_id: str, percentage: int = 10) -> dict[str, Any]:
        """Create a bounded prospective canary plan without changing production policy."""
        if not 1 <= percentage <= 50:
            raise GPUError("POLICY_CANARY_PERCENTAGE_INVALID", "percentage must be between 1 and 50")
        candidate = self.store.object_get(candidate_policy_id)
        if candidate["kind"] != "ResearchPolicy" or str(candidate["project_id"]) != str(project_id):
            raise GPUError("INVALID_POLICY_CANARY_TARGET", candidate_policy_id)
        production = self.ensure_production_policy(project_id)
        if str(production["id"]) == candidate_policy_id:
            raise GPUError("POLICY_CANARY_BASELINE_INVALID", "candidate must differ from current production policy")
        return self._create(
            project_id,
            "PolicyCanary",
            {
                "candidate_policy_id": candidate_policy_id,
                "baseline_policy_id": str(production["id"]),
                "percentage": percentage,
                "scope": candidate["data"].get("applicability", {}).get("scope", "PROJECT"),
                "decision_count": 0,
                "stop_conditions": ["hard epistemic regression", "severe negative transfer", "cost budget exceeded"],
            },
            "POLICY_CANARY_STARTED",
            "ACTIVE",
        )

    def record_shadow(
        self,
        project_id: str,
        production_policy_id: str,
        shadow_policy_id: str,
        decision_id: str,
        production_action: dict[str, Any],
        shadow_action: dict[str, Any],
        observed_production_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist an observational shadow comparison without inventing B's outcome."""
        production = self.store.object_get(production_policy_id)
        shadow = self.store.object_get(shadow_policy_id)
        if any(item["kind"] != "ResearchPolicy" or str(item["project_id"]) != str(project_id) for item in (production, shadow)):
            raise GPUError("INVALID_POLICY_SHADOW_TARGET", "policies must belong to project")
        return self._create(
            project_id,
            "PolicyShadowEvaluation",
            {
                "production_policy_id": production_policy_id,
                "shadow_policy_id": shadow_policy_id,
                "decision_id": decision_id,
                "production_action": production_action,
                "shadow_action": shadow_action,
                "observed_production_result": observed_production_result,
                "counterfactual_status": "COUNTERFACTUAL_UNKNOWN",
                "interpretation": "Only the production action's result is observed; the shadow action was not executed.",
            },
            "POLICY_SHADOW_RECORDED",
            "COMPLETED",
        )

    def policy_diff(self, base_policy_id: str, candidate_policy_id: str) -> dict[str, Any]:
        base, candidate = self.store.object_get(base_policy_id), self.store.object_get(candidate_policy_id)
        if base["kind"] != "ResearchPolicy" or candidate["kind"] != "ResearchPolicy":
            raise GPUError("NOT_A_RESEARCH_POLICY", "Both IDs must be ResearchPolicy records")
        changed = {key: {"from": base["data"].get(key), "to": candidate["data"].get(key)} for key in sorted(set(base["data"]) | set(candidate["data"])) if base["data"].get(key) != candidate["data"].get(key)}
        return {"base_policy_id": base_policy_id, "candidate_policy_id": candidate_policy_id, "changes": changed}

    def export_policy(self, policy_id: str, provider: str | None = None) -> dict[str, Any]:
        """Export a portable semantic policy; adapters are data, never instructions to execute."""
        policy = self.store.object_get(policy_id)
        if policy["kind"] != "ResearchPolicy":
            raise GPUError("NOT_A_RESEARCH_POLICY", policy_id)
        adapter_name = (provider or "generic").strip().lower()
        adapter = policy["data"].get("provider_adapters", {}).get(adapter_name, {})
        experiments = self._objects(str(policy["project_id"]), "PolicyExperiment")
        scorecards = [
            {"id": str(experiment["id"]), "decision": experiment["data"].get("decision"), "results": experiment["data"].get("results")}
            for experiment in experiments
            if experiment["data"].get("baseline_policy_id") == policy_id
        ]
        return {
            "policy_id": policy_id,
            "version": policy["data"].get("version"),
            "semantic_policy": {
                key: value
                for key, value in policy["data"].items()
                if key.endswith("_policy") or key in {"applicability", "known_failure_modes"}
            },
            "provider_compiled_form": {"provider": adapter_name, "adapter": adapter},
            "benchmark_scorecards": scorecards,
            "provenance": policy["data"].get("provenance", {}),
            "known_limitations": policy["data"].get("known_failure_modes", []),
        }

    def record_hindsight(
        self,
        policy_id: str,
        observed_improvement: float | None,
        observed_cost: float | None,
        unexpected_failure: str | None = None,
    ) -> dict[str, Any]:
        """Record post-promotion operational evidence without changing scientific state."""
        policy = self.store.object_get(policy_id)
        if policy["kind"] != "ResearchPolicy":
            raise GPUError("NOT_A_RESEARCH_POLICY", policy_id)
        history = list(policy["data"].get("post_promotion_hindsight", []))
        history.append({
            "recorded_at": datetime.now(UTC).isoformat(),
            "observed_improvement": observed_improvement,
            "observed_cost": observed_cost,
            "unexpected_failure": unexpected_failure,
        })
        predicted = policy["data"].get("benchmark_results", {}).get("strong_next_action_recall")
        calibration = None
        if isinstance(predicted, (int, float)) and observed_improvement is not None:
            calibration = observed_improvement - predicted
        updated = self.store.object_update(
            policy_id,
            {"post_promotion_hindsight": history, "policy_calibration_error": calibration},
            policy["status"],
            "RESEARCH_POLICY_HINDSIGHT_RECORDED",
        )
        self._create(
            str(policy["project_id"]),
            "PolicyHindsight",
            {
                "policy_id": policy_id,
                "predicted_benefit": predicted,
                "observed_improvement": observed_improvement,
                "actual_cost": observed_cost,
                "unexpected_failure": unexpected_failure,
                "scope": policy["data"].get("applicability", {}).get("scope", "PROJECT"),
                "calibration_error": calibration,
            },
            "RESEARCH_POLICY_HINDSIGHT_RECORDED",
            "COMPLETED",
        )
        return updated
