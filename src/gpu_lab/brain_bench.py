from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import GPUError


class ProvenanceKind(StrEnum):
    HISTORICAL_FACT = "HISTORICAL_FACT"
    RECONSTRUCTED_INFERENCE = "RECONSTRUCTED_INFERENCE"


class BenchmarkSplit(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION = "VALIDATION"
    HELD_OUT = "HELD_OUT"


class BenchmarkPolicy(StrEnum):
    CURRENT_BRAIN_V1 = "CURRENT_BRAIN_V1"
    CHEAPEST_FEASIBLE_ACTION = "CHEAPEST_FEASIBLE_ACTION"
    MAX_EXPECTED_INFORMATION_ACTION = "MAX_EXPECTED_INFORMATION_ACTION"
    LLM_DIRECT_WITHOUT_STRUCTURED_MEMORY = "LLM_DIRECT_WITHOUT_STRUCTURED_MEMORY"
    RANDOM_VALID_ACTION = "RANDOM_VALID_ACTION"
    BRAIN_V1_5 = "BRAIN_v1_5"
    BRAIN_V2_STRATEGY_AUGMENTED = "BRAIN_v2_STRATEGY_AUGMENTED"
    BRAIN_V3_1_DISCOVERY_SEARCH = "BRAIN_v3.1_DISCOVERY_SEARCH"
    BRAIN_V3_3_DISTRIBUTED_DISCOVERY = "BRAIN_v3.3_DISTRIBUTED_DISCOVERY"
    BRAIN_V3_4_DISTRIBUTED_CORRECTION = "BRAIN_v3.4_DISTRIBUTED_CORRECTION"


class SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ProvenanceKind
    source: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    captured_at: datetime
    note: str | None = None

    @field_validator("captured_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must include a UTC offset")
        return value.astimezone(UTC)


class BenchmarkAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1)
    action_type: str = Field(min_length=1)
    feasible: bool = True
    hypothesis_ids: list[str] = Field(default_factory=list)
    expected_information_gain: float = Field(default=0.0, ge=0.0)
    compute_cost: float = Field(default=0.0, ge=0.0)
    engineering_cost: float = Field(default=0.0, ge=0.0)
    execution_risk: float = Field(default=0.0, ge=0.0)
    tags: list[str] = Field(default_factory=list)
    prediction: str | None = None
    scientific_distance: str | None = None

    @property
    def total_cost(self) -> float:
        return self.compute_cost + self.engineering_cost + self.execution_risk


class EvaluationRubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_negative_memory: bool = True
    require_scope_preservation: bool = True
    require_reproduction_gate: bool = True
    require_null_model: bool = False
    minimum_hypothesis_niches: int = Field(default=1, ge=0)
    architecture_action_types: list[str] = Field(
        default_factory=lambda: ["ARCHITECTURE_DESIGN", "TRAINING_RUN"]
    )


class BenchmarkEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    benchmark_split: BenchmarkSplit = BenchmarkSplit.DEVELOPMENT
    episode_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    cutoff_timestamp: datetime
    scientific_question: str = Field(min_length=1)
    visible_state: dict[str, Any]
    hidden_future_state: dict[str, Any]
    known_dead_lineages: list[str] = Field(default_factory=list)
    known_active_hypotheses: list[str] = Field(default_factory=list)
    known_unknowns: list[str] = Field(default_factory=list)
    candidate_actions: list[BenchmarkAction] = Field(min_length=1)
    strong_next_actions: list[str] = Field(min_length=1)
    acceptable_next_actions: list[str] = Field(default_factory=list)
    bad_next_actions: list[str] = Field(default_factory=list)
    forbidden_future_records: list[str] = Field(default_factory=list)
    source_provenance: list[SourceProvenance] = Field(min_length=1)
    evaluation_rubric: EvaluationRubric = Field(default_factory=EvaluationRubric)
    v31_context: dict[str, Any] = Field(default_factory=dict)
    v33_context: dict[str, Any] = Field(default_factory=dict)
    # Correction fixtures encode only pre-cutoff critique/verification facts.
    # A policy never receives adjudication or outcome facts from the future.
    v34_context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("cutoff_timestamp")
    @classmethod
    def require_aware_cutoff(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("cutoff_timestamp must include a UTC offset")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_action_references(self) -> BenchmarkEpisode:
        action_ids = {action.action_id for action in self.candidate_actions}
        referenced = set(
            self.strong_next_actions + self.acceptable_next_actions + self.bad_next_actions
        )
        missing = sorted(referenced - action_ids)
        if missing:
            raise ValueError(f"benchmark action references are missing: {', '.join(missing)}")
        if set(self.strong_next_actions) & set(self.bad_next_actions):
            raise ValueError("an action cannot be both strong and bad")
        return self

    def visible_payload(self) -> dict[str, Any]:
        """Return the only episode payload a policy is permitted to observe."""
        return {
            "episode_id": self.episode_id,
            "project_id": self.project_id,
            "domain": self.domain,
            "cutoff_timestamp": self.cutoff_timestamp.isoformat(),
            "scientific_question": self.scientific_question,
            "visible_state": self.visible_state,
            "known_dead_lineages": self.known_dead_lineages,
            "known_active_hypotheses": self.known_active_hypotheses,
            "known_unknowns": self.known_unknowns,
            "candidate_actions": [
                action.model_dump(
                    mode="json",
                    exclude={
                        "tags",
                        "expected_information_gain",
                        "compute_cost",
                        "engineering_cost",
                        "execution_risk",
                    },
                )
                for action in self.candidate_actions
            ],
            "v31_context": self.v31_context,
            "v33_context": self.v33_context,
            "v34_context": self.v34_context,
        }


class BenchmarkDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_action_id: str
    retrieved_record_ids: list[str] = Field(default_factory=list)
    considered_hypothesis_ids: list[str] = Field(default_factory=list)
    considered_null_models: list[str] = Field(default_factory=list)
    prediction: str | None = None
    claimed_scope: str | None = None
    expected_information_gain: float | None = Field(default=None, ge=0.0)
    realized_information_gain: float | None = Field(default=None, ge=0.0)
    decision_relevance: float | None = Field(default=None, ge=0.0)
    gpu_hours: float | None = Field(default=None, ge=0.0)
    cross_project_transfer: str | None = Field(
        default=None, pattern="^(NONE|POSITIVE|NEGATIVE)$"
    )
    strategy_reused: bool = False
    strategy_reuse_succeeded: bool | None = None


class MetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float | None
    passed: bool | None = None
    details: str


class BenchmarkScorecard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    policy: BenchmarkPolicy
    selected_action_id: str
    metrics: dict[str, MetricResult]


class AggregateMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: float | None
    observations: int


class BenchmarkAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scorecards: int
    metrics: dict[str, AggregateMetric]


class BenchmarkPolicyRunner(Protocol):
    def __call__(self, visible_episode: dict[str, Any]) -> BenchmarkDecision: ...


class ResearchBrainBench:
    """Leakage-resistant episode loading, policy selection, and transparent scoring."""

    def __init__(self, root: Path):
        self.root = root

    def load_episode(self, episode: str | Path) -> BenchmarkEpisode:
        path = Path(episode)
        if not path.is_absolute() and path.parent == Path("."):
            path = self.root / path
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return BenchmarkEpisode.model_validate(raw)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise GPUError(
                "BRAIN_BENCH_INVALID_EPISODE",
                f"Invalid benchmark episode {path.name}: {exc}",
                retryable=False,
            ) from exc

    def load_all(self) -> list[BenchmarkEpisode]:
        episodes = [self.load_episode(path) for path in sorted(self.root.glob("*.json"))]
        identifiers = [episode.episode_id for episode in episodes]
        if len(identifiers) != len(set(identifiers)):
            raise GPUError(
                "BRAIN_BENCH_DUPLICATE_EPISODE",
                "Benchmark episode IDs must be unique",
                retryable=False,
            )
        return episodes

    def get_episode(self, episode_id: str) -> BenchmarkEpisode:
        for episode in self.load_all():
            if episode.episode_id == episode_id:
                return episode
        raise GPUError(
            "BRAIN_BENCH_EPISODE_NOT_FOUND",
            episode_id,
            retryable=False,
        )

    @staticmethod
    def _seed(episode: BenchmarkEpisode) -> int:
        digest = hashlib.sha256(episode.episode_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8])

    @classmethod
    def builtin_policy_decision(
        cls, episode: BenchmarkEpisode, policy: BenchmarkPolicy
    ) -> BenchmarkDecision:
        """Run an explicitly heuristic policy using only the blinded episode payload.

        This does not call an LLM and never consults hidden labels, tags, costs, or
        expected-information values. Its purpose is a reproducible v1/v1.5/v2
        comparison, not a claim that the frozen episodes are training data.
        """
        payload = episode.visible_payload()
        actions = [item for item in payload["candidate_actions"] if item.get("feasible", True)]
        if not actions:
            raise GPUError(
                "BRAIN_BENCH_NO_VALID_ACTION",
                f"Benchmark episode {episode.episode_id} has no feasible candidate action",
                retryable=False,
            )
        reproduction = str(payload["visible_state"].get("reproduction_status", "UNKNOWN")).upper()
        dead = set(payload["known_dead_lineages"])
        if policy == BenchmarkPolicy.CURRENT_BRAIN_V1:
            eligible = (
                [item for item in actions if item["action_type"] == "REPRODUCTION"]
                if reproduction != "REPRODUCED"
                else actions
            )
            selected = min(
                eligible,
                key=lambda item: (
                    item["action_type"] in {"TRAINING_RUN", "ARCHITECTURE_DESIGN", "CLAIM_PROMOTION"},
                    item["action_id"],
                ),
            )
            return BenchmarkDecision(selected_action_id=selected["action_id"])
        if policy == BenchmarkPolicy.BRAIN_V1_5:
            eligible = [
                item
                for item in actions
                if not (set(item.get("hypothesis_ids", [])) & dead)
                and item["action_type"] not in {"CLAIM_PROMOTION", "ARCHITECTURE_DESIGN"}
            ]
            if reproduction != "REPRODUCED":
                reproduction_actions = [
                    item for item in eligible if item["action_type"] == "REPRODUCTION"
                ]
                if reproduction_actions:
                    eligible = reproduction_actions
            selected = min(
                eligible or actions,
                key=lambda item: (
                    item["action_type"] in {"TRAINING_RUN"},
                    item["action_type"] not in {"FROZEN_DIAGNOSTIC", "HYPOTHESIS_GENERATION", "EVIDENCE_REVIEW"},
                    item["action_id"],
                ),
            )
            return BenchmarkDecision(
                selected_action_id=selected["action_id"],
                retrieved_record_ids=sorted(dead),
                considered_null_models=["required-null"]
                if episode.evaluation_rubric.require_null_model
                else [],
            )
        if policy == BenchmarkPolicy.BRAIN_V2_STRATEGY_AUGMENTED:
            v15 = cls.builtin_policy_decision(episode, BenchmarkPolicy.BRAIN_V1_5)
            selected = next(
                item for item in actions if item["action_id"] == v15.selected_action_id
            )
            action_types = {item["action_type"] for item in actions}
            if (
                episode.evaluation_rubric.require_null_model
                and "NULL_MODEL_TEST" in action_types
            ):
                selected = next(item for item in actions if item["action_type"] == "NULL_MODEL_TEST")
            return BenchmarkDecision(
                selected_action_id=selected["action_id"],
                retrieved_record_ids=sorted(dead),
                considered_null_models=["required-null"]
                if episode.evaluation_rubric.require_null_model
                else [],
                strategy_reused=True,
            )
        if policy == BenchmarkPolicy.BRAIN_V3_1_DISCOVERY_SEARCH:
            context = payload.get("v31_context", {})
            regime = str(context.get("expected_search_regime", "EXPLOIT"))
            if context.get("single_path_prerequisite"):
                selected = next(
                    (item for item in actions if item["action_type"] in {"ARTIFACT_ANALYSIS", "REPRODUCTION"}),
                    actions[0],
                )
            else:
                preferred = (
                    ["ORTHOGONAL", "FAR", "MID", "NEAR"]
                    if regime == "PARADIGM_RESET"
                    else ["FAR", "ORTHOGONAL", "MID", "NEAR"]
                    if regime == "DIVERGENT_SEARCH"
                    else ["MID", "NEAR", "FAR", "ORTHOGONAL"]
                )
                selected = next(
                    (item for distance in preferred for item in actions if item.get("scientific_distance") == distance),
                    actions[0],
                )
            return BenchmarkDecision(
                selected_action_id=selected["action_id"],
                retrieved_record_ids=sorted(dead),
                strategy_reused=True,
            )
        if policy == BenchmarkPolicy.BRAIN_V3_3_DISTRIBUTED_DISCOVERY:
            context = payload.get("v33_context", {})
            if not context.get("generation_complete", True):
                selected = next(
                    (item for item in actions if item["action_type"] == "DISCOVERY_GENERATION"),
                    actions[0],
                )
                return BenchmarkDecision(selected_action_id=selected["action_id"], strategy_reused=True)
            survivors = set(context.get("qd_survivor_action_ids", []))
            selected = next((item for item in actions if item["action_id"] in survivors), None)
            if selected is None:
                return cls.builtin_policy_decision(episode, BenchmarkPolicy.BRAIN_V3_1_DISCOVERY_SEARCH)
            return BenchmarkDecision(
                selected_action_id=selected["action_id"],
                retrieved_record_ids=sorted(dead),
                strategy_reused=True,
            )
        if policy == BenchmarkPolicy.BRAIN_V3_4_DISTRIBUTED_CORRECTION:
            context = payload.get("v34_context", {})
            phase = str(context.get("correction_phase", "")).upper()
            required_type = {
                "INDEPENDENT_CRITIQUE": "CORRECTION_CHALLENGE",
                "VERIFICATION": "CORRECTION_VERIFICATION",
                "EXPERIMENT_REQUIRED": "DISCRIMINATING_TEST_DESIGN",
            }.get(phase)
            selected = next(
                (item for item in actions if required_type and item["action_type"] == required_type),
                None,
            )
            if selected is None:
                return cls.builtin_policy_decision(episode, BenchmarkPolicy.BRAIN_V3_3_DISTRIBUTED_DISCOVERY)
            return BenchmarkDecision(
                selected_action_id=selected["action_id"],
                retrieved_record_ids=sorted(dead),
                strategy_reused=True,
            )
        if policy == BenchmarkPolicy.LLM_DIRECT_WITHOUT_STRUCTURED_MEMORY:
            selected = min(actions, key=lambda item: item["action_id"])
            return BenchmarkDecision(selected_action_id=selected["action_id"])
        raise GPUError(
            "BRAIN_BENCH_POLICY_RUNNER_REQUIRED",
            f"{policy.value} requires an explicit policy runner for {episode.episode_id}",
            retryable=False,
        )

    @classmethod
    def baseline_decision(
        cls,
        episode: BenchmarkEpisode,
        policy: BenchmarkPolicy,
        runner: BenchmarkPolicyRunner | None = None,
    ) -> BenchmarkDecision:
        feasible = [action for action in episode.candidate_actions if action.feasible]
        if not feasible:
            raise GPUError(
                "BRAIN_BENCH_NO_VALID_ACTION",
                f"Benchmark episode {episode.episode_id} has no feasible candidate action",
                retryable=False,
            )
        if policy == BenchmarkPolicy.CHEAPEST_FEASIBLE_ACTION:
            selected = min(feasible, key=lambda action: (action.total_cost, action.action_id))
            return BenchmarkDecision(selected_action_id=selected.action_id)
        if policy == BenchmarkPolicy.MAX_EXPECTED_INFORMATION_ACTION:
            selected = max(
                feasible,
                key=lambda action: (action.expected_information_gain, -action.total_cost),
            )
            return BenchmarkDecision(
                selected_action_id=selected.action_id,
                expected_information_gain=selected.expected_information_gain,
            )
        if policy == BenchmarkPolicy.RANDOM_VALID_ACTION:
            selected = random.Random(cls._seed(episode)).choice(feasible)
            return BenchmarkDecision(selected_action_id=selected.action_id)
        if policy in {
            BenchmarkPolicy.CURRENT_BRAIN_V1,
            BenchmarkPolicy.BRAIN_V1_5,
            BenchmarkPolicy.BRAIN_V2_STRATEGY_AUGMENTED,
            BenchmarkPolicy.BRAIN_V3_1_DISCOVERY_SEARCH,
            BenchmarkPolicy.BRAIN_V3_3_DISTRIBUTED_DISCOVERY,
            BenchmarkPolicy.BRAIN_V3_4_DISTRIBUTED_CORRECTION,
            BenchmarkPolicy.LLM_DIRECT_WITHOUT_STRUCTURED_MEMORY,
        } and runner is None:
            return cls.builtin_policy_decision(episode, policy)
        if runner is None:
            raise GPUError(
                "BRAIN_BENCH_POLICY_RUNNER_REQUIRED",
                f"{policy.value} requires an explicit policy runner for {episode.episode_id}",
                retryable=False,
            )
        return runner(episode.visible_payload())

    def compare_builtin_policies(self) -> dict[str, Any]:
        """Score all required reproducible baselines without exposing answer keys to a policy."""
        policies = [
            BenchmarkPolicy.CURRENT_BRAIN_V1,
            BenchmarkPolicy.CHEAPEST_FEASIBLE_ACTION,
            BenchmarkPolicy.MAX_EXPECTED_INFORMATION_ACTION,
            BenchmarkPolicy.LLM_DIRECT_WITHOUT_STRUCTURED_MEMORY,
            BenchmarkPolicy.RANDOM_VALID_ACTION,
            BenchmarkPolicy.BRAIN_V1_5,
            BenchmarkPolicy.BRAIN_V2_STRATEGY_AUGMENTED,
            BenchmarkPolicy.BRAIN_V3_1_DISCOVERY_SEARCH,
            BenchmarkPolicy.BRAIN_V3_3_DISTRIBUTED_DISCOVERY,
            BenchmarkPolicy.BRAIN_V3_4_DISTRIBUTED_CORRECTION,
        ]
        episodes = self.load_all()
        results = {}
        for policy in policies:
            cards = [
                self.score(episode, policy, self.baseline_decision(episode, policy))
                for episode in episodes
            ]
            results[policy.value] = self.aggregate(cards).model_dump(mode="json")
        return {
            "benchmark_version": "brain-bench-v3-4",
            "episode_count": len(episodes),
            "results": results,
            "warning": "These are sourced historical scorecards; no model policy is claimed validated by this comparison alone.",
        }

    @staticmethod
    def score(
        episode: BenchmarkEpisode,
        policy: BenchmarkPolicy,
        decision: BenchmarkDecision,
    ) -> BenchmarkScorecard:
        actions = {action.action_id: action for action in episode.candidate_actions}
        selected = actions.get(decision.selected_action_id)
        if selected is None:
            raise GPUError(
                "BRAIN_BENCH_UNKNOWN_ACTION",
                "Policy selected an action outside the frozen candidate set: "
                f"{episode.episode_id}/{decision.selected_action_id}",
                retryable=False,
            )

        retrieved = set(decision.retrieved_record_ids)
        future = retrieved & set(episode.forbidden_future_records)
        resurrected = set(selected.hypothesis_ids) & set(episode.known_dead_lineages)
        reproduction_status = str(
            episode.visible_state.get("reproduction_status", "UNKNOWN")
        ).upper()
        reproduction_gate_ok = (
            reproduction_status == "REPRODUCED" or selected.action_type == "REPRODUCTION"
        )
        strong = decision.selected_action_id in episode.strong_next_actions
        bad = decision.selected_action_id in episode.bad_next_actions
        acceptable = strong or decision.selected_action_id in episode.acceptable_next_actions
        cheap_falsifier = "CHEAP_FALSIFIER" in selected.tags
        architecture_too_early = (
            selected.action_type in episode.evaluation_rubric.architecture_action_types
            and "ARCHITECTURE_AUTHORIZED" not in selected.tags
        )
        negative_memory_reused = bool(
            set(episode.known_dead_lineages) & set(decision.retrieved_record_ids)
        )
        null_covered = bool(decision.considered_null_models) or not (
            episode.evaluation_rubric.require_null_model
        )
        niches = {
            item.split(":", 1)[0]
            for item in decision.considered_hypothesis_ids
            if item
        }
        required_niches = episode.evaluation_rubric.minimum_hypothesis_niches
        prediction_specificity = 1.0 if (decision.prediction or selected.prediction) else 0.0
        expected_information = (
            decision.expected_information_gain
            if decision.expected_information_gain is not None
            else selected.expected_information_gain
        )
        realized_information = decision.realized_information_gain
        relevance = decision.decision_relevance
        gpu_hours = decision.gpu_hours
        information_per_gpu_hour = (
            realized_information / gpu_hours
            if realized_information is not None and gpu_hours is not None and gpu_hours > 0
            else None
        )

        def metric(value: float | None, passed: bool | None, details: str) -> MetricResult:
            return MetricResult(value=value, passed=passed, details=details)

        metrics = {
            "dead_idea_resurrection_rate": metric(
                float(bool(resurrected)), not resurrected, f"resurrected={sorted(resurrected)}"
            ),
            "future_information_leakage_rate": metric(
                float(bool(future)), not future, f"future_records={sorted(future)}"
            ),
            "unjustified_promotion_rate": metric(
                float("PROMOTE_WITHOUT_EVIDENCE" in selected.tags),
                "PROMOTE_WITHOUT_EVIDENCE" not in selected.tags,
                "derived from the frozen action tags",
            ),
            "scope_violation_rate": metric(
                float("SCOPE_VIOLATION" in selected.tags),
                "SCOPE_VIOLATION" not in selected.tags,
                f"claimed_scope={decision.claimed_scope!r}",
            ),
            "strong_next_action_recall": metric(float(strong), strong, "selected strong action"),
            "acceptable_next_action_rate": metric(
                float(acceptable), acceptable, "selected strong or acceptable action"
            ),
            "bad_action_selection_rate": metric(float(bad), not bad, "selected bad action"),
            "cheap_falsifier_selection_rate": metric(
                float(cheap_falsifier), None, "selected action tagged CHEAP_FALSIFIER"
            ),
            "architecture_too_early_rate": metric(
                float(architecture_too_early),
                not architecture_too_early,
                "architecture action requires ARCHITECTURE_AUTHORIZED",
            ),
            "duplicate_experiment_rate": metric(
                float("DUPLICATE_EXPERIMENT" in selected.tags),
                "DUPLICATE_EXPERIMENT" not in selected.tags,
                "derived from the frozen action tags",
            ),
            "reproduction_gate_compliance": metric(
                float(reproduction_gate_ok), reproduction_gate_ok, f"status={reproduction_status}"
            ),
            "negative_memory_reuse": metric(
                float(negative_memory_reused),
                negative_memory_reused
                if episode.evaluation_rubric.require_negative_memory
                else None,
                "dead lineage present in retrieved record IDs",
            ),
            "null_model_coverage": metric(
                float(null_covered), null_covered, "null model considered when required"
            ),
            "competing_hypothesis_diversity": metric(
                float(len(niches)),
                len(niches) >= required_niches,
                f"niches={len(niches)}, required={required_niches}",
            ),
            "prediction_specificity": metric(
                prediction_specificity,
                bool(prediction_specificity),
                "decision or selected action contains a prediction",
            ),
            "expected_information_gain": metric(
                expected_information, None, "transparent action/policy estimate"
            ),
            "realized_information_gain": metric(
                realized_information, None, "post-outcome value when available"
            ),
            "decision_relevance": metric(relevance, None, "post-outcome value when available"),
            "gpu_hours_per_resolved_uncertainty": metric(
                gpu_hours, None, "post-outcome GPU hours when available"
            ),
            "information_gain_per_gpu_hour": metric(
                information_per_gpu_hour, None, "realized information divided by GPU hours"
            ),
            "zero_information_decision_rate": metric(
                (
                    float(decision.realized_information_gain == 0.0)
                    if decision.realized_information_gain is not None
                    else None
                ),
                None,
                "only meaningful after outcome assessment",
            ),
            "cross_project_positive_transfer": metric(
                (
                    float(decision.cross_project_transfer == "POSITIVE")
                    if decision.cross_project_transfer is not None
                    else None
                ),
                None,
                "positive transfer must be established by a held-out transfer test",
            ),
            "cross_project_negative_transfer": metric(
                (
                    float(decision.cross_project_transfer == "NEGATIVE")
                    if decision.cross_project_transfer is not None
                    else None
                ),
                None,
                "negative transfer must be preserved rather than averaged away",
            ),
            "strategy_reuse_success_rate": metric(
                (
                    float(decision.strategy_reuse_succeeded)
                    if decision.strategy_reused
                    and decision.strategy_reuse_succeeded is not None
                    else None
                ),
                None,
                "only meaningful when a strategy was reused and its outcome inspected",
            ),
        }
        return BenchmarkScorecard(
            episode_id=episode.episode_id,
            policy=policy,
            selected_action_id=decision.selected_action_id,
            metrics=metrics,
        )

    @staticmethod
    def aggregate(scorecards: list[BenchmarkScorecard]) -> BenchmarkAggregate:
        if not scorecards:
            raise GPUError(
                "BRAIN_BENCH_EMPTY_SCORECARDS",
                "At least one scorecard is required",
                retryable=False,
            )
        metric_names = set.intersection(*(set(card.metrics) for card in scorecards))
        metrics = {}
        for name in sorted(metric_names):
            values = [
                card.metrics[name].value
                for card in scorecards
                if card.metrics[name].value is not None
            ]
            metrics[name] = AggregateMetric(
                mean=sum(values) / len(values) if values else None,
                observations=len(values),
            )
        return BenchmarkAggregate(scorecards=len(scorecards), metrics=metrics)
