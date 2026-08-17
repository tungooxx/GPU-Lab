import math
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .discovery import (
    BRAIN_POLICY_VERSION,
    SearchRegime,
    choose_regime,
    classify_scientific_distance,
    fallback_candidates,
    frontier_gap,
    portfolio_critique,
    stagnation_state,
)
from .epistemics import normalize_scientific_scope, scope_is_empirically_bounded
from .errors import GPUError
from .research import ResearchStore
from .strategy import SCORING_POLICY_VERSION, STRATEGY_POLICY_VERSION, ResearchStrategyService

VerificationStatus = Literal[
    "IMPLEMENTED_UNVERIFIED",
    "VERIFIED_UNIT",
    "VERIFIED_INTEGRATION",
    "VERIFIED_REAL",
    "RESULT_NOT_INSPECTED",
    "RESULT_INSPECTED",
    "BLOCKED",
]

EDGE_STATUSES = {
    "OBSERVED_ASSOCIATION",
    "HYPOTHESIZED_CAUSAL",
    "INTERVENTION_SUPPORTED",
    "WEAKENED",
    "REFUTED",
    "UNKNOWN",
}
CAUSAL_RELATIONS = {
    "CAUSES",
    "INFLUENCES",
    "MEDIATES",
    "ENABLES",
    "INHIBITS",
    "ASSOCIATED_WITH",
    "TRANSMITS_INFORMATION_TO",
}
AGENDA_STATUSES = {"OPEN", "ACTIVE", "RESOLVED", "DEFERRED", "BLOCKED"}
ACTION_TYPES = {
    "LITERATURE_SEARCH",
    "EVIDENCE_REVIEW",
    "REPRODUCTION",
    "ARTIFACT_ANALYSIS",
    "FROZEN_DIAGNOSTIC",
    "CAUSAL_INTERVENTION",
    "ABLATION",
    "REPLICATION",
    "GENERALIZATION",
    "TRAINING_RUN",
    "NOVELTY_CHECK",
    "CODE_INSPECTION",
    "NULL_MODEL_TEST",
    "MATCHED_CONTROL",
    "RANDOM_CONTROL",
    "MAGNITUDE_MATCHED_CONTROL",
    "PORTFOLIO_REFINEMENT",
}
EXECUTABLE_ACTIONS = {
    "REPRODUCTION",
    "FROZEN_DIAGNOSTIC",
    "REPLICATION",
    "GENERALIZATION",
    "TRAINING_RUN",
    "CAUSAL_INTERVENTION",
    "ABLATION",
    "NULL_MODEL_TEST",
    "MATCHED_CONTROL",
    "RANDOM_CONTROL",
    "MAGNITUDE_MATCHED_CONTROL",
}


class ActionScore(BaseModel):
    scientific_importance: float = Field(ge=0.1, le=5)
    expected_discrimination: float = Field(ge=0.1, le=5)
    expected_information_gain: float = Field(ge=0.1, le=5)
    feasibility: float = Field(ge=0.1, le=5)
    compute_cost: float = Field(ge=0.1, le=5)
    engineering_cost: float = Field(ge=0.1, le=5)
    execution_risk: float = Field(ge=0.1, le=5)
    decision_relevance: float = Field(default=1, ge=0.1, le=5)

    @property
    def priority(self) -> float:
        numerator = (
            self.scientific_importance
            * self.expected_discrimination
            * self.expected_information_gain
            * self.feasibility
            * self.decision_relevance
        )
        denominator = self.compute_cost * self.engineering_cost * self.execution_risk
        return round(numerator / denominator, 6)


class ActionCandidate(BaseModel):
    action_type: str
    question_addressed: str
    hypotheses_discriminated: list[str] = Field(default_factory=list)
    predicted_outcomes: list[str] = Field(default_factory=list)
    required_resources: list[str] = Field(default_factory=list)
    score: ActionScore
    payload: dict[str, Any] = Field(default_factory=dict)
    available: bool = True
    blocked_reason: str | None = None

    def checked(self) -> "ActionCandidate":
        if self.action_type not in ACTION_TYPES:
            raise GPUError("INVALID_RESEARCH_ACTION_TYPE", self.action_type)
        return self

    def persisted_data(self) -> dict[str, Any]:
        data = self.model_dump()
        data["priority"] = self.score.priority if self.available else 0.0
        return data


class ResearchBrain:
    """Native deterministic scientific policy over canonical PostgreSQL state."""

    def __init__(self, store: ResearchStore):
        self.store = store
        self.strategy = ResearchStrategyService(store)

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """Return a lossless-enough JSON representation for decision snapshots."""
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, uuid.UUID):
            return str(value)
        return value

    @staticmethod
    def _state_snapshot(state: dict) -> dict:
        """Persist a traceable planning snapshot without embedding every object payload."""
        snapshot: dict[str, Any] = {}
        for key, value in state["canonical_state"].items():
            if isinstance(value, list):
                snapshot[key] = [
                    {
                        field: str(item[field]) if field == "id" else item[field]
                        for field in ("id", "kind", "status")
                        if field in item
                    }
                    for item in value
                ]
            else:
                snapshot[key] = value
        return snapshot

    def world_model_create(self, project_id: str, name: str, scope: str) -> dict:
        return self.store.world_model_create_atomic(project_id, name, scope)

    def world_model_get(self, world_model_id: str, as_of: str | None = None) -> dict:
        temporal = {"as_of": as_of} if as_of is not None else {}
        model = self._expect(world_model_id, "WorldModel", as_of)
        nodes = [
            self.store.object_get(item, **temporal) for item in model["data"].get("node_ids", [])
        ]
        edges = [
            self.store.object_get(item, **temporal) for item in model["data"].get("edge_ids", [])
        ]
        versions = self.store.objects_list(
            model["project_id"],
            "WorldModelVersion",
            limit=None,
            data_filters={"world_model_id": world_model_id},
            **temporal,
        )
        return {"world_model": model, "nodes": nodes, "edges": edges, "versions": versions}

    def world_entity_create(
        self,
        world_model_id: str,
        kind: str,
        name: str,
        description: str,
        attributes: dict[str, Any] | None = None,
    ) -> dict:
        allowed = {
            "ScientificVariable",
            "Phenomenon",
            "Mechanism",
            "MechanismState",
            "InformationPath",
            "Assumption",
            "Scope",
            "Intervention",
            "ExpectedEffect",
        }
        if kind not in allowed:
            raise GPUError("INVALID_WORLD_ENTITY_KIND", kind)
        self._expect(world_model_id, "WorldModel")
        created = self.store.world_model_child_create(
            world_model_id,
            kind,
            {
                "world_model_id": world_model_id,
                "name": name,
                "description": description,
                "attributes": attributes or {},
            },
            f"{kind.upper()}_CREATED",
            "node_ids",
            {"nodes_added": [name]},
        )
        return {"entity": created["child"], "version": created["version"]}

    def causal_edge_create(
        self,
        world_model_id: str,
        source_id: str,
        target_id: str,
        relation: str,
        status: str,
        supporting_ids: list[str] | None = None,
        against_ids: list[str] | None = None,
        unresolved_prediction_ids: list[str] | None = None,
        decision_id: str | None = None,
        scope: str | dict[str, Any] | None = None,
        supporting_evidence_family_ids: list[str] | None = None,
        contradicting_evidence_family_ids: list[str] | None = None,
    ) -> dict:
        if status not in EDGE_STATUSES:
            raise GPUError("INVALID_CAUSAL_EDGE_STATUS", status)
        if relation not in CAUSAL_RELATIONS:
            raise GPUError("INVALID_CAUSAL_RELATION", relation)
        model = self._expect(world_model_id, "WorldModel")
        source, target = self.store.object_get(source_id), self.store.object_get(target_id)
        if (
            source["project_id"] != model["project_id"]
            or target["project_id"] != model["project_id"]
        ):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", "World-model nodes must share a project")
        model_nodes = {str(item) for item in model["data"].get("node_ids", [])}
        if source_id not in model_nodes or target_id not in model_nodes:
            raise GPUError(
                "WORLD_MODEL_NODE_REQUIRED",
                "Causal-edge endpoints must already belong to the selected WorldModel",
            )
        evidence = list(dict.fromkeys([*(supporting_ids or []), *(against_ids or [])]))
        predictions = list(dict.fromkeys(unresolved_prediction_ids or []))
        if supporting_evidence_family_ids or contradicting_evidence_family_ids:
            raise GPUError(
                "CAUSAL_EVIDENCE_FAMILY_REQUIRES_RESULT_ASSESSMENT",
                "EvidenceFamily support is derived from an inspected result, not supplied directly",
            )
        if status == "INTERVENTION_SUPPORTED":
            raise GPUError(
                "CAUSAL_PROMOTION_REQUIRES_RESULT_ASSESSMENT",
                "Use brain_result_assess so intervention provenance, controls, and scope are verified atomically",
            )
        references = [
            *evidence,
            *predictions,
            *([decision_id] if decision_id else []),
        ]
        self._validate_references(model["project_id"], references)
        if decision_id:
            self._expect(decision_id, "ResearchDecision")
        if (
            status in {"OBSERVED_ASSOCIATION", "INTERVENTION_SUPPORTED", "WEAKENED", "REFUTED"}
            and not evidence
        ):
            raise GPUError("CAUSAL_EDGE_EVIDENCE_REQUIRED", status)
        if status in {"HYPOTHESIZED_CAUSAL", "UNKNOWN"} and not (evidence or predictions):
            raise GPUError(
                "CAUSAL_EDGE_PROVENANCE_REQUIRED",
                "Hypothesized edges need evidence or an unresolved prediction",
            )
        normalized_scope = normalize_scientific_scope(scope)
        edge_data = {
            "world_model_id": world_model_id,
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation,
            "edge_status": status,
            "supporting_ids": supporting_ids or [],
            "against_ids": against_ids or [],
            "unresolved_prediction_ids": predictions,
            "decision_id": decision_id,
            "scope": normalized_scope,
            "support_level": "NONE",
            "supporting_evidence_family_ids": [],
            "contradicting_evidence_family_ids": [],
            "independent_support_family_count": 0,
            "independent_contradicting_family_count": 0,
            "replication_count": 0,
            "matched_control_count": 0,
            "counter_intervention_count": 0,
        }
        created = self.store.world_model_child_create(
            world_model_id,
            "CausalEdge",
            edge_data,
            "CAUSAL_EDGE_CREATED",
            "edge_ids",
            {"edges_added": [{"source_id": source_id, "target_id": target_id}]},
            evidence,
            decision_id,
        )
        return {"edge": created["child"], "version": created["version"]}

    def causal_edge_update(
        self,
        edge_id: str,
        status: str,
        rationale: str,
        supporting_ids: list[str] | None = None,
        against_ids: list[str] | None = None,
        decision_id: str | None = None,
        scope: str | dict[str, Any] | None = None,
        supporting_evidence_family_ids: list[str] | None = None,
        contradicting_evidence_family_ids: list[str] | None = None,
    ) -> dict:
        if status not in EDGE_STATUSES:
            raise GPUError("INVALID_CAUSAL_EDGE_STATUS", status)
        if supporting_evidence_family_ids or contradicting_evidence_family_ids:
            raise GPUError(
                "CAUSAL_EVIDENCE_FAMILY_REQUIRES_RESULT_ASSESSMENT",
                "EvidenceFamily support is derived from an inspected result, not supplied directly",
            )
        if status == "INTERVENTION_SUPPORTED":
            raise GPUError(
                "CAUSAL_PROMOTION_REQUIRES_RESULT_ASSESSMENT",
                "Use brain_result_assess so intervention provenance, controls, and scope are verified atomically",
            )
        edge = self._expect(edge_id, "CausalEdge")
        model = self._expect(edge["data"]["world_model_id"], "WorldModel")
        supporting_additions = list(dict.fromkeys(supporting_ids or []))
        against_additions = list(dict.fromkeys(against_ids or []))
        evidence = [*supporting_additions, *against_additions]
        references = [
            *evidence,
            *([decision_id] if decision_id else []),
        ]
        self._validate_references(model["project_id"], references)
        if decision_id:
            self._expect(decision_id, "ResearchDecision")
        normalized_scope = normalize_scientific_scope(
            scope if scope is not None else edge["data"].get("scope")
        )
        return self.store.causal_edge_update_atomic(
            edge_id,
            {
                "edge_status": status,
                "supporting_ids": supporting_additions,
                "against_ids": against_additions,
                "decision_id": decision_id,
                "last_update_rationale": rationale,
                "scope": normalized_scope,
            },
            "RESULT_INSPECTED",
            {
                "edges_status_changed": [
                    {"edge_id": edge_id, "from": edge["data"]["edge_status"], "to": status}
                ]
            },
            evidence,
            decision_id,
            expected_edge_status=str(edge["data"].get("edge_status")),
        )

    def agenda_create(self, project_id: str, name: str) -> dict:
        return self._json_safe(self.store.agenda_create_once(project_id, name))

    def agenda_item_create(
        self,
        agenda_id: str,
        question: str,
        importance: float,
        uncertainty: float,
        scientific_scope: str,
        blocking_hypothesis_ids: list[str] | None = None,
        related_anomaly_ids: list[str] | None = None,
        related_contradiction_ids: list[str] | None = None,
        candidate_experiments: list[dict[str, Any]] | None = None,
        reproduction_required: bool = False,
    ) -> dict:
        agenda = self._expect(agenda_id, "ResearchAgenda")
        references = [
            *(blocking_hypothesis_ids or []),
            *(related_anomaly_ids or []),
            *(related_contradiction_ids or []),
        ]
        self._validate_references(agenda["project_id"], references)
        experiments = candidate_experiments or []
        for candidate in experiments:
            if "action_type" not in candidate:
                raise GPUError("INVALID_RESEARCH_ACTION_TYPE", "Candidate action_type is required")
            self._configured_candidate(candidate, question, blocking_hypothesis_ids or [])
        return self.store.agenda_item_create_atomic(
            agenda_id,
            {
                "agenda_id": agenda_id,
                "question": question,
                "importance": self._bounded(importance),
                "uncertainty": self._bounded(uncertainty),
                "scientific_scope": scientific_scope,
                "blocking_hypothesis_ids": blocking_hypothesis_ids or [],
                "related_anomaly_ids": related_anomaly_ids or [],
                "related_contradiction_ids": related_contradiction_ids or [],
                "candidate_experiments": experiments,
                "reproduction_required": reproduction_required,
            },
        )

    def agenda_item_update(self, agenda_item_id: str, status: str, rationale: str) -> dict:
        if status not in AGENDA_STATUSES:
            raise GPUError("INVALID_AGENDA_STATUS", status)
        self._expect(agenda_item_id, "AgendaItem")
        return self.store.object_update(
            agenda_item_id,
            {"status_rationale": rationale},
            status,
            "AGENDA_ITEM_STATUS_CHANGED",
        )

    def _portfolio_data(self, project_id: str, as_of: str | None = None) -> dict:
        temporal = {"as_of": as_of} if as_of is not None else {}
        hypotheses = self.store.objects_identifiers(project_id, "Hypothesis", **temporal)
        negative = self.store.objects_identifiers(project_id, "NegativeResult", **temporal)
        niches = self.store.objects_list(project_id, "HypothesisNiche", limit=None, **temporal)
        return {
            "active_hypothesis_ids": [
                str(item["id"])
                for item in hypotheses
                if item["status"] in {"ACTIVE", "SURVIVES_INITIAL_TEST"}
            ],
            "refuted_hypothesis_ids": [
                str(item["id"]) for item in hypotheses if item["status"] == "REFUTED"
            ],
            "negative_result_ids": [str(item["id"]) for item in negative],
            "niches": [
                {
                    "niche_id": str(item["id"]),
                    "name": item["data"].get("name"),
                    "active_best_hypothesis_id": item["data"].get(
                        "active_best_hypothesis_id"
                    ),
                }
                for item in niches
            ],
            "updated_at": datetime.now(UTC).isoformat(),
        }

    def portfolio_get(self, project_id: str) -> dict:
        portfolios = self.store.objects_list(project_id, "HypothesisPortfolio", {"ACTIVE"}, 1)
        if portfolios:
            return portfolios[0]
        return {
            "id": None,
            "project_id": project_id,
            "kind": "HypothesisPortfolio",
            "status": "NOT_MATERIALIZED",
            "data": self._portfolio_data(project_id),
        }

    def _portfolio_refresh(self, project_id: str) -> dict:
        portfolio_data = self._portfolio_data(project_id)
        portfolios = self.store.objects_list(project_id, "HypothesisPortfolio", {"ACTIVE"}, 1)
        if portfolios:
            return self.store.object_update(
                str(portfolios[0]["id"]),
                portfolio_data,
                "ACTIVE",
                "HYPOTHESIS_PORTFOLIO_REFRESHED",
            )
        return self.store.object_create(
            project_id,
            "HypothesisPortfolio",
            portfolio_data,
            "HYPOTHESIS_PORTFOLIO_CREATED",
        )

    def _discovery_portfolio(
        self,
        project_id: str,
        agenda_item: dict[str, Any],
        candidates: list[dict[str, Any]],
        hypotheses: list[dict[str, Any]],
        prerequisite: bool,
        related_dead: list[dict[str, Any]],
        as_of: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Create a validated, multi-distance portfolio without authorizing a run."""
        temporal = {"as_of": as_of} if as_of is not None else {}
        decisions = self.store.objects_list(project_id, "ResearchDecision", limit=None, **temporal)
        negatives = self.store.objects_list(project_id, "NegativeResult", limit=None, **temporal)
        breakthroughs = self.store.objects_list(project_id, "BreakthroughSignal", limit=None, **temporal)
        metrics = agenda_item["data"].get("frontier_metrics", [])
        if not isinstance(metrics, list):
            metrics = []
        frontier = frontier_gap(metrics)
        stagnation = stagnation_state(decisions, negatives)
        mechanism_unknown = not bool(agenda_item["data"].get("mechanism_resolved"))
        regime = choose_regime(
            prerequisite=prerequisite,
            mechanism_unknown=mechanism_unknown,
            frontier=frontier,
            stagnation=stagnation,
        )
        baseline = candidates[0] if candidates else None
        for candidate in candidates:
            candidate.update(classify_scientific_distance(candidate, baseline))
            candidate["generation_source"] = candidate.get("payload", {}).get(
                "generation_source", "AGENDA_CONFIGURED"
            )
        critique = portfolio_critique(candidates, prerequisite)
        if not critique["adequate"] and not prerequisite:
            generated = fallback_candidates(
                agenda_item["data"]["question"],
                [str(item["id"]) for item in hypotheses],
                baseline,
                regime["search_regime"],
            )
            for item in generated:
                candidate = ActionCandidate(**item).checked().persisted_data()
                candidate.update(classify_scientific_distance(candidate, baseline))
                candidate["generation_source"] = "DETERMINISTIC_DISCOVERY_FALLBACK"
                candidates.append(candidate)
            critique = portfolio_critique(candidates, False)
        high_breakthroughs = [
            item for item in breakthroughs if item.get("data", {}).get("discovery_value") == "HIGH"
        ]
        if high_breakthroughs and not prerequisite:
            branches = high_breakthroughs[0]["data"].get("branch_recommendations", [])
            for branch in branches:
                dimensions = {"architecture_family": "mechanism-derived redesign"}
                if branch == "search_different_architecture_or_representation":
                    dimensions = {"representation": "alternative representation family"}
                elif branch == "reformulate_joint_objective":
                    dimensions = {"causal_object": "reformulated joint objective"}
                item = ActionCandidate(
                    action_type="LITERATURE_SEARCH",
                    question_addressed=agenda_item["data"]["question"],
                    hypotheses_discriminated=[str(hypothesis["id"]) for hypothesis in hypotheses],
                    predicted_outcomes=["Specify a distinct branch implied by the partial breakthrough."],
                    required_resources=["breakthrough evidence", "human review"],
                    payload={"scientific_dimensions": dimensions, "breakthrough_branch": branch, "generation_source": "BREAKTHROUGH_BRANCHING", "requires_preregistration": True, "non_executing_discovery_candidate": True},
                    score=ActionScore(scientific_importance=4, expected_discrimination=3, expected_information_gain=3, feasibility=4, compute_cost=0.2, engineering_cost=0.5, execution_risk=0.2, decision_relevance=4),
                ).checked().persisted_data()
                item.update(classify_scientific_distance(item, baseline))
                item["generation_source"] = "BREAKTHROUGH_BRANCHING"
                candidates.append(item)
            critique = portfolio_critique(candidates, False)
        # Diminishing returns must change selection behavior, not merely a label.
        for candidate in candidates:
            distance = candidate["scientific_distance"]
            multiplier = 1.0
            if regime["search_regime"] == SearchRegime.DIVERGENT_SEARCH.value:
                multiplier = 0.55 if distance == "NEAR" else 1.2 if distance in {"FAR", "ORTHOGONAL"} else 1.0
            elif regime["search_regime"] == SearchRegime.PARADIGM_RESET.value:
                multiplier = 0.35 if distance == "NEAR" else 1.3 if distance in {"FAR", "ORTHOGONAL"} else 0.8
            candidate["strategic_priority_multiplier"] = multiplier
            candidate["priority"] = round(float(candidate["priority"]) * multiplier, 6)
        # Fallback ideas are deliberate portfolio members, not executable or
        # self-justifying research actions.  Selecting them was the v3.1-v1
        # local-search loop: a zero-information cycle could lead to another
        # non-executing fallback with no concrete preregistered design.
        if not prerequisite and not any(
            not item.get("payload", {}).get("non_executing_discovery_candidate", False)
            and item.get("available", True)
            for item in candidates
        ):
            refinement = ActionCandidate(
                action_type="PORTFOLIO_REFINEMENT",
                question_addressed=agenda_item["data"]["question"],
                hypotheses_discriminated=[str(item["id"]) for item in hypotheses],
                predicted_outcomes=["Convert a scientifically distant idea into a concrete, preregisterable candidate before a new decision."],
                required_resources=["human scientific review", "existing evidence"],
                payload={"mode": "REQUIRES_CONCRETE_PREREGISTERED_CANDIDATE", "non_scientific_process_action": True, "does_not_authorize_execution": True},
                score=ActionScore(scientific_importance=1, expected_discrimination=1, expected_information_gain=1, feasibility=5, compute_cost=0.1, engineering_cost=0.2, execution_risk=0.1, decision_relevance=1),
            ).checked().persisted_data()
            refinement.update(classify_scientific_distance(refinement, baseline))
            refinement["generation_source"] = "PORTFOLIO_GOVERNANCE"
            refinement["strategic_priority_multiplier"] = 1.0
            candidates.append(refinement)
        portfolio = {
            "project_id": project_id,
            "agenda_item_id": str(agenda_item["id"]),
            "search_regime": regime["search_regime"],
            "portfolio_type": "SINGLE_PATH_PREREQUISITE" if prerequisite else "OPEN_ENDED_DISCOVERY",
            "valid_candidate_indexes": [index for index, item in enumerate(candidates) if item.get("available", True)],
            "rejected_candidate_indexes": [],
            "distance_coverage": critique["distance_coverage"],
            "representation_coverage": sorted({str(item.get("payload", {}).get("scientific_dimensions", {}).get("representation", "UNSPECIFIED")) for item in candidates}),
            "lineage_coverage": sorted({str(item.get("payload", {}).get("scientific_dimensions", {}).get("architecture_family", "UNSPECIFIED")) for item in candidates}),
            "generation_sources": sorted({str(item.get("generation_source", "UNKNOWN")) for item in candidates}),
            "negative_memory_constraints_applied": {"related_dead_count": len(related_dead), "local_search_saturation": stagnation["local_search_saturation"], "required_search_radius": stagnation["required_search_radius"]},
            "breakthrough_context": [
                {"id": str(item["id"]), "discovery_value": item["data"].get("discovery_value"), "type": item["data"].get("type")}
                for item in breakthroughs[-10:]
            ],
            "portfolio_critic_result": critique,
            "brain_policy_version": BRAIN_POLICY_VERSION,
        }
        return candidates, portfolio, frontier, {**stagnation, "search_regime": regime, "portfolio_critic": critique}

    def brain_step(
        self, project_id: str, as_of: str | None = None, persist: bool = True
    ) -> dict:
        if as_of is not None and persist:
            raise GPUError(
                "TEMPORAL_DECISION_MUST_BE_DRY_RUN",
                "Historical Brain evaluation cannot mutate canonical scientific state",
            )
        temporal = {"as_of": as_of} if as_of is not None else {}
        started = datetime.now(UTC)
        brain_step_id, request_id = str(uuid.uuid4()), str(uuid.uuid4())
        state = self.store.state_get(project_id, **temporal)
        models = self.store.objects_list(project_id, "WorldModel", limit=1, **temporal)
        agendas = self.store.objects_list(
            project_id, "ResearchAgenda", {"ACTIVE"}, 1, **temporal
        )
        if not models or not agendas:
            missing = [
                name
                for name, values in (("WorldModel", models), ("ResearchAgenda", agendas))
                if not values
            ]
            raise GPUError("BRAIN_STATE_INCOMPLETE", "Missing: " + ", ".join(missing))
        model, agenda = models[0], agendas[0]
        items = self.store.objects_list(
            project_id,
            "AgendaItem",
            {"OPEN", "ACTIVE", "BLOCKED"},
            limit=None,
            data_filters={"agenda_id": str(agenda["id"])},
            **temporal,
        )
        if not items:
            raise GPUError("RESEARCH_AGENDA_EMPTY", str(agenda["id"]))
        agenda_item = max(
            items,
            key=lambda item: item["data"].get("importance", 1) * item["data"].get("uncertainty", 1),
        )
        portfolio = (
            self._portfolio_refresh(project_id)
            if persist
            else {
                "id": None,
                "kind": "HypothesisPortfolio",
                "status": "TEMPORAL_PREVIEW",
                "data": self._portfolio_data(project_id, as_of),
            }
        )
        hypotheses = self.store.objects_list(
            project_id,
            "Hypothesis",
            {"ACTIVE", "SURVIVES_INITIAL_TEST"},
            limit=None,
            **temporal,
        )
        comparative_lessons = self.store.objects_list(
            project_id, "ComparativeLesson", limit=None, **temporal
        )
        meta_lessons = self.store.objects_list(
            project_id, "MetaLesson", limit=None, **temporal
        )
        related_dead = self._related_dead_ideas(project_id, hypotheses, as_of)
        candidates = self._candidate_actions(project_id, agenda_item, hypotheses, as_of)
        base_candidate_data = [
            {
                **candidate.persisted_data(),
                "brain_step_id": brain_step_id,
                "agenda_item_id": str(agenda_item["id"]),
            }
            for candidate in candidates
        ]
        hard_gate = len(candidates) == 1 and candidates[0].action_type in {
            "ARTIFACT_ANALYSIS",
            "REPRODUCTION",
        }
        base_candidate_data, candidate_portfolio, frontier, discovery_state = self._discovery_portfolio(
            project_id,
            agenda_item,
            base_candidate_data,
            hypotheses,
            hard_gate,
            related_dead,
            as_of,
        )
        situation_data = self.strategy.construct_situation_data(
            project_id,
            agenda_item,
            model,
            base_candidate_data,
            len(related_dead),
            as_of,
        )
        situation_data.update(
            {
                "search_regime": candidate_portfolio["search_regime"],
                "frontier_gap": frontier,
                "stagnation_state": discovery_state,
                "state_freshness": state["state_freshness"],
            }
        )
        strategy_retrieval = self.strategy.retrieve(project_id, situation_data, as_of)
        agenda_telemetry = self.strategy.agenda_telemetry(
            project_id, agenda_item, as_of
        )
        candidate_data = self.strategy.adjust_candidates(
            base_candidate_data,
            strategy_retrieval,
            agenda_telemetry,
            hard_gate,
        )
        selectable = [
            index
            for index, item in enumerate(candidate_data)
            if not item.get("payload", {}).get("non_executing_discovery_candidate", False)
            and item.get("available", True)
        ]
        selected_index = max(selectable, key=lambda index: candidate_data[index]["priority"])
        selected = candidate_data[selected_index]
        runner_ups = sorted(
            (
                (index, item)
                for index, item in enumerate(candidate_data)
                if index != selected_index
            ),
            key=lambda item: item[1]["priority"],
            reverse=True,
        )
        runner_up_index, runner_up = runner_ups[0] if runner_ups else (None, None)
        candidate_comparison = [self._candidate_comparison(item, hypotheses) for item in candidate_data]
        prospective_hindsight = self._prospective_hindsight_attack(
            selected, hypotheses, runner_up
        )
        decision_data = {
            "request_id": request_id,
            "brain_step_id": brain_step_id,
            "agenda_item_id": str(agenda_item["id"]),
            "question": agenda_item["data"]["question"],
            "state_snapshot": {
                "world_model_id": str(model["id"]),
                "world_model_version_id": model["data"].get("current_version_id"),
                "agenda_id": str(agenda["id"]),
                "portfolio_id": str(portfolio["id"]) if portfolio["id"] else None,
                "research_state": self._state_snapshot(state),
                "state_freshness": state["state_freshness"],
                "comparative_lesson_ids": [str(item["id"]) for item in comparative_lessons],
                "meta_lesson_ids": [str(item["id"]) for item in meta_lessons],
            },
            "evidence_considered": self._evidence_ids(state),
            "hypotheses_affected": [str(item["id"]) for item in hypotheses],
            "dead_ideas_retrieved": related_dead,
            "comparative_lessons": comparative_lessons,
            "meta_lessons": meta_lessons,
            "research_situation": situation_data,
            "strategy_patterns_retrieved": strategy_retrieval["applied"],
            "strategy_patterns_rejected": strategy_retrieval["rejected"],
            "agenda_diminishing_returns": agenda_telemetry,
            "brain_policy_version": BRAIN_POLICY_VERSION,
            "strategy_policy_version": STRATEGY_POLICY_VERSION,
            "scoring_policy_version": SCORING_POLICY_VERSION,
            "rationale": self._decision_rationale(selected, related_dead, runner_up),
            "search_regime": candidate_portfolio["search_regime"],
            "search_regime_reason": discovery_state["search_regime"]["reason"],
            "candidate_portfolio": candidate_portfolio,
            "candidate_portfolio_size": len(candidate_data),
            "scientific_distance_selected": selected.get("scientific_distance"),
            "scientific_distance_distribution": candidate_portfolio["distance_coverage"],
            "frontier_gap": frontier,
            "stagnation_state": discovery_state,
            "breakthrough_context": candidate_portfolio["breakthrough_context"],
            "central_uncertainty": agenda_item["data"].get(
                "central_uncertainty", agenda_item["data"]["question"]
            ),
            "runner_up_candidate_index": runner_up_index,
            "runner_up_action_type": runner_up.get("action_type") if runner_up else None,
            "why_selected_over_runner_up": (
                self._runner_up_reason(selected, runner_up) if runner_up else None
            ),
            "why_cheaper_alternative_insufficient": (
                "The selected action targets the recorded uncertainty with greater discriminating value under the active search regime."
            ),
            "result_that_changes_belief": selected.get("predicted_outcomes", []),
            "pass_means": "The frozen, preregistered prediction is supported only after artifact inspection.",
            "fail_means": "The stated hypothesis may be weakened or refuted; strategic discovery value remains independent.",
            "remains_unresolved": agenda_item["data"].get("central_uncertainty", agenda_item["data"]["question"]),
            "candidate_comparison": candidate_comparison,
            "prospective_hindsight": prospective_hindsight,
            "hard_blocks_checked": [
                item["code"]
                for item in self._critics(selected, related_dead)[-1].get("findings", [])
                if item.get("hard_block")
            ],
            "expected_information_gain": self._information_gain_label(
                selected["score"]["expected_information_gain"]
            ),
            "estimated_compute": selected["score"]["compute_cost"],
            "estimated_cost": {
                "compute": selected["score"]["compute_cost"],
                "engineering": selected["score"]["engineering_cost"],
            },
            "critics": self._critics(selected, related_dead),
            "status": "SELECTED",
            "actual_information_gain": None,
            "outcome": None,
            "hindsight_assessment": None,
            "duration_ms": 0,
        }
        decision_data["duration_ms"] = int((datetime.now(UTC) - started).total_seconds() * 1000)
        if not persist:
            return {
                "brain_step_id": brain_step_id,
                "decision_id": None,
                "agenda_item": agenda_item,
                "question": agenda_item["data"]["question"],
                "scientific_state": state["canonical_state"],
                "world_model": model,
                "competing_hypotheses": hypotheses,
                "dead_ideas_retrieved": related_dead,
                "research_situation": situation_data,
                "strategy_patterns_retrieved": strategy_retrieval,
                "agenda_diminishing_returns": agenda_telemetry,
                "candidate_portfolio": candidate_portfolio,
                "brain_policy_version": BRAIN_POLICY_VERSION,
                "search_regime": candidate_portfolio["search_regime"],
                "frontier_gap": frontier,
                "stagnation_state": discovery_state,
                "state_freshness": state["state_freshness"],
                "candidate_actions": candidate_data,
                "selected_action": selected,
                "reason": decision_data["rationale"],
                "expected_information_gain": decision_data["expected_information_gain"],
                "estimated_cost": decision_data["estimated_cost"],
                "requires_human_approval": False,
                "verification_status": "TEMPORAL_DRY_RUN",
                "as_of": as_of,
            }
        persisted_step = self.store.brain_decision_create(
            project_id,
            candidate_data,
            selected_index,
            decision_data,
            situation_data,
            candidate_portfolio,
        )
        decision = persisted_step["decision"]
        persisted = persisted_step["candidates"]
        selected = persisted_step["selected"]
        return {
            "brain_step_id": brain_step_id,
            "decision_id": decision["id"],
            "agenda_item": agenda_item,
            "question": agenda_item["data"]["question"],
            "scientific_state": state["canonical_state"],
            "world_model": model,
            "competing_hypotheses": hypotheses,
            "dead_ideas_retrieved": related_dead,
            "research_situation": persisted_step["situation"],
            "strategy_patterns_retrieved": strategy_retrieval,
            "agenda_diminishing_returns": agenda_telemetry,
            "candidate_portfolio": persisted_step["portfolio"],
            "brain_policy_version": BRAIN_POLICY_VERSION,
            "search_regime": candidate_portfolio["search_regime"],
            "frontier_gap": frontier,
            "stagnation_state": discovery_state,
            "state_freshness": state["state_freshness"],
            "candidate_actions": persisted,
            "selected_action": selected,
            "reason": decision_data["rationale"],
            "expected_information_gain": decision_data["expected_information_gain"],
            "estimated_cost": decision_data["estimated_cost"],
            "requires_human_approval": False,
            "verification_status": "IMPLEMENTED_UNVERIFIED",
        }

    def decision_approve(self, decision_id: str, approver: str, rationale: str) -> dict:
        """Record explicit human approval for the selected bounded research action."""
        decision = self._expect(decision_id, "ResearchDecision")
        if decision["status"] not in {"SELECTED", "APPROVED"}:
            raise GPUError("RESEARCH_DECISION_NOT_APPROVABLE", decision["status"])
        if not approver.strip() or not rationale.strip():
            raise GPUError("RESEARCH_APPROVAL_INCOMPLETE", "Approver and rationale are required")
        if decision["status"] == "APPROVED":
            return {**decision, "idempotent_replay": True}
        return self.store.object_update(
            decision_id,
            {
                "approval": {
                    "approver": approver.strip(),
                    "rationale": rationale.strip(),
                    "approved_at": datetime.now(UTC).isoformat(),
                    "selected_action_id": decision["data"]["selected_action"]["id"],
                }
            },
            "APPROVED",
            "RESEARCH_DECISION_APPROVED",
        )

    def legacy_run_provenance_repair(
        self, run_id: str, agenda_item_id: str, rationale: str
    ) -> dict:
        """Attach inspect-only reconstructed provenance to one pre-binding run."""
        run = self._expect(run_id, "ExperimentRun")
        if run["status"] not in {"completed", "RESULT_NOT_INSPECTED", "failed", "cancelled", "unknown"}:
            raise GPUError("LEGACY_RUN_NOT_INSPECTABLE", run["status"])
        agenda_item = self._expect(agenda_item_id, "AgendaItem")
        experiment = self._expect(str(run["data"].get("experiment_id")), "Experiment")
        hypothesis_id = str(experiment["data"].get("hypothesis_id"))
        hypothesis = self._expect(hypothesis_id, "Hypothesis")
        project_id = str(run["project_id"])
        if any(str(item["project_id"]) != project_id for item in (agenda_item, experiment, hypothesis)):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", "Legacy repair inputs must share a project")
        existing_id = run["data"].get("decision_id")
        if existing_id:
            existing = self.store.object_get(str(existing_id))
            if existing["kind"] == "ResearchDecision":
                legacy = existing["data"].get("legacy_provenance", {})
                if legacy.get("run_id") == run_id and legacy.get("reconstructed") is True:
                    return {"run": run, "decision": existing, "idempotent_replay": True}
                raise GPUError("LEGACY_RUN_ALREADY_HAS_DECISION", run_id)
            abandonment = run["data"].get("legacy_abandonment", {})
            if abandonment.get("verified_missing_backing_job") is not True:
                raise GPUError("LEGACY_RUN_INVALID_DECISION_PROVENANCE", str(existing_id))
        if not rationale.strip():
            raise GPUError("LEGACY_REPAIR_RATIONALE_REQUIRED", "Provide the provenance reconstruction reason")
        question = agenda_item["data"].get("question") or experiment["data"].get("plan", {}).get(
            "research_question", "Legacy result inspection"
        )
        decision = self.store.object_create(
            project_id,
            "ResearchDecision",
            {
                "agenda_item_id": agenda_item_id,
                "question": question,
                "hypotheses_affected": [hypothesis_id],
                "selected_action": {
                    "id": f"legacy-inspect:{run_id}",
                    "action_type": "ARTIFACT_ANALYSIS",
                    "question_addressed": question,
                    "payload": {"run_id": run_id, "mode": "INSPECT_LEGACY_RESULT"},
                },
                "legacy_provenance": {
                    "reconstructed": True,
                    "run_id": run_id,
                    "experiment_id": str(experiment["id"]),
                    "artifact_count": len(run["data"].get("artifacts", [])),
                    "superseded_pre_decision_id": str(existing_id) if existing_id else None,
                    "superseded_pre_decision_kind": existing["kind"] if existing_id else None,
                    "rationale": rationale.strip(),
                    "limitations": [
                        "Decision was reconstructed after execution.",
                        "Repair authorizes artifact inspection only; it does not validate the result.",
                    ],
                },
            },
            "LEGACY_PROVENANCE_RECONSTRUCTED",
            "SELECTED",
        )
        repaired_run = self.store.object_update(
            run_id,
            {
                "decision_id": str(decision["id"]),
                "legacy_provenance": {
                    "reconstructed": True,
                    "decision_id": str(decision["id"]),
                    "superseded_pre_decision_id": str(existing_id) if existing_id else None,
                },
            },
            run["status"],
            "EXPERIMENT_RUN_LEGACY_PROVENANCE_REPAIRED",
        )
        return {"run": repaired_run, "decision": decision, "idempotent_replay": False}

    def legacy_reserved_run_abandon(
        self, run_id: str, job_id: str, rationale: str, technical_non_scientific: bool = False
    ) -> dict:
        """Cancel a server-verified unsubmitted reservation without execution evidence."""
        run = self._expect(run_id, "ExperimentRun")
        is_abandoned_replay = (
            run["status"] == "cancelled"
            and run["data"].get("legacy_abandonment", {}).get("verified_missing_backing_job") is True
        )
        if run["status"] != "RESERVED" and not is_abandoned_replay:
            raise GPUError("LEGACY_RUN_NOT_ABANDONABLE", run["status"])
        if run["data"].get("job_id") != job_id:
            raise GPUError("LEGACY_RUN_JOB_MISMATCH", job_id)
        if not rationale.strip():
            raise GPUError("LEGACY_ABANDON_RATIONALE_REQUIRED", "Provide the missing-job rationale")
        decision_id = run["data"].get("decision_id")
        decision_kind = None
        if decision_id:
            try:
                decision = self.store.object_get(str(decision_id))
            except GPUError as exc:
                if exc.error_type != "RESEARCH_OBJECT_NOT_FOUND":
                    raise
            else:
                decision_kind = decision["kind"]
                if (
                    decision_kind == "ResearchDecision"
                    and not technical_non_scientific
                    and not is_abandoned_replay
                ):
                    raise GPUError("LEGACY_RUN_HAS_RESEARCH_DECISION", str(decision_id))
        provenance = {
            "pre_research_decision": decision_kind != "ResearchDecision",
            "original_decision_id": str(decision_id) if decision_id else None,
            "original_decision_kind": decision_kind,
        }
        if technical_non_scientific:
            provenance["technical_non_scientific"] = True
        return self.store.legacy_reserved_run_abandon(
            run_id,
            job_id,
            rationale.strip(),
            provenance,
        )

    def execution_decision_bind(
        self, experiment_id: str, decision_id: str, request_fingerprint: str
    ) -> dict:
        """Bind an executable Brain decision to one exact preregistered request."""
        experiment = self._expect(experiment_id, "Experiment")
        decision = self._expect(decision_id, "ResearchDecision")
        if str(experiment["project_id"]) != str(decision["project_id"]):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", "Experiment and decision differ")
        selected = decision["data"].get("selected_action", {})
        if selected.get("action_type") not in EXECUTABLE_ACTIONS:
            raise GPUError(
                "RESEARCH_DECISION_NOT_EXECUTABLE",
                selected.get("action_type", "UNKNOWN"),
            )
        hypothesis_id = str(experiment["data"].get("hypothesis_id"))
        if hypothesis_id not in {
            str(item) for item in decision["data"].get("hypotheses_affected", [])
        }:
            raise GPUError("RESEARCH_DECISION_MISMATCH", "Decision does not cover the hypothesis")
        plan = experiment["data"].get("plan", {})
        if selected.get("question_addressed") != plan.get("research_question"):
            raise GPUError("RESEARCH_DECISION_MISMATCH", "Decision question differs from the plan")
        binding = {
            "experiment_id": experiment_id,
            "request_fingerprint": request_fingerprint,
        }
        prior = decision["data"].get("execution_binding")
        if prior:
            if prior != binding:
                raise GPUError(
                    "RESEARCH_DECISION_ALREADY_BOUND",
                    "Decision is already bound to another execution request",
                )
            return {**decision, "idempotent_replay": True}
        return self.store.object_update(
            decision_id,
            {"execution_binding": binding},
            decision["status"],
            "RESEARCH_DECISION_BOUND_TO_EXPERIMENT",
        )

    def authorize_execution(
        self, experiment_id: str, decision_id: str, request_fingerprint: str
    ) -> dict:
        """Verify that an execution matches its decision and has required human approval."""
        experiment = self._expect(experiment_id, "Experiment")
        decision = self._expect(decision_id, "ResearchDecision")
        if str(experiment["project_id"]) != str(decision["project_id"]):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", "Experiment and decision differ")
        hypothesis_id = str(experiment["data"].get("hypothesis_id"))
        if hypothesis_id not in {str(item) for item in decision["data"].get("hypotheses_affected", [])}:
            raise GPUError("RESEARCH_DECISION_MISMATCH", "Decision does not cover the hypothesis")
        selected = decision["data"].get("selected_action", {})
        if selected.get("action_type") not in EXECUTABLE_ACTIONS:
            raise GPUError(
                "RESEARCH_DECISION_NOT_EXECUTABLE",
                selected.get("action_type", "UNKNOWN"),
            )
        plan = experiment["data"].get("plan", {})
        if selected.get("question_addressed") != plan.get("research_question"):
            raise GPUError("RESEARCH_DECISION_MISMATCH", "Decision question differs from the plan")
        binding = decision["data"].get("execution_binding", {})
        if binding.get("experiment_id") != experiment_id or binding.get(
            "request_fingerprint"
        ) != request_fingerprint:
            raise GPUError(
                "RESEARCH_EXECUTION_NOT_BOUND",
                "Create a decision bound to this exact experiment command before execution",
            )
        return {
            "decision_id": decision_id,
            "action_type": selected.get("action_type"),
            "requires_human_approval": False,
            "approved": True,
        }

    def result_assess(
        self,
        run_id: str,
        decision_id: str,
        hypothesis_id: str,
        agenda_item_id: str,
        prediction_outcome: str,
        guard_condition_outcome: str,
        condition_evaluations: dict[str, bool],
        evidence_supporting: list[str],
        evidence_against: list[str],
        unexpected_observations: list[str],
        alternative_explanations: list[str],
        scope: str | dict[str, Any],
        hypothesis_transition: str,
        rationale: str,
        causal_edge_id: str | None = None,
        causal_edge_status: str | None = None,
        actual_information_gain: str = "MEDIUM",
        information_gain_basis: list[str] | None = None,
        guard_passed: bool | None = None,
        matched_control_passed: bool | None = None,
    ) -> dict:
        run = self._expect(run_id, "ExperimentRun")
        decision = self._expect(decision_id, "ResearchDecision")
        hypothesis = self._expect(hypothesis_id, "Hypothesis")
        agenda_item = self._expect(agenda_item_id, "AgendaItem")
        project_id = str(run["project_id"])
        if any(
            str(item["project_id"]) != project_id for item in (decision, hypothesis, agenda_item)
        ):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", "Assessment inputs must share a project")
        if run["status"] not in {
            "completed",
            "RESULT_NOT_INSPECTED",
            "failed",
            "cancelled",
            "unknown",
            "RESULT_INSPECTED",
        }:
            raise GPUError("EXPERIMENT_RESULT_NOT_READY", run["status"])
        if str(decision["data"].get("agenda_item_id")) != agenda_item_id:
            raise GPUError("RESEARCH_DECISION_MISMATCH", "Agenda item differs from decision")
        if hypothesis_id not in {
            str(item) for item in decision["data"].get("hypotheses_affected", [])
        }:
            raise GPUError("RESEARCH_DECISION_MISMATCH", "Hypothesis differs from decision")
        if str(run["data"].get("decision_id")) != decision_id:
            raise GPUError("RESEARCH_DECISION_MISMATCH", "Run was not authorized by this decision")
        experiment_id = str(run["data"].get("experiment_id"))
        experiment = self._expect(experiment_id, "Experiment")
        if str(experiment["data"].get("hypothesis_id")) != hypothesis_id:
            raise GPUError("EXPERIMENT_HYPOTHESIS_MISMATCH", hypothesis_id)
        pass_condition = experiment["data"].get("plan", {}).get("pass_condition")
        predictions = self.store.objects_list(
            project_id,
            "Prediction",
            limit=None,
            data_filters={"experiment_id": experiment_id, "frozen": True},
        )
        if not pass_condition or not predictions:
            raise GPUError("EXPERIMENT_GUARD_NOT_PREREGISTERED", experiment_id)
        if any(item["data"].get("pass_condition") != pass_condition for item in predictions):
            raise GPUError("EXPERIMENT_GUARD_MISMATCH", experiment_id)
        if guard_passed is not None:
            if not isinstance(guard_passed, bool):
                raise GPUError(
                    "EXPERIMENT_GUARD_EVALUATION_INVALID",
                    "guard_passed must be a boolean",
                )
            # Bind the supplied outcome to the server-side preregistered text.
            # This avoids requiring clients to round-trip long Unicode condition keys.
            condition_evaluations = {pass_condition: guard_passed}
        elif set(condition_evaluations) != {pass_condition} or not all(
            isinstance(value, bool) for value in condition_evaluations.values()
        ):
            raise GPUError(
                "EXPERIMENT_GUARD_EVALUATION_INVALID",
                "Evaluate the exact frozen pass_condition once with a boolean result",
            )
        allowed_transitions = {
            "SUPPORTED",
            "WEAKENED",
            "REFUTED",
            "SURVIVES_INITIAL_TEST",
            "INCONCLUSIVE",
            "BLOCKED",
        }
        if hypothesis_transition not in allowed_transitions:
            raise GPUError("INVALID_SCIENTIFIC_STATUS", hypothesis_transition)
        if actual_information_gain not in {
            "HIGH",
            "MEDIUM",
            "LOW",
            "ZERO",
            "INVALID",
            "UNKNOWN",
        }:
            raise GPUError("INVALID_INFORMATION_GAIN", actual_information_gain)
        guard_passed = condition_evaluations[pass_condition]
        successful_run = run["status"] in {"completed", "RESULT_NOT_INSPECTED"}
        basis = list(information_gain_basis or [])
        technical_non_scientific = (
            not successful_run and actual_information_gain in {"ZERO", "INVALID", "UNKNOWN"}
        )
        if technical_non_scientific:
            if causal_edge_id or causal_edge_status:
                raise GPUError(
                    "TECHNICAL_INSPECTION_CANNOT_UPDATE_CAUSAL_EDGE",
                    "A failed technical run cannot update a causal edge",
                )
            result = self.store.technical_result_inspection_apply(
                run_id=run_id,
                decision_id=decision_id,
                actual_information_gain=actual_information_gain,
                information_gain_basis=basis,
                inspection={
                    "prediction_outcome": prediction_outcome,
                    "guard_condition_outcome": guard_condition_outcome,
                    "frozen_pass_condition": pass_condition,
                    "condition_evaluations": condition_evaluations,
                    "guard_passed": guard_passed,
                    "scope": normalize_scientific_scope(scope),
                    "rationale": rationale,
                },
            )
            return {
                **result,
                "verification_status": "TECHNICAL_FAILURE_INSPECTED",
            }
        if hypothesis_transition in {"SUPPORTED", "SURVIVES_INITIAL_TEST"} and not guard_passed:
            raise GPUError(
                "EXPERIMENT_GUARD_NOT_PASSED",
                "Positive transitions require the frozen pass condition to evaluate true",
            )
        if hypothesis_transition in {"SUPPORTED", "SURVIVES_INITIAL_TEST"} and not successful_run:
            raise GPUError(
                "FAILED_RUN_CANNOT_SUPPORT_HYPOTHESIS",
                "Failed, cancelled, and unknown runs may only weaken, refute, block, or remain inconclusive",
            )
        if causal_edge_id or causal_edge_status:
            if not causal_edge_id or not causal_edge_status:
                raise GPUError(
                    "CAUSAL_EDGE_UPDATE_INCOMPLETE", "Both causal_edge_id and status are required"
                )
            if causal_edge_status == "INTERVENTION_SUPPORTED" and not successful_run:
                raise GPUError("FAILED_RUN_CANNOT_SUPPORT_CAUSAL_EDGE", run["status"])
            if causal_edge_status not in EDGE_STATUSES:
                raise GPUError("INVALID_CAUSAL_EDGE_STATUS", causal_edge_status)
            edge = self._expect(causal_edge_id, "CausalEdge")
            if str(edge["project_id"]) != project_id:
                raise GPUError("RESEARCH_PROJECT_MISMATCH", causal_edge_id)
        normalized_scope = normalize_scientific_scope(scope)
        if causal_edge_status == "INTERVENTION_SUPPORTED" and not scope_is_empirically_bounded(
            normalized_scope
        ):
            raise GPUError(
                "CAUSAL_PROMOTION_SCOPE_REQUIRED",
                "Intervention support requires explicit model/architecture, intervention, and metric scope",
            )
        if matched_control_passed is not None and not isinstance(matched_control_passed, bool):
            raise GPUError(
                "MATCHED_CONTROL_EVALUATION_INVALID",
                "matched_control_passed must be a boolean when supplied",
            )
        control_preregistered = bool(experiment["data"].get("plan", {}).get("control"))
        if (
            causal_edge_status == "INTERVENTION_SUPPORTED"
            and control_preregistered
            and matched_control_passed is not True
        ):
            raise GPUError(
                "CAUSAL_PROMOTION_MATCHED_CONTROL_REQUIRED",
                "The preregistered matched control must be explicitly assessed as passed",
            )
        if causal_edge_status == "INTERVENTION_SUPPORTED":
            unresolved_nulls = [
                item
                for item in self.store.objects_list(
                    project_id,
                    "NullModel",
                    {"ACTIVE", "OPEN", "PROPOSED"},
                    limit=None,
                )
                if str(item["data"].get("target_entity_id"))
                in {hypothesis_id, str(causal_edge_id)}
                and item["data"].get("strength") == "STRONG"
                and item["data"].get("tested") is not True
                and isinstance(item["data"].get("estimated_cost"), (int, float))
                and not isinstance(item["data"].get("estimated_cost"), bool)
                and float(item["data"]["estimated_cost"]) <= 2
            ]
            if unresolved_nulls:
                raise GPUError(
                    "CAUSAL_PROMOTION_STRONG_NULL_UNTESTED",
                    "Test the strong cheap null model(s) before causal promotion: "
                    + ", ".join(str(item["id"]) for item in unresolved_nulls),
                )
        evidence_data = {
            "source_type": "ExperimentRun",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "job_id": run["data"].get("job_id"),
            "prediction_outcome": prediction_outcome,
            "guard_condition_outcome": guard_condition_outcome,
            "frozen_pass_condition": pass_condition,
            "condition_evaluations": condition_evaluations,
            "guard_passed": guard_passed,
            "supporting_observations": evidence_supporting,
            "against_observations": evidence_against,
            "unexpected_observations": unexpected_observations,
            "alternative_explanations": alternative_explanations,
            "scope": normalized_scope,
            "matched_control_preregistered": control_preregistered,
            "matched_control_passed": matched_control_passed,
            "artifacts": run["data"].get("artifacts", []),
            "exit_code": run["data"].get("exit_code"),
            "extraction_method": "explicit_result_assessment",
            "extracted_at": datetime.now(UTC).isoformat(),
        }
        result = self.store.result_assessment_apply(
            run_id=run_id,
            decision_id=decision_id,
            hypothesis_id=hypothesis_id,
            agenda_item_id=agenda_item_id,
            evidence_data=evidence_data,
            hypothesis_transition=hypothesis_transition,
            rationale=rationale,
            inspection={
                "decision_id": decision_id,
                "prediction_outcome": prediction_outcome,
                "guard_condition_outcome": guard_condition_outcome,
                "frozen_pass_condition": pass_condition,
                "condition_evaluations": condition_evaluations,
                "guard_passed": guard_passed,
                "scope": normalized_scope,
            },
            agenda_status=(
                "ACTIVE"
                if hypothesis_transition in {"INCONCLUSIVE", "BLOCKED"}
                else "RESOLVED"
            ),
            actual_information_gain=actual_information_gain,
            information_gain_basis=basis,
            causal_edge_id=causal_edge_id,
            causal_edge_status=causal_edge_status,
        )
        return {**result, "verification_status": "RESULT_INSPECTED"}

    def _candidate_actions(
        self,
        project_id: str,
        agenda_item: dict,
        hypotheses: list[dict],
        as_of: str | None = None,
    ) -> list[ActionCandidate]:
        temporal = {"as_of": as_of} if as_of is not None else {}
        uninspected = self.store.experiment_run_first(
            project_id,
            {"completed", "RESULT_NOT_INSPECTED"},
            inspected=False,
            **temporal,
        )
        unfinished = self.store.experiment_run_first(
            project_id, {"RESERVED", "running", "unknown"}, **temporal
        )
        failed_uninspected = self.store.experiment_run_first(
            project_id, {"failed", "cancelled"}, inspected=False, **temporal
        )
        question = agenda_item["data"]["question"]
        hypothesis_ids = [str(item["id"]) for item in hypotheses]
        if uninspected is not None:
            return [
                ActionCandidate(
                    action_type="ARTIFACT_ANALYSIS",
                    question_addressed=question,
                    hypotheses_discriminated=hypothesis_ids,
                    predicted_outcomes=["Inspect completed evidence before launching new work"],
                    required_resources=["experiment logs", "artifacts"],
                    payload={"run_id": str(uninspected["id"]), "mode": "INSPECT_RESULT"},
                    score=ActionScore(
                        scientific_importance=5,
                        expected_discrimination=5,
                        expected_information_gain=5,
                        feasibility=5,
                        compute_cost=0.1,
                        engineering_cost=0.2,
                        execution_risk=0.1,
                    ),
                ).checked()
            ]
        if failed_uninspected is not None:
            return [
                ActionCandidate(
                    action_type="ARTIFACT_ANALYSIS",
                    question_addressed=question,
                    hypotheses_discriminated=hypothesis_ids,
                    predicted_outcomes=["Inspect failure evidence and preserve negative knowledge"],
                    required_resources=["experiment logs", "failure artifacts"],
                    payload={
                        "run_id": str(failed_uninspected["id"]),
                        "mode": "INSPECT_FAILURE",
                    },
                    score=ActionScore(
                        scientific_importance=5,
                        expected_discrimination=4,
                        expected_information_gain=4,
                        feasibility=5,
                        compute_cost=0.1,
                        engineering_cost=0.2,
                        execution_risk=0.1,
                    ),
                ).checked()
            ]
        if unfinished is not None:
            return [
                ActionCandidate(
                    action_type="ARTIFACT_ANALYSIS",
                    question_addressed=question,
                    hypotheses_discriminated=hypothesis_ids,
                    predicted_outcomes=["Recover and synchronize unfinished execution"],
                    required_resources=["job status"],
                    payload={"run_id": str(unfinished["id"]), "mode": "RECOVER_UNFINISHED"},
                    score=ActionScore(
                        scientific_importance=5,
                        expected_discrimination=4,
                        expected_information_gain=4,
                        feasibility=5,
                        compute_cost=0.1,
                        engineering_cost=0.1,
                        execution_risk=0.1,
                    ),
                ).checked()
            ]
        reproductions = self.store.objects_list(
            project_id, "Reproduction", limit=None, **temporal
        )
        baseline_reproduced = any(item["status"] == "REPRODUCED" for item in reproductions)
        needs_reproduction = (
            bool(agenda_item["data"].get("reproduction_required")) and not baseline_reproduced
        ) or (bool(reproductions) and not baseline_reproduced)
        if needs_reproduction:
            return [
                ActionCandidate(
                    action_type="REPRODUCTION",
                    question_addressed=question,
                    hypotheses_discriminated=hypothesis_ids,
                    predicted_outcomes=["Confirm baseline before internal causal intervention"],
                    required_resources=["canonical runtime", "baseline checkpoint"],
                    payload={"reproduction_ids": [str(item["id"]) for item in reproductions]},
                    score=ActionScore(
                        scientific_importance=5,
                        expected_discrimination=4,
                        expected_information_gain=5,
                        feasibility=4,
                        compute_cost=1,
                        engineering_cost=1,
                        execution_risk=0.5,
                    ),
                ).checked()
            ]
        configured = agenda_item["data"].get("candidate_experiments", [])
        null_models = self.store.objects_list(
            project_id,
            "NullModel",
            {"ACTIVE", "OPEN", "PROPOSED"},
            limit=None,
            **temporal,
        )
        strong_cheap_nulls = [
            item
            for item in null_models
            if item["data"].get("strength") == "STRONG"
            and isinstance(item["data"].get("estimated_cost"), (int, float))
            and not isinstance(item["data"].get("estimated_cost"), bool)
            and float(item["data"]["estimated_cost"]) <= 2
            and item["data"].get("tested") is not True
        ]
        if strong_cheap_nulls:
            null_candidates = [
                ActionCandidate(
                    action_type=item["data"].get("action_type", "NULL_MODEL_TEST"),
                    question_addressed=question,
                    hypotheses_discriminated=hypothesis_ids,
                    predicted_outcomes=[item["data"].get("expected_outcome", "")],
                    required_resources=["matched null control", "preregistered metric"],
                    payload={
                        "null_model_id": str(item["id"]),
                        "target_entity_id": item["data"].get("target_entity_id"),
                        "control": item["data"].get("discriminating_control"),
                    },
                    score=ActionScore(
                        scientific_importance=5,
                        expected_discrimination=5,
                        expected_information_gain=5,
                        feasibility=4,
                        compute_cost=max(float(item["data"]["estimated_cost"]), 0.1),
                        engineering_cost=1,
                        execution_risk=0.5,
                        decision_relevance=5,
                    ),
                ).checked()
                for item in strong_cheap_nulls
            ]
            configured_candidates = [
                self._configured_candidate(item, question, hypothesis_ids)
                for item in configured
            ]
            return [*null_candidates, *configured_candidates]
        literature_requested = any(
            item.get("action_type") == "LITERATURE_SEARCH" for item in configured
        )
        candidate_evidence = self.store.objects_list(
            project_id, "EvidenceUnit", {"CANDIDATE"}, limit=None, **temporal
        )
        if literature_requested and candidate_evidence:
            return [
                ActionCandidate(
                    action_type="EVIDENCE_REVIEW",
                    question_addressed=question,
                    hypotheses_discriminated=hypothesis_ids,
                    predicted_outcomes=[
                        "Validate candidate provenance and decide whether a scoped claim survives"
                    ],
                    required_resources=["candidate evidence", "claim validation"],
                    payload={
                        "evidence_ids": [str(item["id"]) for item in candidate_evidence],
                        "mode": "VALIDATE_LITERATURE_CANDIDATES",
                    },
                    score=ActionScore(
                        scientific_importance=4,
                        expected_discrimination=3,
                        expected_information_gain=3,
                        feasibility=5,
                        compute_cost=0.1,
                        engineering_cost=0.2,
                        execution_risk=0.2,
                    ),
                ).checked()
            ]
        candidates = [self._configured_candidate(item, question, hypothesis_ids) for item in configured]
        if any(candidate.available for candidate in candidates):
            return candidates
        if configured:
            return [
                ActionCandidate(
                    action_type="LITERATURE_SEARCH",
                    question_addressed=question,
                    hypotheses_discriminated=hypothesis_ids,
                    predicted_outcomes=[
                        "Gather evidence while the required execution provider is unavailable"
                    ],
                    required_resources=["literature provider"],
                    payload={
                        "mode": "ALTERNATIVE_ACTION",
                        "non_executing_discovery_candidate": True,
                        "does_not_authorize_execution": True,
                        "blocked_actions": [
                            {
                                "action_type": item.get("action_type"),
                                "blocked_reason": item.get(
                                    "blocked_reason", "required provider unavailable"
                                ),
                            }
                            for item in configured
                        ],
                    },
                    score=ActionScore(
                        scientific_importance=3,
                        expected_discrimination=2,
                        expected_information_gain=2,
                        feasibility=4,
                        compute_cost=0.2,
                        engineering_cost=0.5,
                        execution_risk=0.2,
                    ),
                ).checked(),
                *candidates,
            ]
        return [
            ActionCandidate(
                action_type="FROZEN_DIAGNOSTIC",
                question_addressed=question,
                hypotheses_discriminated=hypothesis_ids,
                predicted_outcomes=["A cheap frozen test separates competing mechanisms"],
                required_resources=["existing checkpoint"],
                score=ActionScore(
                    scientific_importance=4,
                    expected_discrimination=4,
                    expected_information_gain=4,
                    feasibility=5,
                    compute_cost=0.5,
                    engineering_cost=0.5,
                    execution_risk=0.5,
                ),
            ).checked(),
            ActionCandidate(
                action_type="LITERATURE_SEARCH",
                question_addressed=question,
                hypotheses_discriminated=hypothesis_ids,
                predicted_outcomes=["Find missing prior evidence or an existing falsifier"],
                required_resources=["literature provider"],
                score=ActionScore(
                    scientific_importance=3,
                    expected_discrimination=2,
                    expected_information_gain=2,
                    feasibility=2,
                    compute_cost=0.2,
                    engineering_cost=1,
                    execution_risk=1,
                ),
            ).checked(),
        ]

    @staticmethod
    def _configured_candidate(
        item: dict[str, Any], question: str, hypothesis_ids: list[str]
    ) -> ActionCandidate:
        score = item.get("score", {})
        try:
            return ActionCandidate(
                action_type=item["action_type"],
                question_addressed=item.get("question_addressed", question),
                hypotheses_discriminated=item.get("hypotheses_discriminated", hypothesis_ids),
                predicted_outcomes=item.get("predicted_outcomes", []),
                required_resources=item.get("required_resources", []),
                payload=item.get("payload", {}),
                available=item.get("available", True),
                blocked_reason=item.get("blocked_reason"),
                score=ActionScore(
                    scientific_importance=score.get("scientific_importance", 3),
                    expected_discrimination=score.get("expected_discrimination", 3),
                    expected_information_gain=score.get("expected_information_gain", 3),
                    feasibility=score.get("feasibility", 3),
                    compute_cost=score.get("compute_cost", 1),
                    engineering_cost=score.get("engineering_cost", 1),
                    execution_risk=score.get("execution_risk", 1),
                ),
            ).checked()
        except (KeyError, ValidationError) as exc:
            raise GPUError("INVALID_RESEARCH_ACTION", str(exc)) from exc

    def _related_dead_ideas(
        self, project_id: str, hypotheses: list[dict], as_of: str | None = None
    ) -> list[dict]:
        temporal = {"as_of": as_of} if as_of is not None else {}
        found: dict[str, dict] = {}
        candidates = [
            *self.store.objects_list(project_id, "Hypothesis", limit=None, **temporal),
            *self.store.objects_list(project_id, "NegativeResult", limit=None, **temporal),
        ]
        for hypothesis in hypotheses:
            mechanism = hypothesis["data"].get("mechanism", "")
            query_terms = self.store.terms(mechanism)
            scored = []
            for related in candidates:
                text = related["data"].get("mechanism") or related["data"].get("proposal", "")
                terms = self.store.terms(text)
                union = query_terms | terms
                overlap = len(query_terms & terms) / len(union) if union else 0.0
                if overlap:
                    scored.append(
                        {
                            **related,
                            "lexical_similarity": round(overlap, 3),
                            "containment_similarity": round(
                                len(query_terms & terms) / len(query_terms) if query_terms else 0.0,
                                3,
                            ),
                        }
                    )
            for related in sorted(
                scored, key=lambda item: item["lexical_similarity"], reverse=True
            )[:20]:
                if related["kind"] == "NegativeResult" or related["status"] == "REFUTED":
                    found[str(related["id"])] = {
                        "id": str(related["id"]),
                        "kind": related["kind"],
                        "status": related["status"],
                        "failed_assumption": related["data"].get("failed_assumption"),
                        "revisit_condition": related["data"].get("revisit_condition"),
                        "containment_similarity": related["containment_similarity"],
                        "related_hypothesis_id": str(hypothesis["id"]),
                    }
        return list(found.values())

    def _validate_references(self, project_id: str, identifiers: list[str]) -> None:
        unique = list(dict.fromkeys(identifiers))
        records = self.store.references_get(unique)
        for identifier in unique:
            item = records.get(str(identifier))
            if not item:
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", identifier)
            if str(item["project_id"]) != str(project_id):
                raise GPUError("RESEARCH_PROJECT_MISMATCH", identifier)

    def _expect(self, object_id: str, kind: str, as_of: str | None = None) -> dict:
        temporal = {"as_of": as_of} if as_of is not None else {}
        item = self.store.object_get(object_id, **temporal)
        if item["kind"] != kind:
            raise GPUError(f"NOT_A_{kind.upper()}", object_id)
        return item

    @staticmethod
    def _bounded(value: float) -> float:
        if not math.isfinite(value) or not 0 <= value <= 5:
            raise GPUError("INVALID_AGENDA_SCORE", "Use a finite score between 0 and 5")
        return value

    @staticmethod
    def _information_gain_label(value: float) -> str:
        return "HIGH" if value >= 4 else "MEDIUM" if value >= 2 else "LOW"

    @staticmethod
    def _evidence_ids(state: dict) -> list[str]:
        canonical = state["canonical_state"]
        records = [
            *canonical["supported_claims"],
            *canonical["weakened_claims"],
            *canonical["refuted_claims"],
            *canonical["negative_results"],
        ]
        return [str(item["id"]) for item in records]

    @staticmethod
    def _decision_rationale(
        selected: dict, related_dead: list[dict], runner_up: dict | None = None
    ) -> str:
        reason = (
            f"Selected {selected['action_type']} to discriminate the recorded agenda uncertainty "
            f"with expected discrimination {selected.get('score', {}).get('expected_discrimination')} "
            f"and information gain {selected.get('score', {}).get('expected_information_gain')}; "
            f"it is a {selected.get('scientific_distance', 'NEAR')} scientific alternative. "
            f"Transparent priority remains supporting evidence "
            f"(final={selected['priority']}, base={selected.get('base_priority', selected['priority'])}, "
            f"strategy+={selected.get('positive_strategy_adjustment', 0)}, "
            f"strategy-={selected.get('negative_strategy_adjustment', 0)}, "
            f"diminishing={selected.get('diminishing_return_adjustment', 0)})."
        )
        if runner_up:
            reason += " " + ResearchBrain._runner_up_reason(selected, runner_up)
        if related_dead:
            reason += f" Retrieved {len(related_dead)} related dead idea(s) before selection."
        return reason

    @staticmethod
    def _runner_up_reason(selected: dict, runner_up: dict) -> str:
        selected_disc = selected.get("score", {}).get("expected_discrimination")
        runner_disc = runner_up.get("score", {}).get("expected_discrimination")
        selected_types = set(selected.get("hypotheses_discriminated", []))
        runner_types = set(runner_up.get("hypotheses_discriminated", []))
        if selected_disc != runner_disc:
            comparison = (
                f"expected discrimination {selected_disc} versus {runner_disc}"
            )
        elif selected_types != runner_types:
            comparison = "different competing-hypothesis coverage"
        else:
            comparison = "the retained candidate comparison and explicit score components"
        return (
            f"Compared runner-up {runner_up.get('action_type')}; selected action was preferred "
            f"for {comparison}, not merely because its numeric priority was higher."
        )

    @staticmethod
    def _candidate_comparison(candidate: dict, hypotheses: list[dict]) -> dict[str, Any]:
        payload = candidate.get("payload", {})
        predicted = candidate.get("predicted_outcomes", [])
        return {
            "candidate_id": candidate.get("id"),
            "action_type": candidate.get("action_type"),
            "hypotheses_discriminated": candidate.get(
                "hypotheses_discriminated", [str(item["id"]) for item in hypotheses]
            ),
            "hypotheses_not_discriminated": payload.get("hypotheses_not_discriminated", []),
            "decisive_prediction": predicted[0] if predicted else None,
            "expected_if_pass": payload.get("expected_if_pass", predicted),
            "expected_if_fail": payload.get("expected_if_fail", ["The scoped prediction is not supported"]),
            "expected_belief_change_if_pass": payload.get("expected_belief_change_if_pass"),
            "expected_belief_change_if_fail": payload.get("expected_belief_change_if_fail"),
            "strongest_null": payload.get("strongest_null"),
            "major_confounds": payload.get("confounds", []),
            "scientific_value": candidate.get("score", {}).get("decision_relevance"),
            "cost": candidate.get("score", {}).get("compute_cost"),
            "reason_not_selected": candidate.get("reason_not_selected"),
        }

    @staticmethod
    def _prospective_hindsight_attack(
        selected: dict, hypotheses: list[dict], runner_up: dict | None
    ) -> dict[str, Any]:
        payload = selected.get("payload", {})
        guard = payload.get("guard_condition") or payload.get("guard")
        prediction = selected.get("predicted_outcomes", [])
        failure_explanations = payload.get(
            "failure_explanations",
            ["implementation failure", "weak intervention", "invalid control", "metric insensitivity"],
        )
        hard_block = not bool(prediction) or (len(hypotheses) > 1 and not selected.get("hypotheses_discriminated"))
        return {
            "if_pass": payload.get("expected_belief_change_if_pass", "Update only the scoped mechanism if guards pass."),
            "if_fail": payload.get("expected_belief_change_if_fail", "Weaken or leave unresolved; do not overclaim refutation."),
            "would_failure_refute": bool(payload.get("failure_refutes_hypothesis", False)),
            "failure_explanations_to_rule_out": failure_explanations,
            "guard_to_check": guard,
            "prediction_to_check": prediction[0] if prediction else None,
            "strongest_null": payload.get("strongest_null"),
            "runner_up_action_type": runner_up.get("action_type") if runner_up else None,
            "recommendation": "BLOCK" if hard_block else "ACCEPT",
            "hard_block": hard_block,
        }

    @staticmethod
    def _critics(selected: dict, related_dead: list[dict]) -> list[dict]:
        predicted = selected.get("predicted_outcomes", [])
        hypotheses = selected.get("hypotheses_discriminated", [])
        scientific_findings = []
        if len(hypotheses) < 2:
            scientific_findings.append(
                {
                    "code": "LOW_DISCRIMINATION",
                    "severity": "WARNING",
                    "description": "The candidate does not explicitly separate at least two competing hypotheses.",
                    "hard_block": False,
                }
            )
        if not predicted:
            scientific_findings.append(
                {
                    "code": "MISSING_DECISIVE_PREDICTION",
                    "severity": "ERROR",
                    "description": "No prospective outcome is recorded for the selected action.",
                    "hard_block": True,
                }
            )
        if selected.get("action_type") in {"TRAINING_RUN", "GENERALIZATION"}:
            scientific_findings.append(
                {
                    "code": "ARCHITECTURE_PREMATURITY_REVIEW",
                    "severity": "WARNING",
                    "description": "Training/generalization should remain blocked until the scoped mechanism test is resolved.",
                    "hard_block": False,
                }
            )
        return [
            {
                "operator": "ProximityCritic",
                "advisory": True,
                "finding": "RELATED_DEAD_IDEAS" if related_dead else "NO_CLOSE_DEAD_IDEA_FOUND",
                "related_ids": [item["id"] for item in related_dead],
            },
            {
                "operator": "ExperimentalDesignCritic",
                "advisory": True,
                "finding": "PREREGISTRATION_REQUIRED",
                "action_type": selected["action_type"],
            },
            {
                "operator": "ReproducibilityCritic",
                "advisory": True,
                "finding": "BASELINE_GATE_APPLIED",
            },
            {
                "operator": "ScientificDiscriminationCritic",
                "advisory": not any(item.get("hard_block") for item in scientific_findings),
                "recommendation": "BLOCK"
                if any(item.get("hard_block") for item in scientific_findings)
                else "ACCEPT",
                "fatal_flaws": [item["description"] for item in scientific_findings if item["severity"] == "ERROR"],
                "hypotheses_not_distinguished": [] if len(hypotheses) >= 2 else hypotheses,
                "outcomes_explained_by_multiple_hypotheses": [],
                "strongest_null_explanation": selected.get("payload", {}).get("strongest_null"),
                "missing_control": selected.get("payload", {}).get("control"),
                "confounded_variables": selected.get("payload", {}).get("confounds", []),
                "invalid_intervention_risk": selected.get("payload", {}).get("invalid_intervention_risk"),
                "measurement_artifact_risk": selected.get("payload", {}).get("measurement_artifact_risk"),
                "scope_overreach_risk": selected.get("payload", {}).get("scope_overreach_risk"),
                "architecture_prematurity": any(
                    item["code"] == "ARCHITECTURE_PREMATURITY_REVIEW"
                    for item in scientific_findings
                ),
                "cheaper_decisive_test": selected.get("payload", {}).get("cheaper_decisive_test"),
                "runner_up_preference": None,
                "findings": scientific_findings,
            },
        ]
