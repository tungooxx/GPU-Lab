import math
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .errors import GPUError
from .research import ResearchStore

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
}
EXECUTABLE_ACTIONS = {
    "REPRODUCTION",
    "FROZEN_DIAGNOSTIC",
    "REPLICATION",
    "GENERALIZATION",
    "TRAINING_RUN",
    "CAUSAL_INTERVENTION",
    "ABLATION",
}


class ActionScore(BaseModel):
    scientific_importance: float = Field(ge=0.1, le=5)
    expected_discrimination: float = Field(ge=0.1, le=5)
    expected_information_gain: float = Field(ge=0.1, le=5)
    feasibility: float = Field(ge=0.1, le=5)
    compute_cost: float = Field(ge=0.1, le=5)
    engineering_cost: float = Field(ge=0.1, le=5)
    execution_risk: float = Field(ge=0.1, le=5)

    @property
    def priority(self) -> float:
        numerator = (
            self.scientific_importance
            * self.expected_discrimination
            * self.expected_information_gain
            * self.feasibility
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

    def world_model_create(self, project_id: str, name: str, scope: str) -> dict:
        return self.store.world_model_create_atomic(project_id, name, scope)

    def world_model_get(self, world_model_id: str) -> dict:
        model = self._expect(world_model_id, "WorldModel")
        nodes = [self.store.object_get(item) for item in model["data"].get("node_ids", [])]
        edges = [self.store.object_get(item) for item in model["data"].get("edge_ids", [])]
        versions = self.store.objects_list(
            model["project_id"],
            "WorldModelVersion",
            limit=None,
            data_filters={"world_model_id": world_model_id},
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
        references = [*evidence, *predictions, *([decision_id] if decision_id else [])]
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
    ) -> dict:
        if status not in EDGE_STATUSES:
            raise GPUError("INVALID_CAUSAL_EDGE_STATUS", status)
        edge = self._expect(edge_id, "CausalEdge")
        model = self._expect(edge["data"]["world_model_id"], "WorldModel")
        supporting = list(
            dict.fromkeys([*edge["data"].get("supporting_ids", []), *(supporting_ids or [])])
        )
        against = list(dict.fromkeys([*edge["data"].get("against_ids", []), *(against_ids or [])]))
        evidence = [*supporting, *against]
        references = [*evidence, *([decision_id] if decision_id else [])]
        self._validate_references(model["project_id"], references)
        if decision_id:
            self._expect(decision_id, "ResearchDecision")
        if (
            status
            in {"OBSERVED_ASSOCIATION", "INTERVENTION_SUPPORTED", "WEAKENED", "REFUTED"}
            and not evidence
        ):
            raise GPUError("CAUSAL_EDGE_EVIDENCE_REQUIRED", status)
        return self.store.causal_edge_update_atomic(
            edge_id,
            {
                "edge_status": status,
                "supporting_ids": supporting,
                "against_ids": against,
                "decision_id": decision_id,
                "last_update_rationale": rationale,
            },
            "VERIFIED_REAL" if status == "INTERVENTION_SUPPORTED" else "RESULT_INSPECTED",
            {
                "edges_status_changed": [
                    {"edge_id": edge_id, "from": edge["data"]["edge_status"], "to": status}
                ]
            },
            evidence,
            decision_id,
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

    def _portfolio_data(self, project_id: str) -> dict:
        hypotheses = self.store.objects_identifiers(project_id, "Hypothesis")
        negative = self.store.objects_identifiers(project_id, "NegativeResult")
        niches = self.store.objects_list(project_id, "HypothesisNiche", limit=None)
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

    def brain_step(self, project_id: str) -> dict:
        started = datetime.now(UTC)
        brain_step_id, request_id = str(uuid.uuid4()), str(uuid.uuid4())
        state = self.store.state_get(project_id)
        models = self.store.objects_list(project_id, "WorldModel", limit=1)
        agendas = self.store.objects_list(project_id, "ResearchAgenda", {"ACTIVE"}, 1)
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
        )
        if not items:
            raise GPUError("RESEARCH_AGENDA_EMPTY", str(agenda["id"]))
        agenda_item = max(
            items,
            key=lambda item: item["data"].get("importance", 1) * item["data"].get("uncertainty", 1),
        )
        portfolio = self._portfolio_refresh(project_id)
        hypotheses = self.store.objects_list(
            project_id, "Hypothesis", {"ACTIVE", "SURVIVES_INITIAL_TEST"}, limit=None
        )
        comparative_lessons = self.store.objects_list(
            project_id, "ComparativeLesson", limit=None
        )
        meta_lessons = self.store.objects_list(project_id, "MetaLesson", limit=None)
        related_dead = self._related_dead_ideas(project_id, hypotheses)
        candidates = self._candidate_actions(project_id, agenda_item, hypotheses)
        candidate_data = [
            {
                **candidate.persisted_data(),
                "brain_step_id": brain_step_id,
                "agenda_item_id": str(agenda_item["id"]),
            }
            for candidate in candidates
        ]
        selected_index = max(range(len(candidate_data)), key=lambda index: candidate_data[index]["priority"])
        selected = candidate_data[selected_index]
        decision_data = {
            "request_id": request_id,
            "brain_step_id": brain_step_id,
            "agenda_item_id": str(agenda_item["id"]),
            "question": agenda_item["data"]["question"],
            "state_snapshot": {
                "world_model_id": str(model["id"]),
                "world_model_version_id": model["data"].get("current_version_id"),
                "agenda_id": str(agenda["id"]),
                "portfolio_id": str(portfolio["id"]),
                "research_state": self._json_safe(state["canonical_state"]),
                "comparative_lesson_ids": [str(item["id"]) for item in comparative_lessons],
                "meta_lesson_ids": [str(item["id"]) for item in meta_lessons],
            },
            "evidence_considered": self._evidence_ids(state),
            "hypotheses_affected": [str(item["id"]) for item in hypotheses],
            "dead_ideas_retrieved": related_dead,
            "comparative_lessons": comparative_lessons,
            "meta_lessons": meta_lessons,
            "rationale": self._decision_rationale(selected, related_dead),
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
        persisted_step = self.store.brain_decision_create(
            project_id,
            candidate_data,
            selected_index,
            decision_data,
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
        scope: str,
        hypothesis_transition: str,
        rationale: str,
        causal_edge_id: str | None = None,
        causal_edge_status: str | None = None,
        actual_information_gain: str = "MEDIUM",
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
        if set(condition_evaluations) != {pass_condition} or not all(
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
        if actual_information_gain not in {"HIGH", "MEDIUM", "LOW"}:
            raise GPUError("INVALID_INFORMATION_GAIN", actual_information_gain)
        guard_passed = condition_evaluations[pass_condition]
        successful_run = run["status"] in {"completed", "RESULT_NOT_INSPECTED"}
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
            "scope": scope,
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
                "scope": scope,
            },
            agenda_status=(
                "ACTIVE"
                if hypothesis_transition in {"INCONCLUSIVE", "BLOCKED"}
                else "RESOLVED"
            ),
            actual_information_gain=actual_information_gain,
            causal_edge_id=causal_edge_id,
            causal_edge_status=causal_edge_status,
        )
        return {**result, "verification_status": "RESULT_INSPECTED"}

    def _candidate_actions(
        self, project_id: str, agenda_item: dict, hypotheses: list[dict]
    ) -> list[ActionCandidate]:
        uninspected = self.store.experiment_run_first(
            project_id, {"completed", "RESULT_NOT_INSPECTED"}, inspected=False
        )
        unfinished = self.store.experiment_run_first(
            project_id, {"RESERVED", "running", "unknown"}
        )
        failed_uninspected = self.store.experiment_run_first(
            project_id, {"failed", "cancelled"}, inspected=False
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
        reproductions = self.store.objects_list(project_id, "Reproduction", limit=None)
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
        literature_requested = any(
            item.get("action_type") == "LITERATURE_SEARCH" for item in configured
        )
        candidate_evidence = self.store.objects_list(
            project_id, "EvidenceUnit", {"CANDIDATE"}, limit=None
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

    def _related_dead_ideas(self, project_id: str, hypotheses: list[dict]) -> list[dict]:
        found: dict[str, dict] = {}
        candidates = [
            *self.store.objects_list(project_id, "Hypothesis", limit=None),
            *self.store.objects_list(project_id, "NegativeResult", limit=None),
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

    def _expect(self, object_id: str, kind: str) -> dict:
        item = self.store.object_get(object_id)
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
    def _decision_rationale(selected: dict, related_dead: list[dict]) -> str:
        reason = (
            f"Selected {selected['action_type']} because its heuristic information-per-cost "
            f"priority ({selected['priority']}) is highest for the active agenda item."
        )
        if related_dead:
            reason += f" Retrieved {len(related_dead)} related dead idea(s) before selection."
        return reason

    @staticmethod
    def _critics(selected: dict, related_dead: list[dict]) -> list[dict]:
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
        ]
