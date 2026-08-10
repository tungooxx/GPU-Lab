import math
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

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

    def checked(self) -> "ActionCandidate":
        if self.action_type not in ACTION_TYPES:
            raise GPUError("INVALID_RESEARCH_ACTION_TYPE", self.action_type)
        return self

    def persisted_data(self) -> dict[str, Any]:
        data = self.model_dump()
        data["priority"] = self.score.priority
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
        model = self.store.object_create(
            project_id,
            "WorldModel",
            {
                "name": name,
                "scope": scope,
                "node_ids": [],
                "edge_ids": [],
                "current_version_id": None,
                "version": 0,
            },
            "WORLD_MODEL_CREATED",
            "IMPLEMENTED_UNVERIFIED",
        )
        version = self._version_world_model(
            model,
            {"world_model_created": True},
            evidence_ids=[],
            decision_id=None,
        )
        return {"world_model": self.store.object_get(model["id"]), "version": version}

    def world_model_get(self, world_model_id: str) -> dict:
        model = self._expect(world_model_id, "WorldModel")
        nodes = [self.store.object_get(item) for item in model["data"].get("node_ids", [])]
        edges = [self.store.object_get(item) for item in model["data"].get("edge_ids", [])]
        versions = self.store.objects_list(model["project_id"], "WorldModelVersion")
        versions = [item for item in versions if item["data"].get("world_model_id") == world_model_id]
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
        model = self._expect(world_model_id, "WorldModel")
        entity = self.store.object_create(
            model["project_id"],
            kind,
            {
                "world_model_id": world_model_id,
                "name": name,
                "description": description,
                "attributes": attributes or {},
            },
            f"{kind.upper()}_CREATED",
            "IMPLEMENTED_UNVERIFIED",
        )
        node_ids = [*model["data"].get("node_ids", []), entity["id"]]
        updated = self.store.object_update(
            world_model_id,
            {"node_ids": node_ids},
            model["status"],
            "WORLD_MODEL_NODE_ADDED",
        )
        version = self._version_world_model(
            {**model, "data": updated["data"]},
            {"nodes_added": [entity["id"]]},
            evidence_ids=[],
            decision_id=None,
        )
        return {"entity": entity, "version": version}

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
        model = self._expect(world_model_id, "WorldModel")
        source, target = self.store.object_get(source_id), self.store.object_get(target_id)
        if source["project_id"] != model["project_id"] or target["project_id"] != model["project_id"]:
            raise GPUError("RESEARCH_PROJECT_MISMATCH", "World-model nodes must share a project")
        evidence = list(dict.fromkeys([*(supporting_ids or []), *(against_ids or [])]))
        predictions = list(dict.fromkeys(unresolved_prediction_ids or []))
        self._validate_references(model["project_id"], [*evidence, *predictions])
        if (
            status
            in {"OBSERVED_ASSOCIATION", "INTERVENTION_SUPPORTED", "WEAKENED", "REFUTED"}
            and not evidence
        ):
            raise GPUError("CAUSAL_EDGE_EVIDENCE_REQUIRED", status)
        if status == "HYPOTHESIZED_CAUSAL" and not (evidence or predictions):
            raise GPUError(
                "CAUSAL_EDGE_PROVENANCE_REQUIRED",
                "Hypothesized edges need evidence or an unresolved prediction",
            )
        edge = self.store.object_create(
            model["project_id"],
            "CausalEdge",
            {
                "world_model_id": world_model_id,
                "source_id": source_id,
                "target_id": target_id,
                "relation": relation,
                "edge_status": status,
                "supporting_ids": supporting_ids or [],
                "against_ids": against_ids or [],
                "unresolved_prediction_ids": predictions,
                "decision_id": decision_id,
            },
            "CAUSAL_EDGE_CREATED",
            "IMPLEMENTED_UNVERIFIED",
        )
        edge_ids = [*model["data"].get("edge_ids", []), edge["id"]]
        updated = self.store.object_update(
            world_model_id,
            {"edge_ids": edge_ids},
            model["status"],
            "WORLD_MODEL_EDGE_ADDED",
        )
        version = self._version_world_model(
            {**model, "data": updated["data"]},
            {"edges_added": [edge["id"]]},
            evidence_ids=evidence,
            decision_id=decision_id,
        )
        return {"edge": edge, "version": version}

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
        against = list(
            dict.fromkeys([*edge["data"].get("against_ids", []), *(against_ids or [])])
        )
        evidence = [*supporting, *against]
        self._validate_references(model["project_id"], evidence)
        if status in {"INTERVENTION_SUPPORTED", "WEAKENED", "REFUTED"} and not evidence:
            raise GPUError("CAUSAL_EDGE_EVIDENCE_REQUIRED", status)
        updated = self.store.object_update(
            edge_id,
            {
                "edge_status": status,
                "supporting_ids": supporting,
                "against_ids": against,
                "decision_id": decision_id,
                "last_update_rationale": rationale,
            },
            "VERIFIED_REAL" if status == "INTERVENTION_SUPPORTED" else "RESULT_INSPECTED",
            "CAUSAL_EDGE_STATUS_CHANGED",
        )
        version = self._version_world_model(
            model,
            {
                "edges_status_changed": [
                    {"edge_id": edge_id, "from": edge["data"]["edge_status"], "to": status}
                ]
            },
            evidence_ids=evidence,
            decision_id=decision_id,
        )
        return {"edge": updated, "version": version}

    def agenda_create(self, project_id: str, name: str) -> dict:
        existing = self.store.objects_list(project_id, "ResearchAgenda", {"ACTIVE"}, 1)
        if existing:
            return {**existing[0], "idempotent_replay": True}
        return self.store.object_create(
            project_id,
            "ResearchAgenda",
            {"name": name, "item_ids": []},
            "RESEARCH_AGENDA_CREATED",
        )

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
        item = self.store.object_create(
            agenda["project_id"],
            "AgendaItem",
            {
                "agenda_id": agenda_id,
                "question": question,
                "importance": self._bounded(importance),
                "uncertainty": self._bounded(uncertainty),
                "scientific_scope": scientific_scope,
                "blocking_hypothesis_ids": blocking_hypothesis_ids or [],
                "related_anomaly_ids": related_anomaly_ids or [],
                "related_contradiction_ids": related_contradiction_ids or [],
                "candidate_experiments": candidate_experiments or [],
                "reproduction_required": reproduction_required,
            },
            "AGENDA_ITEM_CREATED",
            "OPEN",
        )
        self.store.object_update(
            agenda_id,
            {"item_ids": [*agenda["data"].get("item_ids", []), item["id"]]},
            "ACTIVE",
            "RESEARCH_AGENDA_UPDATED",
        )
        return item

    def agenda_item_update(self, agenda_item_id: str, status: str, rationale: str) -> dict:
        if status not in AGENDA_STATUSES:
            raise GPUError("INVALID_AGENDA_STATUS", status)
        return self.store.object_update(
            agenda_item_id,
            {"status_rationale": rationale},
            status,
            "AGENDA_ITEM_STATUS_CHANGED",
        )

    def portfolio_get(self, project_id: str) -> dict:
        hypotheses = self.store.objects_list(project_id, "Hypothesis")
        negative = self.store.objects_list(project_id, "NegativeResult")
        portfolio_data = {
            "active_hypothesis_ids": [
                str(item["id"])
                for item in hypotheses
                if item["status"] in {"ACTIVE", "SURVIVES_INITIAL_TEST"}
            ],
            "refuted_hypothesis_ids": [
                str(item["id"]) for item in hypotheses if item["status"] == "REFUTED"
            ],
            "negative_result_ids": [str(item["id"]) for item in negative],
            "updated_at": datetime.now(UTC).isoformat(),
        }
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
            missing = [name for name, values in (("WorldModel", models), ("ResearchAgenda", agendas)) if not values]
            raise GPUError("BRAIN_STATE_INCOMPLETE", "Missing: " + ", ".join(missing))
        model, agenda = models[0], agendas[0]
        items = [
            item
            for item in self.store.objects_list(project_id, "AgendaItem")
            if item["data"].get("agenda_id") == str(agenda["id"])
            and item["status"] in {"OPEN", "ACTIVE", "BLOCKED"}
        ]
        if not items:
            raise GPUError("RESEARCH_AGENDA_EMPTY", str(agenda["id"]))
        agenda_item = max(
            items,
            key=lambda item: item["data"].get("importance", 1)
            * item["data"].get("uncertainty", 1),
        )
        portfolio = self.portfolio_get(project_id)
        hypotheses = self.store.objects_list(
            project_id, "Hypothesis", {"ACTIVE", "SURVIVES_INITIAL_TEST"}
        )
        related_dead = self._related_dead_ideas(project_id, hypotheses)
        candidates = self._candidate_actions(project_id, agenda_item, hypotheses)
        persisted = []
        for candidate in candidates:
            action = self.store.object_create(
                project_id,
                "ResearchActionCandidate",
                {
                    **candidate.persisted_data(),
                    "brain_step_id": brain_step_id,
                    "agenda_item_id": str(agenda_item["id"]),
                },
                "RESEARCH_ACTION_CANDIDATE_CREATED",
                "PROPOSED",
            )
            persisted.append({**candidate.persisted_data(), "id": action["id"]})
        selected = max(persisted, key=lambda item: item["priority"])
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
            },
            "evidence_considered": self._evidence_ids(state),
            "hypotheses_affected": [str(item["id"]) for item in hypotheses],
            "dead_ideas_retrieved": related_dead,
            "candidate_action_ids": [item["id"] for item in persisted],
            "candidate_actions": persisted,
            "selected_action": selected,
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
        decision_data["duration_ms"] = int(
            (datetime.now(UTC) - started).total_seconds() * 1000
        )
        decision = self.store.object_create(
            project_id,
            "ResearchDecision",
            decision_data,
            "RESEARCH_DECISION_SELECTED",
            "SELECTED",
        )
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
            "requires_human_approval": selected["action_type"]
            in {"TRAINING_RUN", "CAUSAL_INTERVENTION", "ABLATION"},
            "verification_status": "IMPLEMENTED_UNVERIFIED",
        }

    def result_assess(
        self,
        run_id: str,
        decision_id: str,
        hypothesis_id: str,
        agenda_item_id: str,
        prediction_outcome: str,
        guard_condition_outcome: str,
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
        if any(str(item["project_id"]) != project_id for item in (decision, hypothesis, agenda_item)):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", "Assessment inputs must share a project")
        if run["status"] not in {"completed", "RESULT_NOT_INSPECTED"}:
            raise GPUError("EXPERIMENT_RESULT_NOT_READY", run["status"])
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
        evidence = self.store.object_create(
            project_id,
            "EvidenceUnit",
            {
                "source_type": "ExperimentRun",
                "run_id": run_id,
                "experiment_id": run["data"].get("experiment_id"),
                "job_id": run["data"].get("job_id"),
                "prediction_outcome": prediction_outcome,
                "guard_condition_outcome": guard_condition_outcome,
                "supporting_observations": evidence_supporting,
                "against_observations": evidence_against,
                "unexpected_observations": unexpected_observations,
                "alternative_explanations": alternative_explanations,
                "scope": scope,
                "artifacts": run["data"].get("artifacts", []),
                "exit_code": run["data"].get("exit_code"),
                "extraction_method": "explicit_result_assessment",
                "extracted_at": datetime.now(UTC).isoformat(),
            },
            "EXPERIMENT_EVIDENCE_INSPECTED",
            "VERIFIED_REAL",
        )
        assessed_hypothesis = self.store.assess(hypothesis_id, hypothesis_transition, rationale)
        inspected_run = self.store.object_update(
            run_id,
            {
                "inspection": {
                    "evidence_id": evidence["id"],
                    "decision_id": decision_id,
                    "prediction_outcome": prediction_outcome,
                    "guard_condition_outcome": guard_condition_outcome,
                    "scope": scope,
                }
            },
            "RESULT_INSPECTED",
            "EXPERIMENT_RESULT_INSPECTED",
        )
        agenda = self.agenda_item_update(
            agenda_item_id,
            "RESOLVED" if hypothesis_transition != "INCONCLUSIVE" else "ACTIVE",
            rationale,
        )
        edge_update = None
        if causal_edge_id or causal_edge_status:
            if not causal_edge_id or not causal_edge_status:
                raise GPUError(
                    "CAUSAL_EDGE_UPDATE_INCOMPLETE", "Both causal_edge_id and status are required"
                )
            supporting = [evidence["id"]] if causal_edge_status == "INTERVENTION_SUPPORTED" else []
            against = [evidence["id"]] if causal_edge_status in {"WEAKENED", "REFUTED"} else []
            edge_update = self.causal_edge_update(
                causal_edge_id,
                causal_edge_status,
                rationale,
                supporting,
                against,
                decision_id,
            )
        updated_decision = self.store.object_update(
            decision_id,
            {
                "actual_information_gain": actual_information_gain,
                "outcome": {
                    "run_id": run_id,
                    "evidence_id": evidence["id"],
                    "hypothesis_transition": hypothesis_transition,
                    "prediction_outcome": prediction_outcome,
                },
                "hindsight_assessment": rationale,
            },
            "COMPLETED",
            "RESEARCH_DECISION_OUTCOME_RECORDED",
        )
        self.store.project_state_update(
            project_id,
            {
                "highest_value_unknown": None,
                "established_facts": [
                    *self.store.state_get(project_id)["state"].get("established_facts", []),
                    {
                        "evidence_id": evidence["id"],
                        "statement": prediction_outcome,
                        "scope": scope,
                        "verification_status": "VERIFIED_REAL",
                    },
                ],
            },
        )
        return {
            "run": inspected_run,
            "evidence": evidence,
            "hypothesis": assessed_hypothesis,
            "agenda_item": agenda,
            "world_model_update": edge_update,
            "decision": updated_decision,
            "verification_status": "RESULT_INSPECTED",
        }

    def _candidate_actions(
        self, project_id: str, agenda_item: dict, hypotheses: list[dict]
    ) -> list[ActionCandidate]:
        runs = self.store.objects_list(project_id, "ExperimentRun")
        uninspected = [
            run
            for run in runs
            if run["status"] in {"completed", "RESULT_NOT_INSPECTED"}
            and not run["data"].get("inspection")
        ]
        unfinished = [
            run
            for run in runs
            if run["status"] in {"RESERVED", "SUBMITTED", "running", "unknown"}
        ]
        question = agenda_item["data"]["question"]
        hypothesis_ids = [str(item["id"]) for item in hypotheses]
        if uninspected:
            return [
                ActionCandidate(
                    action_type="ARTIFACT_ANALYSIS",
                    question_addressed=question,
                    hypotheses_discriminated=hypothesis_ids,
                    predicted_outcomes=["Inspect completed evidence before launching new work"],
                    required_resources=["experiment logs", "artifacts"],
                    payload={"run_id": str(uninspected[0]["id"]), "mode": "INSPECT_RESULT"},
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
        if unfinished:
            return [
                ActionCandidate(
                    action_type="ARTIFACT_ANALYSIS",
                    question_addressed=question,
                    hypotheses_discriminated=hypothesis_ids,
                    predicted_outcomes=["Recover and synchronize unfinished execution"],
                    required_resources=["job status"],
                    payload={"run_id": str(unfinished[0]["id"]), "mode": "RECOVER_UNFINISHED"},
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
        reproductions = self.store.objects_list(project_id, "Reproduction")
        baseline_reproduced = any(item["status"] == "REPRODUCED" for item in reproductions)
        needs_reproduction = (
            bool(agenda_item["data"].get("reproduction_required"))
            and not baseline_reproduced
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
        candidates = [
            self._configured_candidate(item, question, hypothesis_ids)
            for item in configured
            if item.get("available", True)
        ]
        if candidates:
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
                ).checked()
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
        return ActionCandidate(
            action_type=item["action_type"],
            question_addressed=item.get("question_addressed", question),
            hypotheses_discriminated=item.get("hypotheses_discriminated", hypothesis_ids),
            predicted_outcomes=item.get("predicted_outcomes", []),
            required_resources=item.get("required_resources", []),
            payload=item.get("payload", {}),
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

    def _related_dead_ideas(self, project_id: str, hypotheses: list[dict]) -> list[dict]:
        found: dict[str, dict] = {}
        for hypothesis in hypotheses:
            mechanism = hypothesis["data"].get("mechanism", "")
            for related in self.store.related_hypotheses(project_id, mechanism, 20):
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

    def _version_world_model(
        self,
        model: dict,
        changes: dict[str, Any],
        evidence_ids: list[str],
        decision_id: str | None,
    ) -> dict:
        current_version = int(model["data"].get("version", 0))
        version = self.store.object_create(
            str(model["project_id"]),
            "WorldModelVersion",
            {
                "world_model_id": str(model["id"]),
                "version": current_version + 1,
                "parent_version_id": model["data"].get("current_version_id"),
                "changes": changes,
                "evidence_ids": evidence_ids,
                "decision_id": decision_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            "WORLD_MODEL_VERSION_CREATED",
            "IMPLEMENTED_UNVERIFIED",
        )
        self.store.object_update(
            str(model["id"]),
            {"current_version_id": version["id"], "version": current_version + 1},
            model["status"],
            "WORLD_MODEL_VERSION_ADVANCED",
        )
        return version

    def _validate_references(self, project_id: str, identifiers: list[str]) -> None:
        for identifier in identifiers:
            item = self.store.object_get(identifier)
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
