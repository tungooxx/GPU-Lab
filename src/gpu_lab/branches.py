import math
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from .errors import GPUError
from .research import ResearchStore

RECOVERABLE_RUN_STATUSES = {"RESERVED", "RUNNING", "running", "unknown"}
INSPECTABLE_RUN_STATUSES = {
    "RESULT_NOT_INSPECTED",
    "completed",
    "failed",
    "cancelled",
}
_KIND_ERRORS = {
    "ExperimentBranch": "NOT_AN_EXPERIMENT_BRANCH",
    "ExperimentNode": "NOT_AN_EXPERIMENT_NODE",
    "ExperimentRun": "NOT_AN_EXPERIMENT_RUN",
    "Hypothesis": "NOT_A_HYPOTHESIS",
}


class BranchNodeDraft(BaseModel):
    branch_action: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=10_000)
    predicted_outcomes: dict[str, Any] = Field(min_length=1)
    scientific_importance: float = Field(ge=1, le=5)
    expected_discrimination: float = Field(ge=1, le=5)
    expected_information_gain: float = Field(ge=1, le=5)
    feasibility: float = Field(ge=1, le=5)
    compute_cost: float = Field(gt=0, le=1_000_000)
    engineering_cost: float = Field(gt=0, le=1_000_000)
    execution_risk: float = Field(gt=0, le=5)
    experiment_id: str | None = None

    @field_validator(
        "scientific_importance",
        "expected_discrimination",
        "expected_information_gain",
        "feasibility",
        "compute_cost",
        "engineering_cost",
        "execution_risk",
    )
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("branch score components must be finite")
        return value


class ComparativeLessonDraft(BaseModel):
    code_delta: dict[str, Any]
    config_delta: dict[str, Any]
    state_delta: dict[str, Any]
    data_delta: dict[str, Any]
    metric_delta: dict[str, Any] = Field(min_length=1)
    shared_conditions: list[str] = Field(min_length=1, max_length=100)
    candidate_causal_difference: str = Field(min_length=10, max_length=10_000)
    scope: str = Field(min_length=1, max_length=10_000)
    confidence: str = Field(pattern=r"^(LOW|MEDIUM|HIGH)$")
    remaining_confounds: list[str] = Field(default_factory=list, max_length=100)


class ExperimentBranchService:
    """Deterministic branch records and comparison policy; no MCTS or truth promotion."""

    def __init__(self, store: ResearchStore):
        self.store = store

    def create(
        self,
        project_id: str,
        hypothesis_id: str,
        objective: str,
        budget: dict[str, Any],
    ) -> dict[str, Any]:
        hypothesis = self._expect(hypothesis_id, project_id, "Hypothesis")
        if hypothesis["status"] not in {"ACTIVE", "SURVIVES_INITIAL_TEST", "SUPPORTED"}:
            raise GPUError("BRANCH_HYPOTHESIS_NOT_ACTIVE", hypothesis["status"])
        if not objective.strip() or not budget:
            raise GPUError("INVALID_EXPERIMENT_BRANCH", "Objective and explicit budget are required")
        return self.store.object_create(
            project_id,
            "ExperimentBranch",
            {
                "hypothesis_id": hypothesis_id,
                "objective": objective.strip(),
                "budget": budget,
                "policy": "DETERMINISTIC_HEURISTIC",
                "warning": "Branch priority guides testing; it does not establish scientific truth.",
            },
            "EXPERIMENT_BRANCH_CREATED",
        )

    def node_add(
        self,
        branch_id: str,
        draft_data: dict[str, Any],
        parent_node_id: str | None = None,
    ) -> dict[str, Any]:
        branch = self.store.object_get(branch_id)
        if branch["kind"] != "ExperimentBranch":
            raise GPUError("NOT_AN_EXPERIMENT_BRANCH", branch_id)
        draft = self._node_draft(draft_data)
        score = self._score(draft)
        data = {
            **draft.model_dump(mode="json"),
            "branch_id": branch_id,
            "parent_node_id": parent_node_id,
            "priority_score": score,
            "result": None,
            "scientific_interpretation": None,
            "actual_cost": None,
            "information_gained": None,
        }
        return self.store.experiment_branch_node_create(
            str(branch["project_id"]), branch_id, data, parent_node_id, draft.experiment_id
        )

    def get(self, branch_id: str) -> dict[str, Any]:
        branch = self.store.object_get(branch_id)
        if branch["kind"] != "ExperimentBranch":
            raise GPUError("NOT_AN_EXPERIMENT_BRANCH", branch_id)
        project_id = str(branch["project_id"])
        nodes = self.store.objects_list(
            project_id, "ExperimentNode", limit=None, data_filters={"branch_id": branch_id}
        )
        relations = self.store.objects_list(
            project_id, "BranchRelation", limit=None, data_filters={"branch_id": branch_id}
        )
        lessons = self.store.objects_list(
            project_id, "ComparativeLesson", limit=None, data_filters={"branch_id": branch_id}
        )
        return {**branch, "nodes": nodes, "relations": relations, "comparative_lessons": lessons}

    def next_action(self, branch_id: str) -> dict[str, Any]:
        branch = self.get(branch_id)
        experiment_ids = {
            str(node["data"]["experiment_id"])
            for node in branch["nodes"]
            if node["data"].get("experiment_id")
        }
        runs_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in self.store.objects_list(
            str(branch["project_id"]), "ExperimentRun", limit=None
        ):
            experiment_id = run["data"].get("experiment_id")
            if experiment_id and str(experiment_id) in experiment_ids:
                runs_by_experiment[str(experiment_id)].append(run)
        planned, inspectable_nodes, recoverable_nodes = [], [], []
        for node in branch["nodes"]:
            if node["status"] == "RESULT_INSPECTED":
                continue
            experiment_id = node["data"].get("experiment_id")
            runs = runs_by_experiment.get(str(experiment_id), []) if experiment_id else []
            if any(run["status"] in INSPECTABLE_RUN_STATUSES for run in runs):
                run = min(
                    (run for run in runs if run["status"] in INSPECTABLE_RUN_STATUSES),
                    key=lambda item: str(item["id"]),
                )
                inspectable_nodes.append((node, run))
            elif any(run["status"] in RECOVERABLE_RUN_STATUSES for run in runs):
                run = min(
                    (run for run in runs if run["status"] in RECOVERABLE_RUN_STATUSES),
                    key=lambda item: str(item["id"]),
                )
                recoverable_nodes.append((node, run))
            else:
                planned.append(node)
        if inspectable_nodes:
            node, run = min(inspectable_nodes, key=lambda pair: str(pair[0]["id"]))
            return self._recommend(branch_id, "INSPECT_RESULT", node, run_id=str(run["id"]))
        if recoverable_nodes:
            node, run = min(recoverable_nodes, key=lambda pair: str(pair[0]["id"]))
            return self._recommend(
                branch_id, "RECOVER_UNFINISHED", node, run_id=str(run["id"])
            )
        if planned:
            selected = max(
                planned,
                key=lambda item: (float(item["data"]["priority_score"]), str(item["id"])),
            )
            return self._recommend(branch_id, "EXECUTE_BRANCH_NODE", selected)
        inspected = [item for item in branch["nodes"] if item["status"] == "RESULT_INSPECTED"]
        if len(inspected) >= 2 and not branch["comparative_lessons"]:
            return {
                "branch_id": branch_id,
                "action": "COMPARE_RESULTS",
                "node_ids": [str(item["id"]) for item in inspected],
                "reason": "Multiple inspected results exist without a comparative lesson.",
            }
        return {
            "branch_id": branch_id,
            "action": "BRANCH_COMPLETE",
            "reason": "No unfinished, inspectable, planned, or uncompared nodes remain.",
        }

    def record_result(
        self,
        node_id: str,
        run_id: str,
        result: dict[str, Any],
        scientific_interpretation: str,
        actual_cost: dict[str, Any],
        information_gained: str,
    ) -> dict[str, Any]:
        node = self.store.object_get(node_id)
        if node["kind"] != "ExperimentNode":
            raise GPUError("NOT_AN_EXPERIMENT_NODE", node_id)
        run = self._expect(run_id, str(node["project_id"]), "ExperimentRun")
        if run["status"] != "RESULT_INSPECTED":
            raise GPUError("BRANCH_RESULT_NOT_INSPECTED", run_id)
        node_experiment_id = node["data"].get("experiment_id")
        run_experiment_id = run["data"].get("experiment_id")
        if (
            not node_experiment_id
            or not run_experiment_id
            or str(run_experiment_id) != str(node_experiment_id)
        ):
            raise GPUError("BRANCH_RUN_MISMATCH", run_id)
        canonical_result = run["data"].get("inspection")
        if not isinstance(canonical_result, dict) or not canonical_result:
            raise GPUError("BRANCH_RESULT_NOT_INSPECTED", run_id)
        if result != canonical_result:
            raise GPUError("BRANCH_RESULT_MISMATCH", run_id)
        if not scientific_interpretation.strip() or not information_gained.strip():
            raise GPUError("INCOMPLETE_BRANCH_RESULT", node_id)
        return self.store.object_update(
            node_id,
            {
                "run_id": run_id,
                "result": canonical_result,
                "scientific_interpretation": scientific_interpretation.strip(),
                "actual_cost": actual_cost,
                "information_gained": information_gained.strip(),
            },
            "RESULT_INSPECTED",
            "EXPERIMENT_BRANCH_RESULT_RECORDED",
        )

    def compare(
        self,
        branch_id: str,
        node_a_id: str,
        node_b_id: str,
        lesson_data: dict[str, Any],
    ) -> dict[str, Any]:
        branch = self.store.object_get(branch_id)
        if branch["kind"] != "ExperimentBranch":
            raise GPUError("NOT_AN_EXPERIMENT_BRANCH", branch_id)
        node_a = self._expect(node_a_id, str(branch["project_id"]), "ExperimentNode")
        node_b = self._expect(node_b_id, str(branch["project_id"]), "ExperimentNode")
        if node_a_id == node_b_id or any(
            str(item["data"].get("branch_id")) != branch_id for item in (node_a, node_b)
        ):
            raise GPUError("INVALID_BRANCH_COMPARISON", branch_id)
        if any(item["status"] != "RESULT_INSPECTED" for item in (node_a, node_b)):
            raise GPUError("BRANCH_RESULT_NOT_INSPECTED", branch_id)
        try:
            lesson = ComparativeLessonDraft.model_validate(lesson_data)
        except ValidationError as exc:
            raise GPUError("INVALID_COMPARATIVE_LESSON", str(exc)) from exc
        return self.store.comparative_lesson_create(
            str(branch["project_id"]),
            branch_id,
            node_a_id,
            node_b_id,
            {
                **lesson.model_dump(mode="json"),
                "branch_id": branch_id,
                "experiment_a": node_a["data"].get("experiment_id"),
                "experiment_b": node_b["data"].get("experiment_id"),
                "node_a_id": node_a_id,
                "node_b_id": node_b_id,
                "warning": "Candidate causal difference retains stated confounds; it is not proof.",
            },
        )

    def _expect(self, object_id: str, project_id: str, kind: str) -> dict[str, Any]:
        item = self.store.object_get(object_id)
        if item["kind"] != kind:
            raise GPUError(_KIND_ERRORS.get(kind, f"NOT_A_{kind.upper()}"), object_id)
        if str(item["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", object_id)
        return item

    @staticmethod
    def _node_draft(data: dict[str, Any]) -> BranchNodeDraft:
        try:
            return BranchNodeDraft.model_validate(data)
        except ValidationError as exc:
            raise GPUError("INVALID_EXPERIMENT_BRANCH_NODE", str(exc)) from exc

    @staticmethod
    def _score(draft: BranchNodeDraft) -> float:
        value = (
            draft.scientific_importance
            * draft.expected_discrimination
            * draft.expected_information_gain
            * draft.feasibility
            / (draft.compute_cost * draft.engineering_cost * draft.execution_risk)
        )
        return round(value, 6)

    @staticmethod
    def _recommend(
        branch_id: str, action: str, node: dict[str, Any], run_id: str | None = None
    ) -> dict[str, Any]:
        return {
            "branch_id": branch_id,
            "action": action,
            "node_id": str(node["id"]),
            "run_id": run_id,
            "branch_action": node["data"]["branch_action"],
            "priority_score": node["data"]["priority_score"],
            "score_components": {
                key: node["data"][key]
                for key in (
                    "scientific_importance",
                    "expected_discrimination",
                    "expected_information_gain",
                    "feasibility",
                    "compute_cost",
                    "engineering_cost",
                    "execution_risk",
                )
            },
        }
