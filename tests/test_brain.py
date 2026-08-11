import json
from pathlib import Path

import pytest

from gpu_lab.brain import ActionScore, ResearchBrain
from gpu_lab.errors import GPUError


def test_action_score_rewards_information_and_penalizes_cost():
    cheap_discriminator = ActionScore(
        scientific_importance=5,
        expected_discrimination=5,
        expected_information_gain=5,
        feasibility=5,
        compute_cost=0.5,
        engineering_cost=0.5,
        execution_risk=0.5,
    )
    expensive_training = ActionScore(
        scientific_importance=5,
        expected_discrimination=3,
        expected_information_gain=3,
        feasibility=2,
        compute_cost=5,
        engineering_cost=5,
        execution_risk=3,
    )

    assert cheap_discriminator.priority > expensive_training.priority


def test_configured_action_rejects_unknown_action_type():
    with pytest.raises(GPUError) as error:
        ResearchBrain._configured_candidate(
            {"action_type": "DECLARE_TRUE", "score": {}},
            "Does the intervention discriminate mechanisms?",
            [],
        )
    assert error.value.error_type == "INVALID_RESEARCH_ACTION_TYPE"


def test_causal_edge_status_requires_finite_agenda_scores():
    with pytest.raises(GPUError) as error:
        ResearchBrain._bounded(float("nan"))
    assert error.value.error_type == "INVALID_AGENDA_SCORE"


class CandidateStore:
    def __init__(self, *, reproductions=None, runs=None, evidence=None):
        self.reproductions = reproductions or []
        self.runs = runs or []
        self.evidence = evidence or []

    def objects_list(self, _project_id, kind, *_args, **_kwargs):
        if kind == "ExperimentRun":
            return self.runs
        if kind == "Reproduction":
            return self.reproductions
        if kind == "EvidenceUnit":
            return self.evidence
        return []

    def objects_identifiers(self, _project_id, kind, _statuses=None):
        return [
            {"id": item["id"], "status": item["status"]}
            for item in self.objects_list(_project_id, kind)
        ]

    def experiment_run_first(self, _project_id, statuses, inspected=None):
        return next(
            (
                run
                for run in self.runs
                if run["status"] in statuses
                and (
                    inspected is None
                    or bool(run.get("data", {}).get("inspection")) is inspected
                )
            ),
            None,
        )


def _hasi_episode():
    path = Path(__file__).parents[1] / "research_bench" / "hasi_before_intervention.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_hasi_benchmark_enforces_reproduction_then_causal_intervention():
    episode = _hasi_episode()
    agenda = {"data": episode["historical_state"]["agenda_item"]}
    hypotheses = episode["historical_state"]["active_hypotheses"]
    brain = ResearchBrain(CandidateStore(reproductions=[{"id": "baseline", "status": "PARTIAL"}]))

    before = brain._candidate_actions("project", agenda, hypotheses)
    assert before[0].action_type == "REPRODUCTION"

    brain.store.reproductions[0]["status"] = "REPRODUCED"
    after = brain._candidate_actions("project", agenda, hypotheses)
    selected = max(after, key=lambda candidate: candidate.score.priority)
    assert selected.action_type == "CAUSAL_INTERVENTION"
    assert selected.action_type in episode["known_good_next_tests"]
    assert selected.action_type not in episode["known_bad_decisions"]


def test_provider_failure_selects_an_alternative_action():
    agenda = {
        "data": {
            "question": "Does the state intervention change the output?",
            "candidate_experiments": [
                {
                    "action_type": "CAUSAL_INTERVENTION",
                    "available": False,
                    "blocked_reason": "GPU unavailable",
                }
            ],
        }
    }
    brain = ResearchBrain(CandidateStore())

    selected = brain._candidate_actions("project", agenda, [])[0]

    assert selected.action_type == "LITERATURE_SEARCH"
    assert selected.payload["mode"] == "ALTERNATIVE_ACTION"
    assert selected.payload["blocked_actions"][0]["blocked_reason"] == "GPU unavailable"


def test_portfolio_get_does_not_mutate_scientific_state():
    store = CandidateStore()
    brain = ResearchBrain(store)

    portfolio = brain.portfolio_get("project")

    assert portfolio["status"] == "NOT_MATERIALIZED"
    assert portfolio["data"]["active_hypothesis_ids"] == []


def test_literature_candidates_change_next_action_to_evidence_review():
    brain = ResearchBrain(
        CandidateStore(
            evidence=[{"id": "evidence-1", "kind": "EvidenceUnit", "status": "CANDIDATE"}]
        )
    )
    agenda = {
        "data": {
            "question": "What does prior work establish?",
            "candidate_experiments": [{"action_type": "LITERATURE_SEARCH"}],
        }
    }

    selected = brain._candidate_actions("project", agenda, [])[0]

    assert selected.action_type == "EVIDENCE_REVIEW"
    assert selected.payload["evidence_ids"] == ["evidence-1"]


class AuthorizationStore:
    def __init__(self, action_type="FROZEN_DIAGNOSTIC", fingerprint="bound-request"):
        self.objects = {
            "experiment": {
                "id": "experiment",
                "project_id": "project",
                "kind": "Experiment",
                "status": "PREREGISTERED",
                "data": {
                    "hypothesis_id": "hypothesis",
                    "plan": {"research_question": "Does the intervention discriminate?"},
                },
            },
            "decision": {
                "id": "decision",
                "project_id": "project",
                "kind": "ResearchDecision",
                "status": "SELECTED",
                "data": {
                    "hypotheses_affected": ["hypothesis"],
                    "selected_action": {
                        "id": "action",
                        "action_type": action_type,
                        "question_addressed": "Does the intervention discriminate?",
                    },
                    "execution_binding": {
                        "experiment_id": "experiment",
                        "request_fingerprint": fingerprint,
                    },
                },
            },
        }

    def object_get(self, object_id):
        return self.objects[object_id]


def test_non_execution_decision_cannot_authorize_command():
    brain = ResearchBrain(AuthorizationStore(action_type="LITERATURE_SEARCH"))

    with pytest.raises(GPUError) as error:
        brain.authorize_execution("experiment", "decision", "bound-request")

    assert error.value.error_type == "RESEARCH_DECISION_NOT_EXECUTABLE"


def test_execution_decision_requires_exact_bound_command():
    brain = ResearchBrain(AuthorizationStore())

    with pytest.raises(GPUError) as error:
        brain.authorize_execution("experiment", "decision", "different-request")

    assert error.value.error_type == "RESEARCH_EXECUTION_NOT_BOUND"
    authorized = brain.authorize_execution("experiment", "decision", "bound-request")
    assert authorized["action_type"] == "FROZEN_DIAGNOSTIC"


class AssessmentStore:
    def __init__(self):
        self.applied = None
        self.objects = {
            "run": {
                "id": "run",
                "project_id": "project",
                "kind": "ExperimentRun",
                "status": "completed",
                "data": {"decision_id": "decision", "experiment_id": "experiment"},
            },
            "decision": {
                "id": "decision",
                "project_id": "project",
                "kind": "ResearchDecision",
                "status": "SELECTED",
                "data": {"agenda_item_id": "agenda", "hypotheses_affected": ["hypothesis"]},
            },
            "hypothesis": {
                "id": "hypothesis",
                "project_id": "project",
                "kind": "Hypothesis",
                "status": "ACTIVE",
                "data": {},
            },
            "agenda": {
                "id": "agenda",
                "project_id": "project",
                "kind": "AgendaItem",
                "status": "ACTIVE",
                "data": {},
            },
            "experiment": {
                "id": "experiment",
                "project_id": "project",
                "kind": "Experiment",
                "status": "PREREGISTERED",
                "data": {
                    "hypothesis_id": "hypothesis",
                    "plan": {"pass_condition": "metric > 0"},
                },
            },
        }

    def object_get(self, object_id):
        return self.objects[object_id]

    def objects_list(self, *_args, **_kwargs):
        return [{"data": {"pass_condition": "metric > 0"}}]

    def result_assessment_apply(self, **kwargs):
        self.applied = kwargs
        return {"run": {"status": "RESULT_INSPECTED"}}


@pytest.mark.parametrize("transition", ["INCONCLUSIVE", "BLOCKED"])
def test_nonconclusive_assessment_keeps_agenda_active(transition):
    store = AssessmentStore()
    brain = ResearchBrain(store)

    brain.result_assess(
        run_id="run",
        decision_id="decision",
        hypothesis_id="hypothesis",
        agenda_item_id="agenda",
        prediction_outcome="No conclusive result",
        guard_condition_outcome="Guard evaluated",
        condition_evaluations={"metric > 0": False},
        evidence_supporting=[],
        evidence_against=[],
        unexpected_observations=[],
        alternative_explanations=[],
        scope="fixture",
        hypothesis_transition=transition,
        rationale="More work is required",
    )

    assert store.applied["agenda_status"] == "ACTIVE"
