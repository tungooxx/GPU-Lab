import json
from pathlib import Path

import pytest

from gpu_lab.brain import ActionScore, ResearchBrain
from gpu_lab.brain_bench import BenchmarkEpisode
from gpu_lab.errors import GPUError
from gpu_lab.research import _json_document


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


def test_research_json_document_normalizes_postgres_uuid_values():
    from uuid import uuid4

    identifier = uuid4()

    assert _json_document({"id": identifier, "nested": [identifier]}) == {
        "id": str(identifier),
        "nested": [str(identifier)],
    }


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


def test_state_snapshot_keeps_object_identity_without_copying_large_payloads():
    state = {
        "canonical_state": {
            "research_question": "What happened?",
            "active_hypotheses": [
                {
                    "id": "hypothesis-1",
                    "kind": "Hypothesis",
                    "status": "ACTIVE",
                    "data": {"large": "x" * 100_000},
                }
            ],
        }
    }

    snapshot = ResearchBrain._state_snapshot(state)

    assert snapshot["active_hypotheses"] == [
        {"id": "hypothesis-1", "kind": "Hypothesis", "status": "ACTIVE"}
    ]


def test_state_snapshot_preserves_scalar_canonical_entries():
    state = {"canonical_state": {"established_facts": ["fact-a", "fact-b"]}}
    snapshot = ResearchBrain._state_snapshot(state)
    assert snapshot["established_facts"] == ["fact-a", "fact-b"]


class CandidateStore:
    def __init__(self, *, reproductions=None, runs=None, evidence=None, null_models=None):
        self.reproductions = reproductions or []
        self.runs = runs or []
        self.evidence = evidence or []
        self.null_models = null_models or []

    def objects_list(self, _project_id, kind, *_args, **_kwargs):
        if kind == "ExperimentRun":
            return self.runs
        if kind == "Reproduction":
            return self.reproductions
        if kind == "EvidenceUnit":
            return self.evidence
        if kind == "NullModel":
            return self.null_models
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
    return BenchmarkEpisode.model_validate(json.loads(path.read_text(encoding="utf-8")))


def test_hasi_benchmark_enforces_reproduction_then_causal_intervention():
    episode = _hasi_episode()
    state_substitution = next(
        action for action in episode.candidate_actions if action.action_id == "state-substitution"
    )
    agenda = {
        "data": {
            "question": episode.scientific_question,
            "reproduction_gate_scope": "BASELINE_COMPARISON",
            "candidate_experiments": [
                {
                    "action_type": state_substitution.action_type,
                    "predicted_outcomes": [state_substitution.prediction],
                    "payload": {"benchmark_action_id": state_substitution.action_id},
                    "score": {
                        "scientific_importance": 5,
                        "expected_discrimination": 5,
                        "expected_information_gain": state_substitution.expected_information_gain,
                        "feasibility": 5,
                        "compute_cost": state_substitution.compute_cost,
                        "engineering_cost": state_substitution.engineering_cost,
                        "execution_risk": state_substitution.execution_risk,
                    },
                }
            ],
        }
    }
    hypotheses = [
        {"id": hypothesis_id, "status": "ACTIVE", "data": {"mechanism": hypothesis_id}}
        for hypothesis_id in episode.known_active_hypotheses
    ]
    brain = ResearchBrain(CandidateStore(reproductions=[{"id": "baseline", "status": "PARTIAL"}]))

    before = brain._candidate_actions("project", agenda, hypotheses)
    assert before[0].action_type == "REPRODUCTION"

    brain.store.reproductions[0]["status"] = "REPRODUCED"
    after = brain._candidate_actions("project", agenda, hypotheses)
    selected = max(after, key=lambda candidate: candidate.score.priority)
    assert selected.action_type == "CAUSAL_INTERVENTION"
    assert selected.payload["benchmark_action_id"] == "state-substitution"


def test_incomplete_reproduction_does_not_block_causal_development():
    agenda = {
        "data": {
            "question": "Does the frozen causal intervention discriminate mechanisms?",
            # Legacy state may retain this boolean.  Without an explicit
            # publication/baseline scope it is not an execution prerequisite.
            "reproduction_required": True,
            "candidate_experiments": [
                {
                    "action_type": "CAUSAL_INTERVENTION",
                    "prediction": "The preregistered metric differs from control.",
                    "payload": {"development_stage": "CAUSAL_DEVELOPMENT"},
                }
            ],
        }
    }
    brain = ResearchBrain(CandidateStore(reproductions=[{"id": "baseline", "status": "PARTIAL"}]))

    selected = brain._candidate_actions("project", agenda, [])

    assert selected[0].action_type == "CAUSAL_INTERVENTION"


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


def test_operationally_inspected_abandoned_run_does_not_block_new_experiment():
    agenda = {
        "data": {
            "question": "Does the frozen intervention discriminate?",
            "candidate_experiments": [
                {"action_type": "CAUSAL_INTERVENTION", "available": True}
            ],
        }
    }
    brain = ResearchBrain(
        CandidateStore(
            runs=[
                {
                    "id": "abandoned-run",
                    "status": "cancelled",
                    "data": {"inspection": {"mode": "EXECUTION_NOT_SUBMITTED"}},
                }
            ]
        )
    )

    selected = brain._candidate_actions("project", agenda, [])[0]

    assert selected.action_type == "CAUSAL_INTERVENTION"


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


def test_strong_cheap_null_model_preempts_architecture_or_causal_work():
    brain = ResearchBrain(
        CandidateStore(
            null_models=[
                {
                    "id": "null-1",
                    "kind": "NullModel",
                    "status": "ACTIVE",
                    "data": {
                        "strength": "STRONG",
                        "estimated_cost": 0.5,
                        "tested": False,
                        "action_type": "MAGNITUDE_MATCHED_CONTROL",
                        "expected_outcome": "Random magnitude-matched substitution mimics target.",
                        "discriminating_control": "Match perturbation norm.",
                        "target_entity_id": "hypothesis",
                    },
                }
            ]
        )
    )
    agenda = {
        "data": {
            "question": "Does the intervention identify the mechanism?",
            "candidate_experiments": [{"action_type": "CAUSAL_INTERVENTION"}],
        }
    }

    selected = brain._candidate_actions("project", agenda, [])[0]

    assert selected.action_type == "MAGNITUDE_MATCHED_CONTROL"
    assert selected.payload["null_model_id"] == "null-1"


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


def test_causal_execution_is_authorized_without_a_separate_approval():
    brain = ResearchBrain(AuthorizationStore(action_type="CAUSAL_INTERVENTION"))

    authorized = brain.authorize_execution("experiment", "decision", "bound-request")

    assert authorized["requires_human_approval"] is False
    assert authorized["approved"] is True


class FrozenExperimentDecisionStore:
    def __init__(self):
        self.persisted = None
        self.objects = {
            "experiment": {
                "id": "experiment",
                "project_id": "project",
                "kind": "Experiment",
                "status": "PREREGISTERED",
                "data": {
                    "hypothesis_id": "hypothesis",
                    "frozen": True,
                    "plan": {
                        "research_question": "Does frozen MSRS A0C distinguish the proposed mechanism?",
                        "prediction": "The preregistered A0C metric improves over the matched control.",
                        "pass_condition": "Metric exceeds the frozen threshold.",
                        "fail_condition": "Metric does not exceed the frozen threshold.",
                        "interpretation_if_pass": "Support the scoped MSRS mechanism.",
                        "interpretation_if_fail": "Do not support the scoped MSRS mechanism.",
                    },
                },
            },
            "hypothesis": {
                "id": "hypothesis",
                "project_id": "project",
                "kind": "Hypothesis",
                "status": "ACTIVE",
                "data": {},
            },
        }

    def object_get(self, object_id):
        return self.objects[object_id]

    def brain_decision_create(self, project_id, candidates, selected_index, decision_data):
        self.persisted = (project_id, candidates, selected_index, decision_data)
        selected = {**candidates[selected_index], "id": "candidate"}
        return {"decision": {"id": "decision", "data": {**decision_data, "selected_action": selected}}}


def test_experiment_execution_decision_uses_frozen_question_and_prediction():
    store = FrozenExperimentDecisionStore()
    brain = ResearchBrain(store)
    brain.brain_step = lambda _project_id, persist=False: {
        "agenda_item": {"id": "agenda"},
        "scientific_state": {"research_question": "agenda-level question"},
    }

    result = brain.experiment_execution_decision_create("project", "experiment")

    selected = result["decision"]["data"]["selected_action"]
    assert selected["question_addressed"] == "Does frozen MSRS A0C distinguish the proposed mechanism?"
    assert selected["predicted_outcomes"] == ["The preregistered A0C metric improves over the matched control."]
    assert selected["payload"]["prospective_prediction"] == selected["predicted_outcomes"][0]
    assert result["decision"]["data"]["question"] == selected["question_addressed"]


def test_experiment_execution_decision_preserves_discovery_gate():
    store = FrozenExperimentDecisionStore()
    brain = ResearchBrain(store)

    def blocked(*_args, **_kwargs):
        raise GPUError("DISCOVERY_ROUND_INCOMPLETE", "Complete FAR first")

    brain.brain_step = blocked
    with pytest.raises(GPUError) as error:
        brain.experiment_execution_decision_create("project", "experiment")

    assert error.value.error_type == "DISCOVERY_ROUND_INCOMPLETE"
    assert store.persisted is None


class LegacyRepairStore:
    def __init__(self):
        self.created = []
        self.objects = {
            "run": {
                "id": "run",
                "project_id": "project",
                "kind": "ExperimentRun",
                "status": "completed",
                "data": {"experiment_id": "experiment", "artifacts": [{"path": "stdout.log"}]},
            },
            "experiment": {
                "id": "experiment",
                "project_id": "project",
                "kind": "Experiment",
                "status": "ACTIVE",
                "data": {"hypothesis_id": "hypothesis", "plan": {"research_question": "Q?"}},
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
                "status": "OPEN",
                "data": {"question": "Q?"},
            },
        }

    def object_get(self, object_id):
        return self.objects[object_id]

    def object_create(self, project_id, kind, data, event_type, status="ACTIVE"):
        item = {
            "id": "reconstructed-decision",
            "project_id": project_id,
            "kind": kind,
            "status": status,
            "data": data,
        }
        self.objects[item["id"]] = item
        self.created.append((item, event_type))
        return item

    def object_update(self, object_id, data_update, status, event_type):
        item = self.objects[object_id]
        item["data"] = {**item["data"], **data_update}
        item["status"] = status
        return item


def test_legacy_run_provenance_repair_reconstructs_inspection_decision_once():
    store = LegacyRepairStore()
    brain = ResearchBrain(store)

    repaired = brain.legacy_run_provenance_repair("run", "agenda", "Historical run predates binding")
    replay = brain.legacy_run_provenance_repair("run", "agenda", "Historical run predates binding")

    assert repaired["decision"]["data"]["legacy_provenance"]["reconstructed"] is True
    assert repaired["decision"]["data"]["selected_action"]["action_type"] == "ARTIFACT_ANALYSIS"
    assert store.objects["run"]["data"]["decision_id"] == "reconstructed-decision"
    assert replay["idempotent_replay"] is True


class LegacyAbandonStore(LegacyRepairStore):
    def __init__(self):
        super().__init__()
        self.objects["run"]["status"] = "RESERVED"
        self.objects["run"]["data"].update(
            {"job_id": "missing-job", "decision_id": "prediction"}
        )
        self.objects["prediction"] = {
            "id": "prediction",
            "project_id": "project",
            "kind": "Prediction",
            "status": "ACTIVE",
            "data": {},
        }
        self.abandon_args = None

    def legacy_reserved_run_abandon(self, run_id, job_id, rationale, provenance):
        self.abandon_args = (run_id, job_id, rationale, provenance)
        self.objects[run_id]["status"] = "cancelled"
        self.objects[run_id]["data"]["legacy_abandonment"] = {
            "verified_missing_backing_job": True,
            "job_id": job_id,
        }
        return {"id": run_id, "status": "cancelled"}


def test_legacy_reserved_run_abandon_preserves_pre_decision_provenance():
    store = LegacyAbandonStore()
    result = ResearchBrain(store).legacy_reserved_run_abandon(
        "run", "missing-job", "No local job was ever submitted"
    )

    assert result["status"] == "cancelled"
    assert store.abandon_args == (
        "run",
        "missing-job",
        "No local job was ever submitted",
        {
            "pre_research_decision": True,
            "original_decision_id": "prediction",
            "original_decision_kind": "Prediction",
        },
    )


def test_technical_abandonment_preserves_existing_research_decision_without_evidence():
    store = LegacyAbandonStore()
    store.objects["decision"] = {
        "id": "decision", "project_id": "project", "kind": "ResearchDecision",
        "status": "APPROVED", "data": {},
    }
    store.objects["run"]["data"]["decision_id"] = "decision"

    ResearchBrain(store).legacy_reserved_run_abandon(
        "run", "missing-job", "Invalid environment name before submit", technical_non_scientific=True
    )

    assert store.abandon_args[3] == {
        "pre_research_decision": False,
        "technical_non_scientific": True,
        "original_decision_id": "decision",
        "original_decision_kind": "ResearchDecision",
    }


def test_legacy_abandonment_replay_allows_reconstructed_decision():
    store = LegacyAbandonStore()
    store.objects["run"]["status"] = "cancelled"
    store.objects["run"]["data"]["legacy_abandonment"] = {
        "verified_missing_backing_job": True
    }
    store.objects["run"]["data"]["decision_id"] = "reconstructed-decision"
    store.objects["reconstructed-decision"] = {
        "id": "reconstructed-decision",
        "project_id": "project",
        "kind": "ResearchDecision",
        "status": "SELECTED",
        "data": {},
    }

    ResearchBrain(store).legacy_reserved_run_abandon(
        "run", "missing-job", "No local job was ever submitted"
    )

    assert store.abandon_args[3]["original_decision_kind"] == "ResearchDecision"


def test_legacy_repair_replaces_only_abandoned_pre_decision_provenance():
    store = LegacyAbandonStore()
    store.objects["run"]["status"] = "cancelled"
    store.objects["run"]["data"]["legacy_abandonment"] = {
        "verified_missing_backing_job": True
    }

    repaired = ResearchBrain(store).legacy_run_provenance_repair(
        "run", "agenda", "Repair a pre-ResearchDecision reservation"
    )

    assert repaired["decision"]["data"]["legacy_provenance"]["superseded_pre_decision_id"] == "prediction"
    assert repaired["decision"]["data"]["legacy_provenance"]["superseded_pre_decision_kind"] == "Prediction"


def test_legacy_repair_rejects_unabandoned_pre_decision_provenance():
    store = LegacyAbandonStore()
    store.objects["run"]["status"] = "cancelled"

    with pytest.raises(GPUError, match="prediction"):
        ResearchBrain(store).legacy_run_provenance_repair("run", "agenda", "No proof")


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


def test_assessment_binds_guard_boolean_to_frozen_condition():
    store = AssessmentStore()
    brain = ResearchBrain(store)

    brain.result_assess(
        run_id="run",
        decision_id="decision",
        hypothesis_id="hypothesis",
        agenda_item_id="agenda",
        prediction_outcome="Guard failed",
        guard_condition_outcome="FAIL",
        # Simulate a transport that cannot preserve a Unicode condition key.
        condition_evaluations={"metric ? 0": False},
        evidence_supporting=[],
        evidence_against=[],
        unexpected_observations=[],
        alternative_explanations=[],
        scope="fixture",
        hypothesis_transition="BLOCKED",
        rationale="The frozen guard failed",
        guard_passed=False,
    )

    assert store.applied["evidence_data"]["condition_evaluations"] == {"metric > 0": False}


def test_assessment_normalizes_legacy_array_guard_evaluation():
    store = AssessmentStore()
    brain = ResearchBrain(store)

    brain.result_assess(
        run_id="run",
        decision_id="decision",
        hypothesis_id="hypothesis",
        agenda_item_id="agenda",
        prediction_outcome="Guard failed",
        guard_condition_outcome="FAIL",
        condition_evaluations=[{"passed": False}],
        evidence_supporting=[],
        evidence_against=[],
        unexpected_observations=[],
        alternative_explanations=[],
        scope="fixture",
        hypothesis_transition="BLOCKED",
        rationale="The frozen guard failed",
    )

    assert store.applied["evidence_data"]["condition_evaluations"] == {"metric > 0": False}
