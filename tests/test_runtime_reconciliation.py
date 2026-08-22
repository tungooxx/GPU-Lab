import pytest

from gpu_lab import server
from gpu_lab.brain import ResearchBrain
from gpu_lab.research import ResearchStore


def test_inspected_technical_unknown_run_is_not_operationally_active():
    run = {
        "status": "unknown",
        "data": {
            "inspection": {
                "technical_non_scientific": True,
                "scientific_result": "NOT_ASSESSED",
                "prediction_outcome": "TECHNICAL_STALE_DUPLICATE_NO_RESULT",
                "actual_information_gain": "ZERO",
            }
        },
    }
    assert ResearchStore.run_has_terminal_technical_inspection(run)
    assert not ResearchStore.experiment_run_is_operationally_active(run)


def test_agenda_candidate_normalizes_prediction_and_experiment_id_shorthand():
    candidate = ResearchBrain._configured_candidate(
        {
            "action_type": "CAUSAL_INTERVENTION",
            "prediction": "The treatment improves the held-out metric.",
            "experiment_id": "experiment-id",
            "score": {},
        },
        "Does treatment help?",
        [],
    )
    assert candidate.predicted_outcomes == ["The treatment improves the held-out metric."]
    assert candidate.payload["experiment_id"] == "experiment-id"


class _OrphanResearch:
    def run_resolve(self, _run_id):
        return {"run_id": "run-id", "job_id": "local-job"}

    def run_reconcile_orphan(self, *args):
        self.args = args
        return {"id": args[0], "status": args[2], "data": {}}


class _MissingLocal:
    def job_status(self, _job_id):
        return {"error": {"type": "JOB_NOT_FOUND"}}


class _LabLifecycle:
    def experiment_run_terminal(self, _run_id, _status):
        return {"result_ready": 0}


@pytest.mark.asyncio
async def test_orphan_reconciliation_requires_explicit_non_scientific_confirmation(monkeypatch):
    research = _OrphanResearch()
    monkeypatch.setattr(server, "research", lambda: research)
    monkeypatch.setattr(server, "local", _MissingLocal())
    monkeypatch.setattr(server, "lab", lambda: _LabLifecycle())

    blocked = await server.research_experiment_reconcile_orphan("run-id", "dead worker")
    assert blocked["error"]["type"] == "TECHNICAL_NON_SCIENTIFIC_CONFIRMATION_REQUIRED"

    result = await server.research_experiment_reconcile_orphan(
        "run-id", "confirmed missing process and no result artifact", technical_non_scientific=True
    )
    assert result["run"]["status"] == "TECHNICAL_ORPHANED"
    assert research.args[3] == "missing"
