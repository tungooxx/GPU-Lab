import pytest

from gpu_lab import server


class FakeResearch:
    def __init__(self, mapping):
        self.mapping = mapping
        self.updates = []
        self.artifacts = []

    def run_resolve(self, _identifier):
        return self.mapping

    def run_update(self, run_id, result):
        self.updates.append((run_id, result))
        return {"id": run_id, "status": result["status"], "data": result}

    def object_create(self, *args):
        self.artifacts.append(args)
        return {"id": "artifact"}

    def artifact_record(self, *args):
        self.artifacts.append(args)
        return {"id": "artifact", "idempotent_replay": False}

    def edge_create(self, *_args):
        return None


class LocalMustNotRun:
    def job_status(self, _job_id):
        raise AssertionError("reserved or inspected executions must not query a local job")

    def artifacts(self, _job_id):
        raise AssertionError("reserved or inspected executions must not enumerate artifacts")

    async def status(self):
        raise AssertionError("reserved or inspected executions must not query runtime status")


def _mapping(status):
    return {
        "experiment_id": "experiment-id",
        "run_id": "run-id",
        "job_id": "local_reserved_job",
        "idempotency_key": "attempt-id",
        "status": status,
        "run": {
            "id": "run-id",
            "project_id": "project-id",
            "status": status,
            "data": {"experiment_id": "experiment-id", "job_id": "local_reserved_job"},
        },
    }


@pytest.mark.asyncio
async def test_sync_returns_reserved_mapping_without_missing_job_exception(monkeypatch):
    research = FakeResearch(_mapping("RESERVED"))
    monkeypatch.setattr(server, "research", lambda: research)
    monkeypatch.setattr(server, "local", LocalMustNotRun())

    result = await server.research_experiment_sync(run_id="run-id")

    assert result["status"] == "RESERVED"
    assert result["run_id"] == "run-id"
    assert result["job_id"] == "local_reserved_job"
    assert result["retry_safe"] is True
    assert result["recovery_action"] == "RETRY_EXECUTION"
    assert research.updates == []


@pytest.mark.asyncio
async def test_sync_preserves_inspected_run_and_does_not_duplicate_artifacts(monkeypatch):
    research = FakeResearch(_mapping("RESULT_INSPECTED"))
    monkeypatch.setattr(server, "research", lambda: research)
    monkeypatch.setattr(server, "local", LocalMustNotRun())

    result = await server.research_experiment_sync(job_id="local_reserved_job")

    assert result["status"] == "RESULT_INSPECTED"
    assert result["run"]["status"] == "RESULT_INSPECTED"
    assert research.updates == []
    assert research.artifacts == []
