import pytest

from gpu_lab import server


class FakeResearch:
    def __init__(self, mapping):
        self.mapping = mapping
        self.updates = []
        self.artifacts = []
        self.promotions = 0

    def run_resolve(self, _identifier):
        return self.mapping

    def run_update(self, run_id, result):
        self.updates.append((run_id, result))
        return {"id": run_id, "status": result["status"], "data": result}

    def run_mark_submitted(self, _run_id):
        self.promotions += 1
        self.mapping = {
            **self.mapping,
            "status": "running",
            "run": {**self.mapping["run"], "status": "running"},
        }
        return self.mapping

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


class MissingLocal:
    def job_status(self, _job_id):
        return {"error": {"type": "JOB_NOT_FOUND", "message": "missing"}}


class RunningLocal:
    def job_status(self, job_id):
        return {"job_id": job_id, "status": "running", "exit_code": None, "logs_tail": ""}

    def artifacts(self, _job_id):
        return []

    async def status(self):
        return {"instance_id": "local"}


class UnprovenLocal:
    def __init__(self, status):
        self.status = status

    def job_status(self, job_id):
        return {"job_id": job_id, "status": self.status, "exit_code": None, "logs_tail": ""}

    async def submit(self, *_args):
        return {"job_id": "local_reserved_job", "status": self.status}


class AuthorizingBrain:
    def authorize_execution(self, *_args):
        return {"authorized": True}


class LegacyReservedResearch:
    def object_get(self, _run_id):
        return {
            "id": "run-id",
            "kind": "ExperimentRun",
            "status": "RESERVED",
            "data": {"job_id": "missing-job"},
        }


class LegacyAbandonBrain:
    def __init__(self):
        self.args = None

    def legacy_reserved_run_abandon(self, *args):
        self.args = args
        return {"id": args[0], "status": "cancelled"}


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
    monkeypatch.setattr(server, "local", MissingLocal())

    result = await server.research_experiment_sync(run_id="run-id")

    assert result["status"] == "RESERVED"
    assert result["run_id"] == "run-id"
    assert result["job_id"] == "local_reserved_job"
    assert result["retry_safe"] is True
    assert result["recovery_action"] == "RETRY_EXECUTION"
    assert research.updates == []


@pytest.mark.asyncio
async def test_sync_promotes_reserved_mapping_when_local_job_exists(monkeypatch):
    research = FakeResearch(_mapping("RESERVED"))
    monkeypatch.setattr(server, "research", lambda: research)
    monkeypatch.setattr(server, "local", RunningLocal())

    result = await server.research_experiment_sync(run_id="run-id")

    assert result["status"] == "running"
    assert result["run_id"] == "run-id"
    assert research.updates[0][1]["status"] == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize("runner_status", ["queued", "unknown"])
async def test_sync_does_not_promote_reserved_mapping_without_launch_proof(
    monkeypatch, runner_status
):
    research = FakeResearch(_mapping("RESERVED"))
    monkeypatch.setattr(server, "research", lambda: research)
    monkeypatch.setattr(server, "local", UnprovenLocal(runner_status))

    result = await server.research_experiment_sync(run_id="run-id")

    assert result["status"] == "RESERVED"
    assert result["runner_status"] == runner_status
    assert result["recovery_action"] == "RETRY_EXECUTION"
    assert research.promotions == 0


@pytest.mark.asyncio
async def test_execute_does_not_mark_queued_replay_as_started(monkeypatch):
    research = FakeResearch(_mapping("RESERVED"))
    research.run_reserve = lambda *_args: research.mapping
    monkeypatch.setattr(server.settings, "gpu_lab_enable_local_runner", True)
    monkeypatch.setattr(server, "research", lambda: research)
    monkeypatch.setattr(server, "brain", lambda: AuthorizingBrain())
    monkeypatch.setattr(server, "local", UnprovenLocal("queued"))

    result = await server.research_experiment_execute(
        "experiment-id", "decision-id", "echo ok", execution_attempt_uuid="attempt-id"
    )

    assert result["status"] == "RESERVED"
    assert result["runner_status"] == "queued"
    assert result["recovery_action"] == "RETRY_EXECUTION"
    assert research.promotions == 0


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


@pytest.mark.asyncio
async def test_legacy_reserved_run_abandon_requires_server_verified_missing_job(monkeypatch):
    brain = LegacyAbandonBrain()
    monkeypatch.setattr(server.settings, "gpu_lab_enable_local_runner", True)
    monkeypatch.setattr(server, "research", lambda: LegacyReservedResearch())
    monkeypatch.setattr(server, "brain", lambda: brain)
    monkeypatch.setattr(server, "local", MissingLocal())

    result = await server.legacy_reserved_run_abandon("run-id", "No local job was submitted")

    assert result["status"] == "cancelled"
    assert brain.args == ("run-id", "missing-job", "No local job was submitted")


@pytest.mark.asyncio
async def test_legacy_reserved_run_abandon_refuses_when_job_exists(monkeypatch):
    brain = LegacyAbandonBrain()
    monkeypatch.setattr(server.settings, "gpu_lab_enable_local_runner", True)
    monkeypatch.setattr(server, "research", lambda: LegacyReservedResearch())
    monkeypatch.setattr(server, "brain", lambda: brain)
    monkeypatch.setattr(server, "local", RunningLocal())

    result = await server.legacy_reserved_run_abandon("run-id", "No local job was submitted")

    assert result["error"]["type"] == "LEGACY_RUN_BACKING_JOB_NOT_PROVEN_ABSENT"
    assert brain.args is None
