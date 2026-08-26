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

    def object_get(self, experiment_id):
        return {
            "id": experiment_id,
            "project_id": "project-id",
            "data": {"plan": {}},
        }

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


class CapturingLocal(UnprovenLocal):
    def __init__(self, status):
        super().__init__(status)
        self.submit_args = None

    async def submit(self, *args):
        self.submit_args = args
        return await super().submit(*args)


class RemoteRunner:
    def __init__(self):
        self.submit_args = None
        self.repo = type("AuditRepo", (), {"audit": staticmethod(lambda *_args, **_kwargs: None)})()

    async def experiment_submit(self, *args):
        self.submit_args = args
        return {"job_id": args[-1], "status": "running", "instance": {"id": args[0]}}

    async def experiment_status(self, job_id):
        return {"job_id": job_id, "status": "completed", "exit_code": 0, "logs_tail": ["ok"]}

    async def artifact_list(self, _job_id):
        return []

    async def gpu_status(self, instance_id):
        return {"instance_id": instance_id}


class CancellationIncompleteRemote(RemoteRunner):
    async def experiment_status(self, job_id):
        return {"job_id": job_id, "status": "cancellation_incomplete", "cancellation_incomplete": True, "process_group_alive": True, "exit_code": None, "logs_tail": []}


class TerminalLab:
    def experiment_run_terminal(self, *_args):
        return {"status": "ok"}


class AuthorizingBrain:
    def authorize_execution(self, *_args):
        return {"authorized": True}


class BindingBrain:
    def __init__(self):
        self.args = None

    def execution_decision_bind(self, *args):
        self.args = args
        return {"id": args[1], "data": {"execution_binding": {"experiment_id": args[0]}}}


class DecisionCreatingBrain(BindingBrain):
    def experiment_execution_decision_create(self, _project_id, _experiment_id):
        return {
            "decision": {"id": "decision-id"},
            "large_durable_trace": "x" * 100_000,
        }


class LegacyReservedResearch:
    def object_get(self, _run_id):
        return {
            "id": "run-id",
            "kind": "ExperimentRun",
            "status": "RESERVED",
            "data": {"job_id": "missing-job"},
        }


class LegacyAbandonedResearch(LegacyReservedResearch):
    def object_get(self, _run_id):
        return {
            "id": "run-id",
            "kind": "ExperimentRun",
            "status": "cancelled",
            "data": {
                "job_id": "missing-job",
                "legacy_abandonment": {"verified_missing_backing_job": True},
            },
        }


class LegacyAbandonBrain:
    def __init__(self):
        self.args = None

    def legacy_reserved_run_abandon(self, *args):
        self.args = args
        return {"id": args[0], "status": "cancelled"}


class DecisionReservedResearch(LegacyReservedResearch):
    def object_get(self, _run_id):
        return {
            "id": "run-id",
            "kind": "ExperimentRun",
            "status": "RESERVED",
            "data": {"job_id": "missing-job", "decision_id": "decision-id"},
        }


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
async def test_execution_decision_bind_exposes_exact_request_fingerprint(monkeypatch):
    brain = BindingBrain()
    monkeypatch.setattr(server, "brain", lambda: brain)

    result = await server.execution_decision_bind(
        "experiment-id", "decision-id", "python run.py", "/workspace", {"MODE": "test"}, "torch-env"
    )

    assert result["id"] == "decision-id"
    assert brain.args[:2] == ("experiment-id", "decision-id")
    assert len(brain.args[2]) == 64
    assert brain.args[2] == server._execution_action_fingerprint(
        "experiment-id", "python run.py", "/workspace", {"MODE": "test"}, "torch-env"
    )


@pytest.mark.asyncio
async def test_decision_create_returns_compact_execution_handoff(monkeypatch):
    brain = DecisionCreatingBrain()
    monkeypatch.setattr(server, "brain", lambda: brain)

    result = await server.research_decision_create(
        "project-id", "experiment-id", "python run.py", "/workspace", {"MODE": "test"}, "torch-env"
    )

    assert result["decision_id"] == "decision-id"
    assert result["experiment_id"] == "experiment-id"
    assert result["next_tool"] == "research_experiment_execute"
    assert result["execution_binding"] == {"experiment_id": "experiment-id"}
    assert "large_durable_trace" not in result
    assert brain.args[:2] == ("experiment-id", "decision-id")


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
async def test_execute_normalizes_local_instance_label_before_reservation(monkeypatch):
    research = FakeResearch(_mapping("RESERVED"))
    captured_reservation = {}
    def reserve(*args):
        captured_reservation["execution"] = args[-1]
        return research.mapping
    research.run_reserve = reserve
    runner = CapturingLocal("queued")
    monkeypatch.setattr(server.settings, "gpu_lab_enable_local_runner", True)
    monkeypatch.setattr(server, "research", lambda: research)
    monkeypatch.setattr(server, "brain", lambda: AuthorizingBrain())
    monkeypatch.setattr(server, "local", runner)

    result = await server.research_experiment_execute(
        "experiment-id", "decision-id", "echo ok", execution_attempt_uuid="attempt-id", instance_id="local"
    )

    assert result["status"] == "RESERVED"
    assert captured_reservation["execution"]["executor"] == "local"
    assert captured_reservation["execution"]["instance_id"] is None
    assert captured_reservation["execution"]["requested_instance_id"] == "local"
    assert runner.submit_args is not None


@pytest.mark.asyncio
async def test_execute_uses_canonical_remote_submission_and_absolute_python(monkeypatch):
    research = FakeResearch(_mapping("RESERVED"))
    research.run_reserve = lambda *_args: research.mapping
    runner = RemoteRunner()
    monkeypatch.setattr(server, "research", lambda: research)
    monkeypatch.setattr(server, "brain", lambda: AuthorizingBrain())
    monkeypatch.setattr(server, "svc", lambda: runner)

    result = await server.research_experiment_execute(
        "experiment-id", "decision-id", "python train.py", "/workspace/repos/plancarry",
        python_env="/workspace/repos/plancarry/.venv/bin/python",
        execution_attempt_uuid="attempt-id", instance_id="vast_1",
    )

    assert result["execution_attempt_uuid"] == "attempt-id"
    assert runner.submit_args[0] == "vast_1"
    assert runner.submit_args[-1] == "local_reserved_job"
    assert "VIRTUAL_ENV=" in runner.submit_args[2]


@pytest.mark.asyncio
async def test_sync_uses_remote_status_artifacts_and_runtime(monkeypatch):
    mapping = _mapping("running")
    mapping["run"]["data"].update({"executor": "vast", "instance_id": "vast_1"})
    research = FakeResearch(mapping)
    runner = RemoteRunner()
    monkeypatch.setattr(server, "research", lambda: research)
    monkeypatch.setattr(server, "svc", lambda: runner)
    monkeypatch.setattr(server, "lab", lambda: TerminalLab())

    result = await server.research_experiment_sync(run_id="run-id")

    assert result["status"] == "completed"
    assert research.updates[0][1]["runtime"] == {"instance_id": "vast_1"}


@pytest.mark.asyncio
async def test_sync_reports_incomplete_cancellation_without_marking_unknown(monkeypatch):
    mapping = _mapping("running")
    mapping["run"]["data"].update({"executor": "vast", "instance_id": "vast_1"})
    research = FakeResearch(mapping)
    monkeypatch.setattr(server, "research", lambda: research)
    monkeypatch.setattr(server, "svc", lambda: CancellationIncompleteRemote())

    result = await server.research_experiment_sync(run_id="run-id")

    assert result["status"] == "CANCELLATION_INCOMPLETE"
    assert result["recovery_action"] == "CANCEL_PROCESS_GROUP"
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


@pytest.mark.asyncio
async def test_legacy_reserved_run_abandon_requires_server_verified_missing_job(monkeypatch):
    brain = LegacyAbandonBrain()
    monkeypatch.setattr(server.settings, "gpu_lab_enable_local_runner", True)
    monkeypatch.setattr(server, "research", lambda: LegacyReservedResearch())
    monkeypatch.setattr(server, "brain", lambda: brain)
    monkeypatch.setattr(server, "local", MissingLocal())

    result = await server.legacy_reserved_run_abandon("run-id", "No local job was submitted")

    assert result["status"] == "cancelled"
    assert brain.args == ("run-id", "missing-job", "No local job was submitted", False)


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


@pytest.mark.asyncio
async def test_legacy_reserved_run_abandon_allows_only_verified_abandonment_replay(monkeypatch):
    brain = LegacyAbandonBrain()
    monkeypatch.setattr(server.settings, "gpu_lab_enable_local_runner", True)
    monkeypatch.setattr(server, "research", lambda: LegacyAbandonedResearch())
    monkeypatch.setattr(server, "brain", lambda: brain)
    monkeypatch.setattr(server, "local", MissingLocal())

    result = await server.legacy_reserved_run_abandon("run-id", "No local job was submitted")

    assert result["status"] == "cancelled"
    assert brain.args == ("run-id", "missing-job", "No local job was submitted", False)


@pytest.mark.asyncio
async def test_technical_abandonment_explicitly_allows_pre_submit_decision(monkeypatch):
    brain = LegacyAbandonBrain()
    monkeypatch.setattr(server.settings, "gpu_lab_enable_local_runner", True)
    monkeypatch.setattr(server, "research", lambda: DecisionReservedResearch())
    monkeypatch.setattr(server, "brain", lambda: brain)
    monkeypatch.setattr(server, "local", MissingLocal())

    result = await server.legacy_reserved_run_abandon(
        "run-id", "Invalid environment name before job submission", technical_non_scientific=True
    )

    assert result["status"] == "cancelled"
    assert brain.args == (
        "run-id", "missing-job", "Invalid environment name before job submission", True
    )
