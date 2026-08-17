import asyncio
from pathlib import Path

import pytest

from gpu_lab.config import Settings
from gpu_lab.db import Repository
from gpu_lab.local_runner import LocalRunner
from gpu_lab.models import Job
from gpu_lab.server import _mcp_client_denied, _normalise_mcp_accept_header, call, scrub


class _Process:
    pid = 424242

    async def wait(self):
        return 0


def _runner(tmp_path: Path) -> LocalRunner:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return LocalRunner(
        Settings(
            gpu_lab_enable_local_runner=True,
            gpu_lab_local_workspace=workspace,
            gpu_lab_local_env_root=tmp_path / "envs",
            gpu_lab_database_url=f"sqlite:///{tmp_path / 'gpu-lab.db'}",
        ),
        Repository(tmp_path / "gpu-lab.db"),
    )


def test_requirements_path_accepts_file_or_directory(tmp_path):
    runner = _runner(tmp_path)
    requirements = runner.workspace / "project" / "requirements.txt"
    requirements.parent.mkdir()
    requirements.write_text("pytest\n")

    assert runner._requirements_path("project") == requirements
    assert runner._requirements_path("project/requirements.txt") == requirements


def test_local_job_environment_excludes_gateway_secrets(monkeypatch, tmp_path):
    runner = _runner(tmp_path)
    monkeypatch.setenv("PATH", "/usr/local/bin")
    monkeypatch.setenv("VAST_API_KEY", "vast-secret")
    monkeypatch.setenv("GPU_LAB_APPROVAL_SECRET", "approval-secret")
    monkeypatch.setenv("GPU_LAB_RESEARCH_DATABASE_URL", "postgresql://secret")
    monkeypatch.setenv("GPU_LAB_LITERATURE_WORKER_TOKEN", "literature-secret")

    environment = runner._job_environment({"PYTHONPATH": "/project", "EXPERIMENT_SEED": "7"})

    assert environment["PATH"] == "/usr/local/bin"
    assert environment["PYTHONPATH"] == "/project"
    assert environment["EXPERIMENT_SEED"] == "7"
    assert not {
        "VAST_API_KEY",
        "GPU_LAB_APPROVAL_SECRET",
        "GPU_LAB_RESEARCH_DATABASE_URL",
        "GPU_LAB_LITERATURE_WORKER_TOKEN",
    } & environment.keys()


def test_parse_gpu_metrics_keeps_only_complete_numeric_nvidia_rows():
    metrics = LocalRunner._parse_gpu_metrics(
        "0, NVIDIA RTX 4090, 24564, 12000, 87, 64\n"
        "malformed row\n"
        "1, NVIDIA RTX 4090, unknown, 200, 5, 41\n"
    )

    assert metrics == [
        {
            "index": 0,
            "name": "NVIDIA RTX 4090",
            "memory_total_mb": 24564,
            "memory_used_mb": 12000,
            "utilization_percent": 87,
            "temperature_c": 64,
        }
    ]


def test_job_status_can_skip_expensive_log_reading(tmp_path):
    runner = _runner(tmp_path)
    job = Job(
        job_id="local_status_without_logs",
        instance_id="local",
        repo_path=str(runner.workspace),
        command="echo done",
        status="completed",
    )
    runner.repo.save_job(job)
    jobdir = runner.workspace / ".gpu-lab" / "jobs" / job.job_id
    jobdir.mkdir(parents=True)
    (jobdir / "stdout.log").write_text("large output is deliberately omitted")
    (jobdir / "exit_code").write_text("0")

    status = runner.job_status(job.job_id, include_logs=False)

    assert status == {"job_id": job.job_id, "status": "completed", "exit_code": None}


def test_mcp_wildcard_accept_header_allows_json_response():
    headers = [(b"accept", b"text/html, */*"), (b"content-type", b"application/json")]

    assert _normalise_mcp_accept_header(headers) == [
        (b"accept", b"application/json"),
        (b"content-type", b"application/json"),
    ]


def test_audit_scrub_redacts_bearer_and_assignment_credentials():
    value = (
        "Authorization: Bearer top-secret api_key=abc password:xyz token=qwerty "
        "client_secret=hidden --secret=also-hidden"
    )

    scrubbed = scrub(value)

    assert "top-secret" not in scrubbed
    assert "abc" not in scrubbed
    assert "xyz" not in scrubbed
    assert "qwerty" not in scrubbed
    assert "hidden" not in scrubbed
    assert "also-hidden" not in scrubbed


@pytest.mark.asyncio
async def test_call_audits_only_scrubbed_arguments(monkeypatch):
    audits = []

    class FakeRepository:
        def audit(self, *args):
            audits.append(args)

    class FakeService:
        repo = FakeRepository()

    monkeypatch.setattr("gpu_lab.server.svc", lambda: FakeService())

    await call(
        lambda value, *, authorization: {"ok": bool(value and authorization)},
        "client_secret=hidden",
        authorization="Bearer top-secret",
    )

    payload = str(audits[0][1])
    assert "hidden" not in payload
    assert "top-secret" not in payload
    assert "[REDACTED]" in payload


def test_mcp_network_policy_blocks_only_the_isolated_worker_subnet():
    denied = "172.29.0.0/24,172.30.0.0/24"

    assert _mcp_client_denied("172.29.0.12", denied) is True
    assert _mcp_client_denied("172.30.0.12", denied) is True
    assert _mcp_client_denied("172.28.0.12", denied) is False
    assert _mcp_client_denied("127.0.0.1", denied) is False
    assert _mcp_client_denied("not-an-ip", denied) is True


@pytest.mark.asyncio
async def test_local_submit_replays_reserved_job_id(monkeypatch, tmp_path):
    runner = _runner(tmp_path)
    calls = 0

    async def fake_subprocess(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert args[0].startswith("sh -c ")
        return _Process()

    monkeypatch.setattr("gpu_lab.local_runner.asyncio.create_subprocess_shell", fake_subprocess)
    first = await runner.submit("echo ok", job_id="local_execution_attempt_1")
    monkeypatch.setattr(
        runner,
        "job_status",
        lambda job_id: {"job_id": job_id, "status": "running", "exit_code": None},
    )
    second = await runner.submit("echo ok", job_id="local_execution_attempt_1")

    assert first["job_id"] == second["job_id"]
    assert second["idempotent_replay"] is True
    assert calls == 1


@pytest.mark.asyncio
async def test_concurrent_local_submit_spawns_canonical_job_once(monkeypatch, tmp_path):
    runner = _runner(tmp_path)
    calls = 0

    async def fake_subprocess(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Process()

    monkeypatch.setattr("gpu_lab.local_runner.asyncio.create_subprocess_shell", fake_subprocess)
    monkeypatch.setattr(runner, "_process_identity", lambda _pid: "start-ticks")
    monkeypatch.setattr(
        runner,
        "job_status",
        lambda job_id: {"job_id": job_id, "status": "running", "exit_code": None},
    )

    first, second = await asyncio.gather(
        runner.submit("echo ok", job_id="local_concurrent_attempt"),
        runner.submit("echo ok", job_id="local_concurrent_attempt"),
    )

    assert first["job_id"] == second["job_id"]
    assert second["idempotent_replay"] is True
    assert calls == 1


@pytest.mark.asyncio
async def test_separate_runners_atomically_claim_canonical_job(monkeypatch, tmp_path):
    first_runner = _runner(tmp_path)
    second_runner = _runner(tmp_path)
    spawn_entered = asyncio.Event()
    allow_spawn = asyncio.Event()
    calls = 0

    async def fake_subprocess(*args, **kwargs):
        nonlocal calls
        calls += 1
        spawn_entered.set()
        await allow_spawn.wait()
        return _Process()

    monkeypatch.setattr("gpu_lab.local_runner.asyncio.create_subprocess_shell", fake_subprocess)
    monkeypatch.setattr(first_runner, "_process_identity", lambda _pid: "start-ticks")

    first_task = asyncio.create_task(
        first_runner.submit("echo ok", job_id="local_cross_process_attempt")
    )
    await spawn_entered.wait()
    replay = await second_runner.submit(
        "echo ok", job_id="local_cross_process_attempt"
    )
    allow_spawn.set()
    first = await first_task

    assert first["status"] == "running"
    assert replay["status"] == "queued"
    assert replay["idempotent_replay"] is True
    assert calls == 1


@pytest.mark.asyncio
async def test_status_poll_cannot_invalidate_active_submission_claim(monkeypatch, tmp_path):
    first_runner = _runner(tmp_path)
    second_runner = _runner(tmp_path)
    spawn_entered = asyncio.Event()
    allow_spawn = asyncio.Event()
    calls = 0

    async def fake_subprocess(*args, **kwargs):
        nonlocal calls
        calls += 1
        spawn_entered.set()
        await allow_spawn.wait()
        return _Process()

    monkeypatch.setattr("gpu_lab.local_runner.asyncio.create_subprocess_shell", fake_subprocess)
    monkeypatch.setattr(first_runner, "_process_identity", lambda _pid: "start-ticks")

    first_task = asyncio.create_task(
        first_runner.submit("echo ok", job_id="local_polled_submission")
    )
    await spawn_entered.wait()
    polled = second_runner.job_status("local_polled_submission")
    replay = await second_runner.submit("echo ok", job_id="local_polled_submission")
    allow_spawn.set()
    first = await first_task

    assert polled["status"] == "queued"
    assert replay["status"] == "queued"
    assert replay["idempotent_replay"] is True
    assert first["status"] == "running"
    assert calls == 1


def test_reconcile_marks_foreign_running_job_unknown(tmp_path):
    runner = _runner(tmp_path)
    job = Job(
        job_id="local_previous_runner",
        instance_id="local",
        repo_path=str(runner.workspace),
        command="sleep 300",
        status="running",
        remote_pid=1,
        metadata={"runner_instance_id": "previous-container"},
    )
    runner.repo.save_job(job)

    result = runner.reconcile_jobs()

    assert result == {"reconciled": 1, "completed": 0, "failed": 0, "unknown": 1}
    assert runner.repo.get_job(job.job_id).status == "unknown"


def test_reconcile_finalizes_job_from_persisted_exit_code(tmp_path):
    runner = _runner(tmp_path)
    job = Job(
        job_id="local_finished_while_gateway_down",
        instance_id="local",
        repo_path=str(runner.workspace),
        command="echo done",
        status="running",
        remote_pid=424242,
        metadata={"runner_instance_id": "previous-container"},
    )
    runner.repo.save_job(job)
    jobdir = runner.workspace / ".gpu-lab" / "jobs" / job.job_id
    jobdir.mkdir(parents=True)
    (jobdir / "exit_code").write_text("0")

    result = runner.reconcile_jobs()

    assert result == {"reconciled": 1, "completed": 1, "failed": 0, "unknown": 0}
    recovered = runner.repo.get_job(job.job_id)
    assert recovered.status == "completed"
    assert recovered.exit_code == 0


@pytest.mark.asyncio
async def test_env_prepare_uses_explicit_python_and_exact_requirements(monkeypatch, tmp_path):
    runner = _runner(tmp_path)
    requirements = runner.workspace / "requirements-vrc.txt"
    requirements.write_text("torch==2.6.0\n")
    calls = []

    async def fake_subprocess(*args):
        calls.append(args)
        return _Process()

    monkeypatch.setattr("gpu_lab.local_runner.asyncio.create_subprocess_exec", fake_subprocess)
    result = await runner.env_prepare(
        "vrc-py313-torch260-cu124",
        "requirements-vrc.txt",
        "python3.13",
    )

    assert calls[0][:3] == ("python3.13", "-m", "venv")
    assert calls[1][-2:] == ("-r", str(requirements))
    assert result["name"] == "vrc-py313-torch260-cu124"
