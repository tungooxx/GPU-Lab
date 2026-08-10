from pathlib import Path

import pytest

from gpu_lab.config import Settings
from gpu_lab.db import Repository
from gpu_lab.local_runner import LocalRunner
from gpu_lab.server import _normalise_mcp_accept_header


class _Process:
    pid = 424242

    async def wait(self):
        return 0


def _runner(tmp_path: Path) -> LocalRunner:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
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


def test_mcp_wildcard_accept_header_allows_json_response():
    headers = [(b"accept", b"text/html, */*"), (b"content-type", b"application/json")]

    assert _normalise_mcp_accept_header(headers) == [
        (b"accept", b"application/json"),
        (b"content-type", b"application/json"),
    ]


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
