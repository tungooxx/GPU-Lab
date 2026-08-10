import asyncio
import os
import shlex
import signal
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .db import Repository
from .errors import GPUError
from .models import Job


class LocalRunner:
    """Run detached Linux experiments only within the mounted local workspace."""

    def __init__(self, settings: Settings, repo: Repository):
        self.settings, self.repo = settings, repo
        self.workspace = settings.gpu_lab_local_workspace.resolve()

    def _require_enabled(self) -> None:
        if not self.settings.gpu_lab_enable_local_runner:
            raise GPUError("LOCAL_RUNNER_DISABLED", "Set GPU_LAB_ENABLE_LOCAL_RUNNER=true")
        if not self.workspace.is_dir():
            raise GPUError("LOCAL_WORKSPACE_MISSING", f"Workspace is unavailable: {self.workspace}")

    def _path(self, working_directory: str) -> Path:
        candidate = (self.workspace / working_directory).resolve()
        if not candidate.is_relative_to(self.workspace) or not candidate.is_dir():
            raise GPUError("INVALID_LOCAL_PATH", "Working directory must be inside the local workspace")
        return candidate

    async def status(self) -> dict:
        self._require_enabled()
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise GPUError("NVIDIA_SMI_UNAVAILABLE", "nvidia-smi is unavailable in the local runtime") from exc
        out, err = await proc.communicate()
        runtime = await asyncio.create_subprocess_shell(
            "python3 -c \"import sys; print(sys.executable); print(sys.version.split()[0]); import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())\"",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        runtime_out, _ = await runtime.communicate()
        return {
            "instance_id": "local",
            "workspace": str(self.workspace),
            "gpu": out.decode().strip() if proc.returncode == 0 else None,
            "gpu_error": err.decode().strip() if proc.returncode else None,
            "python_runtime": runtime_out.decode().strip().splitlines(),
            "environment_root": str(self.settings.gpu_lab_local_env_root),
        }

    async def submit(
        self,
        command: str,
        working_directory: str = ".",
        name: str | None = None,
        env: dict[str, str] | None = None,
        python_env: str | None = None,
    ) -> dict:
        self._require_enabled()
        workdir = self._path(working_directory)
        job_id = "local_" + uuid.uuid4().hex[:16]
        jobdir = self.workspace / ".gpu-lab" / "jobs" / job_id
        jobdir.mkdir(parents=True, exist_ok=False)
        stdout, stderr, exit_code = jobdir / "stdout.log", jobdir / "stderr.log", jobdir / "exit_code"
        script = jobdir / "command.sh"
        script.write_text(command)
        script.chmod(0o700)
        run_env = {**os.environ, **{k: v for k, v in (env or {}).items() if k.replace("_", "a").isalnum()}}
        run_env["GPU_LAB_JOB_DIR"] = str(jobdir)
        if python_env:
            venv = self.settings.gpu_lab_local_env_root / python_env
            if not venv.is_dir() or "/" in python_env or "\\" in python_env:
                raise GPUError("LOCAL_ENV_NOT_FOUND", f"No persistent environment named {python_env}")
            run_env["PATH"] = f"{venv / 'bin'}:{run_env.get('PATH', '')}"
            run_env["VIRTUAL_ENV"] = str(venv)
        wrapper = f"bash {shlex.quote(str(script))}; code=$?; printf '%s' $code > {shlex.quote(str(exit_code))}; exit $code"
        with stdout.open("wb") as stdout_handle, stderr.open("wb") as stderr_handle:
            process = await asyncio.create_subprocess_shell(
                f"sh -lc {shlex.quote(wrapper)}",
                cwd=workdir,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                env=run_env,
            )
        job = Job(
            job_id=job_id,
            instance_id="local",
            name=name,
            repo_path=str(workdir),
            command=command,
            status="running",
            remote_pid=process.pid,
            started_at=datetime.now(UTC),
        )
        self.repo.save_job(job)
        return {"job_id": job_id, "status": "running", "logs": str(jobdir)}

    async def env_prepare(self, name: str, requirements_path: str | None = None) -> dict:
        self._require_enabled()
        if not name.replace("-", "").replace("_", "").isalnum():
            raise GPUError("INVALID_ENV_NAME", "Environment name contains unsafe characters")
        env_dir = self.settings.gpu_lab_local_env_root / name
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        command = f"python3 -m venv {shlex.quote(str(env_dir))}"
        if requirements_path:
            requirements = self._path(requirements_path)
            command += f" && {shlex.quote(str(env_dir / 'bin' / 'pip'))} install -r {shlex.quote(str(requirements / 'requirements.txt'))}"
        proc = await asyncio.create_subprocess_shell(command)
        code = await proc.wait()
        if code:
            raise GPUError("LOCAL_ENV_PREPARE_FAILED", f"Environment setup exited with {code}")
        return {"name": name, "path": str(env_dir), "python": str(env_dir / 'bin' / 'python')}

    def logs(self, job_id: str, tail: int = 200) -> dict:
        status = self.job_status(job_id)
        return {**status, "logs_tail": status["logs_tail"].splitlines()[-max(1, min(tail, 2000)):]}

    def artifacts(self, job_id: str) -> list[dict]:
        root = self.workspace / ".gpu-lab" / "jobs" / job_id
        if not root.is_dir():
            raise GPUError("JOB_NOT_FOUND", f"No local job named {job_id}")
        return [{"path": str(item.relative_to(root)), "size": item.stat().st_size} for item in root.rglob("*") if item.is_file()]

    def artifact_read(self, job_id: str, path: str, max_bytes: int = 65536) -> dict:
        root = self.workspace / ".gpu-lab" / "jobs" / job_id
        file = (root / path).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise GPUError("ARTIFACT_NOT_FOUND", path)
        limit = min(max(1, max_bytes), self.settings.gpu_lab_max_text_artifact_bytes)
        content = file.read_bytes()[:limit]
        return {"job_id": job_id, "path": path, "truncated": file.stat().st_size > limit, "content": content.decode(errors="replace")}

    def job_status(self, job_id: str) -> dict:
        job = self.repo.get_job(job_id)
        if not job or job.instance_id != "local":
            raise GPUError("JOB_NOT_FOUND", f"No local job named {job_id}")
        jobdir = self.workspace / ".gpu-lab" / "jobs" / job_id
        code_file = jobdir / "exit_code"
        if code_file.is_file():
            try:
                job.exit_code = int(code_file.read_text().strip())
                job.status = "completed" if job.exit_code == 0 else "failed"
                job.completed_at = job.completed_at or datetime.now(UTC)
            except ValueError:
                pass
        else:
            if not job.remote_pid or job.remote_pid <= 0:
                job.status = "unknown"
            else:
                try:
                    os.kill(job.remote_pid, 0)
                except OSError:
                    job.status = "unknown"
        self.repo.save_job(job)
        logs = ""
        for file in (jobdir / "stdout.log", jobdir / "stderr.log"):
            if file.exists():
                logs += file.read_text(errors="replace")[-65536:]
        return {"job_id": job_id, "status": job.status, "exit_code": job.exit_code, "logs_tail": logs}

    def cancel(self, job_id: str) -> dict:
        job = self.repo.get_job(job_id)
        if not job or job.instance_id != "local":
            raise GPUError("JOB_NOT_FOUND", f"No local job named {job_id}")
        if not job.remote_pid or job.remote_pid <= 0:
            raise GPUError("JOB_PID_UNKNOWN", f"No valid process ID for {job_id}")
        try:
            os.killpg(job.remote_pid, signal.SIGTERM)
        except (PermissionError, ProcessLookupError):
            pass
        job.status, job.completed_at = "cancelled", datetime.now(UTC)
        self.repo.save_job(job)
        return {"job_id": job_id, "status": "cancelled"}
