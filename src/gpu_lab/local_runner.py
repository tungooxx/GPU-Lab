import asyncio
import hashlib
import os
import re
import shlex
import signal
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .db import Repository
from .errors import GPUError
from .models import Job

_JOB_ENV_ALLOWLIST = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "TERM",
    "TMPDIR",
    "TZ",
)


class LocalRunner:
    """Run detached Linux experiments only within the mounted local workspace."""

    def __init__(self, settings: Settings, repo: Repository):
        self.settings, self.repo = settings, repo
        self.workspace = settings.gpu_lab_local_workspace.resolve()
        # Docker's HOSTNAME is stable across a Python worker restart but changes
        # when the container/process namespace is replaced.
        self.runner_instance_id = (
            os.environ.get("GPU_LAB_RUNNER_INSTANCE_ID")
            or os.environ.get("HOSTNAME")
            or os.environ.get("COMPUTERNAME")
            or "local-host"
        )
        self.submission_owner_id = uuid.uuid4().hex
        self.submission_owner_pid = os.getpid()
        self.submission_owner_identity = self._process_identity(self.submission_owner_pid)
        self._submission_lock = asyncio.Lock()

    @staticmethod
    def _process_identity(pid: int) -> str | None:
        """Return Linux process start ticks so PID reuse cannot impersonate a job."""
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().split()
            return fields[21]
        except (OSError, IndexError):
            return None

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

    def _environment_path(self, name: str) -> Path:
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            raise GPUError("INVALID_ENV_NAME", "Environment name contains unsafe characters")
        root = self.settings.gpu_lab_local_env_root.resolve()
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root):
            raise GPUError("INVALID_ENV_NAME", "Environment must be inside the environment root")
        return candidate

    def _environment_from_python(self, value: str) -> Path:
        """Resolve either a persistent environment name or its absolute Python path."""
        if not Path(value).is_absolute():
            return self._environment_path(value)
        executable = Path(value).resolve()
        root = self.settings.gpu_lab_local_env_root.resolve()
        if not executable.is_relative_to(root):
            raise GPUError(
                "INVALID_PYTHON_ENV_PATH",
                "Absolute environment paths must be inside the environment root",
            )
        # local_status exposes the canonical environment directory, while some
        # reviewed manifests persist its bin/python path.  Both identify the
        # same persistent environment and must be executable inputs.
        if executable.is_dir():
            return executable
        if executable.name in {"python", "python3"} and executable.parent.name == "bin":
            return executable.parent.parent
        raise GPUError(
            "INVALID_PYTHON_ENV_PATH",
            "Absolute environment paths must be an environment directory or its bin/python executable",
        )

    def _requirements_path(self, value: str) -> Path:
        candidate = (self.workspace / value).resolve()
        if not candidate.is_relative_to(self.workspace):
            raise GPUError("INVALID_LOCAL_PATH", "Requirements must be inside the local workspace")
        if candidate.is_dir():
            candidate = candidate / "requirements.txt"
        if not candidate.is_file():
            raise GPUError("REQUIREMENTS_NOT_FOUND", f"Requirements file is unavailable: {candidate}")
        return candidate

    @staticmethod
    def _job_environment(env: dict[str, str] | None) -> dict[str, str]:
        """Build a non-secret process environment for user-submitted commands."""
        run_env = {
            name: value
            for name in _JOB_ENV_ALLOWLIST
            if isinstance((value := os.environ.get(name)), str)
        }
        run_env.update(
            {
                key: value
                for key, value in (env or {}).items()
                if isinstance(key, str)
                and isinstance(value, str)
                and key.replace("_", "a").isalnum()
            }
        )
        return run_env

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
            "canonical_vrc_environment": {
                "name": self.settings.gpu_lab_canonical_vrc_env,
                "path": str(
                    self._environment_path(self.settings.gpu_lab_canonical_vrc_env)
                ),
                "expected_runtime": "Python 3.13 + PyTorch 2.6.0+cu124",
                "verified_prediction_maxabs": 0,
            },
        }

    @staticmethod
    def _parse_gpu_metrics(output: str) -> list[dict]:
        """Parse the small, unit-free ``nvidia-smi`` telemetry response."""
        metrics = []
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6:
                continue
            try:
                index, memory_total, memory_used, utilization, temperature = map(
                    int, (fields[0], fields[2], fields[3], fields[4], fields[5])
                )
            except ValueError:
                continue
            metrics.append(
                {
                    "index": index,
                    "name": fields[1],
                    "memory_total_mb": memory_total,
                    "memory_used_mb": memory_used,
                    "utilization_percent": utilization,
                    "temperature_c": temperature,
                }
            )
        return metrics

    async def gpu_metrics(self) -> list[dict]:
        """Return current local GPU load for the read-only web monitor."""
        self._require_enabled()
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise GPUError("NVIDIA_SMI_UNAVAILABLE", "nvidia-smi is unavailable in the local runtime") from exc
        output, error = await proc.communicate()
        if proc.returncode:
            raise GPUError("NVIDIA_SMI_FAILED", error.decode(errors="replace").strip())
        return self._parse_gpu_metrics(output.decode(errors="replace"))

    async def submit(
        self,
        command: str,
        working_directory: str = ".",
        name: str | None = None,
        env: dict[str, str] | None = None,
        python_env: str | None = None,
        job_id: str | None = None,
    ) -> dict:
        async with self._submission_lock:
            return await self._submit_locked(
                command, working_directory, name, env, python_env, job_id
            )

    async def _submit_locked(
        self,
        command: str,
        working_directory: str,
        name: str | None,
        env: dict[str, str] | None,
        python_env: str | None,
        job_id: str | None,
    ) -> dict:
        self._require_enabled()
        workdir = self._path(working_directory)
        job_id = job_id or "local_" + uuid.uuid4().hex[:16]
        if not re.fullmatch(r"local_[a-zA-Z0-9_-]{8,80}", job_id):
            raise GPUError("INVALID_LOCAL_JOB_ID", "Local job ID is invalid")
        fingerprint = hashlib.sha256(
            repr((command, str(workdir), sorted((env or {}).items()), python_env)).encode()
        ).hexdigest()
        run_env = self._job_environment(env)
        if python_env:
            venv = self._environment_from_python(python_env)
            if not venv.is_dir():
                raise GPUError("LOCAL_ENV_NOT_FOUND", f"No persistent environment named {python_env}")
            run_env["PATH"] = f"{venv / 'bin'}:{run_env.get('PATH', '')}"
            run_env["VIRTUAL_ENV"] = str(venv)
        jobdir = self.workspace / ".gpu-lab" / "jobs" / job_id
        jobdir.mkdir(parents=True, exist_ok=True)
        stdout, stderr, exit_code = jobdir / "stdout.log", jobdir / "stderr.log", jobdir / "exit_code"
        script = jobdir / "command.sh"
        if script.exists() and script.read_text() != command:
            raise GPUError("LOCAL_JOB_ID_REUSED", "Job script differs from its reserved command")
        script.write_text(command)
        script.chmod(0o700)
        run_env["GPU_LAB_JOB_DIR"] = str(jobdir)
        pid_file = jobdir / "process.pid"
        wrapper = (
            f"printf '%s' $$ > {shlex.quote(str(pid_file))}; "
            f"bash {shlex.quote(str(script))}; code=$?; "
            f"printf '%s' $code > {shlex.quote(str(exit_code))}; exit $code"
        )
        job = Job(
            job_id=job_id,
            instance_id="local",
            name=name,
            repo_path=str(workdir),
            command=command,
            status="queued",
            started_at=datetime.now(UTC),
            metadata={
                "request_fingerprint": fingerprint,
                "python_env": python_env,
                "runner_instance_id": self.runner_instance_id,
                "submission_owner_id": self.submission_owner_id,
                "submission_owner_pid": self.submission_owner_pid,
                "submission_owner_identity": self.submission_owner_identity,
                "submission_claimed_at": datetime.now(UTC).isoformat(),
            },
        )
        claim, existing = self.repo.claim_job(job)
        if claim == "conflict":
            raise GPUError("LOCAL_JOB_ID_REUSED", "Job ID is bound to a different command")
        if claim == "existing":
            # A queued row is an active inter-process submission claim. Do not
            # probe it here: job_status would turn the pre-spawn row into unknown
            # and allow a second process to launch the same command.
            current = (
                {"status": "queued"}
                if existing.status == "queued"
                else self.job_status(job_id)
            )
            return {
                "job_id": job_id,
                "status": current["status"],
                "logs": str(jobdir),
                "idempotent_replay": True,
            }
        try:
            with stdout.open("wb") as stdout_handle, stderr.open("wb") as stderr_handle:
                process = await asyncio.create_subprocess_shell(
                    # A login shell may replace PATH from its profile.  That defeats
                    # the selected persistent Python environment, which is supplied
                    # through run_env above.  The wrapper has no need for login-shell
                    # initialization, so preserve the explicit execution environment.
                    f"sh -c {shlex.quote(wrapper)}",
                    cwd=workdir,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                    env=run_env,
                )
        except Exception:
            job.status = "unknown"
            self.repo.save_job(job)
            raise
        job.status = "running"
        job.remote_pid = process.pid
        job.metadata["process_identity"] = self._process_identity(process.pid)
        self.repo.save_job(job)
        return {"job_id": job_id, "status": "running", "logs": str(jobdir)}

    async def env_prepare(
        self,
        name: str,
        requirements_path: str | None = None,
        python_executable: str = "python3",
    ) -> dict:
        self._require_enabled()
        env_dir = self._environment_path(name)
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        if not re.fullmatch(r"[a-zA-Z0-9_./+-]+", python_executable):
            raise GPUError("INVALID_PYTHON_EXECUTABLE", "Python executable contains unsafe characters")
        proc = await asyncio.create_subprocess_exec(
            python_executable, "-m", "venv", str(env_dir)
        )
        code = await proc.wait()
        if code:
            raise GPUError("LOCAL_ENV_PREPARE_FAILED", f"Environment setup exited with {code}")
        if requirements_path:
            requirements = self._requirements_path(requirements_path)
            install = await asyncio.create_subprocess_exec(
                str(env_dir / "bin" / "pip"), "install", "-r", str(requirements)
            )
            install_code = await install.wait()
            if install_code:
                raise GPUError(
                    "LOCAL_ENV_REQUIREMENTS_FAILED",
                    f"Requirements installation exited with {install_code}",
                )
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

    def job_status(self, job_id: str, include_logs: bool = True) -> dict:
        job = self.repo.get_job(job_id)
        if not job or job.instance_id != "local":
            raise GPUError("JOB_NOT_FOUND", f"No local job named {job_id}")
        jobdir = self.workspace / ".gpu-lab" / "jobs" / job_id
        code_file = jobdir / "exit_code"
        pid_file = jobdir / "process.pid"
        if not job.remote_pid and pid_file.is_file():
            try:
                job.remote_pid = int(pid_file.read_text().strip())
            except ValueError:
                pass
        # A successful cancellation is terminal even when SIGTERM/SIGKILL
        # prevented the child from writing its normal exit-code file.
        if job.status == "cancelled":
            result = {"job_id": job_id, "status": "cancelled", "exit_code": job.exit_code}
            if include_logs:
                logs = ""
                for file in (jobdir / "stdout.log", jobdir / "stderr.log"):
                    if file.exists():
                        logs += file.read_text(errors="replace")[-65536:]
                result["logs_tail"] = logs
            return result
        if code_file.is_file():
            try:
                job.exit_code = int(code_file.read_text().strip())
                job.status = "completed" if job.exit_code == 0 else "failed"
                job.completed_at = job.completed_at or datetime.now(UTC)
            except ValueError:
                pass
        elif job.status == "queued":
            # ``queued`` is the durable pre-spawn claim. A status reader must
            # not invalidate a live claimant and make the ID reclaimable while
            # its subprocess creation is still in flight.
            # A client/gateway may disconnect after the child is spawned but
            # before this coroutine can persist ``running``.  The detached
            # wrapper writes process.pid before executing user code, so recover
            # that durable launch proof instead of turning a real job unknown.
            process_identity = (
                self._process_identity(job.remote_pid)
                if job.remote_pid and job.remote_pid > 0
                else None
            )
            if process_identity:
                job.status = "running"
                job.metadata.update(
                    {
                        "process_identity": process_identity,
                        "runner_instance_id": self.runner_instance_id,
                        "recovered_from_durable_pid": True,
                    }
                )
            else:
                owner_pid = job.metadata.get("submission_owner_pid")
                owner_identity = job.metadata.get("submission_owner_identity")
                same_owner = job.metadata.get("submission_owner_id") == self.submission_owner_id
                owner_alive = (
                    job.metadata.get("runner_instance_id") == self.runner_instance_id
                    and isinstance(owner_pid, int)
                    and bool(owner_identity)
                    and self._process_identity(owner_pid) == owner_identity
                )
                try:
                    claimed_at = datetime.fromisoformat(job.metadata["submission_claimed_at"])
                    fresh_claim = (datetime.now(UTC) - claimed_at).total_seconds() < 60
                except (KeyError, TypeError, ValueError):
                    fresh_claim = False
                if not (same_owner or owner_alive or fresh_claim):
                    job.status = "unknown"
        else:
            launched_by_current_runner = (
                job.metadata.get("runner_instance_id") == self.runner_instance_id
            )
            current_identity = (
                self._process_identity(job.remote_pid)
                if job.remote_pid and job.remote_pid > 0
                else None
            )
            if (
                not launched_by_current_runner
                or not current_identity
                or current_identity != job.metadata.get("process_identity")
            ):
                job.status = "unknown"
            else:
                try:
                    os.kill(job.remote_pid, 0)
                except OSError:
                    job.status = "unknown"
        self.repo.save_job(job)
        result = {"job_id": job_id, "status": job.status, "exit_code": job.exit_code}
        if include_logs:
            logs = ""
            for file in (jobdir / "stdout.log", jobdir / "stderr.log"):
                if file.exists():
                    logs += file.read_text(errors="replace")[-65536:]
            result["logs_tail"] = logs
        return result

    def reconcile_jobs(self) -> dict[str, int]:
        """Refresh persisted non-final jobs after a gateway start or restart."""
        summary = {"reconciled": 0, "completed": 0, "failed": 0, "unknown": 0}
        if not self.settings.gpu_lab_enable_local_runner or not self.workspace.is_dir():
            return summary
        for job in self.repo.list_jobs(instance_id="local", limit=10000):
            if job.status not in {"queued", "running", "unknown"}:
                continue
            outcome = self.job_status(job.job_id)
            summary["reconciled"] += 1
            if outcome["status"] in summary:
                summary[outcome["status"]] += 1
        return summary

    def cancel(self, job_id: str) -> dict:
        job = self.repo.get_job(job_id)
        if not job or job.instance_id != "local":
            raise GPUError("JOB_NOT_FOUND", f"No local job named {job_id}")
        if not job.remote_pid or job.remote_pid <= 0:
            raise GPUError("JOB_PID_UNKNOWN", f"No valid process ID for {job_id}")
        def group_alive() -> bool:
            try:
                os.kill(-job.remote_pid, 0)
                return True
            except (PermissionError, ProcessLookupError):
                return False
        try:
            os.killpg(job.remote_pid, signal.SIGTERM)
        except (PermissionError, ProcessLookupError):
            pass
        for _ in range(5):
            if not group_alive():
                break
            __import__("time").sleep(1)
        if group_alive():
            try:
                os.killpg(job.remote_pid, signal.SIGKILL)
            except (PermissionError, ProcessLookupError):
                pass
            __import__("time").sleep(1)
        if group_alive():
            job.metadata.update({"cancellation_requested_at": datetime.now(UTC).isoformat(), "cancellation_incomplete": True})
            self.repo.save_job(job)
            return {"job_id": job_id, "status": "cancellation_incomplete", "terminated": False, "process_group_alive": True, "recovery_action": "CANCEL_PROCESS_GROUP"}
        job.status, job.completed_at = "cancelled", datetime.now(UTC)
        job.metadata.pop("cancellation_requested_at", None)
        job.metadata.pop("cancellation_incomplete", None)
        self.repo.save_job(job)
        return {"job_id": job_id, "status": "cancelled", "terminated": True, "process_group_alive": False}
