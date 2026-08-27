import mimetypes
import re
import uuid
from datetime import UTC, datetime
from pathlib import PurePosixPath

from .config import Settings
from .db import Repository
from .errors import GPUError
from .models import Instance, Job
from .providers import VastProvider
from .ssh import SSHClient, q


class GPUService:
    def __init__(self, settings: Settings):
        self.settings, self.repo, self.ssh = (
            settings,
            Repository(settings.db_path),
            SSHClient(settings),
        )
        self.provider = VastProvider(settings.vast_api_key) if settings.vast_api_key else None

    def _provider(self) -> VastProvider:
        if not self.provider:
            raise GPUError(
                "PROVIDER_NOT_CONFIGURED", "VAST_API_KEY is required for Vast operations"
            )
        return self.provider

    def _instance(self, ident: str) -> Instance:
        item = self.repo.get_instance(ident)
        if not item:
            raise GPUError("INSTANCE_NOT_FOUND", f"No locally known instance: {ident}")
        return item

    def _remote_path(self, path: str) -> str:
        root, candidate = PurePosixPath(self.settings.gpu_lab_remote_root), PurePosixPath(path)
        if not candidate.is_relative_to(root):
            raise GPUError("INVALID_PATH", "Path must be inside the configured remote workspace")
        return str(candidate)

    async def gpu_list(self) -> list[dict]:
        if self.provider:
            visible = await self.provider.list_instances()
            visible_ids = {item.id for item in visible}
            for item in visible:
                self.repo.save_instance(item)
            # Preserve historical instance/job references, but never present an
            # instance missing from Vast as live.  A later provider refresh can
            # revive the record if it reappears (for example after scheduling).
            for item in self.repo.list_instances():
                if item.provider == "vast" and item.id not in visible_ids:
                    if item.status == "destroyed":
                        continue
                    item.status = "provider_missing"
                    item.metadata = {
                        **item.metadata,
                        "provider_visible": False,
                        "provider_missing_at": datetime.now(UTC).isoformat(),
                    }
                    self.repo.save_instance(item)
        return [x.model_dump(mode="json", exclude={"metadata"}) for x in self.repo.list_instances()]

    async def gpu_status(self, instance_id: str) -> dict:
        item = self._instance(instance_id)
        if self.provider:
            try:
                item = await self.provider.get_instance(instance_id)
                self.repo.save_instance(item)
            except GPUError:
                pass
        result = {
            "instance_id": item.id,
            "provider": item.provider,
            "provider_status": item.status,
            "ssh_reachable": False,
        }
        try:
            out, _, _ = await self.ssh.run(
                item,
                "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version --format=csv,noheader,nounits; hostname; df -BG / | tail -1; free -g | awk '/Mem:/ {print $7}'; nvcc --version 2>/dev/null | tail -1",
                30,
            )
            lines = out.strip().splitlines()
            gpu = [x.strip() for x in lines[0].split(",")]
            result.update(
                {
                    "ssh_reachable": True,
                    "gpu": {
                        "name": gpu[0],
                        "count": 1,
                        "memory_total_mb": int(gpu[1]),
                        "memory_used_mb": int(gpu[2]),
                        "utilization_percent": int(gpu[3]),
                        "temperature_c": int(gpu[4]),
                    },
                    "system": {
                        "hostname": lines[1],
                        "disk_free_gb": int(re.search(r"(\d+)G", lines[2]).group(1)),
                        "ram_free_gb": int(lines[3]),
                    },
                    "driver": gpu[5],
                    "cuda": lines[4] if len(lines) > 4 else None,
                }
            )
        except GPUError as exc:
            result["ssh_error"] = exc.message
        return result

    async def gpu_search(
        self, gpu_name: str | None, min_vram_gb: int | None, max_hourly_price: float | None
    ) -> list[dict]:
        offers = await self._provider().search_offers(
            gpu_name=gpu_name, min_vram_gb=min_vram_gb, max_hourly_price=max_hourly_price
        )
        return [
            {
                "offer_id": str(o.get("id")),
                "gpu": o.get("gpu_name"),
                "gpu_count": o.get("num_gpus"),
                "vram_mb": o.get("gpu_ram"),
                "hourly_price": o.get("dph_total", o.get("dph_base")),
                "verified": o.get("verified", False),
                "reliability": o.get("reliability"),
                "raw": o,
            }
            for o in offers[:25]
        ]

    async def gpu_create(
        self, offer_id: str, disk_gb: int = 100, image: str | None = None, label: str | None = None
    ) -> dict:
        item = await self._provider().create_instance(
            offer_id, disk_gb=disk_gb, image=image, label=label
        )
        self.repo.save_instance(item)
        return {
            "instance": item.model_dump(mode="json", exclude={"metadata"}),
            "chosen_offer_id": offer_id,
            "reason": "explicit offer selected by caller",
        }

    async def gpu_start(self, instance_id: str) -> dict:
        item = await self._provider().start_instance(instance_id)
        self.repo.save_instance(item)
        return {
            "instance": item.model_dump(mode="json", exclude={"metadata"}),
            "requested_state": "running",
            "note": "Vast may report scheduling until host resources are available.",
        }

    async def gpu_stop(self, instance_id: str) -> dict:
        await self._provider().stop_instance(instance_id)
        item = self._instance(instance_id)
        item.status = "stopped"
        self.repo.save_instance(item)
        return {"instance_id": instance_id, "status": "stopped"}

    async def gpu_destroy(self, instance_id: str, confirmation: str) -> dict:
        if confirmation != "DESTROY":
            raise GPUError(
                "CONFIRMATION_REQUIRED", "Pass confirmation='DESTROY' to destroy an instance"
            )
        item = self._instance(instance_id)
        already_deleted = False
        try:
            await self._provider().destroy_instance(instance_id)
        except GPUError as exc:
            if exc.error_type != "PROVIDER_NOT_FOUND":
                raise
            already_deleted = True
        item.status = "destroyed"
        item.metadata = {
            **item.metadata,
            "provider_visible": False,
            "provider_deleted_at": datetime.now(UTC).isoformat(),
            "provider_deletion_confirmed": not already_deleted,
            "provider_already_deleted": already_deleted,
        }
        self.repo.save_instance(item)
        return {
            "instance_id": instance_id,
            "status": "ALREADY_DELETED" if already_deleted else "destroyed",
            "historical_record_retained": True,
        }

    async def repo_checkout(
        self,
        instance_id: str,
        repo_url: str,
        commit: str | None = None,
        branch: str | None = None,
        name: str | None = None,
    ) -> dict:
        item = self._instance(instance_id)
        name = name or repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise GPUError("INVALID_REPOSITORY_NAME", "Repository name contains unsafe characters")
        path = self._remote_path(f"{self.settings.gpu_lab_remote_root}/repos/{name}")
        fetch_ref = q(commit or branch or "HEAD")
        command = (
            f'mkdir -p {q(self.settings.gpu_lab_remote_root + "/repos")} && '
            f'if [ -d {q(path + "/.git")} ]; then '
            f'cd {q(path)} && test -z "$(git status --porcelain)" || exit 42; '
            f'else git clone --depth 1 --no-checkout {q(repo_url)} {q(path)}; fi && '
            f'cd {q(path)} && git fetch --depth 1 origin {fetch_ref} && '
            f'resolved_ref=$(git rev-parse --verify FETCH_HEAD^{{commit}}) && '
            f'git checkout --detach "$resolved_ref" && git rev-parse HEAD'
        )
        out, err, code = await self.ssh.run(item, command, 300)
        if code == 42:
            raise GPUError("DIRTY_WORKTREE", "Refusing to overwrite a dirty worktree")
        if code:
            raise GPUError("REPOSITORY_CHECKOUT_FAILED", err.strip() or out.strip())
        return {
            "repo_path": path,
            "commit": out.strip().splitlines()[-1],
            "branch": branch,
            "dirty": False,
            "repo_url": repo_url,
        }

    async def env_prepare(self, instance_id: str, repo_path: str, strategy: str = "auto") -> dict:
        item = self._instance(instance_id)
        path = self._remote_path(repo_path)
        cmd = f"cd {q(path)}; if [ -f uv.lock ]; then uv sync; elif [ -f pyproject.toml ]; then uv sync; elif [ -f requirements.txt ]; then python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; elif [ -f environment.yml ]; then conda env update -f environment.yml; else exit 44; fi; python3 --version; (uv pip freeze 2>/dev/null || .venv/bin/pip freeze 2>/dev/null || pip freeze) | sha256sum; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1; python3 -c 'import torch; print(torch.__version__,torch.cuda.is_available())' 2>/dev/null || true"
        out, err, code = await self.ssh.run(item, cmd, 1800)
        if code == 44:
            raise GPUError("ENVIRONMENT_NOT_FOUND", "No supported environment manifest found")
        if code:
            raise GPUError("ENVIRONMENT_PREPARE_FAILED", err.strip() or out.strip())
        lines = out.strip().splitlines()
        return {
            "repo_path": path,
            "strategy": strategy,
            "python": lines[-4] if len(lines) >= 4 else None,
            "package_lock_fingerprint": lines[-3] if len(lines) >= 3 else None,
            "driver": lines[-2] if len(lines) >= 2 else None,
            "pytorch": lines[-1] if lines else None,
        }

    async def experiment_submit(
        self,
        instance_id: str,
        repo_path: str,
        command: str,
        name: str | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
        artifact_patterns: list[str] | None = None,
        metadata: dict | None = None,
        job_id: str | None = None,
    ) -> dict:
        item = self._instance(instance_id)
        repo_path = self._remote_path(repo_path)
        # The canonical Research OS executor reserves this identity before a
        # process is launched.  Keeping it here makes remote submission
        # idempotent at the same boundary as local submission.
        job_id = job_id or "exp_" + uuid.uuid4().hex[:16]
        if not re.fullmatch(r"(?:exp|remote)_[A-Za-z0-9_-]{8,80}", job_id):
            raise GPUError("INVALID_REMOTE_JOB_ID", "Remote job ID is invalid")
        existing = self.repo.get_job(job_id)
        if existing:
            if (
                existing.instance_id != instance_id
                or existing.repo_path != repo_path
                or existing.command != command
            ):
                raise GPUError("REMOTE_JOB_ID_REUSED", "Job ID is bound to a different remote command")
            return {
                "job_id": job_id,
                "status": existing.status,
                "experiment": existing.name,
                "instance": {"id": instance_id, "gpu": item.gpu_model},
                "idempotent_replay": True,
            }
        jobdir = f"{self.settings.gpu_lab_remote_root}/jobs/{job_id}"
        safe_env = " ".join(
            f"{k}={q(v)}"
            for k, v in (env or {}).items()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
        )
        job = Job(
            job_id=job_id,
            instance_id=instance_id,
            name=name,
            repo_path=repo_path,
            command=command,
            remote_session=job_id,
            started_at=datetime.now(UTC),
            metadata={**(metadata or {}), "artifact_patterns": artifact_patterns or []},
        )
        # Quote the pane command once as the final tmux argument.  Do not wrap
        # it in literal single quotes: ``q(inner)`` may itself contain quotes.
        inner = (
            f"echo $$ > {q(jobdir + '/process_group.pid')}; "
            + (safe_env + " " if safe_env else "")
            + command
            + f"; code=$?; echo $code > {q(jobdir + '/exit_code')}; exit $code"
        )
        pane_command = (
            f"setsid sh -c {q(inner)} "
            f"> {q(jobdir + '/stdout.log')} "
            f"2> {q(jobdir + '/stderr.log')}"
        )
        bootstrap = (
            f"mkdir -p {q(jobdir)}/{{metrics,plots,checkpoints,artifacts}}; "
            f"printf '%s\\n' {q(command)} > {q(jobdir + '/command.sh')}; "
            f"printf '%s' {q(job.model_dump_json())} > {q(jobdir + '/metadata.json')}; "
            f"cd {q(repo_path)} && tmux new-session -d -s {q(job_id)} {q(pane_command)} "
            f"&& tmux display-message -p -t {q(job_id)} '#{{pane_pid}}'"
        )
        out, err, code = await self.ssh.run(item, bootstrap, 60)
        if code:
            raise GPUError("JOB_SUBMIT_FAILED", err.strip() or out.strip())
        job.remote_pid = int(out.strip().splitlines()[-1])
        job.status = "running"
        self.repo.save_job(job)
        self.repo.event(job_id, "submitted", "Detached tmux job created")
        return {
            "job_id": job_id,
            "status": "running",
            "experiment": name,
            "instance": {"id": instance_id, "gpu": item.gpu_model},
            "git": {},
            "runtime": {"elapsed_seconds": 0},
        }

    async def experiment_status(self, job_id: str) -> dict:
        job = self.repo.get_job(job_id)
        if not job:
            raise GPUError("JOB_NOT_FOUND", f"No job named {job_id}")
        item = self._instance(job.instance_id)
        jd = f"{self.settings.gpu_lab_remote_root}/jobs/{job_id}"
        out, _, _ = await self.ssh.run(
            item,
            f"pgid=''; [ -f {q(jd + '/process_group.pid')} ] && pgid=$(cat {q(jd + '/process_group.pid')} 2>/dev/null || true); case \"$pgid\" in ''|*[!0-9]*) group_alive=false;; *) kill -0 -- -$pgid 2>/dev/null && group_alive=true || group_alive=false;; esac; if [ \"$group_alive\" = true ]; then echo process_running; elif tmux has-session -t {q(job_id)} 2>/dev/null; then echo running; elif [ -f {q(jd + '/exit_code')} ]; then echo exit:$(cat {q(jd + '/exit_code')}); else echo unknown; fi; tail -n 50 {q(jd + '/stdout.log')} {q(jd + '/stderr.log')} 2>/dev/null",
            30,
        )
        first, *logs = out.splitlines()
        if first in {"running", "process_running"}:
            job.status = "running"
        elif first.startswith("exit:"):
            job.exit_code = int(first.split(":", 1)[1])
            job.status = "completed" if job.exit_code == 0 else "failed"
            job.completed_at = datetime.now(UTC)
        else:
            job.status = "unknown"
        cancellation_incomplete = bool(
            first == "process_running" and job.metadata.get("cancellation_requested_at")
        )
        self.repo.save_job(job)
        return {
            "job_id": job_id,
            "status": "cancellation_incomplete" if cancellation_incomplete else job.status,
            "cancellation_incomplete": cancellation_incomplete,
            "process_group_alive": first == "process_running",
            "pid": job.remote_pid,
            "exit_code": job.exit_code,
            "start_time": job.started_at,
            "end_time": job.completed_at,
            "logs_tail": logs[-50:],
            "artifact_count": len(await self.artifact_list(job_id)),
        }

    async def experiment_logs(self, job_id: str, tail: int = 200, stream: str = "combined") -> dict:
        job = self.repo.get_job(job_id)
        if not job:
            raise GPUError("JOB_NOT_FOUND", f"No job named {job_id}")
        if stream not in {"stdout", "stderr", "combined"}:
            raise GPUError("INVALID_STREAM", "stream must be stdout, stderr, or combined")
        tail = max(1, min(tail, self.settings.gpu_lab_max_log_lines))
        jd = f"{self.settings.gpu_lab_remote_root}/jobs/{job_id}"
        files = (
            [jd + f"/{stream}.log"]
            if stream != "combined"
            else [jd + "/stdout.log", jd + "/stderr.log"]
        )
        out, _, _ = await self.ssh.run(
            self._instance(job.instance_id), f"tail -n {tail} " + " ".join(q(x) for x in files), 30
        )
        return {
            "job_id": job_id,
            "stream": stream,
            "truncated_to_lines": tail,
            "logs": out[-262144:],
        }

    async def experiment_cancel(self, job_id: str) -> dict:
        job = self.repo.get_job(job_id)
        if not job:
            raise GPUError("JOB_NOT_FOUND", f"No job named {job_id}")
        item = self._instance(job.instance_id)
        jobdir = f"{self.settings.gpu_lab_remote_root}/jobs/{job_id}"
        # The job command is launched by ``setsid`` and writes its session / process
        # group leader.  Killing tmux alone only removes the wrapper pane and can
        # orphan CUDA children.  TERM first, then KILL only if the group survives.
        cancel_command = (
            f"pgfile={q(jobdir + '/process_group.pid')}; "
            "pgid=''; [ -f \"$pgfile\" ] && pgid=$(cat \"$pgfile\" 2>/dev/null || true); "
            f"case \"$pgid\" in ''|*[!0-9]*) pgid={int(job.remote_pid or 0)};; esac; "
            "if [ \"$pgid\" -gt 0 ] 2>/dev/null; then "
            "kill -TERM -- -$pgid 2>/dev/null || true; "
            "for n in 1 2 3 4 5; do kill -0 -- -$pgid 2>/dev/null || break; sleep 1; done; "
            "if kill -0 -- -$pgid 2>/dev/null; then kill -KILL -- -$pgid 2>/dev/null || true; sleep 1; fi; "
            "fi; "
            f"tmux kill-session -t {q(job_id)} 2>/dev/null || true; "
            "if [ \"$pgid\" -gt 0 ] 2>/dev/null && kill -0 -- -$pgid 2>/dev/null; then echo process_group_alive; "
            f"elif tmux has-session -t {q(job_id)} 2>/dev/null; then echo tmux_alive; else echo terminated; fi"
        )
        out, _, _ = await self.ssh.run(
            item,
            cancel_command,
            30,
        )
        if "terminated" not in out:
            job.metadata["cancellation_requested_at"] = datetime.now(UTC).isoformat()
            job.metadata["cancellation_incomplete"] = True
            self.repo.save_job(job)
            return {"job_id": job_id, "status": "cancellation_incomplete", "terminated": False, "process_group_alive": "process_group_alive" in out, "retry_safe": False, "recovery_action": "CANCEL_PROCESS_GROUP"}
        job.status = "cancelled"
        job.completed_at = datetime.now(UTC)
        job.metadata.pop("cancellation_requested_at", None)
        job.metadata.pop("cancellation_incomplete", None)
        self.repo.save_job(job)
        self.repo.event(job_id, "cancelled", "tmux session and process group terminated")
        return {"job_id": job_id, "status": "cancelled", "terminated": True, "process_group_alive": False}

    async def artifact_list(self, job_id: str) -> list[dict]:
        job = self.repo.get_job(job_id)
        if not job:
            raise GPUError("JOB_NOT_FOUND", f"No job named {job_id}")
        jd = f"{self.settings.gpu_lab_remote_root}/jobs/{job_id}"
        out, _, _ = await self.ssh.run(
            self._instance(job.instance_id),
            f"cd {q(jd)} && find metrics plots checkpoints artifacts -type f -printf '%p|%s|%T@\\n' 2>/dev/null",
            60,
        )
        result = []
        for line in out.splitlines():
            path, size, modified = line.split("|", 2)
            result.append(
                {
                    "path": path,
                    "size": int(size),
                    "modified_at": modified,
                    "mime_type": mimetypes.guess_type(path)[0] or "application/octet-stream",
                }
            )
        return result

    async def artifact_read(self, job_id: str, path: str, max_bytes: int | None = None) -> dict:
        job = self.repo.get_job(job_id)
        if not job:
            raise GPUError("JOB_NOT_FOUND", f"No job named {job_id}")
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise GPUError("INVALID_ARTIFACT_PATH", "Artifact path escapes job directory")
        limit = min(
            max_bytes or self.settings.gpu_lab_max_text_artifact_bytes,
            self.settings.gpu_lab_max_text_artifact_bytes,
        )
        file = f"{self.settings.gpu_lab_remote_root}/jobs/{job_id}/{relative}"
        out, err, code = await self.ssh.run(
            self._instance(job.instance_id), f"test -f {q(file)} && head -c {limit} {q(file)}", 30
        )
        if code:
            raise GPUError("ARTIFACT_NOT_FOUND", err.strip() or path)
        return {
            "job_id": job_id,
            "path": path,
            "truncated": len(out.encode()) >= limit,
            "content": out,
        }

    async def experiment_list(
        self, instance_id: str | None = None, status: str | None = None, limit: int = 50
    ) -> list[dict]:
        return [
            x.model_dump(mode="json")
            for x in self.repo.list_jobs(instance_id, status, min(limit, 100))
        ]

    async def remote_exec(self, instance_id: str, command: str, timeout_seconds: int = 60) -> dict:
        if not self.settings.gpu_lab_enable_remote_exec:
            raise GPUError(
                "REMOTE_EXEC_DISABLED",
                "Set GPU_LAB_ENABLE_REMOTE_EXEC=true to enable this dangerous debugging tool",
            )
        out, err, code = await self.ssh.run(
            self._instance(instance_id), command, min(timeout_seconds, 300)
        )
        self.repo.event(None, "remote_exec", "Remote debug command", {"instance_id": instance_id})
        return {"exit_code": code, "stdout": out[:65536], "stderr": err[:65536]}
