import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import uvicorn
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .errors import GPUError
from .executable_papers import ExecutablePaperCandidate

logger = logging.getLogger(__name__)


class WorkerRequest(BaseModel):
    paper_id: str | None = Field(default=None, max_length=100)
    repository: str | None = Field(default=None, max_length=1000)
    commit: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{40}$")
    tutorials: str | None = Field(default=None, max_length=1000)
    executable_paper_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tool: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,200}$")
    args: dict[str, Any] = Field(default_factory=dict)


class Paper2AgentSubprocessProvider:
    """Runs the official Paper2Agent checkout as a replaceable isolated worker cache."""

    def __init__(self, upstream_root: Path, timeout_seconds: int = 14_400):
        self.upstream_root = upstream_root.resolve()
        self.timeout_seconds = timeout_seconds
        self._coordination_lock = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._build_tasks: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def _repository(value: str) -> str:
        parsed = urlparse(value)
        parts = [part for part in parsed.path.split("/") if part]
        if (
            parsed.scheme != "https"
            or parsed.hostname != "github.com"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
            or len(parts) != 2
            or any(part in {".", ".."} for part in parts)
            or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts)
        ):
            raise GPUError(
                "INVALID_EXECUTABLE_PAPER_REPOSITORY",
                "Paper2Agent accepts only public https://github.com/<owner>/<repo> repositories",
            )
        return f"https://github.com/{parts[0]}/{parts[1].removesuffix('.git')}.git"

    async def _cleanup_build(
        self, build_id: str, lock: asyncio.Lock, task: asyncio.Task[None] | None
    ) -> None:
        async with self._coordination_lock:
            if task is not None and task.done() and self._build_tasks.get(build_id) is task:
                self._build_tasks.pop(build_id, None)
            if (
                self._locks.get(build_id) is lock
                and not lock.locked()
                and build_id not in self._build_tasks
            ):
                self._locks.pop(build_id, None)

    def _build_done(
        self, build_id: str, lock: asyncio.Lock, task: asyncio.Task[None]
    ) -> None:
        try:
            error = task.exception()
        except asyncio.CancelledError:
            error = None
        if error is not None:
            logger.error(
                "Paper2Agent build task failed build_id=%s error_type=%s",
                build_id,
                type(error).__name__,
            )
        asyncio.create_task(self._cleanup_build(build_id, lock, task))

    @staticmethod
    def _build_id(paper_id: str, repository: str, commit: str, tutorials: str | None) -> str:
        return hashlib.sha256(
            json.dumps(
                [paper_id, repository, commit.lower(), tutorials], separators=(",", ":")
            ).encode()
        ).hexdigest()

    async def _run(
        self,
        *command: str,
        cwd: Path | None = None,
        include_model_credentials: bool = False,
    ) -> tuple[int, str, str]:
        allowed_environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "HOME", "LANG", "LC_ALL"}
        }
        if include_model_credentials:
            allowed_environment.update(
                {
                    key: value
                    for key, value in os.environ.items()
                    if key == "CLAUDE_CONFIG_DIR" or key.startswith("ANTHROPIC_")
                }
            )
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd or self.upstream_root),
            env=allowed_environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise GPUError(
                "EXECUTABLE_PAPER_BUILD_TIMEOUT",
                "Paper2Agent exceeded its configured build deadline",
                retryable=True,
            ) from exc
        return (
            process.returncode or 0,
            stdout.decode("utf-8", "replace")[-100_000:],
            stderr.decode("utf-8", "replace")[-100_000:],
        )

    async def _remote_head(self, repository: str) -> str:
        code, stdout, _ = await self._run("git", "ls-remote", repository, "HEAD")
        if code or not stdout.strip():
            raise GPUError(
                "EXECUTABLE_PAPER_REPOSITORY_UNAVAILABLE",
                "Could not resolve the requested public repository",
                retryable=True,
            )
        return stdout.split()[0].lower()

    def _project(self, build_id: str) -> Path:
        path = (self.upstream_root / "projects" / build_id).resolve()
        if not path.is_relative_to(self.upstream_root / "projects"):
            raise GPUError("INVALID_EXECUTABLE_PAPER", build_id)
        return path

    def _runtime(self, build_id: str) -> tuple[Path, Path, Path]:
        project = self._project(build_id)
        servers = list((project / "src").glob("*_mcp.py"))
        pythons = list(project.glob("*-env/bin/python"))
        if len(servers) != 1 or len(pythons) != 1:
            raise GPUError(
                "EXECUTABLE_PAPER_INCOMPLETE",
                "Paper2Agent did not produce exactly one MCP server and environment",
            )
        server, python = servers[0].resolve(), pythons[0].resolve()
        if not server.is_relative_to(project) or not python.is_relative_to(project):
            raise GPUError(
                "EXECUTABLE_PAPER_PATH_ESCAPE",
                "Generated MCP runtime paths must remain inside their isolated project",
            )
        return project, server, python

    @asynccontextmanager
    async def _session(self, build_id: str):
        project, server, python = self._runtime(build_id)
        params = StdioServerParameters(
            command=str(python),
            args=[str(server)],
            cwd=project,
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(project)},
        )
        async with stdio_client(params) as streams, ClientSession(
            *streams, read_timeout_seconds=timedelta(seconds=300)
        ) as session:
            await session.initialize()
            yield session

    async def build(
        self,
        paper_id: str,
        repository: str,
        commit: str,
        tutorials: str | None = None,
    ) -> ExecutablePaperCandidate:
        if not _credential_configured():
            raise GPUError(
                "EXECUTABLE_PAPER_CREDENTIAL_REQUIRED",
                "Configure a task-scoped Anthropic API key or authenticate Claude Code in its isolated volume",
            )
        repository = self._repository(repository)
        commit = commit.lower()
        if await self._remote_head(repository) != commit:
            raise GPUError(
                "EXECUTABLE_PAPER_COMMIT_NOT_HEAD",
                "The official Paper2Agent pipeline cannot pin arbitrary target refs; pass the exact current default-branch HEAD",
            )
        build_id = self._build_id(paper_id, repository, commit, tutorials)
        async with self._coordination_lock:
            lock = self._locks.setdefault(build_id, asyncio.Lock())
        task = None
        try:
            async with lock:
                project = self._project(build_id)
                completed = project / ".gpu-lab-build-complete"
                if not completed.exists():
                    task = self._build_tasks.get(build_id)
                    if task is None or task.done():
                        task = asyncio.create_task(
                            self._build_once(build_id, repository, commit, tutorials)
                        )
                        self._build_tasks[build_id] = task
                        task.add_done_callback(
                            lambda finished, build_id=build_id, lock=lock: self._build_done(
                                build_id, lock, finished
                            )
                        )
                else:
                    task = None
            if task is not None:
                # A disconnected/retried HTTP request must not cancel a costly in-flight build.
                await asyncio.shield(task)
        finally:
            await self._cleanup_build(build_id, lock, task)
        if not completed.is_file() or completed.read_text(encoding="ascii").strip() != commit:
            raise GPUError(
                "EXECUTABLE_PAPER_BUILD_INCOMPLETE",
                "Paper2Agent did not persist the expected completion marker",
                retryable=True,
            )
        _, server, python = self._runtime(build_id)
        return ExecutablePaperCandidate(
            provider_build_id=build_id,
            repository=repository,
            commit=commit,
            environment={"python": str(python), "isolated": True},
            generated_mcp=str(server),
        )

    async def _build_once(
        self, build_id: str, repository: str, commit: str, tutorials: str | None
    ) -> None:
        project = self._project(build_id)
        command = [
            "bash",
            str(self.upstream_root / "Paper2Agent.sh"),
            "--project_dir",
            f"projects/{build_id}",
            "--github_url",
            repository,
        ]
        if tutorials:
            command.extend(["--tutorials", tutorials])
        code, stdout, stderr = await self._run(*command, include_model_credentials=True)
        project.mkdir(parents=True, exist_ok=True)
        (project / "gpu-lab-paper2agent.log").write_text(
            stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8"
        )
        if code:
            raise GPUError(
                "EXECUTABLE_PAPER_BUILD_FAILED",
                "Paper2Agent failed; inspect its worker log without promoting scientific state",
                retryable=False,
            )
        clone = project / "repo" / repository.rsplit("/", 1)[-1].removesuffix(".git")
        if not clone.is_dir():
            raise GPUError(
                "EXECUTABLE_PAPER_BUILD_INCOMPLETE",
                "Paper2Agent did not produce the expected repository checkout",
                retryable=False,
            )
        check_code, actual, _ = await self._run("git", "rev-parse", "HEAD", cwd=clone.resolve())
        if check_code or actual.strip().lower() != commit:
            raise GPUError(
                "EXECUTABLE_PAPER_COMMIT_MISMATCH",
                "The generated agent does not match the requested repository commit",
            )
        (project / ".gpu-lab-build-complete").write_text(commit, encoding="ascii")

    async def inspect_tools(self, executable_paper_id: str) -> list[dict[str, Any]]:
        async with self._session(executable_paper_id) as session:
            result = await session.list_tools()
        return [tool.model_dump(mode="json") for tool in result.tools]

    async def verify(self, executable_paper_id: str) -> dict[str, Any]:
        tools = await self.inspect_tools(executable_paper_id)
        project, _, _ = self._runtime(executable_paper_id)
        report = project / "reports" / "coverage_and_quality_report.md"
        return {
            "verification_status": "VERIFIED_INTEGRATION" if tools else "IMPLEMENTED_UNVERIFIED",
            "mcp_initialization": "passed",
            "tool_count": len(tools),
            "paper2agent_report_present": report.is_file(),
            "warning": "Scientific behavior still requires a GPU-Lab reproduction smoke.",
        }

    async def invoke(
        self, executable_paper_id: str, tool: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        async with self._session(executable_paper_id) as session:
            available = {item.name for item in (await session.list_tools()).tools}
            if tool not in available:
                raise GPUError("EXECUTABLE_PAPER_TOOL_NOT_FOUND", tool)
            result = await session.call_tool(tool, args)
        return result.model_dump(mode="json")


WORKER_TOKEN = os.environ.get("GPU_LAB_EXECUTABLE_PAPER_WORKER_TOKEN", "")
provider = Paper2AgentSubprocessProvider(
    Path(os.environ.get("PAPER2AGENT_ROOT", "/opt/paper2agent")),
    int(os.environ.get("PAPER2AGENT_TIMEOUT_SECONDS", "14400")),
)


def _credential_configured() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    config_root = Path(os.environ.get("CLAUDE_CONFIG_DIR", "/home/paperagent/.claude"))
    try:
        return config_root.is_dir() and any(item.is_file() for item in config_root.rglob("*"))
    except OSError:
        return False


def _required(value, field: str):
    if value in (None, ""):
        raise GPUError("INVALID_EXECUTABLE_PAPER_REQUEST", f"{field} is required")
    return value


async def dispatch(request: Request) -> JSONResponse:
    operation = request.path_params["operation"]
    if operation == "health":
        credential_configured = _credential_configured()
        return JSONResponse(
            {
                "result": {
                    "provider": "paper2agent",
                    "status": "ready" if credential_configured else "needs_credentials",
                    "credential_configured": credential_configured,
                    "upstream_commit": os.environ.get("PAPER2AGENT_UPSTREAM_COMMIT"),
                }
            }
        )
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ").encode()
    if not WORKER_TOKEN or not hmac.compare_digest(supplied, WORKER_TOKEN.encode()):
        return JSONResponse({"error": {"type": "UNAUTHORIZED", "message": "Unauthorized"}}, 401)
    try:
        body = WorkerRequest.model_validate(await request.json())
        if operation == "build":
            result = await provider.build(
                _required(body.paper_id, "paper_id"),
                _required(body.repository, "repository"),
                _required(body.commit, "commit"),
                body.tutorials,
            )
        elif operation == "inspect-tools":
            result = await provider.inspect_tools(
                _required(body.executable_paper_id, "executable_paper_id")
            )
        elif operation == "verify":
            result = await provider.verify(
                _required(body.executable_paper_id, "executable_paper_id")
            )
        elif operation == "invoke":
            result = await provider.invoke(
                _required(body.executable_paper_id, "executable_paper_id"),
                _required(body.tool, "tool"),
                body.args,
            )
        else:
            raise GPUError("UNKNOWN_EXECUTABLE_PAPER_OPERATION", operation)
        if isinstance(result, BaseModel):
            result = result.model_dump(mode="json")
        return JSONResponse({"result": result})
    except (GPUError, ValidationError, ValueError) as exc:
        error = exc.response()["error"] if isinstance(exc, GPUError) else {
            "type": "INVALID_EXECUTABLE_PAPER_REQUEST",
            "message": str(exc),
        }
        return JSONResponse({"error": error}, 400)
    except Exception:  # noqa: BLE001 - sanitize all generated-code failures
        return JSONResponse(
            {
                "error": {
                    "type": "EXECUTABLE_PAPER_PROVIDER_FAILURE",
                    "message": "The executable-paper worker failed without changing scientific truth",
                    "retryable": True,
                }
            },
            502,
        )


app = Starlette(routes=[Route("/{operation}", dispatch, methods=["POST"])])


def main() -> None:
    if not WORKER_TOKEN:
        raise RuntimeError("GPU_LAB_EXECUTABLE_PAPER_WORKER_TOKEN is required")
    uvicorn.run(app, host="0.0.0.0", port=8020, access_log=False)


if __name__ == "__main__":
    main()
