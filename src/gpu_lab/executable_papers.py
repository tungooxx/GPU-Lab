import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, Field

from .errors import GPUError
from .research import ResearchStore


class ExecutablePaperCandidate(BaseModel):
    provider_build_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository: str
    commit: str
    environment: dict[str, Any] = Field(default_factory=dict)
    available_tools: list[dict[str, Any]] = Field(default_factory=list)
    generated_mcp: str | None = None
    verification_status: str = "IMPLEMENTED_UNVERIFIED"
    test_results: dict[str, Any] = Field(default_factory=dict)
    reproduction_candidates: list[dict[str, Any]] = Field(default_factory=list)
    warning: str = (
        "Generated paper tools are unverified executable candidates, not scientific truth."
    )


@runtime_checkable
class ExecutablePaperProvider(Protocol):
    async def build(
        self,
        paper_id: str,
        repository: str,
        commit: str,
        tutorials: str | None = None,
    ) -> ExecutablePaperCandidate: ...

    async def inspect_tools(self, executable_paper_id: str) -> list[dict[str, Any]]: ...

    async def verify(self, executable_paper_id: str) -> dict[str, Any]: ...

    async def invoke(
        self, executable_paper_id: str, tool: str, args: dict[str, Any]
    ) -> dict[str, Any]: ...


class HttpExecutablePaperProvider:
    """Task-scoped client for an isolated executable-paper worker."""

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout_seconds: int = 14_400,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not base_url.startswith(("http://", "https://")):
            raise GPUError("INVALID_EXECUTABLE_PAPER_WORKER_URL", base_url)
        if not token:
            raise GPUError(
                "EXECUTABLE_PAPER_WORKER_TOKEN_REQUIRED",
                "Configure a task-scoped token for the isolated executable-paper worker",
            )
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def _call(self, operation: str, payload: dict[str, Any]) -> Any:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                response = await client.post(
                    f"{self.base_url}/{operation}",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.token}"},
                )
        except httpx.HTTPError as exc:
            raise GPUError(
                "EXECUTABLE_PAPER_PROVIDER_UNAVAILABLE",
                f"The isolated executable-paper worker failed during {operation}: {exc}",
                retryable=True,
            ) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise GPUError(
                "EXECUTABLE_PAPER_PROVIDER_INVALID_RESPONSE",
                f"The isolated worker returned non-JSON status {response.status_code}",
                retryable=response.status_code >= 500,
            ) from exc
        if not isinstance(data, dict):
            raise GPUError(
                "EXECUTABLE_PAPER_PROVIDER_INVALID_RESPONSE",
                "The isolated worker returned a non-object body",
                retryable=response.status_code >= 500,
            )
        if data.get("error"):
            error = data["error"]
            if not isinstance(error, dict):
                raise GPUError(
                    "EXECUTABLE_PAPER_PROVIDER_INVALID_RESPONSE",
                    "The isolated worker returned a non-object error",
                    retryable=response.status_code >= 500,
                )
            raise GPUError(
                error.get("type", "EXECUTABLE_PAPER_PROVIDER_ERROR"),
                error.get("message", "Executable-paper worker error"),
                error.get("retryable", False),
            )
        if response.is_error or "result" not in data:
            raise GPUError(
                "EXECUTABLE_PAPER_PROVIDER_INVALID_RESPONSE",
                f"The isolated worker returned HTTP {response.status_code} without a result",
                retryable=response.status_code >= 500,
            )
        return data["result"]

    async def health(self) -> dict[str, Any]:
        result = await self._call("health", {})
        if not isinstance(result, dict):
            raise GPUError(
                "EXECUTABLE_PAPER_PROVIDER_INVALID_RESPONSE",
                "The isolated worker returned a non-object health result",
            )
        return result

    async def build(
        self,
        paper_id: str,
        repository: str,
        commit: str,
        tutorials: str | None = None,
    ) -> ExecutablePaperCandidate:
        return ExecutablePaperCandidate.model_validate(
            await self._call(
                "build",
                {
                    "paper_id": paper_id,
                    "repository": repository,
                    "commit": commit,
                    "tutorials": tutorials,
                },
            )
        )

    async def inspect_tools(self, executable_paper_id: str) -> list[dict[str, Any]]:
        result = await self._call("inspect-tools", {"executable_paper_id": executable_paper_id})
        if not isinstance(result, list):
            raise GPUError(
                "EXECUTABLE_PAPER_PROVIDER_INVALID_RESPONSE",
                "inspect-tools did not return a list",
            )
        return result

    async def verify(self, executable_paper_id: str) -> dict[str, Any]:
        result = await self._call("verify", {"executable_paper_id": executable_paper_id})
        if not isinstance(result, dict):
            raise GPUError(
                "EXECUTABLE_PAPER_PROVIDER_INVALID_RESPONSE", "verify did not return an object"
            )
        return result

    async def invoke(
        self, executable_paper_id: str, tool: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        result = await self._call(
            "invoke",
            {"executable_paper_id": executable_paper_id, "tool": tool, "args": args},
        )
        if not isinstance(result, dict):
            raise GPUError(
                "EXECUTABLE_PAPER_PROVIDER_INVALID_RESPONSE", "invoke did not return an object"
            )
        return result


class ExecutablePaperService:
    """Imports worker outcomes into Research OS without trusting generated-paper state."""

    def __init__(self, store: ResearchStore, provider: ExecutablePaperProvider):
        self.store = store
        self.provider = provider

    @staticmethod
    def _approval_hash(action: str, parameters: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                {"action": action, "parameters": parameters},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def approve(
        self,
        project_id: str,
        action: str,
        parameters: dict[str, Any],
        approver: str,
        rationale: str,
        ttl_minutes: int = 30,
    ) -> dict[str, Any]:
        action = action.upper()
        if action not in {"BUILD", "INVOKE"}:
            raise GPUError("INVALID_EXECUTABLE_PAPER_APPROVAL_ACTION", action)
        if not approver.strip() or not rationale.strip():
            raise GPUError("RESEARCH_APPROVAL_INCOMPLETE", "Approver and rationale are required")
        if not 1 <= ttl_minutes <= 60:
            raise GPUError("INVALID_APPROVAL_TTL", "Approval TTL must be between 1 and 60 minutes")
        self.store.project_get(project_id)
        approved_at = datetime.now(UTC)
        return self.store.object_create(
            project_id,
            "ActionApproval",
            {
                "action": action,
                "parameter_hash": self._approval_hash(action, parameters),
                "approver": approver.strip(),
                "rationale": rationale.strip(),
                "approved_at": approved_at.isoformat(),
                "expires_at": (approved_at + timedelta(minutes=ttl_minutes)).isoformat(),
            },
            "EXECUTABLE_PAPER_ACTION_APPROVED",
            "APPROVED",
        )

    def _require_approval(
        self,
        approval_id: str,
        project_id: str,
        action: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        if not approval_id:
            raise GPUError(
                "HUMAN_APPROVAL_REQUIRED",
                "A matching server-verified approval is required for this exact action",
            )
        try:
            approval_id = str(uuid.UUID(approval_id))
        except ValueError as exc:
            raise GPUError(
                "HUMAN_APPROVAL_REQUIRED",
                "A matching server-verified approval is required for this exact action",
            ) from exc
        approval = self.store.object_get(approval_id)
        if (
            approval["kind"] != "ActionApproval"
            or approval["status"] != "APPROVED"
            or str(approval["project_id"]) != str(project_id)
            or approval["data"].get("action") != action
            or approval["data"].get("parameter_hash") != self._approval_hash(action, parameters)
        ):
            raise GPUError(
                "HUMAN_APPROVAL_REQUIRED",
                "A matching server-verified approval is required for this exact action",
            )
        try:
            expires_at = datetime.fromisoformat(approval["data"]["expires_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GPUError("INVALID_ACTION_APPROVAL", approval_id) from exc
        if expires_at <= datetime.now(UTC):
            raise GPUError("ACTION_APPROVAL_EXPIRED", approval_id)
        return self.store.object_update(
            approval_id,
            {"last_used_at": datetime.now(UTC).isoformat()},
            "APPROVED",
            "EXECUTABLE_PAPER_APPROVAL_USED",
        )

    async def build(
        self,
        project_id: str,
        paper_id: str,
        repository: str,
        commit: str,
        tutorials: str | None = None,
        approval_id: str | None = None,
    ) -> dict[str, Any]:
        paper = self.store.object_get(paper_id)
        if str(paper["project_id"]) != project_id or paper["kind"] != "Paper":
            raise GPUError("INVALID_EXECUTABLE_PAPER_SOURCE", paper_id)
        parameters = {
            "project_id": project_id,
            "paper_id": paper_id,
            "repository": repository,
            "commit": commit,
            "tutorials": tutorials,
        }
        self._require_approval(approval_id or "", project_id, "BUILD", parameters)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "repository": repository,
                    "commit": commit,
                    "tutorials": tutorials,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        exact = self.store.executable_paper_by_fingerprint(project_id, fingerprint)
        if exact:
            return {**exact, "idempotent_replay": True}
        candidate = await self.provider.build(paper_id, repository, commit, tutorials)
        expected_repository = repository.lower().rstrip("/").removesuffix(".git").rstrip("/")
        actual_repository = (
            candidate.repository.lower().rstrip("/").removesuffix(".git").rstrip("/")
        )
        if candidate.commit.lower() != commit.lower() or actual_repository != expected_repository:
            raise GPUError(
                "EXECUTABLE_PAPER_BUILD_IDENTITY_MISMATCH",
                "The worker result does not match the requested repository and commit",
            )
        record = self.store.object_create(
            project_id,
            "ExecutablePaper",
            {
                **candidate.model_dump(mode="json"),
                "paper_id": paper_id,
                "build_fingerprint": fingerprint,
                "provider": "paper2agent",
            },
            "EXECUTABLE_PAPER_BUILT",
            candidate.verification_status,
        )
        self.store.edge_create(paper_id, record["id"], "HAS_EXECUTABLE_CANDIDATE")
        return record

    async def inspect_tools(self, executable_paper_id: str) -> dict[str, Any]:
        record = self._record(executable_paper_id)
        tools = await self.provider.inspect_tools(record["data"]["provider_build_id"])
        return self.store.object_update(
            executable_paper_id,
            {**record["data"], "available_tools": tools},
            record["status"],
            "EXECUTABLE_PAPER_TOOLS_INSPECTED",
        )

    async def verify(self, executable_paper_id: str) -> dict[str, Any]:
        record = self._record(executable_paper_id)
        result = await self.provider.verify(record["data"]["provider_build_id"])
        status = result.get("verification_status", "IMPLEMENTED_UNVERIFIED")
        if status not in {"IMPLEMENTED_UNVERIFIED", "VERIFIED_UNIT", "VERIFIED_INTEGRATION"}:
            raise GPUError(
                "INVALID_EXECUTABLE_PAPER_VERIFICATION",
                "A generated paper worker cannot promote itself to VERIFIED_REAL",
            )
        return self.store.object_update(
            executable_paper_id,
            {**record["data"], "test_results": result, "verification_status": status},
            status,
            "EXECUTABLE_PAPER_VERIFIED",
        )

    async def invoke(
        self,
        executable_paper_id: str,
        tool: str,
        args: dict[str, Any],
        approval_id: str | None = None,
    ) -> dict[str, Any]:
        record = self._record(executable_paper_id)
        parameters = {
            "executable_paper_id": executable_paper_id,
            "tool": tool,
            "args": args,
        }
        self._require_approval(
            approval_id or "", str(record["project_id"]), "INVOKE", parameters
        )
        if record["status"] not in {"VERIFIED_UNIT", "VERIFIED_INTEGRATION"}:
            raise GPUError(
                "EXECUTABLE_PAPER_NOT_VERIFIED",
                "Inspect and verify the generated paper MCP before invoking it",
            )
        advertised = {item.get("name") for item in record["data"].get("available_tools", [])}
        if tool not in advertised:
            raise GPUError(
                "EXECUTABLE_PAPER_TOOL_NOT_INSPECTED",
                "Invoke only a tool captured by executable_paper_inspect_tools",
            )
        result = await self.provider.invoke(record["data"]["provider_build_id"], tool, args)
        return {
            "executable_paper_id": executable_paper_id,
            "tool": tool,
            "result": result,
            "warning": "Invocation output is not evidence until registered and assessed.",
        }

    def _record(self, executable_paper_id: str) -> dict[str, Any]:
        record = self.store.object_get(executable_paper_id)
        if record["kind"] != "ExecutablePaper":
            raise GPUError("INVALID_EXECUTABLE_PAPER", executable_paper_id)
        return record
