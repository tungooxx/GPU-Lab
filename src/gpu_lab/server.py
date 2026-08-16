import argparse
import hashlib
import hmac
import inspect
import ipaddress
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.types import ASGIApp

from .brain import ResearchBrain
from .brain_bench import BenchmarkPolicy, ResearchBrainBench
from .branches import ExperimentBranchService
from .config import Settings
from .dashboard import DASHBOARD_HTML
from .embeddings import EmbeddingService, LocalHashEmbeddingProvider
from .engineering import CodingExecutionPolicy, EngineeringService
from .epistemics import EpistemicService
from .errors import GPUError
from .executable_papers import ExecutablePaperService, HttpExecutablePaperProvider
from .literature import HttpLiteratureProvider, LiteratureService
from .local_runner import LocalRunner
from .meta_controller import MetaResearchController
from .meta_research import MetaResearchService
from .policy_lab import PolicyLabService
from .qd import HypothesisQDService
from .research import ResearchStore
from .research_operators import HttpResearchOperatorProvider, ResearchOperatorService
from .service import GPUService
from .strategy import ResearchStrategyService
from .terminal import TERMINAL_HTML

logger = logging.getLogger(__name__)

settings, service, research_store, research_brain, brain_bench_service, epistemic_service, literature_service, executable_paper_service, qd_service, branch_service, meta_research_service, strategy_service, policy_lab_service = (
    Settings(),
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
)
meta_controller_service: MetaResearchController | None = None
_singleton_lock = threading.RLock()
embedding_service: EmbeddingService | None = None
engineering_service: EngineeringService | None = None
research_operator_service: ResearchOperatorService | None = None
instructions = (
    "Safe, structured remote GPU experiment control plane. Credentials are never returned. "
    "Before research_experiment_execute, call research_decision_create and pass its decision_id; "
    "when a selected preregistered experiment is executable, execute it without waiting for a separate approval call."
)
if settings.gpu_lab_enable_local_runner:
    instructions += (
        " The local Linux research workspace is /workspace/local-vlm; use it as the base "
        "repository for local research experiments, and keep all local work inside that workspace."
    )
mcp = FastMCP(
    "GPU Lab",
    host=settings.fastmcp_host,
    port=settings.fastmcp_port,
    json_response=True,
    stateless_http=True,
    instructions=instructions,
)


_READ_ONLY_TOOLS = {
    "gpu_list",
    "vast_gpu_status",
    "gpu_search",
    "experiment_status",
    "experiment_logs",
    "experiment_list",
    "activity_recent",
    "research_state_get",
    "research_object_get",
    "research_benchmark_list",
    "research_benchmark_episode_get",
    "research_benchmark_policy_run",
    "research_benchmark_compare",
    "improve_status",
    "policy_get",
    "policy_compare",
    "policy_experiment_get",
    "independent_evidence_count",
    "supporting_evidence_families",
    "contradicting_evidence_families",
    "group_evidence_by_origin",
    "belief_audit",
    "world_model_consistency_check",
    "world_model_get",
    "hypothesis_portfolio_get",
    "literature_provider_status",
    "literature_search",
    "literature_ask",
    "executable_paper_provider_status",
    "claim_search",
    "claim_get_evidence",
    "claim_compare",
    "hypothesis_related",
    "hypothesis_niche_list",
    "hypothesis_qd_screen",
    "experiment_branch_get",
    "experiment_branch_next",
    "research_progress",
    "meta_lesson_list",
    "experiment_priority",
    "research_events",
    "paper_search",
    "paper_get",
    "paper_evidence_search",
    "research_semantic_search",
    "research_embedding_status",
    "research_embedding_search",
    "research_operator_status",
    "research_null_model_critique",
    "research_operator_critique",
    "research_strategy_list",
    "research_strategy_dataset_export",
    "decision_epistemic_audit",
    "paper_ask",
    "reproduction_status",
    "reproduction_plan",
    "artifact_list",
    "artifact_read",
    "local_status",
    "local_experiment_status",
    "local_experiment_logs",
    "local_artifact_list",
    "local_artifact_read",
    "engineering_task_get",
    "engineering_context_get",
    "engineering_result_get",
    "engineering_task_verify",
}
_DESTRUCTIVE_TOOLS = {
    "gpu_destroy",
    "executable_paper_build",
    "executable_paper_invoke",
    "research_hypothesis_generate",
    "research_null_model_create",
    "research_null_model_test",
    "research_decision_outcome_assess",
    "executable_paper_action_approve",
}
_OPEN_WORLD_TOOLS = {
    "gpu_list",
    "vast_gpu_status",
    "gpu_search",
    "gpu_create",
    "gpu_stop",
    "gpu_destroy",
    "repo_checkout",
    "env_prepare",
    "experiment_submit",
    "experiment_status",
    "experiment_logs",
    "experiment_cancel",
    "experiment_list",
    "artifact_list",
    "artifact_read",
    "remote_exec",
    "local_experiment_submit",
    "local_env_prepare",
    "literature_search",
    "literature_ask",
    "literature_gather",
    "brain_literature_resolve",
    "executable_paper_build",
    "executable_paper_inspect_tools",
    "executable_paper_verify",
    "executable_paper_invoke",
}
_ACRONYMS = {"api": "API", "gpu": "GPU", "id": "ID", "ssh": "SSH", "url": "URL", "vlm": "VLM"}


class GenericToolResult(BaseModel):
    """Stable structured envelope for dynamically shaped GPU Lab results."""

    result: Any


_GENERIC_RESULT_SCHEMA = GenericToolResult.model_json_schema()


def _normalise_mcp_accept_header(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """Treat an HTTP wildcard Accept header as accepting the JSON MCP response.

    Some connector preflight clients send ``Accept: */*`` rather than naming
    ``application/json``.  That is valid HTTP content negotiation, but the
    current Python MCP transport checks only explicit media-type prefixes.
    Keep the normal MCP JSON body validation unchanged; this only expands a
    wildcard response preference for the ``/mcp`` route.
    """
    accept = next(
        (value.decode("latin-1") for name, value in headers if name.lower() == b"accept"), ""
    )
    if "application/json" in accept.lower() or "*/*" not in accept:
        return headers
    return [
        (name, b"application/json" if name.lower() == b"accept" else value)
        for name, value in headers
    ]


class McpAcceptCompatibilityMiddleware(BaseHTTPMiddleware):
    """Allow standards-compliant wildcard Accept headers on the MCP endpoint."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/mcp":
            request.scope["headers"] = _normalise_mcp_accept_header(request.scope["headers"])
        return await call_next(request)


def _mcp_client_denied(host: str | None, cidrs: str) -> bool:
    """Return whether a client belongs to an explicitly blocked internal network."""
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
        return any(
            address in ipaddress.ip_network(item.strip(), strict=False)
            for item in cidrs.split(",")
            if item.strip()
        )
    except ValueError:
        return True


class McpClientNetworkPolicyMiddleware(BaseHTTPMiddleware):
    """Prevent isolated worker networks from calling the execution-bearing MCP endpoint."""

    async def dispatch(self, request: Request, call_next):
        client_host = request.client.host if request.client else None
        if request.url.path == "/mcp" and _mcp_client_denied(
            client_host, settings.gpu_lab_denied_mcp_client_cidrs
        ):
            return JSONResponse({"error": "MCP client network is not authorized"}, status_code=403)
        return await call_next(request)


class McpRequestObservabilityMiddleware(BaseHTTPMiddleware):
    """Correlate every MCP request that reaches this process with its final HTTP result."""

    async def dispatch(self, request: Request, call_next):
        request_id = _safe_request_id(request.headers.get("x-request-id"))
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "MCP request failed before a response request_id=%s method=%s path=%s",
                request_id,
                request.method,
                request.url.path,
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Request-ID"] = request_id
        if request.url.path == "/mcp":
            logger.info(
                "MCP request complete request_id=%s method=%s status=%s duration_ms=%s",
                request_id,
                request.method,
                response.status_code,
                duration_ms,
            )
        return response


def _safe_request_id(value: str | None) -> str:
    """Accept a trace ID only when it is safe to reflect into headers and logs."""
    if value and len(value) <= 128 and re.fullmatch(r"[A-Za-z0-9._-]+", value):
        return value
    return str(uuid.uuid4())


def _tool_title(name: str) -> str:
    return " ".join(_ACRONYMS.get(word, word.capitalize()) for word in name.split("_"))


def _apply_tool_metadata() -> None:
    """Fill MCP metadata for every tool, including conditional local/remote tools."""
    for name, tool in mcp._tool_manager._tools.items():
        tool.title = tool.title or _tool_title(name)
        tool.description = (
            tool.description or f"Perform the GPU Lab {tool.title.lower()} operation."
        )
        tool.annotations = ToolAnnotations(
            readOnlyHint=name in _READ_ONLY_TOOLS,
            destructiveHint=name in _DESTRUCTIVE_TOOLS,
            openWorldHint=name in _OPEN_WORLD_TOOLS,
        )
        tool.fn_metadata.output_schema = _GENERIC_RESULT_SCHEMA
        tool.fn_metadata.output_model = GenericToolResult
        tool.fn_metadata.wrap_output = True
        tool.__dict__.pop("output_schema", None)


def svc() -> GPUService:
    global service
    if service is None:
        service = GPUService(settings)
    return service


def research() -> ResearchStore:
    global research_store
    if not settings.gpu_lab_research_database_url:
        raise GPUError("RESEARCH_DATABASE_NOT_CONFIGURED", "Set GPU_LAB_RESEARCH_DATABASE_URL")
    if research_store is None:
        with _singleton_lock:
            if research_store is None:
                research_store = ResearchStore(settings.gpu_lab_research_database_url)
    return research_store


def engineering() -> EngineeringService:
    global engineering_service
    if engineering_service is None:
        with _singleton_lock:
            if engineering_service is None:
                engineering_service = EngineeringService(research())
    return engineering_service


def initialize_research_runtime() -> ResearchStore | None:
    """Run migrations and deferred temporal recovery before MCP begins accepting requests."""
    if not settings.gpu_lab_research_database_url:
        return None
    return research()


def brain() -> ResearchBrain:
    global research_brain
    if research_brain is None:
        with _singleton_lock:
            if research_brain is None:
                research_brain = ResearchBrain(research())
    return research_brain


def brain_bench() -> ResearchBrainBench:
    global brain_bench_service
    if brain_bench_service is None:
        with _singleton_lock:
            if brain_bench_service is None:
                brain_bench_service = ResearchBrainBench(
                    Path(settings.gpu_lab_research_bench_dir)
                )
    return brain_bench_service


def policy_lab() -> PolicyLabService:
    global policy_lab_service
    if policy_lab_service is None:
        with _singleton_lock:
            if policy_lab_service is None:
                policy_lab_service = PolicyLabService(
                    research(),
                    brain_bench(),
                    auto_evaluate=settings.gpu_lab_policy_auto_evaluate,
                    auto_reject=settings.gpu_lab_policy_auto_reject,
                    auto_revise=settings.gpu_lab_policy_auto_revise,
                    max_revisions=settings.gpu_lab_policy_max_revisions,
                    auto_promote_production=settings.gpu_lab_policy_auto_promote_production,
                    engineering_service=engineering(),
                )
    return policy_lab_service


def meta_controller() -> MetaResearchController:
    global meta_controller_service
    if meta_controller_service is None:
        with _singleton_lock:
            if meta_controller_service is None:
                meta_controller_service = MetaResearchController(
                    research(), policy_lab(), mode=settings.gpu_lab_policy_autonomy_mode,
                    candidate_budget=settings.gpu_lab_policy_meta_candidate_budget,
                    benchmark_budget=settings.gpu_lab_policy_meta_benchmark_budget,
                    literature_budget=settings.gpu_lab_policy_meta_literature_budget,
                )
    return meta_controller_service


def epistemics() -> EpistemicService:
    global epistemic_service
    if epistemic_service is None:
        with _singleton_lock:
            if epistemic_service is None:
                epistemic_service = EpistemicService(research())
    return epistemic_service


def embeddings() -> EmbeddingService:
    global embedding_service
    if settings.gpu_lab_embedding_provider != "local-hash":
        raise GPUError(
            "EMBEDDING_PROVIDER_UNAVAILABLE",
            "Automatic embeddings are disabled; structured and lexical retrieval remain available.",
        )
    if embedding_service is None:
        with _singleton_lock:
            if embedding_service is None:
                embedding_service = EmbeddingService(
                    research(),
                    LocalHashEmbeddingProvider(settings.gpu_lab_embedding_dimension),
                )
    return embedding_service


def literature() -> LiteratureService:
    global literature_service
    if settings.gpu_lab_literature_provider != "paperqa-http":
        raise GPUError(
            "LITERATURE_PROVIDER_UNAVAILABLE",
            "Start the isolated literature profile and set provider=paperqa-http.",
        )
    if literature_service is None:
        with _singleton_lock:
            if literature_service is None:
                literature_service = LiteratureService(
                    research(),
                    HttpLiteratureProvider(
                        settings.gpu_lab_literature_worker_url,
                        settings.gpu_lab_literature_worker_token or "",
                    ),
                )
    return literature_service


async def run_meta_research(project_id: str) -> dict[str, Any]:
    """Run one bounded campaign and, when configured, its single prepared literature scout."""
    progress = await call(meta_research().progress, project_id)
    decisions = int(progress.get("metrics", {}).get("scientific_decisions_total", 0)) if isinstance(progress, dict) else 0
    postmortem = None
    postmortem_opportunities = []
    if decisions and decisions % 5 == 0:
        postmortem = await call(meta_research().meta_review, project_id)
        if "error" not in postmortem:
            postmortem_opportunities = await call(
                meta_controller().postmortem_opportunities,
                project_id,
                str(postmortem["id"]),
            )
    result = await call(meta_controller().run_once, project_id)
    request = result.get("literature_request") if isinstance(result, dict) else None
    if not request:
        return {**result, "postmortem": postmortem, "postmortem_opportunities": postmortem_opportunities}
    if settings.gpu_lab_literature_provider != "paperqa-http":
        deferred = await call(
            research().object_update,
            str(request["id"]),
            {"dispatch_status": "DEFERRED", "dispatch_reason": "literature provider unavailable"},
            "DEFERRED",
            "LITERATURE_SCOUT_DEFERRED",
        )
        return {**result, "postmortem": postmortem, "postmortem_opportunities": postmortem_opportunities, "literature_scout": {"status": "DEFERRED", "request": deferred}}
    gathered = await call(literature().gather, project_id, request["data"]["question"])
    if "error" in gathered:
        deferred = await call(
            research().object_update,
            str(request["id"]),
            {"dispatch_status": "UNAVAILABLE", "provider_error": gathered["error"]},
            "DEFERRED",
            "LITERATURE_SCOUT_DEFERRED",
        )
        return {**result, "postmortem": postmortem, "postmortem_opportunities": postmortem_opportunities, "literature_scout": {"status": "UNAVAILABLE", "request": deferred}}
    completed = await call(
        research().object_update,
        str(request["id"]),
        {"dispatch_status": "COMPLETED", "evidence_ids": [str(item["id"]) for item in gathered.get("evidence", [])]},
        "COMPLETED",
        "LITERATURE_SCOUT_COMPLETED",
    )
    transfers = await call(
        meta_controller().literature_scout_complete,
        project_id,
        str(request["id"]),
        [str(item["id"]) for item in gathered.get("evidence", [])],
    )
    return {**result, "postmortem": postmortem, "postmortem_opportunities": postmortem_opportunities, "literature_scout": {"status": "COMPLETED", "request": completed, "gathered": gathered, "policy_transfers": transfers}}


def executable_papers() -> ExecutablePaperService:
    global executable_paper_service
    if settings.gpu_lab_executable_paper_provider != "paper2agent-http":
        raise GPUError(
            "EXECUTABLE_PAPER_PROVIDER_UNAVAILABLE",
            "Start the isolated paper-agents profile and set provider=paper2agent-http.",
        )
    if executable_paper_service is None:
        with _singleton_lock:
            if executable_paper_service is None:
                executable_paper_service = ExecutablePaperService(
                    research(),
                    HttpExecutablePaperProvider(
                        settings.gpu_lab_executable_paper_worker_url,
                        settings.gpu_lab_executable_paper_worker_token or "",
                    ),
                )
    return executable_paper_service


def qd() -> HypothesisQDService:
    global qd_service
    if qd_service is None:
        with _singleton_lock:
            if qd_service is None:
                qd_service = HypothesisQDService(research())
    return qd_service


def research_operators() -> ResearchOperatorService:
    global research_operator_service
    if settings.gpu_lab_research_operator_provider != "literature-http":
        raise GPUError(
            "RESEARCH_OPERATOR_UNAVAILABLE",
            "Start the isolated literature profile and set provider=literature-http.",
        )
    if research_operator_service is None:
        with _singleton_lock:
            if research_operator_service is None:
                research_operator_service = ResearchOperatorService(
                    research(),
                    qd(),
                    HttpResearchOperatorProvider(
                        settings.gpu_lab_literature_worker_url,
                        settings.gpu_lab_literature_worker_token or "",
                    ),
                )
    return research_operator_service


def branches() -> ExperimentBranchService:
    global branch_service
    if branch_service is None:
        with _singleton_lock:
            if branch_service is None:
                branch_service = ExperimentBranchService(research())
    return branch_service


def meta_research() -> MetaResearchService:
    global meta_research_service
    if meta_research_service is None:
        with _singleton_lock:
            if meta_research_service is None:
                meta_research_service = MetaResearchService(research())
    return meta_research_service


def strategy() -> ResearchStrategyService:
    global strategy_service
    if strategy_service is None:
        with _singleton_lock:
            if strategy_service is None:
                strategy_service = ResearchStrategyService(research())
    return strategy_service


async def call(fn, *args, **kwargs):
    tool_name = getattr(fn, "__name__", "unknown")
    started = time.perf_counter()
    arguments = {"args": scrub(args), "kwargs": scrub(kwargs)}
    try:
        result = fn(*args, **kwargs)
        result = await result if inspect.isawaitable(result) else result
        svc().repo.audit(
            tool_name, arguments, "success", int((time.perf_counter() - started) * 1000)
        )
        return result
    except GPUError as exc:
        svc().repo.audit(
            tool_name,
            arguments,
            "error",
            int((time.perf_counter() - started) * 1000),
            exc.message,
        )
        return exc.response()
    except Exception:
        # Do not expose implementation details or leave an MCP request without an audit record.
        logger.exception("Unexpected MCP tool failure: %s", tool_name)
        svc().repo.audit(
            tool_name,
            arguments,
            "error",
            int((time.perf_counter() - started) * 1000),
            "Internal error",
        )
        return {
            "error": {
                "type": "INTERNAL_ERROR",
                "message": "Unexpected server error",
                "retryable": False,
            }
        }


def scrub(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if re.search(r"key|token|secret|password", key, re.IGNORECASE)
            else scrub(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"(?i)(bearer\s+|token\s*[=:]\s*|api[_-]?key\s*[=:]\s*|secret\s*[=:]\s*|password\s*[=:]\s*)[^\s'\"]+",
            r"\1[REDACTED]",
            value,
        )[:4096]
    return value


_STATE_SUMMARY_FIELDS = (
    "statement",
    "mechanism",
    "question",
    "prediction",
    "scope",
    "status_rationale",
    "failed_assumption",
    "revisit_condition",
    "experiment_id",
    "run_id",
    "decision_id",
    "name",
    "title",
)


def _state_object_summary(item: dict[str, Any]) -> dict[str, Any]:
    """Return an MCP-safe object summary; large scientific payloads stay retrievable by ID."""
    data = item.get("data", {})
    summary_data = {
        field: value[:1000] if isinstance(value, str) else value
        for field, value in data.items()
        if field in _STATE_SUMMARY_FIELDS and isinstance(value, (str, int, float, bool, type(None)))
    }
    return {
        key: item[key]
        for key in ("id", "project_id", "kind", "status", "created_at")
        if key in item
    } | {"data": summary_data}


def _compact_research_state(state: dict[str, Any], limit: int = 10) -> dict[str, Any]:
    """Keep state discovery below connector response limits without hiding object identities."""
    limit = min(max(limit, 1), 50)
    objects = state.get("objects", [])
    object_counts: dict[str, int] = {}
    for item in objects:
        kind = str(item.get("kind", "Unknown"))
        object_counts[kind] = object_counts.get(kind, 0) + 1

    canonical_state = _compact_canonical_state(state.get("canonical_state", {}), limit)

    project_state = state.get("state", {})
    return {
        "name": state.get("name"),
        "question": state.get("question"),
        "state": {
            key: value
            for key, value in project_state.items()
            if key
            in {
                "research_question",
                "current_best_explanation",
                "highest_value_unknown",
                "established_facts",
                "next_discriminating_experiments",
            }
        },
        "canonical_state": canonical_state,
        "object_counts": dict(sorted(object_counts.items())),
        "temporal_snapshot": {
            "as_of": state.get("as_of"),
            "valid_from": state.get("valid_from"),
            "committed_at": state.get("committed_at"),
            "legacy_backfill": state.get("legacy_backfill", False),
        },
        "detail_hint": "Use research_object_get with an object ID for its complete persisted record.",
    }


def _compact_canonical_state(canonical_state: dict[str, Any], limit: int) -> dict[str, Any]:
    """Keep the decision context traceable by ID without copying full object payloads."""
    compact: dict[str, Any] = {}
    for key, value in canonical_state.items():
        if isinstance(value, list):
            compact[key] = [
                _state_object_summary(item) if isinstance(item, dict) else item
                for item in value[:limit]
            ]
            if len(value) > limit:
                compact[f"{key}_truncated"] = len(value) - limit
        else:
            compact[key] = value
    return compact


def _compact_brain_step(result: dict[str, Any], limit: int = 10) -> dict[str, Any]:
    """Return an MCP-safe planning recommendation; full decision data remains durable by ID."""
    return {
        "brain_step_id": result["brain_step_id"],
        "decision_id": result["decision_id"],
        "agenda_item": _state_object_summary(result["agenda_item"]),
        "question": result["question"],
        "scientific_state": _compact_canonical_state(result["scientific_state"], limit),
        "world_model": _state_object_summary(result["world_model"]),
        "competing_hypotheses": [
            _state_object_summary(item) for item in result["competing_hypotheses"][:limit]
        ],
        "dead_ideas_retrieved": result["dead_ideas_retrieved"][:limit],
        "research_situation": result.get("research_situation"),
        "strategy_patterns_retrieved": result.get("strategy_patterns_retrieved"),
        "agenda_diminishing_returns": result.get("agenda_diminishing_returns"),
        "candidate_actions": result["candidate_actions"][:limit],
        "selected_action": result["selected_action"],
        "reason": result.get("reason"),
        "expected_information_gain": result.get("expected_information_gain"),
        "estimated_cost": result.get("estimated_cost"),
        "requires_human_approval": result.get("requires_human_approval", False),
        "verification_status": result.get("verification_status"),
        "as_of": result.get("as_of"),
        "detail_hint": (
            "Historical dry run; no ResearchDecision was persisted."
            if result.get("decision_id") is None
            else "Use research_object_get with the decision ID for its complete persisted trace."
        ),
    }


@mcp.tool()
async def gpu_list():
    """List known and provider-visible GPU instances."""
    return await call(svc().gpu_list)


@mcp.tool()
async def vast_gpu_status(instance_id: str):
    """Return provider and nvidia-smi runtime state."""
    return await call(svc().gpu_status, instance_id)


@mcp.tool()
async def gpu_search(
    gpu_name: str | None = None,
    min_vram_gb: int | None = None,
    max_hourly_price: float | None = None,
):
    """Return safely ranked Vast offers; create requires an explicit offer ID."""
    return await call(svc().gpu_search, gpu_name, min_vram_gb, max_hourly_price)


@mcp.tool()
async def gpu_create(
    offer_id: str, disk_gb: int = 100, image: str | None = None, label: str | None = None
):
    """Create a Vast instance from a caller-selected offer."""
    return await call(svc().gpu_create, offer_id, disk_gb, image, label)


@mcp.tool()
async def gpu_stop(instance_id: str):
    """Stop an instance while preserving data when Vast supports it."""
    return await call(svc().gpu_stop, instance_id)


@mcp.tool()
async def gpu_destroy(instance_id: str, confirmation: str):
    """Destroy only with confirmation exactly DESTROY."""
    return await call(svc().gpu_destroy, instance_id, confirmation)


@mcp.tool()
async def repo_checkout(
    instance_id: str,
    repo_url: str,
    commit: str | None = None,
    branch: str | None = None,
    name: str | None = None,
):
    return await call(svc().repo_checkout, instance_id, repo_url, commit, branch, name)


@mcp.tool()
async def env_prepare(instance_id: str, repo_path: str, strategy: str = "auto"):
    return await call(svc().env_prepare, instance_id, repo_path, strategy)


@mcp.tool()
async def experiment_submit(
    instance_id: str,
    repo_path: str,
    command: str,
    name: str | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
    artifact_patterns: list[str] | None = None,
    metadata: dict | None = None,
):
    return await call(
        svc().experiment_submit,
        instance_id,
        repo_path,
        command,
        name,
        env,
        timeout_seconds,
        artifact_patterns,
        metadata,
    )


@mcp.tool()
async def experiment_status(job_id: str):
    return await call(svc().experiment_status, job_id)


@mcp.tool()
async def experiment_logs(job_id: str, tail: int = 200, stream: str = "combined"):
    return await call(svc().experiment_logs, job_id, tail, stream)


@mcp.tool()
async def experiment_cancel(job_id: str):
    return await call(svc().experiment_cancel, job_id)


@mcp.tool()
async def experiment_list(
    instance_id: str | None = None, status: str | None = None, limit: int = 50
):
    return await call(svc().experiment_list, instance_id, status, limit)


@mcp.tool()
async def activity_recent(limit: int = 50):
    """List recent MCP tool calls, their sanitized inputs, outcomes, and durations."""
    return svc().repo.list_audit(min(max(limit, 1), 100))


@mcp.tool()
async def research_project_create(name: str, question: str):
    """Create a canonical research project and its immutable first event."""
    return await call(research().project_create, name, question)


@mcp.tool()
async def research_state_get(project_id: str, limit: int = 10, as_of: str | None = None):
    """Retrieve compact canonical scientific state; fetch complete records individually by ID."""
    state = await call(research().state_get, project_id, as_of)
    if "error" in state:
        return state
    return _compact_research_state(state, limit)


@mcp.tool()
async def research_object_get(object_id: str, as_of: str | None = None):
    """Retrieve one complete persisted research record by its ID."""
    return await call(research().object_get, object_id, as_of)


@mcp.tool()
async def engineering_task_create(
    project_id: str,
    purpose: str,
    task_type: str,
    change_request: str = "",
    repository: str = "",
    repository_root: str = "",
    base_commit: str | None = None,
    relevant_files: list[str] | None = None,
    relevant_symbols: list[str] | None = None,
    scientific_variable_changed: str | None = None,
    scientific_variables_held_fixed: list[str] | None = None,
    scientific_invariants: dict[str, Any] | None = None,
    engineering_invariants: dict[str, Any] | None = None,
    prohibited_changes: list[str] | None = None,
    acceptance_tests: list[str] | None = None,
    baseline_commands: list[str] | None = None,
    targeted_tests: list[str] | None = None,
    broader_tests: list[str] | None = None,
    expected_artifacts: list[str] | None = None,
    implementation_guards: list[dict[str, Any]] | None = None,
    research_decision_id: str | None = None,
    experiment_id: str | None = None,
):
    """Create an engineering-only implementation task; it cannot assess scientific truth."""
    return await call(
        engineering().task_create, project_id, purpose, task_type, change_request, repository,
        repository_root, base_commit, relevant_files, relevant_symbols, scientific_variable_changed,
        scientific_variables_held_fixed, scientific_invariants, engineering_invariants,
        prohibited_changes, acceptance_tests, baseline_commands, targeted_tests, broader_tests,
        expected_artifacts, implementation_guards, research_decision_id, experiment_id,
    )


@mcp.tool()
async def engineering_policy_get():
    """Return the provider-neutral engineering phase contract."""
    return CodingExecutionPolicy.contract()


@mcp.tool()
async def engineering_task_get(task_id: str):
    """Retrieve a durable engineering task."""
    return await call(engineering().task_get, task_id)


@mcp.tool()
async def engineering_task_start(task_id: str, inspection: dict[str, Any], baseline: dict[str, Any]):
    """Record repository inspection and a passing baseline before implementation work."""
    return await call(engineering().task_start, task_id, inspection, baseline)


@mcp.tool()
async def engineering_result_get(result_id: str):
    """Retrieve implementation evidence without treating it as scientific evidence."""
    return await call(engineering().result_get, result_id)


@mcp.tool()
async def engineering_task_verify(task_id: str):
    """Check implementation readiness; this never assesses a hypothesis."""
    return await call(engineering().task_verify, task_id)


@mcp.tool()
async def engineering_context_get(task_id: str):
    """Return compact implementation context without dumping scientific state."""
    return await call(engineering().context_get, task_id)


@mcp.tool()
async def engineering_task_update(task_id: str, status: str, update: dict[str, Any] | None = None):
    """Update engineering task workflow status and provenance only."""
    return await call(engineering().task_update, task_id, status, update)


@mcp.tool()
async def engineering_diff_review(task_id: str, review: dict[str, Any]):
    """Persist a diff review and block material scientific design drift."""
    return await call(engineering().diff_review, task_id, review)


@mcp.tool()
async def engineering_result_record(task_id: str, result: dict[str, Any]):
    """Record implementation evidence and unlock a linked policy benchmark if verified."""
    recorded = await call(engineering().result_record, task_id, result)
    if isinstance(recorded, dict) and "error" in recorded:
        return recorded
    task = await call(engineering().task_get, task_id)
    if isinstance(task, dict) and "error" in task:
        return task
    if not task.get("data", {}).get("policy_patch_id"):
        return recorded
    handoff = await call(policy_lab().code_patch_result_sync, str(task["project_id"]), task_id)
    return {"engineering_result": recorded, "policy_handoff": handoff}


@mcp.tool()
async def research_benchmark_list():
    """List frozen benchmark checkpoints without exposing hidden future evidence or answer keys."""
    episodes = await call(brain_bench().load_all)
    if isinstance(episodes, dict) and "error" in episodes:
        return episodes
    return [
        {
            "episode_id": episode.episode_id,
            "project_id": episode.project_id,
            "domain": episode.domain,
            "cutoff_timestamp": episode.cutoff_timestamp.isoformat(),
            "scientific_question": episode.scientific_question,
            "provenance_kinds": sorted(
                {item.kind.value for item in episode.source_provenance}
            ),
        }
        for episode in episodes
    ]


@mcp.tool()
async def research_benchmark_episode_get(episode_id: str):
    """Return the blinded policy input without hidden future state, rubric, tags, or scores."""
    episode = await call(brain_bench().get_episode, episode_id)
    if isinstance(episode, dict) and "error" in episode:
        return episode
    return episode.visible_payload()


@mcp.tool()
async def research_benchmark_policy_run(episode_id: str, policy: str):
    """Run one blinded deterministic policy without revealing held-out scoring."""
    episode = await call(brain_bench().get_episode, episode_id)
    if isinstance(episode, dict) and "error" in episode:
        return episode
    try:
        policy_name = BenchmarkPolicy(policy)
    except ValueError:
        return {
            "error": {
                "type": "BRAIN_BENCH_INVALID_POLICY",
                "message": policy,
                "retryable": False,
            }
        }
    decision = await call(brain_bench().baseline_decision, episode, policy_name)
    if isinstance(decision, dict) and "error" in decision:
        return decision
    return decision.model_dump(mode="json")


@mcp.tool()
async def research_benchmark_compare():
    """Score required v1, v1.5, v2, and baseline policies only after blinded policy selection."""
    return await call(brain_bench().compare_builtin_policies)


@mcp.tool()
async def improve_start(
    project_id: str,
    idea: str | None = None,
    paper: str | None = None,
    failure: str | None = None,
    component: str | None = None,
    search: bool = False,
    prompt: bool = False,
):
    """Create bounded policy candidates; prompt mode compiles candidates before evaluation."""
    if search:
        if settings.gpu_lab_literature_provider != "paperqa-http":
            raise GPUError(
                "LITERATURE_PROVIDER_UNAVAILABLE",
                "improve --search requires the isolated PaperQA literature provider.",
            )
        weakness_query = failure or idea or component or "research experiment discrimination weakness"
        literature_service = await call(literature)
        if isinstance(literature_service, dict) and "error" in literature_service:
            return literature_service
        literature_result = await call(literature_service.search,
            "Research methods for this measured policy weakness; return competing methods, "
            "limitations, and negative evidence where available: " + weakness_query
        )
        if isinstance(literature_result, dict) and "error" in literature_result:
            return literature_result
        paper = literature_result.answer or "\n".join(
            candidate.source_excerpt for candidate in literature_result.evidence_candidates[:3]
        )
    return await call(
        policy_lab().improve,
        project_id,
        idea=idea,
        paper=paper,
        failure=failure,
        component=component,
        search=search,
        prompt=prompt,
    )


@mcp.tool()
async def improve_status(improvement_run_id: str):
    """Retrieve a durable improvement run and its recommendation."""
    return await call(research().object_get, improvement_run_id)


@mcp.tool()
async def policy_get(policy_id: str):
    """Retrieve a versioned ResearchPolicy without compiling or applying it."""
    return await call(research().object_get, policy_id)


@mcp.tool()
async def policy_compare(base_policy_id: str, candidate_policy_id: str):
    """Show the semantic diff between two durable policy versions."""
    return await call(policy_lab().policy_diff, base_policy_id, candidate_policy_id)


@mcp.tool()
async def policy_export(policy_id: str, provider: str | None = None):
    """Export a portable policy representation without applying or compiling executable text."""
    return await call(policy_lab().export_policy, policy_id, provider)


@mcp.tool()
async def research_policy_context_get(project_id: str, provider: str = "CHATGPT"):
    """Return the active compiled policy context; dynamic research state remains in Research OS tools."""
    policy = await call(policy_lab().ensure_production_policy, project_id)
    if isinstance(policy, dict) and "error" in policy:
        return policy
    return await call(policy_lab().compile_policy, str(policy["id"]), provider)


@mcp.tool()
async def policy_experiment_get(policy_experiment_id: str):
    """Retrieve an isolated meta-research PolicyExperiment."""
    return await call(research().object_get, policy_experiment_id)


@mcp.tool()
async def policy_transfer_classify(
    policy_experiment_id: str,
    project_results: dict[str, bool],
    model_results: dict[str, bool] | None = None,
):
    """Classify isolated policy evidence across projects and optionally across models."""
    return await call(policy_lab().classify_transfer, policy_experiment_id, project_results, model_results)


@mcp.tool()
async def policy_promote(project_id: str, patch_id: str):
    """Explicitly promote a benchmark-supported patch; auto-promotion is disabled by default."""
    return await call(policy_lab().promote, project_id, patch_id)


@mcp.tool()
async def policy_rollback(project_id: str, policy_id: str):
    """Restore a previous policy version while retaining all policy history."""
    return await call(policy_lab().rollback, project_id, policy_id)


@mcp.tool()
async def policy_restrict(project_id: str, policy_id: str, status: str, reason: str):
    """Deprecate or scope-restrict a non-production policy without deleting its lineage."""
    return await call(policy_lab().restrict_policy, project_id, policy_id, status, reason)


@mcp.tool()
async def policy_code_patch_prepare(project_id: str, patch_id: str, code_change: dict[str, Any]):
    """Create a bounded EngineeringTask for a code-bearing policy patch; it is not policy success."""
    return await call(policy_lab().code_patch_prepare, project_id, patch_id, code_change)


@mcp.tool()
async def policy_pin(project_id: str, policy_id: str | None = None):
    """Pin a project policy (or clear the pin) to constrain autonomous policy changes."""
    return await call(meta_controller().policy_pin, project_id, policy_id)


@mcp.tool()
async def policy_feedback_record(project_id: str, feedback: str, target_component: str = "experiment_selection"):
    """Record user feedback as a hypothesis pending inspection of actual outcomes."""
    return await call(meta_controller().feedback_record, project_id, feedback, target_component)


@mcp.tool()
async def policy_feedback_validate(project_id: str, feedback_id: str):
    """Validate feedback against persisted decision outcomes before creating a candidate."""
    return await call(meta_controller().feedback_validate, project_id, feedback_id)


@mcp.tool()
async def policy_ranker_readiness(project_id: str):
    """Assess whether an offline advisory ranker is justified; it never trains or deploys one."""
    return await call(meta_controller().ranker_readiness, project_id)


@mcp.tool()
async def policy_canary_start(project_id: str, candidate_policy_id: str, percentage: int = 10):
    """Start a bounded prospective canary; it does not replace production policy."""
    return await call(policy_lab().start_canary, project_id, candidate_policy_id, percentage)


@mcp.tool()
async def policy_canary_observation_record(
    canary_id: str,
    decision_id: str,
    observed_behavior: dict[str, Any],
    hard_epistemic_regression: bool = False,
):
    """Record a prospective canary observation; hard epistemic regression stops the canary."""
    return await call(
        policy_lab().record_canary_observation,
        canary_id,
        decision_id,
        observed_behavior,
        hard_epistemic_regression=hard_epistemic_regression,
    )


@mcp.tool()
async def policy_shadow_record(
    project_id: str,
    production_policy_id: str,
    shadow_policy_id: str,
    decision_id: str,
    production_action: dict[str, Any],
    shadow_action: dict[str, Any],
    observed_production_result: dict[str, Any] | None = None,
):
    """Record a non-executing policy comparison; shadow outcomes remain counterfactually unknown."""
    return await call(
        policy_lab().record_shadow,
        project_id,
        production_policy_id,
        shadow_policy_id,
        decision_id,
        production_action,
        shadow_action,
        observed_production_result,
    )


@mcp.tool()
async def policy_hindsight_record(
    policy_id: str,
    observed_improvement: float | None,
    observed_cost: float | None,
    unexpected_failure: str | None = None,
    decision_ids: list[str] | None = None,
):
    """Record hindsight, then autonomously assess calibration and rollback evidence."""
    result = await call(
        policy_lab().record_hindsight,
        policy_id,
        observed_improvement,
        observed_cost,
        unexpected_failure,
        decision_ids,
    )
    if "error" in result:
        return result
    project_id = str(result["project_id"])
    calibration = await call(meta_controller().monitor_calibration, project_id)
    regressions = await call(meta_controller().monitor_promotions, project_id)
    return {
        **result,
        "calibration_opportunities": calibration,
        "policy_regressions": regressions,
    }


@mcp.tool()
async def evidence_family_create(
    project_id: str,
    origin_type: str,
    origin_id: str,
    description: str,
    derived_from_evidence_family_id: str | None = None,
    dependency_note: str | None = None,
):
    """Create one empirical-origin family; derived database records remain one confirmation."""
    return await call(
        epistemics().evidence_family_create,
        project_id,
        origin_type,
        origin_id,
        description,
        derived_from_evidence_family_id,
        dependency_note,
    )


@mcp.tool()
async def evidence_family_link(
    family_id: str, entity_id: str, relationship: str = "DERIVED"
):
    """Link a derived, supporting, or contradicting record to its empirical EvidenceFamily."""
    return await call(epistemics().evidence_family_link, family_id, entity_id, relationship)


@mcp.tool()
async def independent_evidence_count(entity_id: str, as_of: str | None = None):
    """Count independent empirical roots, not the number of derived database records."""
    return await call(epistemics().independent_evidence_count, entity_id, as_of)


@mcp.tool()
async def supporting_evidence_families(entity_id: str, as_of: str | None = None):
    """List independent families explicitly linked as support for an entity."""
    return await call(epistemics().supporting_evidence_families, entity_id, as_of)


@mcp.tool()
async def contradicting_evidence_families(entity_id: str, as_of: str | None = None):
    """List independent families explicitly linked against an entity."""
    return await call(epistemics().contradicting_evidence_families, entity_id, as_of)


@mcp.tool()
async def group_evidence_by_origin(entity_id: str, as_of: str | None = None):
    """Group linked families under their dependency roots for transparent anti-double-counting."""
    return await call(epistemics().group_evidence_by_origin, entity_id, as_of)


@mcp.tool()
async def belief_audit(entity_id: str, as_of: str | None = None):
    """Explain independent support, contradictions, scope, and promotion risks for one belief."""
    return await call(epistemics().belief_audit, entity_id, as_of)


@mcp.tool()
async def world_model_consistency_check(project_id: str, as_of: str | None = None):
    """Report typed scientific graph inconsistencies without deleting or promoting state."""
    return await call(epistemics().world_model_consistency_check, project_id, as_of)


@mcp.tool()
async def research_state_update(project_id: str, update: dict):
    """Persist the evidence-backed research focus that guides the next discriminating test."""
    return await call(research().project_state_update, project_id, update)


@mcp.tool()
async def world_model_create(project_id: str, name: str, scope: str):
    """Create the native versioned mechanistic model owned by Research OS."""
    return await call(brain().world_model_create, project_id, name, scope)


@mcp.tool()
async def world_model_get(world_model_id: str, as_of: str | None = None):
    """Retrieve a WorldModel with its typed nodes, causal edges, and version history."""
    return await call(brain().world_model_get, world_model_id, as_of)


@mcp.tool()
async def world_entity_create(
    world_model_id: str,
    kind: str,
    name: str,
    description: str,
    attributes: dict | None = None,
):
    """Add a typed scientific entity to a WorldModel and create a new model version."""
    return await call(
        brain().world_entity_create,
        world_model_id,
        kind,
        name,
        description,
        attributes,
    )


@mcp.tool()
async def causal_edge_create(
    world_model_id: str,
    source_id: str,
    target_id: str,
    relation: str,
    status: str,
    supporting_ids: list[str] | None = None,
    against_ids: list[str] | None = None,
    unresolved_prediction_ids: list[str] | None = None,
    decision_id: str | None = None,
    scope: str | dict | None = None,
    supporting_evidence_family_ids: list[str] | None = None,
    contradicting_evidence_family_ids: list[str] | None = None,
):
    """Create a provenance-bearing causal edge and version the WorldModel."""
    return await call(
        brain().causal_edge_create,
        world_model_id,
        source_id,
        target_id,
        relation,
        status,
        supporting_ids,
        against_ids,
        unresolved_prediction_ids,
        decision_id,
        scope,
        supporting_evidence_family_ids,
        contradicting_evidence_family_ids,
    )


@mcp.tool()
async def causal_edge_update(
    edge_id: str,
    status: str,
    rationale: str,
    supporting_ids: list[str] | None = None,
    against_ids: list[str] | None = None,
    decision_id: str | None = None,
    scope: str | dict | None = None,
    supporting_evidence_family_ids: list[str] | None = None,
    contradicting_evidence_family_ids: list[str] | None = None,
):
    """Apply an evidence-linked causal-edge transition and preserve its model version delta."""
    return await call(
        brain().causal_edge_update,
        edge_id,
        status,
        rationale,
        supporting_ids,
        against_ids,
        decision_id,
        scope,
        supporting_evidence_family_ids,
        contradicting_evidence_family_ids,
    )


@mcp.tool()
async def research_agenda_create(project_id: str, name: str):
    """Create the project ResearchAgenda in canonical PostgreSQL state."""
    return await call(brain().agenda_create, project_id, name)


@mcp.tool()
async def research_agenda_item_create(
    agenda_id: str,
    question: str,
    importance: float,
    uncertainty: float,
    scientific_scope: str,
    blocking_hypothesis_ids: list[str] | None = None,
    related_anomaly_ids: list[str] | None = None,
    related_contradiction_ids: list[str] | None = None,
    candidate_experiments: list[dict] | None = None,
    reproduction_required: bool = False,
):
    """Persist one scored scientific unknown and its candidate experiments."""
    return await call(
        brain().agenda_item_create,
        agenda_id,
        question,
        importance,
        uncertainty,
        scientific_scope,
        blocking_hypothesis_ids,
        related_anomaly_ids,
        related_contradiction_ids,
        candidate_experiments,
        reproduction_required,
    )


@mcp.tool()
async def research_agenda_item_update(agenda_item_id: str, status: str, rationale: str):
    """Apply an explicit provenance-bearing AgendaItem status transition."""
    return await call(brain().agenda_item_update, agenda_item_id, status, rationale)


@mcp.tool()
async def hypothesis_portfolio_get(project_id: str):
    """Read the durable portfolio, or preview its current contents before the first brain step."""
    return await call(brain().portfolio_get, project_id)


@mcp.tool()
async def brain_step(project_id: str, as_of: str | None = None):
    """Persist a current decision, or return a non-mutating decision at historical cutoff as_of."""
    if as_of is None and settings.gpu_lab_embedding_provider != "disabled":
        await call(embeddings().refresh_project, project_id)
    result = await call(brain().brain_step, project_id, as_of, as_of is None)
    if "error" in result:
        return result
    return _compact_brain_step(result)


def _execution_action_fingerprint(
    experiment_id: str,
    command: str,
    working_directory: str,
    env: dict[str, str] | None,
    python_env: str | None,
) -> str:
    action = {
        "experiment_id": experiment_id,
        "command": command,
        "working_directory": working_directory,
        "env": env or {},
        "python_env": python_env,
    }
    return hashlib.sha256(
        json.dumps(action, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@mcp.tool()
async def research_decision_create(
    project_id: str,
    experiment_id: str,
    command: str,
    working_directory: str = ".",
    env: dict[str, str] | None = None,
    python_env: str | None = None,
):
    """Create a compact execution handoff containing the required decision_id.

    Full Brain-step detail remains durable and can be read with
    research_object_get. Pass decision_id to research_experiment_execute with
    the same bound command.
    """
    step = await call(brain().brain_step, project_id)
    if "error" in step:
        return step
    fingerprint = _execution_action_fingerprint(
        experiment_id, command, working_directory, env, python_env
    )
    binding = await call(
        brain().execution_decision_bind,
        experiment_id,
        step["decision_id"],
        fingerprint,
    )
    handoff = {
        "decision_id": step["decision_id"],
        "experiment_id": experiment_id,
        "action_fingerprint": fingerprint,
        "next_tool": "research_experiment_execute",
        "detail_hint": "Use research_object_get with decision_id for the complete persisted Brain trace.",
    }
    if "error" in binding:
        return {**handoff, "execution_binding_error": binding["error"]}
    return {**handoff, "execution_binding": binding["data"]["execution_binding"]}


@mcp.tool()
async def brain_decision_approve(decision_id: str, approver: str, rationale: str):
    """Record explicit human approval for the exact selected Brain action."""
    return await call(brain().decision_approve, decision_id, approver, rationale)


@mcp.tool()
async def execution_decision_bind(
    experiment_id: str,
    decision_id: str,
    command: str,
    working_directory: str = ".",
    env: dict[str, str] | None = None,
    python_env: str | None = None,
):
    """Bind a compatible ResearchDecision to the exact executable request.

    This is safe to use for a RESERVED run whose backing local job is absent:
    it changes only decision provenance, and the caller must still retry the
    canonical execution with its original execution_attempt_uuid.
    """
    fingerprint = _execution_action_fingerprint(
        experiment_id, command, working_directory, env, python_env
    )
    return await call(
        brain().execution_decision_bind,
        experiment_id,
        decision_id,
        fingerprint,
    )


@mcp.tool()
async def legacy_run_provenance_repair(run_id: str, agenda_item_id: str, rationale: str):
    """Reconstruct inspect-only decision provenance for one completed legacy ExperimentRun."""
    return await call(brain().legacy_run_provenance_repair, run_id, agenda_item_id, rationale)


@mcp.tool()
async def legacy_reserved_run_abandon(
    run_id: str, rationale: str, technical_non_scientific: bool = False
):
    """Cancel an unsubmitted reservation only after verifying its local job is absent.

    A linked ResearchDecision requires explicit technical_non_scientific=true.
    The resulting cancellation is operational bookkeeping, never evidence.
    The flag is intentionally not inferred from the error text.
    """
    run = await call(research().object_get, run_id)
    if "error" in run:
        return run
    is_abandoned_replay = (
        run.get("status") == "cancelled"
        and run.get("data", {}).get("legacy_abandonment", {}).get("verified_missing_backing_job")
        is True
    )
    if run.get("kind") != "ExperimentRun" or (
        run.get("status") != "RESERVED" and not is_abandoned_replay
    ):
        return {"error": {"type": "LEGACY_RUN_NOT_ABANDONABLE", "message": run.get("status")}}
    job_id = run.get("data", {}).get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return {"error": {"type": "LEGACY_RUN_JOB_MISSING", "message": run_id}}
    if not settings.gpu_lab_enable_local_runner:
        return {"error": {"type": "LOCAL_RUNNER_REQUIRED", "message": "Cannot verify the backing job"}}
    job = await call(local.job_status, job_id)
    if "error" not in job or job["error"].get("type") != "JOB_NOT_FOUND":
        return {
            "error": {
                "type": "LEGACY_RUN_BACKING_JOB_NOT_PROVEN_ABSENT",
                "message": "The local job exists or could not be verified absent",
            }
        }
    return await call(brain().legacy_reserved_run_abandon, run_id, job_id, rationale, technical_non_scientific)


@mcp.tool()
async def brain_result_assess(
    run_id: str,
    decision_id: str,
    hypothesis_id: str,
    agenda_item_id: str,
    prediction_outcome: str,
    guard_condition_outcome: str,
    condition_evaluations: dict[str, bool],
    evidence_supporting: list[str],
    evidence_against: list[str],
    unexpected_observations: list[str],
    alternative_explanations: list[str],
    scope: str | dict,
    hypothesis_transition: str,
    rationale: str,
    causal_edge_id: str | None = None,
    causal_edge_status: str | None = None,
    actual_information_gain: str = "MEDIUM",
    guard_passed: bool | None = None,
    matched_control_passed: bool | None = None,
):
    """Inspect a real result and explicitly update evidence, belief, agenda, and WorldModel."""
    return await call(
        brain().result_assess,
        run_id=run_id,
        decision_id=decision_id,
        hypothesis_id=hypothesis_id,
        agenda_item_id=agenda_item_id,
        prediction_outcome=prediction_outcome,
        guard_condition_outcome=guard_condition_outcome,
        condition_evaluations=condition_evaluations,
        evidence_supporting=evidence_supporting,
        evidence_against=evidence_against,
        unexpected_observations=unexpected_observations,
        alternative_explanations=alternative_explanations,
        scope=scope,
        hypothesis_transition=hypothesis_transition,
        rationale=rationale,
        causal_edge_id=causal_edge_id,
        causal_edge_status=causal_edge_status,
        actual_information_gain=actual_information_gain,
        guard_passed=guard_passed,
        matched_control_passed=matched_control_passed,
    )


@mcp.tool()
async def claim_create(
    project_id: str, statement: str, scope: str, evidence_ids: list[str] | None = None
):
    evidence = evidence_ids or []
    for evidence_id in evidence:
        item = await call(research().object_get, evidence_id)
        if "error" in item:
            return item
        if str(item["project_id"]) != project_id or item["kind"] != "EvidenceUnit":
            return {"error": {"type": "INVALID_CLAIM_EVIDENCE", "message": evidence_id}}
    result = await call(
        research().object_create,
        project_id,
        "Claim",
        {"statement": statement, "scope": scope, "evidence_ids": evidence},
        "CLAIM_CREATED",
    )
    if "error" not in result:
        for evidence_id in evidence:
            await call(research().edge_create, evidence_id, result["id"], "SUPPORTS")
    return result


@mcp.tool()
async def claim_search(
    project_id: str, query: str, limit: int = 25, as_of: str | None = None
):
    """Find claims by their persisted statement, scope, or attached evidence identifiers."""
    return await call(
        research().search, project_id, query, "Claim", min(max(limit, 1), 100), as_of
    )


@mcp.tool()
async def claim_get_evidence(claim_id: str, as_of: str | None = None):
    """Retrieve the exact evidence units cited by a claim, never an uncited summary."""
    claim = research().object_get(claim_id, as_of)
    if claim["kind"] != "Claim":
        return {"error": {"type": "NOT_A_CLAIM", "message": claim_id}}
    units = [
        research().object_get(item, as_of) for item in claim["data"].get("evidence_ids", [])
    ]
    return {"claim": claim, "evidence": units}


@mcp.tool()
async def claim_compare(
    claim_id: str, other_claim_id: str, as_of: str | None = None
):
    """Compare scope and evidence overlap without inferring agreement from prose alone."""
    first = research().object_get(claim_id, as_of)
    second = research().object_get(other_claim_id, as_of)
    if first["kind"] != "Claim" or second["kind"] != "Claim":
        return {"error": {"type": "NOT_A_CLAIM", "message": "Both inputs must be Claim IDs"}}
    if first["project_id"] != second["project_id"]:
        return {
            "error": {"type": "RESEARCH_PROJECT_MISMATCH", "message": "Claims must share a project"}
        }
    first_evidence = set(first["data"].get("evidence_ids", []))
    second_evidence = set(second["data"].get("evidence_ids", []))
    return {
        "claim": first,
        "other_claim": second,
        "same_scope": first["data"].get("scope") == second["data"].get("scope"),
        "shared_evidence_ids": sorted(first_evidence & second_evidence),
        "only_claim_evidence_ids": sorted(first_evidence - second_evidence),
        "only_other_evidence_ids": sorted(second_evidence - first_evidence),
    }


@mcp.tool()
async def contradiction_create(project_id: str, claim_id: str, other_claim_id: str, rationale: str):
    """Record an explicit conflict between two scoped claims instead of silently averaging them."""
    if claim_id == other_claim_id:
        return {
            "error": {
                "type": "INVALID_CONTRADICTION",
                "message": "A claim cannot contradict itself",
            }
        }
    claim, other_claim = research().object_get(claim_id), research().object_get(other_claim_id)
    if claim["kind"] != "Claim" or other_claim["kind"] != "Claim":
        return {"error": {"type": "NOT_A_CLAIM", "message": "Both inputs must be Claim IDs"}}
    if claim["project_id"] != other_claim["project_id"] or str(claim["project_id"]) != project_id:
        return {"error": {"type": "RESEARCH_PROJECT_MISMATCH", "message": project_id}}
    result = await call(
        research().object_create,
        project_id,
        "Contradiction",
        {"claim_id": claim_id, "other_claim_id": other_claim_id, "rationale": rationale},
        "CONTRADICTION_CREATED",
    )
    if "error" not in result:
        await call(research().edge_create, claim_id, result["id"], "CONTRADICTED_BY")
        await call(research().edge_create, other_claim_id, result["id"], "CONTRADICTED_BY")
    return result


@mcp.tool()
async def hypothesis_create(
    project_id: str,
    mechanism: str,
    prediction: str,
    kill_condition: str,
    parent_ids: list[str] | None = None,
    scientific_difference: str | None = None,
):
    """Create a falsifiable hypothesis after screening related active and failed mechanisms."""
    parents = parent_ids or []
    for parent_id in parents:
        parent = research().object_get(parent_id)
        if str(parent["project_id"]) != project_id or parent["kind"] != "Hypothesis":
            return {"error": {"type": "INVALID_HYPOTHESIS_PARENT", "message": parent_id}}
    related = await call(research().related_hypotheses, project_id, mechanism)
    if isinstance(related, dict) and "error" in related:
        return related
    related = [item for item in related if item["containment_similarity"] >= 0.35]
    close_dead = [
        item
        for item in related
        if item["containment_similarity"] >= 0.6
        and (item["status"] == "REFUTED" or item["kind"] == "NegativeResult")
    ]
    if close_dead and not scientific_difference:
        return {
            "error": {
                "type": "HYPOTHESIS_RESEMBLES_NEGATIVE_KNOWLEDGE",
                "message": "Provide scientific_difference explaining the changed assumption.",
                "related_ids": [str(item["id"]) for item in close_dead],
            }
        }
    result = await call(
        research().object_create,
        project_id,
        "Hypothesis",
        {
            "mechanism": mechanism,
            "prediction": prediction,
            "kill_condition": kill_condition,
            "parent_ids": parents,
            "scientific_difference": scientific_difference,
            "related_hypothesis_ids": [str(item["id"]) for item in related],
        },
        "HYPOTHESIS_CREATED",
    )
    if "error" not in result:
        for parent_id in parents:
            await call(research().edge_create, parent_id, result["id"], "PARENT_OF")
        for item in related:
            await call(research().edge_create, item["id"], result["id"], "RELATED_TO")
    return result


@mcp.tool()
async def hypothesis_related(project_id: str, mechanism: str, limit: int = 10):
    """Retrieve related active and failed mechanisms before proposing a new descendant."""
    return await call(research().related_hypotheses, project_id, mechanism, min(max(limit, 1), 50))


@mcp.tool()
async def hypothesis_niche_create(
    project_id: str, name: str, description: str, diversity_signature: dict
):
    """Create a mechanistic hypothesis niche without assigning scientific truth or probability."""
    return await call(qd().niche_create, project_id, name, description, diversity_signature)


@mcp.tool()
async def hypothesis_niche_list(project_id: str):
    """List the project's mechanistic niches and explicitly selected active representatives."""
    return await call(qd().niche_list, project_id)


@mcp.tool()
async def research_operator_status():
    """Report whether typed model-backed research operators are isolated and available."""
    if settings.gpu_lab_research_operator_provider == "disabled":
        return {
            "status": "disabled",
            "provider": "none",
            "scientific_truth_access": False,
        }
    result = await call(
        HttpLiteratureProvider(
            settings.gpu_lab_literature_worker_url,
            settings.gpu_lab_literature_worker_token or "",
        ).health
    )
    if "error" in result:
        return {
            "status": "unavailable",
            "provider": settings.gpu_lab_research_operator_provider,
            "error": result["error"],
            "scientific_truth_access": False,
        }
    return {
        "status": "ready" if result.get("status") == "ready" else "unavailable",
        "provider": settings.gpu_lab_research_operator_provider,
        "worker": result,
        "scientific_truth_access": False,
    }


@mcp.tool()
async def research_hypothesis_generate(
    project_id: str, agenda_item_id: str, persist: bool = False
):
    """Generate 3-5 typed advisory mechanisms, then run dead-memory and QD screening."""
    return await call(
        research_operators().generate_hypotheses,
        project_id,
        agenda_item_id,
        persist=persist,
    )


@mcp.tool()
async def research_null_model_critique(
    project_id: str, target_claim: str, context: dict
):
    """Generate typed cheap null explanations and controls without scientific promotion."""
    return await call(
        research_operators().null_model_critique,
        project_id,
        target_claim,
        context,
    )


@mcp.tool()
async def research_operator_critique(operator_name: str, project_id: str, context: dict):
    """Run one typed advisory Mechanism, Design, Proximity, or Novelty critic."""
    return await call(
        research_operators().critique,
        operator_name,
        project_id,
        context,
    )


async def _hypothesis_draft_with_embedding(draft: dict) -> dict:
    if draft.get("embedding") or settings.gpu_lab_embedding_provider == "disabled":
        return draft
    try:
        service = embeddings()
        text = service.canonical_text({"kind": "Hypothesis", "data": draft})
        vectors = await service.provider.embed_texts([text])
        return {**draft, "embedding": vectors[0]}
    except Exception:  # noqa: BLE001 - QD remains available if the secondary index fails
        return draft


@mcp.tool()
async def hypothesis_qd_screen(project_id: str, draft: dict):
    """Compare a typed draft with active/dead ideas using retrieval and structured mechanisms."""
    return await call(qd().screen, project_id, await _hypothesis_draft_with_embedding(draft))


@mcp.tool()
async def hypothesis_qd_create(project_id: str, draft: dict):
    """Persist a screened hypothesis with niche, ancestry, similarity, and scientific difference."""
    created = await call(qd().create, project_id, await _hypothesis_draft_with_embedding(draft))
    if "error" not in created and settings.gpu_lab_embedding_provider != "disabled":
        created["automatic_embedding"] = await call(
            embeddings().refresh_object, created["id"]
        )
    return created


@mcp.tool()
async def hypothesis_niche_set_best(niche_id: str, hypothesis_id: str, rationale: str):
    """Select a surviving niche representative for test priority, never as proof of truth."""
    return await call(qd().niche_set_best, niche_id, hypothesis_id, rationale)


@mcp.tool()
async def experiment_branch_create(
    project_id: str, hypothesis_id: str, objective: str, budget: dict
):
    """Create a deterministic experiment branch for one surviving hypothesis and explicit budget."""
    return await call(branches().create, project_id, hypothesis_id, objective, budget)


@mcp.tool()
async def experiment_branch_node_add(
    branch_id: str, draft: dict, parent_node_id: str | None = None
):
    """Add a scored, predicted branch action without executing or interpreting results."""
    return await call(branches().node_add, branch_id, draft, parent_node_id)


@mcp.tool()
async def experiment_branch_get(branch_id: str):
    """Retrieve the branch, nodes, typed relations, and comparative lessons."""
    return await call(branches().get, branch_id)


@mcp.tool()
async def experiment_branch_next(branch_id: str):
    """Select inspect, recover, execute, compare, or complete using a deterministic policy."""
    return await call(branches().next_action, branch_id)


@mcp.tool()
async def experiment_branch_result_record(
    node_id: str,
    run_id: str,
    result: dict,
    scientific_interpretation: str,
    actual_cost: dict,
    information_gained: str,
):
    """Attach only an already-inspected canonical run result to its matching branch node."""
    return await call(
        branches().record_result,
        node_id,
        run_id,
        result,
        scientific_interpretation,
        actual_cost,
        information_gained,
    )


@mcp.tool()
async def experiment_branch_compare(
    branch_id: str, node_a_id: str, node_b_id: str, lesson: dict
):
    """Persist a confound-aware ComparativeLesson across two inspected branch results."""
    return await call(branches().compare, branch_id, node_a_id, node_b_id, lesson)


@mcp.tool()
async def research_progress(project_id: str):
    """Return transparent scientific-progress counts and information-per-GPU-hour when measurable."""
    return await call(meta_research().progress, project_id)


@mcp.tool()
async def meta_review(project_id: str):
    """Persist a process-only MetaLesson and assess readiness for bounded campaign automation."""
    return await call(meta_research().meta_review, project_id)


@mcp.tool()
async def improvement_opportunities(project_id: str):
    """Detect recurring, evidence-backed scientist-behavior weaknesses."""
    return await call(meta_controller().detect_opportunities, project_id)


@mcp.tool()
async def meta_state_get(project_id: str):
    """Return compact meta-science state and active policy health."""
    return await call(meta_controller().state_get, project_id)


@mcp.tool()
async def policy_health_report(project_id: str):
    """Return the compact v3 production-policy health report."""
    return await call(meta_controller().policy_health_report, project_id)


@mcp.tool()
async def benchmark_gap_prepare(project_id: str, benchmark_gap_id: str):
    """Prepare a future-only benchmark authoring proposal; it cannot alter current evaluation."""
    return await call(meta_controller().benchmark_gap_prepare, project_id, benchmark_gap_id)


@mcp.tool()
async def meta_research_roi(project_id: str):
    """Return bounded meta-research cost and observed-yield metrics."""
    return await call(meta_controller().meta_research_roi, project_id)


@mcp.tool()
async def meta_research_run_once(project_id: str):
    """Run at most one bounded autonomous meta-research campaign."""
    return await run_meta_research(project_id)


@mcp.tool()
async def policy_regressions_check(project_id: str):
    """Inspect post-promotion evidence and apply only configured scoped rollback."""
    return await call(meta_controller().monitor_promotions, project_id)


@mcp.tool()
async def autonomy_config_get(project_id: str):
    """Return durable policy-autonomy mode and campaign budgets."""
    return await call(meta_controller().config_get, project_id)


@mcp.tool()
async def autonomy_config_update(project_id: str, update: dict[str, Any]):
    """Update bounded autonomy settings; default mode remains advisory."""
    return await call(meta_controller().config_update, project_id, update)


@mcp.tool()
async def policy_model_change_detect(project_id: str, provider: str, model: str):
    """Create a deduplicated compatibility-evaluation opportunity after a model change."""
    return await call(meta_controller().model_change_detect, project_id, provider, model)


@mcp.tool()
async def provider_adapter_evaluate(project_id: str, candidate_id: str, evidence_ids: list[str], live_result: str):
    """Record durable live PASS/FAIL evidence for a proposed provider adapter."""
    return await call(policy_lab().provider_adapter_evaluate, project_id, candidate_id, evidence_ids, live_result)


@mcp.tool()
async def provider_adapter_promote(project_id: str, candidate_id: str):
    """Promote only a provider adapter that survived live cross-model evaluation."""
    return await call(policy_lab().provider_adapter_promote, project_id, candidate_id)


@mcp.tool()
async def meta_lesson_list(project_id: str):
    """List durable research-process lessons without treating them as scientific evidence."""
    return await call(meta_research().list_lessons, project_id)


@mcp.tool()
async def research_null_model_create(project_id: str, null_model: dict):
    """Register an explicit alternative explanation; it cannot itself promote scientific truth."""
    return await call(strategy().null_model_create, project_id, null_model)


@mcp.tool()
async def research_null_model_test(
    null_model_id: str,
    outcome: str,
    evidence_family_ids: list[str],
    rationale: str,
):
    """Record an evidence-backed null-model result before any related causal promotion."""
    return await call(
        strategy().null_model_test,
        null_model_id,
        outcome,
        evidence_family_ids,
        rationale,
    )


@mcp.tool()
async def research_decision_outcome_assess(
    decision_id: str, assessment: dict, domain: str | None = None
):
    """Persist an outcome, then run one bounded event-driven meta-science pass."""
    result = await call(strategy().decision_outcome_assess, decision_id, assessment, domain)
    if "error" in result:
        return result
    project_id = str(result["outcome"]["project_id"])
    meta_result = await run_meta_research(project_id)
    return {**result, "meta_research": meta_result}


@mcp.tool()
async def research_strategy_list(project_id: str | None = None, as_of: str | None = None):
    """List project, domain, and global research-process patterns with provenance."""
    return await call(strategy().strategy_list, project_id, as_of)


@mcp.tool()
async def research_strategy_dataset_export(project_id: str | None = None):
    """Export versioned observational policy-transition data for offline future evaluation."""
    return await call(strategy().dataset_export, project_id)


@mcp.tool()
async def strategy_learning_eligibility(decision_id: str):
    """Explain whether one decision/outcome may contribute to production strategy memory."""
    return await call(strategy().strategy_learning_eligibility, decision_id)


@mcp.tool()
async def decision_epistemic_audit(decision_id: str):
    """Audit one persisted decision's scientific role, closure, and strategy-learning eligibility."""
    return await call(strategy().decision_epistemic_audit, decision_id)


@mcp.tool()
async def historical_reclassification_report(project_id: str | None = None):
    """Apply idempotent v2.1 epistemic classifications and return aggregate counts."""
    return await call(research().epistemic_reclassification, project_id)


@mcp.tool()
async def experiment_plan_register(project_id: str, hypothesis_id: str, plan: dict):
    """Preregister a frozen experiment plan before results are inspected."""
    required = {
        "research_question",
        "prediction",
        "alternative_hypotheses",
        "intervention",
        "control",
        "primary_metric",
        "secondary_metrics",
        "expected_direction",
        "pass_condition",
        "fail_condition",
        "interpretation_if_pass",
        "interpretation_if_fail",
        "estimated_runtime_minutes",
        "estimated_gpu_cost_usd",
    }
    missing = sorted(required - set(plan))
    if missing:
        return {
            "error": {
                "type": "EXPERIMENT_PLAN_INCOMPLETE",
                "message": f"Missing: {', '.join(missing)}",
            }
        }
    hypothesis = research().object_get(hypothesis_id)
    if str(hypothesis["project_id"]) != project_id or hypothesis["kind"] != "Hypothesis":
        return {"error": {"type": "INVALID_EXPERIMENT_HYPOTHESIS", "message": hypothesis_id}}
    result = await call(
        research().object_create,
        project_id,
        "Experiment",
        {"hypothesis_id": hypothesis_id, "plan": plan, "frozen": True},
        "EXPERIMENT_REGISTERED",
    )
    if "error" not in result:
        prediction = await call(
            research().object_create,
            project_id,
            "Prediction",
            {
                "hypothesis_id": hypothesis_id,
                "experiment_id": result["id"],
                "statement": plan["prediction"],
                "pass_condition": plan["pass_condition"],
                "fail_condition": plan["fail_condition"],
                "frozen": True,
            },
            "PREDICTION_REGISTERED",
        )
        await call(research().edge_create, hypothesis_id, result["id"], "TESTED_BY")
        if "error" not in prediction:
            await call(research().edge_create, hypothesis_id, prediction["id"], "PREDICTS")
            await call(research().edge_create, prediction["id"], result["id"], "TESTED_BY")
            result["prediction"] = prediction
    return result


@mcp.tool()
async def experiment_priority(
    scientific_importance: float,
    hypothesis_discrimination: float,
    expected_information_gain: float,
    compute_cost: float,
    implementation_cost: float,
    execution_risk: float,
):
    """Rank a proposed experiment by explicit information gain per total cost; it does not execute it."""
    values = [
        scientific_importance,
        hypothesis_discrimination,
        expected_information_gain,
        compute_cost,
        implementation_cost,
        execution_risk,
    ]
    if any(value < 0 for value in values):
        return {
            "error": {
                "type": "INVALID_EXPERIMENT_ECONOMICS",
                "message": "Inputs must be non-negative",
            }
        }
    denominator = max(compute_cost * implementation_cost * max(execution_risk, 0.01), 0.01)
    return {
        "priority": scientific_importance
        * hypothesis_discrimination
        * expected_information_gain
        / denominator,
        "formula": "importance * discrimination * information_gain / max(compute_cost * implementation_cost * max(execution_risk, 0.01), 0.01)",
        "recommendation": "Prefer the smallest discriminating test before additional training.",
    }


@mcp.tool()
async def research_events(project_id: str, limit: int = 100, as_of: str | None = None):
    """Retrieve immutable events, optionally bounded at an inclusive historical cutoff."""
    return await call(research().events, project_id, min(max(limit, 1), 100), as_of)


@mcp.tool()
async def research_temporal_finalize_pending():
    """Recover deferred temporal commit markers without mutating historical read operations."""
    return await call(research().temporal_finalize_pending)


@mcp.tool()
async def research_assess(object_id: str, status: str, rationale: str):
    """Update a claim or hypothesis only with an explicit evidence-backed assessment event."""
    return await call(research().assess, object_id, status, rationale)


@mcp.tool()
async def paper_ingest(
    project_id: str, title: str, url: str, card: dict, version: str | None = None
):
    """Persist a paper card; claims must remain separate evidence-backed objects."""
    return await call(
        research().object_create,
        project_id,
        "Paper",
        {"title": title, "url": url, "version": version, "card": card},
        "PAPER_INGESTED",
    )


@mcp.tool()
async def paper_search(
    project_id: str, query: str, limit: int = 25, as_of: str | None = None
):
    return await call(
        research().search, project_id, query, "Paper", min(max(limit, 1), 100), as_of
    )


@mcp.tool()
async def paper_get(paper_id: str, as_of: str | None = None):
    """Return the persisted paper card and provenance URL/version."""
    paper = research().object_get(paper_id, as_of)
    if paper["kind"] != "Paper":
        return {"error": {"type": "NOT_A_PAPER", "message": paper_id}}
    return paper


@mcp.tool()
async def paper_evidence_create(project_id: str, paper_id: str, text: str, locator: dict):
    """Store a source passage with page/section/figure provenance before making a claim."""
    paper = research().object_get(paper_id)
    if str(paper["project_id"]) != project_id or paper["kind"] != "Paper":
        return {"error": {"type": "INVALID_EVIDENCE_PAPER", "message": paper_id}}
    result = await call(
        research().object_create,
        project_id,
        "EvidenceUnit",
        {"paper_id": paper_id, "text": text, "locator": locator},
        "EVIDENCE_UNIT_CREATED",
    )
    if "error" not in result:
        await call(research().edge_create, paper_id, result["id"], "CONTAINS_EVIDENCE")
    return result


@mcp.tool()
async def paper_evidence_search(
    project_id: str, query: str, limit: int = 25, as_of: str | None = None
):
    return await call(
        research().search,
        project_id,
        query,
        "EvidenceUnit",
        min(max(limit, 1), 100),
        as_of,
    )


@mcp.tool()
async def research_embedding_store(object_id: str, embedding: list[float]):
    """Attach a caller-provided embedding to a research object for pgvector retrieval."""
    return await call(research().embedding_set, object_id, embedding)


@mcp.tool()
async def research_embedding_status(project_id: str):
    """Report automatic embedding coverage without treating the secondary index as truth."""
    if settings.gpu_lab_embedding_provider == "disabled":
        return {
            "status": "disabled",
            "fallback": "structured_and_lexical_retrieval",
        }
    return await call(embeddings().project_status, project_id)


@mcp.tool()
async def research_embedding_refresh(project_id: str):
    """Generate or refresh source-hashed embeddings for canonical scientific objects."""
    if settings.gpu_lab_embedding_provider == "disabled":
        return {
            "status": "disabled",
            "fallback": "structured_and_lexical_retrieval",
        }
    return await call(embeddings().refresh_project, project_id)


@mcp.tool()
async def research_embedding_search(
    project_id: str, query: str, kind: str | None = None, limit: int = 25
):
    """Search the automatic vector index, falling back to lexical retrieval on provider failure."""
    if settings.gpu_lab_embedding_provider == "disabled":
        return {
            "mode": "lexical_fallback",
            "hits": await call(
                research().search, project_id, query, kind, min(max(limit, 1), 100)
            ),
        }
    return await call(
        embeddings().search, project_id, query, kind, min(max(limit, 1), 100)
    )


@mcp.tool()
async def research_semantic_search(
    project_id: str,
    embedding: list[float],
    kind: str | None = None,
    limit: int = 25,
    as_of: str | None = None,
):
    """Use pgvector retrieval, excluding object revisions and embeddings newer than as_of."""
    return await call(
        research().semantic_search,
        project_id,
        embedding,
        kind,
        min(max(limit, 1), 100),
        as_of,
    )


@mcp.tool()
async def paper_ask(project_id: str, question: str, limit: int = 8):
    """Return retrieved source passages for a question; this is evidence retrieval, not a conclusion."""
    evidence, seen = [], set()
    for term in ResearchStore.terms(question):
        matches = await call(
            research().search, project_id, term, "EvidenceUnit", min(max(limit, 1), 25)
        )
        if isinstance(matches, dict) and "error" in matches:
            return matches
        for match in matches:
            if match["id"] not in seen:
                evidence.append(match)
                seen.add(match["id"])
                if len(evidence) >= min(max(limit, 1), 25):
                    break
    return {
        "question": question,
        "evidence": evidence,
        "warning": "Retrieved passages are not canonical truth. Create a scoped Claim with explicit evidence IDs before drawing a conclusion.",
    }


@mcp.tool()
async def literature_provider_status():
    """Report isolated PaperQA worker health without exposing model credentials."""
    result = {
        "configured_provider": settings.gpu_lab_literature_provider,
        "worker_url": settings.gpu_lab_literature_worker_url,
        "canonical_truth_owner": "PostgreSQL Research OS",
    }
    if settings.gpu_lab_literature_provider != "paperqa-http":
        return {**result, "status": "disabled"}
    worker = await call(literature().provider.health)
    if "error" not in worker:
        result["worker"] = worker
        result["status"] = "ready"
    else:
        result["status"] = "unavailable"
        result["error"] = worker["error"]
    return result


@mcp.tool()
async def literature_search(query: str, filters: dict | None = None):
    """Search through the optional literature engine without mutating scientific state."""
    try:
        provider = literature().provider
    except GPUError as exc:
        return exc.response()
    result = await call(provider.search, query, filters)
    return result.model_dump(mode="json") if isinstance(result, BaseModel) else result


@mcp.tool()
async def literature_ask(question: str, papers: list[str] | None = None):
    """Ask PaperQA for citation candidates; its answer is never canonical scientific truth."""
    try:
        provider = literature().provider
    except GPUError as exc:
        return exc.response()
    result = await call(provider.ask, question, papers)
    return result.model_dump(mode="json") if isinstance(result, BaseModel) else result


@mcp.tool()
async def literature_gather(
    project_id: str,
    question: str,
    papers: list[str] | None = None,
    claim_statement: str | None = None,
    claim_scope: str | None = None,
):
    """Validate and persist PaperQA evidence candidates, plus an optional unresolved Claim."""
    try:
        service = literature()
    except GPUError as exc:
        return exc.response()
    return await call(
        service.gather,
        project_id,
        question,
        papers,
        claim_statement,
        claim_scope,
    )


@mcp.tool()
async def brain_literature_resolve(
    decision_id: str,
    question: str,
    papers: list[str] | None = None,
    claim_statement: str | None = None,
    claim_scope: str | None = None,
):
    """Execute a selected literature action, import candidates, and recompute the Brain decision."""
    try:
        service = literature()
    except GPUError as exc:
        return exc.response()
    return await call(
        service.resolve_decision,
        brain(),
        decision_id,
        question,
        papers,
        claim_statement,
        claim_scope,
    )


@mcp.tool()
async def executable_paper_provider_status():
    """Report isolated Paper2Agent worker health without exposing coding-agent credentials."""
    result = {
        "configured_provider": settings.gpu_lab_executable_paper_provider,
        "worker_url": settings.gpu_lab_executable_paper_worker_url,
        "canonical_truth_owner": "PostgreSQL Research OS",
    }
    if settings.gpu_lab_executable_paper_provider != "paper2agent-http":
        return {**result, "status": "disabled"}
    try:
        provider = executable_papers().provider
    except GPUError as exc:
        return exc.response()
    worker = await call(provider.health)
    if "error" in worker:
        return {**result, "status": "unavailable", "error": worker["error"]}
    return {**result, "status": worker.get("status", "ready"), "worker": worker}


@mcp.tool()
async def executable_paper_action_approve(
    project_id: str,
    action: str,
    parameters: dict,
    approver: str,
    rationale: str,
    approval_secret: str,
    ttl_minutes: int = 30,
):
    """Create a parameter-bound, expiring approval after authenticating the human approver."""
    configured = settings.gpu_lab_approval_secret or ""
    if not configured or not hmac.compare_digest(
        approval_secret.encode("utf-8", "replace"), configured.encode("utf-8", "replace")
    ):
        return {
            "error": {
                "type": "APPROVAL_AUTHENTICATION_FAILED",
                "message": "The server approval secret is missing or invalid",
            }
        }
    try:
        service = executable_papers()
    except GPUError as exc:
        return exc.response()
    return await call(
        service.approve,
        project_id,
        action,
        parameters,
        approver,
        rationale,
        ttl_minutes,
    )


@mcp.tool()
async def executable_paper_build(
    project_id: str,
    paper_id: str,
    repository: str,
    commit: str,
    tutorials: str | None = None,
    approval_id: str | None = None,
):
    """Build an executable-paper candidate in isolated Paper2Agent; may use paid model access."""
    try:
        service = executable_papers()
    except GPUError as exc:
        return exc.response()
    return await call(
        service.build, project_id, paper_id, repository, commit, tutorials, approval_id
    )


@mcp.tool()
async def executable_paper_inspect_tools(executable_paper_id: str):
    """Initialize a generated paper MCP and persist its advertised tool schemas."""
    try:
        service = executable_papers()
    except GPUError as exc:
        return exc.response()
    return await call(service.inspect_tools, executable_paper_id)


@mcp.tool()
async def executable_paper_verify(executable_paper_id: str):
    """Run integration checks without allowing a generated worker to claim real verification."""
    try:
        service = executable_papers()
    except GPUError as exc:
        return exc.response()
    return await call(service.verify, executable_paper_id)


@mcp.tool()
async def executable_paper_invoke(
    executable_paper_id: str,
    tool: str,
    args: dict,
    approval_id: str | None = None,
):
    """Invoke a verified generated tool; its output remains unassessed and non-canonical."""
    try:
        service = executable_papers()
    except GPUError as exc:
        return exc.response()
    return await call(service.invoke, executable_paper_id, tool, args, approval_id)


@mcp.tool()
async def anomaly_create(
    project_id: str,
    expected: str,
    observed: str,
    scope: str,
    priority: str = "medium",
    model: str | None = None,
    dataset: str | None = None,
    experiment_id: str | None = None,
    affected_claim_ids: list[str] | None = None,
    possible_explanations: list[str] | None = None,
):
    """Persist an unexpected observation with its scope, affected claims, and candidate explanations."""
    claims = affected_claim_ids or []
    for claim_id in claims:
        claim = research().object_get(claim_id)
        if str(claim["project_id"]) != project_id or claim["kind"] != "Claim":
            return {"error": {"type": "INVALID_ANOMALY_CLAIM", "message": claim_id}}
    result = await call(
        research().object_create,
        project_id,
        "Anomaly",
        {
            "expected": expected,
            "observed": observed,
            "scope": scope,
            "priority": priority,
            "model": model,
            "dataset": dataset,
            "experiment_id": experiment_id,
            "affected_claim_ids": claims,
            "possible_explanations": possible_explanations or [],
        },
        "ANOMALY_CREATED",
    )
    if "error" not in result:
        for claim_id in claims:
            await call(research().edge_create, result["id"], claim_id, "AFFECTS")
    return result


@mcp.tool()
async def negative_result_create(
    project_id: str,
    proposal: str,
    prediction: str,
    result: str,
    failed_assumption: str,
    revisit_condition: str,
    why_plausible: str | None = None,
    experiment_id: str | None = None,
    weakened_descendant_ids: list[str] | None = None,
):
    """Store a failed proposal and the evidence needed to avoid resurrecting it without change."""
    descendants = weakened_descendant_ids or []
    for hypothesis_id in descendants:
        hypothesis = research().object_get(hypothesis_id)
        if str(hypothesis["project_id"]) != project_id or hypothesis["kind"] != "Hypothesis":
            return {
                "error": {"type": "INVALID_NEGATIVE_RESULT_DESCENDANT", "message": hypothesis_id}
            }
    result_object = await call(
        research().object_create,
        project_id,
        "NegativeResult",
        {
            "proposal": proposal,
            "why_plausible": why_plausible,
            "prediction": prediction,
            "experiment_id": experiment_id,
            "result": result,
            "failed_assumption": failed_assumption,
            "weakened_descendant_ids": descendants,
            "revisit_condition": revisit_condition,
        },
        "NEGATIVE_RESULT_CREATED",
    )
    if "error" not in result_object:
        for hypothesis_id in descendants:
            await call(research().edge_create, result_object["id"], hypothesis_id, "WEAKENS")
    return result_object


@mcp.tool()
async def lesson_create(
    project_id: str, statement: str, evidence_ids: list[str], confounds: list[str] | None = None
):
    return await call(
        research().object_create,
        project_id,
        "Lesson",
        {"statement": statement, "evidence_ids": evidence_ids, "confounds": confounds or []},
        "LESSON_CREATED",
    )


@mcp.tool()
async def reproduction_prepare(
    project_id: str,
    paper_id: str,
    repository: str,
    commit: str | None,
    dataset: str | None,
    checkpoint: str | None,
    evaluation_command: str | None,
    reported_metric: dict | None,
    tolerance: float | None,
):
    """Register the executable provenance required before attempting reproduction."""
    paper = research().object_get(paper_id)
    if str(paper["project_id"]) != project_id or paper["kind"] != "Paper":
        return {"error": {"type": "INVALID_REPRODUCTION_PAPER", "message": paper_id}}
    data = {
        "paper_id": paper_id,
        "repository": repository,
        "commit": commit,
        "dataset": dataset,
        "checkpoint": checkpoint,
        "evaluation_command": evaluation_command,
        "reported_metric": reported_metric,
        "tolerance": tolerance,
        "status": "PREPARED",
    }
    result = await call(
        research().object_create, project_id, "Reproduction", data, "REPRODUCTION_PREPARED"
    )
    if "error" not in result:
        await call(research().edge_create, paper_id, result["id"], "HAS_REPRODUCTION")
    return result


@mcp.tool()
async def reproduction_status(reproduction_id: str):
    return await call(research().object_get, reproduction_id)


@mcp.tool()
async def reproduction_plan(paper_id: str):
    """Retrieve the prepared executable provenance and outstanding reproduction plans for a paper."""
    paper = research().object_get(paper_id)
    if paper["kind"] != "Paper":
        return {"error": {"type": "NOT_A_PAPER", "message": paper_id}}
    plans = await call(research().search, str(paper["project_id"]), paper_id, "Reproduction", 100)
    if isinstance(plans, dict) and "error" in plans:
        return plans
    return {"paper": paper, "plans": plans}


@mcp.tool()
async def reproduction_run(
    reproduction_id: str,
    command: str,
    working_directory: str = ".",
    env: dict[str, str] | None = None,
    python_env: str | None = None,
):
    """Execute a prepared reproduction with recorded environment and command provenance."""
    if not settings.gpu_lab_enable_local_runner:
        return {"error": {"type": "LOCAL_RUNNER_DISABLED"}}
    reproduction = research().object_get(reproduction_id)
    if reproduction["kind"] != "Reproduction":
        return {"error": {"type": "NOT_A_REPRODUCTION", "message": reproduction_id}}
    if reproduction["status"] not in {"ACTIVE", "PREPARED", "BLOCKED"}:
        return {"error": {"type": "REPRODUCTION_NOT_RUNNABLE", "message": reproduction["status"]}}
    job = await call(
        local.submit,
        command,
        working_directory,
        "reproduction-" + reproduction_id[:8],
        env,
        python_env,
    )
    if "error" in job:
        return job
    return await call(
        research().object_update,
        reproduction_id,
        {
            "status": "RUNNING",
            "job_id": job["job_id"],
            "command": command,
            "working_directory": working_directory,
            "environment": env or {},
            "python_env": python_env,
        },
        "RUNNING",
        "REPRODUCTION_STARTED",
    )


@mcp.tool()
async def reproduction_sync(reproduction_id: str):
    """Persist real execution logs/artifacts before evaluating a reproduction metric."""
    reproduction = research().object_get(reproduction_id)
    if reproduction["kind"] != "Reproduction":
        return {"error": {"type": "NOT_A_REPRODUCTION", "message": reproduction_id}}
    if reproduction["status"] in {"PARTIAL", "FAILED", "REPRODUCED"}:
        return {
            "id": reproduction_id,
            "status": reproduction["status"],
            "data": reproduction["data"],
            "already_final": True,
        }
    job_id = reproduction["data"].get("job_id")
    if not job_id:
        return {"error": {"type": "REPRODUCTION_MISSING_JOB", "message": reproduction_id}}
    outcome, artifacts, runtime = (
        local.job_status(job_id),
        local.artifacts(job_id),
        await local.status(),
    )
    if outcome["status"] == "running":
        return await call(
            research().object_update,
            reproduction_id,
            {"logs_tail": outcome["logs_tail"][-65536:], "runtime": runtime},
            "RUNNING",
            "REPRODUCTION_LOGS_SYNCED",
        )
    status = "FAILED" if outcome["status"] != "completed" else "PARTIAL"
    event = "REPRODUCTION_FAILED" if status == "FAILED" else "REPRODUCTION_EXECUTION_COMPLETED"
    return await call(
        research().object_update,
        reproduction_id,
        {
            "exit_code": outcome["exit_code"],
            "logs_tail": outcome["logs_tail"][-65536:],
            "artifacts": artifacts,
            "runtime": runtime,
        },
        status,
        event,
    )


@mcp.tool()
async def reproduction_compare(
    reproduction_id: str, observed_metric: float, metric_name: str | None = None
):
    """Compare a real observed metric against the prepared paper metric and tolerance."""
    reproduction = research().object_get(reproduction_id)
    if reproduction["kind"] != "Reproduction":
        return {"error": {"type": "NOT_A_REPRODUCTION", "message": reproduction_id}}
    if reproduction["status"] == "FAILED":
        return {
            "error": {
                "type": "REPRODUCTION_EXECUTION_FAILED",
                "message": "A failed run cannot be metric-compared",
            }
        }
    if reproduction["status"] != "PARTIAL":
        return {
            "error": {
                "type": "REPRODUCTION_COMPARISON_UNAVAILABLE",
                "message": "Run reproduction_sync after execution before comparing metrics",
            }
        }
    reported = reproduction["data"].get("reported_metric") or {}
    expected = reported.get("value")
    tolerance = reproduction["data"].get("tolerance")
    if not isinstance(expected, (int, float)) or not isinstance(tolerance, (int, float)):
        return {
            "error": {
                "type": "REPRODUCTION_COMPARISON_UNAVAILABLE",
                "message": "reported_metric.value and tolerance are required",
            }
        }
    difference = abs(observed_metric - expected)
    status = "REPRODUCED" if difference <= tolerance else "PARTIAL"
    return await call(
        research().object_update,
        reproduction_id,
        {
            "observed_metric": {
                "name": metric_name or reported.get("name"),
                "value": observed_metric,
            },
            "difference": difference,
        },
        status,
        "REPRODUCTION_COMPLETED",
    )


@mcp.tool()
async def research_experiment_execute(
    experiment_id: str,
    decision_id: str,
    command: str,
    working_directory: str = ".",
    env: dict[str, str] | None = None,
    python_env: str | None = None,
    execution_attempt_uuid: str | None = None,
    engineering_task_id: str | None = None,
):
    """Run a preregistered experiment with a ResearchDecision created by research_decision_create.

    Retries must reuse execution_attempt_uuid. Every response returns the
    canonical experiment_id, run_id, and job_id after identity reservation.
    """
    if not settings.gpu_lab_enable_local_runner:
        return {"error": {"type": "LOCAL_RUNNER_DISABLED"}}
    if engineering_task_id:
        readiness = await call(
            engineering().assert_ready_for_experiment,
            engineering_task_id,
            experiment_id,
        )
        if "error" in readiness:
            return readiness
    request = {
        "experiment_id": experiment_id,
        "decision_id": decision_id,
        "command": command,
        "working_directory": working_directory,
        "env": env or {},
        "python_env": python_env,
    }
    fingerprint = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    idempotency_key = execution_attempt_uuid or f"auto:{fingerprint}"
    action_fingerprint = _execution_action_fingerprint(
        experiment_id, command, working_directory, env, python_env
    )
    job_id = (
        "local_" + hashlib.sha256(f"{experiment_id}:{idempotency_key}".encode()).hexdigest()[:24]
    )
    reservation = await call(
        research().run_reserve,
        experiment_id,
        idempotency_key,
        job_id,
        fingerprint,
        {
            "executor": "local",
            "decision_id": decision_id,
            "command": command,
            "working_directory": working_directory,
            "environment": env or {},
            "python_env": python_env,
        },
    )
    if "error" in reservation:
        return reservation
    authorization = await call(
        brain().authorize_execution,
        experiment_id,
        decision_id,
        action_fingerprint,
    )
    if "error" in authorization:
        return {
            "experiment_id": reservation["experiment_id"],
            "run_id": reservation["run_id"],
            "job_id": reservation["job_id"],
            "idempotency_key": reservation["idempotency_key"],
            "status": "RESERVED",
            "authorization_error": authorization["error"],
            "retry_safe": True,
        }
    job = await call(
        local.submit,
        command,
        working_directory,
        "research-" + experiment_id[:8],
        env,
        python_env,
        reservation["job_id"],
    )
    if "error" in job:
        return {
            "experiment_id": reservation["experiment_id"],
            "run_id": reservation["run_id"],
            "job_id": reservation["job_id"],
            "idempotency_key": reservation["idempotency_key"],
            "status": "RESERVED",
            "submission_error": job["error"],
            "retry_safe": True,
        }
    if job.get("status") not in {"running", "completed", "failed", "cancelled"}:
        return {
            **reservation,
            "status": "RESERVED",
            "runner_status": job.get("status", "unknown"),
            "job": job,
            "retry_safe": True,
            "recovery_action": "RETRY_EXECUTION",
        }
    mapping = await call(research().run_mark_submitted, reservation["run_id"])
    if "error" in mapping:
        return {
            "experiment_id": reservation["experiment_id"],
            "run_id": reservation["run_id"],
            "job_id": reservation["job_id"],
            "idempotency_key": reservation["idempotency_key"],
            "status": job["status"],
            "mapping_error": mapping["error"],
            "retry_safe": True,
        }
    return {**mapping, "job": job, "retry_safe": True}


@mcp.tool()
async def research_experiment_sync(run_id: str | None = None, job_id: str | None = None):
    """Sync by run ID or job ID and always return the canonical execution mapping."""
    identifier = run_id or job_id
    if not identifier:
        return {"error": {"type": "EXECUTION_IDENTIFIER_REQUIRED"}}
    mapping = await call(research().run_resolve, identifier)
    if "error" in mapping:
        return mapping
    if run_id and job_id:
        job_mapping = await call(research().run_resolve, job_id)
        if "error" in job_mapping:
            return job_mapping
        if job_mapping["run_id"] != mapping["run_id"]:
            return {"error": {"type": "EXECUTION_IDENTIFIER_MISMATCH"}}
    canonical_run_id, canonical_job_id = mapping["run_id"], mapping["job_id"]
    outcome = None
    if mapping["status"] == "RESERVED":
        outcome = await call(local.job_status, canonical_job_id)
        if "error" in outcome:
            return {
                **mapping,
                "retry_safe": True,
                "recovery_action": "RETRY_EXECUTION",
                "message": (
                    "No local process was submitted. Retry research_experiment_execute with the "
                    "same experiment_id, decision_id, command, and execution_attempt_uuid."
                ),
            }
        if outcome.get("status") not in {"running", "completed", "failed", "cancelled"}:
            return {
                **mapping,
                "runner_status": outcome.get("status", "unknown"),
                "retry_safe": True,
                "recovery_action": "RETRY_EXECUTION",
                "message": (
                    "The local job does not prove that a process launched. Retry "
                    "research_experiment_execute with the original execution arguments."
                ),
            }
        promoted = await call(research().run_mark_submitted, canonical_run_id)
        if "error" in promoted:
            return {
                **mapping,
                "retry_safe": True,
                "recovery_action": "RETRY_SYNC",
                "mapping_error": promoted["error"],
            }
        mapping = promoted
    if mapping["status"] == "RESULT_INSPECTED":
        return {**mapping, "retry_safe": True, "recovery_action": "NONE"}
    outcome = outcome or await call(local.job_status, canonical_job_id)
    if "error" in outcome:
        return {
            **mapping,
            "status": "UNKNOWN",
            "runner_status": "missing",
            "retry_safe": True,
            "recovery_action": "INSPECT_OR_RETRY",
            "runner_error": outcome["error"],
        }
    artifacts = await call(local.artifacts, canonical_job_id)
    if isinstance(artifacts, dict) and "error" in artifacts:
        return {
            **mapping,
            "status": "UNKNOWN",
            "runner_status": outcome["status"],
            "retry_safe": True,
            "recovery_action": "INSPECT_OR_RETRY",
            "artifact_error": artifacts["error"],
        }
    runtime = await call(local.status)
    if "error" in runtime:
        runtime = {"error": runtime["error"]}
    runner_status = outcome["status"]
    research_status = runner_status
    if runner_status in {"completed", "failed"}:
        for artifact in artifacts:
            recorded = await call(
                research().artifact_record,
                canonical_run_id,
                canonical_job_id,
                artifact,
            )
            if "error" in recorded:
                return {
                    **mapping,
                    "status": "UNKNOWN",
                    "runner_status": runner_status,
                    "retry_safe": True,
                    "recovery_action": "RETRY_SYNC",
                    "artifact_record_error": recorded["error"],
                }
    updated = await call(
        research().run_update,
        canonical_run_id,
        {
            "status": research_status,
            "runner_status": runner_status,
            "exit_code": outcome["exit_code"],
            "logs_tail": outcome["logs_tail"][-65536:],
            "artifacts": artifacts,
            "runtime": runtime,
        },
    )
    response = {
        "experiment_id": mapping["experiment_id"],
        "run_id": canonical_run_id,
        "job_id": canonical_job_id,
        "idempotency_key": mapping.get("idempotency_key"),
        "status": updated.get("status", research_status),
        "run": updated,
    }
    return response


@mcp.tool()
async def artifact_list(job_id: str):
    return await call(svc().artifact_list, job_id)


@mcp.tool()
async def artifact_read(job_id: str, path: str, max_bytes: int | None = None):
    return await call(svc().artifact_read, job_id, path, max_bytes)


@mcp.tool()
async def remote_exec(instance_id: str, command: str, timeout_seconds: int = 60):
    """Dangerous, bounded non-interactive debug command."""
    return await call(svc().remote_exec, instance_id, command, timeout_seconds)


local = LocalRunner(settings, svc().repo)


@mcp.tool()
async def local_status():
    """Show the mounted local research workspace and container-visible GPU."""
    return await call(local.status)


@mcp.tool()
async def local_experiment_submit(
    command: str,
    working_directory: str = ".",
    name: str | None = None,
    env: dict[str, str] | None = None,
    python_env: str | None = None,
):
    """Start a detached Linux experiment confined to the local research workspace."""
    return await call(local.submit, command, working_directory, name, env, python_env)


@mcp.tool()
async def local_env_prepare(
    name: str,
    requirements_path: str | None = None,
    python_executable: str = "python3",
):
    """Create a persistent local environment from a requirements file or directory."""
    return await call(local.env_prepare, name, requirements_path, python_executable)


@mcp.tool()
async def local_experiment_status(job_id: str):
    return await call(local.job_status, job_id)


@mcp.tool()
async def local_experiment_logs(job_id: str, tail: int = 200):
    return await call(local.logs, job_id, tail)


@mcp.tool()
async def local_artifact_list(job_id: str):
    return await call(local.artifacts, job_id)


@mcp.tool()
async def local_artifact_read(job_id: str, path: str, max_bytes: int = 65536):
    return await call(local.artifact_read, job_id, path, max_bytes)


@mcp.tool()
async def local_experiment_cancel(job_id: str):
    return await call(local.cancel, job_id)


_apply_tool_metadata()


async def gateway_status() -> dict:
    try:
        instances = await svc().gpu_list()
        provider_error = None
    except GPUError as exc:
        instances = []
        provider_error = exc.message
    local_status = None
    if settings.gpu_lab_enable_local_runner:
        try:
            local_status = await local.status()
        except GPUError as exc:
            local_status = {"error": exc.message}
    return {
        "status": "ok",
        "mcp_endpoint": "/mcp",
        "instances": instances,
        "local": local_status,
        "local_runner_enabled": settings.gpu_lab_enable_local_runner,
        "tool_count": len(mcp._tool_manager._tools),
        "provider_error": provider_error,
    }


def gateway_liveness() -> dict:
    """Return an inexpensive readiness response that never depends on a provider API."""
    return {
        "status": "ok",
        "mcp_endpoint": "/mcp",
        "local_runner_enabled": settings.gpu_lab_enable_local_runner,
        "tool_count": len(mcp._tool_manager._tools),
    }


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health(_: Request):
    return JSONResponse(gateway_liveness())


@mcp.custom_route("/status", methods=["GET"], include_in_schema=False)
async def status(_: Request):
    return JSONResponse(await gateway_status())


@mcp.custom_route("/activity", methods=["GET"], include_in_schema=False)
async def activity(_: Request):
    return JSONResponse(svc().repo.list_audit(100))


def _research_map_payload(project_id: str) -> dict[str, Any]:
    """Return the smallest evidence-bearing WorldModel view needed by the dashboard."""
    models = research().objects_list(project_id, "WorldModel", limit=1)
    if not models:
        raise GPUError("WORLD_MODEL_NOT_FOUND", f"No WorldModel exists for project {project_id}")
    graph = brain().world_model_get(str(models[0]["id"]))
    payload = {
        "world_model": _state_object_summary(graph["world_model"]),
        "nodes": [
            {
                "id": str(node["id"]),
                "kind": node["kind"],
                "status": node["status"],
                "data": {
                    key: node["data"].get(key)
                    for key in ("name", "description", "attributes")
                    if key in node["data"]
                },
            }
            for node in graph["nodes"][:200]
        ],
        "edges": [
            {
                "id": str(edge["id"]),
                "kind": edge["kind"],
                "status": edge["status"],
                "data": {
                    key: edge["data"].get(key)
                    for key in (
                        "source_id",
                        "target_id",
                        "relation",
                        "edge_status",
                        "supporting_ids",
                        "against_ids",
                        "unresolved_prediction_ids",
                    )
                    if key in edge["data"]
                },
            }
            for edge in graph["edges"][:300]
        ],
    }
    return ResearchBrain._json_safe(payload)


@mcp.custom_route("/research-map", methods=["GET"], include_in_schema=False)
async def research_map(request: Request):
    project_id = request.query_params.get("project_id", "").strip()
    if not project_id:
        return JSONResponse({"error": "project_id is required"}, status_code=400)
    try:
        return JSONResponse(_research_map_payload(project_id))
    except GPUError as error:
        return JSONResponse({"error": error.message}, status_code=404)


@mcp.custom_route("/terminal", methods=["GET"], include_in_schema=False)
async def terminal(_: Request):
    return HTMLResponse(TERMINAL_HTML)


@mcp.custom_route("/terminal/activity", methods=["GET"], include_in_schema=False)
async def terminal_activity(_: Request):
    return JSONResponse(svc().repo.list_audit(100))


@mcp.custom_route("/terminal/jobs", methods=["GET"], include_in_schema=False)
async def terminal_jobs(_: Request):
    if not settings.gpu_lab_enable_local_runner:
        return JSONResponse([])
    jobs = []
    for job in svc().repo.list_jobs(instance_id="local", limit=30):
        jobs.append(local.job_status(job.job_id))
    return JSONResponse(jobs)


@mcp.custom_route("/", methods=["GET"], include_in_schema=False)
async def dashboard(_: Request):
    return HTMLResponse(DASHBOARD_HTML)


def http_app():
    """Build the streamable HTTP app with connector header compatibility."""
    app = mcp.streamable_http_app()
    app.add_middleware(McpAcceptCompatibilityMiddleware)
    app.add_middleware(McpClientNetworkPolicyMiddleware)
    app.add_middleware(McpRequestObservabilityMiddleware)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    args = parser.parse_args()
    # Complete schema migration/recovery before accepting MCP traffic so read-only tools remain
    # read-only even on their first request.
    initialize_research_runtime()
    reconciliation = local.reconcile_jobs()
    if reconciliation["reconciled"]:
        logger.info("Reconciled persisted local jobs at startup: %s", reconciliation)
    if args.transport == "streamable-http":
        import uvicorn

        uvicorn.run(http_app(), host=settings.fastmcp_host, port=settings.fastmcp_port)
        return
    mcp.run(transport=args.transport)
