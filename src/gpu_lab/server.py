import argparse
import base64
import inspect
import re
import secrets
import time

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from .config import Settings
from .dashboard import DASHBOARD_HTML
from .errors import GPUError
from .local_runner import LocalRunner
from .research import ResearchStore
from .service import GPUService
from .terminal import TERMINAL_HTML

settings, service, research_store = Settings(), None, None
allowed_hosts = [
    host.strip()
    for host in settings.gpu_lab_allowed_hosts.split(",")
    if host.strip()
]
instructions = "Safe, structured remote GPU experiment control plane. Credentials are never returned."
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
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=[
            f"https://{host.removesuffix(':*')}" for host in allowed_hosts if ":*" not in host
        ],
    ),
)


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
        research_store = ResearchStore(settings.gpu_lab_research_database_url)
    return research_store


async def call(fn, *args, **kwargs):
    tool_name = getattr(fn, "__name__", "unknown")
    started = time.perf_counter()
    arguments = {"args": scrub(args), "kwargs": scrub(kwargs)}
    try:
        result = fn(*args, **kwargs)
        result = await result if inspect.isawaitable(result) else result
        svc().repo.audit(tool_name, arguments, "success", int((time.perf_counter() - started) * 1000))
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
    except Exception:  # noqa: BLE001 - MCP boundary must return a structured, audited failure.
        # Do not expose implementation details or leave an MCP request without an audit record.
        svc().repo.audit(tool_name, arguments, "error", int((time.perf_counter() - started) * 1000), "Internal error")
        return {"error": {"type": "INTERNAL_ERROR", "message": "Unexpected server error", "retryable": False}}


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
        return re.sub(r"(?i)(bearer|token|api[_-]?key|password)\\s*[=:]\\s*[^\\s'\\\"]+", r"\\1=[REDACTED]", value)[:4096]
    return value


@mcp.tool()
async def gpu_list():
    """List known and provider-visible GPU instances."""
    return await call(svc().gpu_list)


@mcp.tool()
async def gpu_status(instance_id: str):
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
async def research_state_get(project_id: str):
    """Retrieve canonical scientific state before serious research work."""
    return await call(research().state_get, project_id)


@mcp.tool()
async def research_state_update(project_id: str, update: dict):
    """Persist the evidence-backed research focus that guides the next discriminating test."""
    return await call(research().project_state_update, project_id, update)


@mcp.tool()
async def claim_create(project_id: str, statement: str, scope: str, evidence_ids: list[str] | None = None):
    evidence = evidence_ids or []
    for evidence_id in evidence:
        item = research().object_get(evidence_id)
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
async def claim_search(project_id: str, query: str, limit: int = 25):
    """Find claims by their persisted statement, scope, or attached evidence identifiers."""
    return await call(research().search, project_id, query, "Claim", min(max(limit, 1), 100))


@mcp.tool()
async def claim_get_evidence(claim_id: str):
    """Retrieve the exact evidence units cited by a claim, never an uncited summary."""
    claim = research().object_get(claim_id)
    if claim["kind"] != "Claim":
        return {"error": {"type": "NOT_A_CLAIM", "message": claim_id}}
    units = [research().object_get(item) for item in claim["data"].get("evidence_ids", [])]
    return {"claim": claim, "evidence": units}


@mcp.tool()
async def claim_compare(claim_id: str, other_claim_id: str):
    """Compare scope and evidence overlap without inferring agreement from prose alone."""
    first, second = research().object_get(claim_id), research().object_get(other_claim_id)
    if first["kind"] != "Claim" or second["kind"] != "Claim":
        return {"error": {"type": "NOT_A_CLAIM", "message": "Both inputs must be Claim IDs"}}
    if first["project_id"] != second["project_id"]:
        return {"error": {"type": "RESEARCH_PROJECT_MISMATCH", "message": "Claims must share a project"}}
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
    close_dead = [
        item
        for item in related
        if item["lexical_similarity"] >= 0.6
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
        return {"error": {"type": "EXPERIMENT_PLAN_INCOMPLETE", "message": f"Missing: {', '.join(missing)}"}}
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
        return {"error": {"type": "INVALID_EXPERIMENT_ECONOMICS", "message": "Inputs must be non-negative"}}
    denominator = max(compute_cost * implementation_cost * max(execution_risk, 0.01), 0.01)
    return {
        "priority": scientific_importance * hypothesis_discrimination * expected_information_gain / denominator,
        "formula": "importance × discrimination × information_gain / (compute_cost × implementation_cost × execution_risk)",
        "recommendation": "Prefer the smallest discriminating test before additional training.",
    }


@mcp.tool()
async def research_events(project_id: str, limit: int = 100):
    return await call(research().events, project_id, min(max(limit, 1), 100))


@mcp.tool()
async def research_assess(object_id: str, status: str, rationale: str):
    """Update a claim or hypothesis only with an explicit evidence-backed assessment event."""
    return await call(research().assess, object_id, status, rationale)


@mcp.tool()
async def paper_ingest(project_id: str, title: str, url: str, card: dict, version: str | None = None):
    """Persist a paper card; claims must remain separate evidence-backed objects."""
    return await call(research().object_create, project_id, "Paper", {"title": title, "url": url, "version": version, "card": card}, "PAPER_INGESTED")


@mcp.tool()
async def paper_search(project_id: str, query: str, limit: int = 25):
    return await call(research().search, project_id, query, "Paper", min(max(limit, 1), 100))


@mcp.tool()
async def paper_get(paper_id: str):
    """Return the persisted paper card and provenance URL/version."""
    paper = research().object_get(paper_id)
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
async def paper_evidence_search(project_id: str, query: str, limit: int = 25):
    return await call(research().search, project_id, query, "EvidenceUnit", min(max(limit, 1), 100))


@mcp.tool()
async def research_embedding_store(object_id: str, embedding: list[float]):
    """Attach a caller-provided embedding to a research object for pgvector retrieval."""
    return await call(research().embedding_set, object_id, embedding)


@mcp.tool()
async def research_semantic_search(
    project_id: str, embedding: list[float], kind: str | None = None, limit: int = 25
):
    """Use pgvector cosine distance over persisted research-object embeddings."""
    return await call(research().semantic_search, project_id, embedding, kind, min(max(limit, 1), 100))


@mcp.tool()
async def paper_ask(project_id: str, question: str, limit: int = 8):
    """Return retrieved source passages for a question; this is evidence retrieval, not a conclusion."""
    evidence = await call(research().search, project_id, question, "EvidenceUnit", min(max(limit, 1), 25))
    if isinstance(evidence, dict) and "error" in evidence:
        return evidence
    return {
        "question": question,
        "evidence": evidence,
        "warning": "Retrieved passages are not canonical truth. Create a scoped Claim with explicit evidence IDs before drawing a conclusion.",
    }


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
            return {"error": {"type": "INVALID_NEGATIVE_RESULT_DESCENDANT", "message": hypothesis_id}}
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
async def lesson_create(project_id: str, statement: str, evidence_ids: list[str], confounds: list[str] | None = None):
    return await call(research().object_create, project_id, "Lesson", {"statement": statement, "evidence_ids": evidence_ids, "confounds": confounds or []}, "LESSON_CREATED")


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
    result = await call(research().object_create, project_id, "Reproduction", data, "REPRODUCTION_PREPARED")
    if "error" not in result:
        await call(research().edge_create, paper_id, result["id"], "HAS_REPRODUCTION")
    return result


@mcp.tool()
async def reproduction_status(reproduction_id: str):
    return await call(research().object_get, reproduction_id)


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
    job = await call(local.submit, command, working_directory, "reproduction-" + reproduction_id[:8], env, python_env)
    if "error" in job:
        return job
    return await call(
        research().object_update,
        reproduction_id,
        {"status": "RUNNING", "job_id": job["job_id"], "command": command, "working_directory": working_directory, "environment": env or {}, "python_env": python_env},
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
        return {"id": reproduction_id, "status": reproduction["status"], "data": reproduction["data"], "already_final": True}
    job_id = reproduction["data"].get("job_id")
    if not job_id:
        return {"error": {"type": "REPRODUCTION_MISSING_JOB", "message": reproduction_id}}
    outcome, artifacts, runtime = local.job_status(job_id), local.artifacts(job_id), await local.status()
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
        {"exit_code": outcome["exit_code"], "logs_tail": outcome["logs_tail"][-65536:], "artifacts": artifacts, "runtime": runtime},
        status,
        event,
    )


@mcp.tool()
async def reproduction_compare(reproduction_id: str, observed_metric: float, metric_name: str | None = None):
    """Compare a real observed metric against the prepared paper metric and tolerance."""
    reproduction = research().object_get(reproduction_id)
    if reproduction["kind"] != "Reproduction":
        return {"error": {"type": "NOT_A_REPRODUCTION", "message": reproduction_id}}
    if reproduction["status"] == "FAILED":
        return {"error": {"type": "REPRODUCTION_EXECUTION_FAILED", "message": "A failed run cannot be metric-compared"}}
    reported = reproduction["data"].get("reported_metric") or {}
    expected = reported.get("value")
    tolerance = reproduction["data"].get("tolerance")
    if not isinstance(expected, (int, float)) or not isinstance(tolerance, (int, float)):
        return {"error": {"type": "REPRODUCTION_COMPARISON_UNAVAILABLE", "message": "reported_metric.value and tolerance are required"}}
    difference = abs(observed_metric - expected)
    status = "REPRODUCED" if difference <= tolerance else "PARTIAL"
    return await call(
        research().object_update,
        reproduction_id,
        {"observed_metric": {"name": metric_name or reported.get("name"), "value": observed_metric}, "difference": difference},
        status,
        "REPRODUCTION_COMPLETED",
    )


@mcp.tool()
async def research_experiment_execute(
    experiment_id: str,
    command: str,
    working_directory: str = ".",
    env: dict[str, str] | None = None,
    python_env: str | None = None,
):
    """Run a preregistered experiment through the local GPU-Lab executor."""
    if not settings.gpu_lab_enable_local_runner:
        return {"error": {"type": "LOCAL_RUNNER_DISABLED"}}
    job = await call(
        local.submit,
        command,
        working_directory,
        "research-" + experiment_id[:8],
        env,
        python_env,
    )
    if "error" in job:
        return job
    return await call(
        research().run_create,
        experiment_id,
        {
            "executor": "local",
            "job_id": job["job_id"],
            "command": command,
            "working_directory": working_directory,
            "environment": env or {},
            "python_env": python_env,
        },
    )


@mcp.tool()
async def research_experiment_sync(run_id: str):
    """Retrieve a real job's logs/artifacts and append immutable execution evidence."""
    run = research().object_get(run_id)
    job_id = run["data"].get("job_id")
    if not job_id:
        return {"error": {"type": "RUN_MISSING_JOB"}}
    outcome, artifacts, runtime = local.job_status(job_id), local.artifacts(job_id), await local.status()
    updated = await call(
        research().run_update,
        run_id,
        {
            "status": outcome["status"],
            "exit_code": outcome["exit_code"],
            "logs_tail": outcome["logs_tail"][-65536:],
            "artifacts": artifacts,
            "runtime": runtime,
        },
    )
    if "error" in updated or updated.get("already_final") or outcome["status"] not in {"completed", "failed"}:
        return updated
    for artifact in artifacts:
        recorded = await call(
            research().object_create,
            str(run["project_id"]),
            "Artifact",
            {"run_id": run_id, "job_id": job_id, **artifact},
            "ARTIFACT_RECORDED",
        )
        if "error" not in recorded:
            await call(research().edge_create, run_id, recorded["id"], "PRODUCED")
    return updated


@mcp.tool()
async def artifact_list(job_id: str):
    return await call(svc().artifact_list, job_id)


@mcp.tool()
async def artifact_read(job_id: str, path: str, max_bytes: int | None = None):
    return await call(svc().artifact_read, job_id, path, max_bytes)


if settings.gpu_lab_enable_remote_exec:

    @mcp.tool()
    async def remote_exec(instance_id: str, command: str, timeout_seconds: int = 60):
        """Dangerous, bounded non-interactive debug command."""
        return await call(svc().remote_exec, instance_id, command, timeout_seconds)


if settings.gpu_lab_enable_local_runner:
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
    async def local_env_prepare(name: str, requirements_path: str | None = None):
        return await call(local.env_prepare, name, requirements_path)

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


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health(_: Request):
    return JSONResponse(await gateway_status())


@mcp.custom_route("/activity", methods=["GET"], include_in_schema=False)
async def activity(_: Request):
    return JSONResponse(svc().repo.list_audit(100))


def terminal_allowed(request: Request) -> bool:
    password = settings.gpu_lab_terminal_password
    header = request.headers.get("authorization", "")
    if not password or not header.startswith("Basic "):
        return False
    try:
        username, supplied = base64.b64decode(header[6:]).decode().split(":", 1)
    except (ValueError, UnicodeDecodeError):
        return False
    return username == "gpu-lab" and secrets.compare_digest(supplied, password)


def terminal_unauthorized() -> Response:
    return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="GPU Lab Terminal"'})


@mcp.custom_route("/terminal", methods=["GET"], include_in_schema=False)
async def terminal(request: Request):
    return HTMLResponse(TERMINAL_HTML) if terminal_allowed(request) else terminal_unauthorized()


@mcp.custom_route("/terminal/activity", methods=["GET"], include_in_schema=False)
async def terminal_activity(request: Request):
    return JSONResponse(svc().repo.list_audit(100)) if terminal_allowed(request) else terminal_unauthorized()


@mcp.custom_route("/terminal/jobs", methods=["GET"], include_in_schema=False)
async def terminal_jobs(request: Request):
    if not terminal_allowed(request):
        return terminal_unauthorized()
    if not settings.gpu_lab_enable_local_runner:
        return JSONResponse([])
    jobs = []
    for job in svc().repo.list_jobs(instance_id="local", limit=30):
        jobs.append(local.job_status(job.job_id))
    return JSONResponse(jobs)


@mcp.custom_route("/", methods=["GET"], include_in_schema=False)
async def dashboard(_: Request):
    return HTMLResponse(DASHBOARD_HTML)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    args = parser.parse_args()
    mcp.run(transport=args.transport)
