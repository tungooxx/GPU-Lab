"""Exercise the public v3.3 MCP surface against a disposable PostgreSQL database."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client


def candidate(title: str, signature: dict[str, str]) -> dict:
    return {
        "title": title,
        "mechanism": f"Mechanism for {title}",
        "predictions": [f"Prediction for {title}"],
        "falsifier": f"Falsifier for {title}",
        "diversity_signature": signature,
    }


def payload(result) -> dict:
    if result.isError:
        raise RuntimeError(result.content[0].text)
    if result.structuredContent:
        value = dict(result.structuredContent)
        # FastMCP wraps object-shaped tool output under ``result`` on this
        # transport while keeping scalar results direct.
        return dict(value.get("result", value))
    return json.loads(result.content[0].text)


async def call(session: ClientSession, name: str, arguments: dict) -> dict:
    return payload(await session.call_tool(name, arguments))


async def main() -> None:
    database_url = os.environ["GPU_LAB_TEST_DATABASE_URL"]
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "GPU_LAB_RESEARCH_DATABASE_URL": database_url,
        "GPU_LAB_ENABLE_LOCAL_RUNNER": "false",
        "GPU_LAB_ENABLE_REMOTE_EXEC": "false",
        "LAB_UI_ENABLED": "false",
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "gpu_lab.server", "--transport", "stdio"],
        env=environment,
    )
    # ``gpu_lab.server`` is intentionally module-importable; invoke its main
    # explicitly so this works in both source and packaged test environments.
    params.args = ["-c", "from gpu_lab.server import main; main()", "--transport", "stdio"]
    async with stdio_client(params) as streams, ClientSession(*streams) as session:
        await session.initialize()
        names = {tool.name for tool in (await session.list_tools()).tools}
        expected = {
            "discovery_round_create", "discovery_round_join", "discovery_candidate_submit",
            "discovery_batch_freeze", "discovery_batch_get", "discovery_synthesis_start",
            "discovery_archive_get", "discovery_outcome_get",
        }
        assert expected <= names
        project = await call(session, "research_project_create", {
            "name": f"dde-v33-mcp-smoke-{time.time_ns()}", "question": "MCP integration",
        })
        project_id = project["project_id"]
        first = await call(session, "lab_join", {"project_id": project_id, "runtime_type": "CODEX", "worker_name": "A"})
        second = await call(session, "lab_join", {"project_id": project_id, "runtime_type": "LOCAL_AGENT", "worker_name": "B"})
        round_ = await call(session, "discovery_round_create", {
            "project_id": project_id, "search_regime": "DIVERGENT_SEARCH",
        })
        batch_a = await call(session, "discovery_round_join", {
            "discovery_round_id": round_["id"], "worker_id": first["worker"]["id"],
            "session_id": first["session_id"], "generation_operator": "REPRESENTATION_RESET", "requested_distance": "FAR",
        })
        batch_b = await call(session, "discovery_round_join", {
            "discovery_round_id": round_["id"], "worker_id": second["worker"]["id"],
            "session_id": second["session_id"], "generation_operator": "STRONG_NULL_CONSTRUCTION", "requested_distance": "ORTHOGONAL",
        })
        await call(session, "discovery_candidate_submit", {
            "discovery_round_id": round_["id"], "candidate_batch_id": batch_a["id"],
            "worker_id": first["worker"]["id"], "session_id": first["session_id"],
            "candidate": candidate("representation reset", {"representation": "point-token"}),
        })
        denied = await session.call_tool("discovery_batch_get", {
            "discovery_round_id": round_["id"], "candidate_batch_id": batch_a["id"],
            "requester_session_id": second["session_id"],
        })
        denied_payload = payload(denied)
        assert denied_payload["error"]["type"] == "DISCOVERY_PEER_ISOLATION_ACTIVE"
        await call(session, "discovery_peer_isolation_override", {
            "discovery_round_id": round_["id"], "candidate_batch_id": batch_b["id"],
            "worker_id": second["worker"]["id"], "session_id": second["session_id"],
            "rationale": "Explicit operator test of the audited override path",
        })
        exposed = await call(session, "discovery_batch_get", {
            "discovery_round_id": round_["id"], "candidate_batch_id": batch_a["id"],
            "requester_session_id": second["session_id"],
        })
        assert len(exposed["candidates"]) == 1
        await call(session, "discovery_candidate_submit", {
            "discovery_round_id": round_["id"], "candidate_batch_id": batch_b["id"],
            "worker_id": second["worker"]["id"], "session_id": second["session_id"],
            "candidate": candidate("strong null", {"causal_object": "null-ontology"}),
        })
        for batch, worker in ((batch_a, first), (batch_b, second)):
            await call(session, "discovery_batch_freeze", {
                "discovery_round_id": round_["id"], "candidate_batch_id": batch["id"],
                "worker_id": worker["worker"]["id"], "session_id": worker["session_id"],
            })
        archive = await call(session, "discovery_synthesis_start", {
            "discovery_round_id": round_["id"], "literature_available": False,
        })
        assert archive["data"]["coverage"]["literature_status"] == "UNAVAILABLE_NOVELTY_UNVERIFIED"
        assert len((await call(session, "discovery_archive_get", {"archive_id": archive["id"]}))["survivors"]) == 2
        print("DISCOVERY_V33_MCP_SMOKE_OK")


if __name__ == "__main__":
    asyncio.run(main())
