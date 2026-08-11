import asyncio
import json
import sys

import httpx
import pytest

from gpu_lab.errors import GPUError
from gpu_lab.executable_papers import (
    ExecutablePaperCandidate,
    ExecutablePaperService,
    HttpExecutablePaperProvider,
)
from gpu_lab.paper2agent_worker import Paper2AgentSubprocessProvider


class FakeStore:
    def __init__(self):
        self.created = []
        self.edges = []

    def object_create(self, project_id, kind, data, event_type, status="ACTIVE"):
        item = {
            "id": f"{kind.lower()}-{len(self.created)}",
            "project_id": project_id,
            "kind": kind,
            "status": status,
            "data": data,
            "event_type": event_type,
        }
        self.created.append(item)
        return item

    def object_get(self, object_id):
        return next(item for item in self.created if item["id"] == object_id)

    def object_update(self, object_id, data_update, status, event_type):
        item = self.object_get(object_id)
        item["data"] = {**item["data"], **data_update}
        item["status"] = status
        item["event_type"] = event_type
        return item

    def search(self, project_id, query, kind, _limit):
        return [
            item
            for item in self.created
            if item["project_id"] == project_id
            and item["kind"] == kind
            and query in json.dumps(item["data"])
        ]

    def edge_create(self, source, target, relation):
        self.edges.append((source, target, relation))


class FakeProvider:
    def __init__(self, verification_status="VERIFIED_INTEGRATION"):
        self.verification_status = verification_status
        self.build_calls = 0

    async def build(self, paper_id, repository, commit, tutorials=None):
        self.build_calls += 1
        return ExecutablePaperCandidate(
            provider_build_id="a" * 64,
            repository=repository,
            commit=commit,
            generated_mcp="/isolated/generated_mcp.py",
        )

    async def inspect_tools(self, executable_paper_id):
        assert executable_paper_id == "a" * 64
        return [{"name": "reproduce_baseline", "inputSchema": {"type": "object"}}]

    async def verify(self, executable_paper_id):
        return {"verification_status": self.verification_status, "tool_count": 1}

    async def invoke(self, executable_paper_id, tool, args):
        return {"content": [{"type": "text", "text": f"{tool}:{args['seed']}"}]}


@pytest.mark.asyncio
async def test_executable_paper_service_keeps_generated_state_noncanonical_and_idempotent():
    store = FakeStore()
    paper = store.object_create("project", "Paper", {"title": "Paper"}, "PAPER_CREATED")
    provider = FakeProvider()
    service = ExecutablePaperService(store, provider)

    built = await service.build(
        "project", paper["id"], "https://github.com/owner/repo", "1" * 40
    )
    replay = await service.build(
        "project", paper["id"], "https://github.com/owner/repo", "1" * 40
    )
    inspected = await service.inspect_tools(built["id"])
    verified = await service.verify(built["id"])
    invoked = await service.invoke(built["id"], "reproduce_baseline", {"seed": 7})

    assert provider.build_calls == 1
    assert replay["idempotent_replay"] is True
    assert inspected["data"]["available_tools"][0]["name"] == "reproduce_baseline"
    assert verified["status"] == "VERIFIED_INTEGRATION"
    assert invoked["warning"].endswith("registered and assessed.")
    assert (paper["id"], built["id"], "HAS_EXECUTABLE_CANDIDATE") in store.edges
    assert not any(item["status"] == "VERIFIED_REAL" for item in store.created)


@pytest.mark.asyncio
async def test_generated_worker_cannot_self_promote_to_verified_real():
    store = FakeStore()
    paper = store.object_create("project", "Paper", {}, "PAPER_CREATED")
    service = ExecutablePaperService(store, FakeProvider("VERIFIED_REAL"))
    built = await service.build(
        "project", paper["id"], "https://github.com/owner/repo", "1" * 40
    )

    with pytest.raises(GPUError) as error:
        await service.verify(built["id"])

    assert error.value.error_type == "INVALID_EXECUTABLE_PAPER_VERIFICATION"


@pytest.mark.asyncio
async def test_http_executable_paper_contract_and_scoped_auth():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer scoped-token"
        body = json.loads(request.content)
        assert body["repository"] == "https://github.com/owner/repo"
        return httpx.Response(
            200,
            json={
                "result": {
                    "provider_build_id": "a" * 64,
                    "repository": body["repository"],
                    "commit": body["commit"],
                }
            },
        )

    provider = HttpExecutablePaperProvider(
        "http://paper2agent:8020",
        "scoped-token",
        transport=httpx.MockTransport(handler),
    )

    result = await provider.build("paper", "https://github.com/owner/repo", "1" * 40)

    assert result.provider_build_id == "a" * 64
    assert result.verification_status == "IMPLEMENTED_UNVERIFIED"


@pytest.mark.asyncio
async def test_http_executable_paper_preserves_structured_worker_error():
    provider = HttpExecutablePaperProvider(
        "http://paper2agent:8020",
        "scoped-token",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                400,
                json={
                    "error": {
                        "type": "EXECUTABLE_PAPER_BUILD_FAILED",
                        "message": "build failed",
                        "retryable": False,
                    }
                },
            )
        ),
    )

    with pytest.raises(GPUError) as error:
        await provider.build("paper", "https://github.com/owner/repo", "1" * 40)

    assert error.value.error_type == "EXECUTABLE_PAPER_BUILD_FAILED"
    assert error.value.retryable is False


@pytest.mark.parametrize(
    "repository",
    [
        "http://github.com/owner/repo",
        "https://user:pass@github.com/owner/repo",
        "https://github.com:443/owner/repo",
        "https://example.com/owner/repo",
        "https://github.com/owner/repo/extra",
        "https://github.com/owner/repo?token=secret",
    ],
)
def test_paper2agent_worker_accepts_only_public_github_repositories(tmp_path, repository):
    provider = Paper2AgentSubprocessProvider(tmp_path)

    with pytest.raises(GPUError) as error:
        provider._repository(repository)

    assert error.value.error_type == "INVALID_EXECUTABLE_PAPER_REPOSITORY"


@pytest.mark.asyncio
async def test_paper2agent_build_fails_before_work_without_scoped_credential(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "empty-claude-config"))
    provider = Paper2AgentSubprocessProvider(tmp_path)

    with pytest.raises(GPUError) as error:
        await provider.build("paper", "https://github.com/owner/repo", "1" * 40)

    assert error.value.error_type == "EXECUTABLE_PAPER_CREDENTIAL_REQUIRED"


@pytest.mark.asyncio
async def test_paper2agent_build_survives_disconnected_request(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "task-scoped-test-key")
    provider = Paper2AgentSubprocessProvider(tmp_path)
    commit = "1" * 40
    started = asyncio.Event()
    finish = asyncio.Event()

    async def remote_head(_repository):
        return commit

    async def build_once(build_id, _repository, expected_commit, _tutorials):
        started.set()
        await finish.wait()
        project = provider._project(build_id)
        project.mkdir(parents=True)
        (project / ".gpu-lab-build-complete").write_text(expected_commit, encoding="ascii")

    monkeypatch.setattr(provider, "_remote_head", remote_head)
    monkeypatch.setattr(provider, "_build_once", build_once)
    monkeypatch.setattr(
        provider,
        "_runtime",
        lambda build_id: (
            provider._project(build_id),
            provider._project(build_id) / "generated_mcp.py",
            provider._project(build_id) / "env/bin/python",
        ),
    )

    first = asyncio.create_task(
        provider.build("paper", "https://github.com/owner/repo", commit)
    )
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(
        provider.build("paper", "https://github.com/owner/repo", commit)
    )
    finish.set()
    result = await second

    assert result.commit == commit
    assert not provider._build_tasks


@pytest.mark.asyncio
async def test_paper2agent_generated_mcp_inspect_verify_and_invoke(monkeypatch, tmp_path):
    provider = Paper2AgentSubprocessProvider(tmp_path)
    build_id = "a" * 64
    project = provider._project(build_id)
    project.mkdir(parents=True)
    server = project / "fixture_mcp.py"
    server.write_text(
        """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("paper-fixture")

@mcp.tool()
def reproduce_baseline(seed: int):
    return {"seed": seed, "metric": 0.0}

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(provider, "_runtime", lambda _build_id: (project, server, sys.executable))

    tools = await provider.inspect_tools(build_id)
    verification = await provider.verify(build_id)
    invocation = await provider.invoke(build_id, "reproduce_baseline", {"seed": 11})

    assert [tool["name"] for tool in tools] == ["reproduce_baseline"]
    assert verification["verification_status"] == "VERIFIED_INTEGRATION"
    assert verification["paper2agent_report_present"] is False
    assert json.loads(invocation["content"][0]["text"]) == {"seed": 11, "metric": 0.0}


@pytest.mark.asyncio
async def test_paper2agent_worker_health_is_secret_free_and_routes_require_auth(monkeypatch):
    from gpu_lab import paper2agent_worker

    monkeypatch.setattr(paper2agent_worker, "WORKER_TOKEN", "scoped-token")
    transport = httpx.ASGITransport(app=paper2agent_worker.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        health = await client.post("/health", json={})
        unauthorized = await client.post("/verify", json={"executable_paper_id": "a" * 64})

    assert health.json()["result"]["provider"] == "paper2agent"
    assert health.json()["result"]["status"] in {"ready", "needs_credentials"}
    assert "token" not in json.dumps(health.json()).lower()
    assert unauthorized.status_code == 401
