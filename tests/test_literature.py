import builtins
import json

import httpx
import pytest

from gpu_lab.errors import GPUError
from gpu_lab.literature import (
    EvidenceCandidate,
    HttpLiteratureProvider,
    LiteratureResult,
    LiteratureService,
    PaperQALiteratureProvider,
)


class FakeSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


async def fake_agent_query(*, query, settings):
    assert "causal routing" in query
    assert settings.kwargs["agent"]["index"]["paper_directory"]
    return {
        "formatted_answer": "The intervention is reported with one cited source.",
        "context": [
            {
                "title": "Mechanistic Routing",
                "authors": ["A. Scientist", "B. Researcher"],
                "year": "2026",
                "doi": "10.1000/routing",
                "url": "https://example.test/routing",
                "page": "4",
                "section": "3.2",
                "figure": "2",
                "text": "Frozen state substitution changed the downstream carrier.",
            }
        ],
    }


def fake_docs_factory():
    return object()


@pytest.mark.asyncio
async def test_paperqa_contract_normalizes_answer_and_provenance(tmp_path):
    provider = PaperQALiteratureProvider(
        tmp_path,
        settings_factory=FakeSettings,
        agent_query=fake_agent_query,
        docs_factory=fake_docs_factory,
    )

    result = await provider.retrieve_evidence("causal routing")

    assert result.provider == "paperqa"
    assert result.answer.startswith("The intervention")
    assert len(result.evidence_candidates) == 1
    evidence = result.evidence_candidates[0]
    assert evidence.doi == "10.1000/routing"
    assert evidence.page == "4"
    assert evidence.section == "3.2"
    assert evidence.figure == "2"
    assert evidence.extraction_method == "paperqa"


def test_paperqa_custom_openai_compatible_endpoint_configures_all_llms(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "task-scoped-test-key")
    provider = PaperQALiteratureProvider(
        tmp_path,
        settings_factory=FakeSettings,
        agent_query=fake_agent_query,
        docs_factory=fake_docs_factory,
        model="nghi/gpt-5.5",
        base_url="https://api.example.test/v1/",
        max_retries=3,
    )

    settings = provider._settings().kwargs

    assert settings["llm"] == settings["summary_llm"] == settings["agent_llm"] == "nghi/gpt-5.5"
    params = settings["llm_config"]["model_list"][0]["litellm_params"]
    assert params == {
        "model": "nghi/gpt-5.5",
        "api_base": "https://api.example.test/v1",
        "api_key": "task-scoped-test-key",
        "num_retries": 3,
    }


def test_paperqa_normalizes_current_nested_pqa_session_shape(tmp_path):
    provider = PaperQALiteratureProvider(
        tmp_path,
        settings_factory=FakeSettings,
        agent_query=fake_agent_query,
        docs_factory=fake_docs_factory,
    )
    result = provider._normalize(
        "question",
        {
            "formatted_answer": "answer",
            "context": "formatted context that must not be iterated as characters",
            "contexts": [
                {
                    "id": "pqac-1",
                    "context": "summary",
                    "score": 9,
                    "text": {
                        "text": "verbatim source chunk",
                        "name": "Paper pages 4-5",
                        "doc": {
                            "title": "Nested Paper",
                            "authors": ["Nested Author"],
                            "year": 2026,
                            "doi": "10.1000/nested",
                            "url": "https://example.test/nested",
                            "doc_id": "doc-1",
                            "citation": "Nested Author 2026",
                        },
                    },
                }
            ],
        },
    )

    evidence = result.evidence_candidates[0]
    assert evidence.title == "Nested Paper"
    assert evidence.source_excerpt == "verbatim source chunk"
    assert evidence.page == "Paper pages 4-5"
    assert evidence.provider_locator["context_id"] == "pqac-1"
    assert "text" not in evidence.provider_locator


@pytest.mark.asyncio
async def test_paperqa_unavailable_is_explicit(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def reject_paperqa(name, *args, **kwargs):
        if name == "paperqa":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_paperqa)
    provider = PaperQALiteratureProvider(tmp_path)

    with pytest.raises(GPUError) as error:
        await provider.ask("question")

    assert error.value.error_type == "LITERATURE_PROVIDER_UNAVAILABLE"


class FakeProvider:
    async def retrieve_evidence(self, query, papers=None):
        return LiteratureResult(
            question=query,
            answer="candidate answer",
            provider="fake-paperqa",
            evidence_candidates=[
                EvidenceCandidate(
                    title="Grounded Paper",
                    authors=["Author"],
                    year=2025,
                    doi="10.1000/grounded",
                    page="7",
                    section="Results",
                    source_excerpt="A scoped experimental observation.",
                    extraction_method="fake-paperqa",
                )
            ],
        )


class FakeStore:
    def __init__(self):
        self.created = []
        self.edges = []

    def search(self, _project_id, query, kind, _limit):
        return [
            item
            for item in self.created
            if item["project_id"] == _project_id
            and item["kind"] == kind
            and query.lower() in json.dumps(item["data"]).lower()
        ][:_limit]

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

    def edge_create(self, source, target, relation):
        self.edges.append((source, target, relation))

    def object_get(self, object_id):
        return next(item for item in self.created if item["id"] == object_id)

    def object_update(self, object_id, data_update, status, event_type):
        item = self.object_get(object_id)
        item["data"] = {**item["data"], **data_update}
        item["status"] = status
        item["event_type"] = event_type
        return item


@pytest.mark.asyncio
async def test_literature_service_keeps_provider_output_as_candidates():
    store = FakeStore()
    service = LiteratureService(store, FakeProvider())

    result = await service.gather(
        "project",
        "What evidence exists?",
        claim_statement="The scoped effect was observed.",
        claim_scope="single paper",
    )

    assert result["evidence"][0]["status"] == "CANDIDATE"
    assert result["claim"]["status"] == "ACTIVE"
    assert result["claim"]["data"]["evidence_ids"] == [result["evidence"][0]["id"]]
    assert all(item["status"] not in {"SUPPORTED", "VERIFIED_REAL"} for item in store.created)
    assert (result["evidence"][0]["id"], result["claim"]["id"], "SUPPORTS_CANDIDATE") in store.edges

    replay = await service.gather(
        "project",
        "What evidence exists?",
        claim_statement="The scoped effect was observed.",
        claim_scope="single paper",
    )
    assert replay["evidence"][0]["id"] == result["evidence"][0]["id"]
    assert replay["claim"]["id"] == result["claim"]["id"]


@pytest.mark.asyncio
async def test_literature_service_requires_claim_scope():
    service = LiteratureService(FakeStore(), FakeProvider())

    with pytest.raises(GPUError) as error:
        await service.gather("project", "question", claim_statement="claim")

    assert error.value.error_type == "LITERATURE_CLAIM_INCOMPLETE"


def test_fake_store_search_respects_project_and_limit():
    store = FakeStore()
    for project_id in ("other", "project", "project"):
        store.object_create(project_id, "Paper", {"title": "matching paper"}, "CREATED")

    matches = store.search("project", "matching", "Paper", 1)

    assert len(matches) == 1
    assert matches[0]["project_id"] == "project"


@pytest.mark.asyncio
async def test_http_literature_worker_contract_and_auth_header():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer scoped-token"
        assert json.loads(request.content)["question"] == "grounded question"
        return httpx.Response(
            200,
            json={
                "result": {
                    "question": "grounded question",
                    "answer": "candidate",
                    "evidence_candidates": [],
                    "provider": "paperqa",
                }
            },
        )

    provider = HttpLiteratureProvider(
        "http://literature:8010",
        "scoped-token",
        transport=httpx.MockTransport(handler),
    )

    result = await provider.ask("grounded question")

    assert result.provider == "paperqa"
    assert result.warning.startswith("Provider output")


@pytest.mark.asyncio
async def test_http_literature_worker_preserves_structured_errors():
    provider = HttpLiteratureProvider(
        "http://literature:8010",
        "scoped-token",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                400,
                json={
                    "error": {
                        "type": "INVALID_LITERATURE_REQUEST",
                        "message": "question is required",
                        "retryable": False,
                    }
                },
            )
        ),
    )

    with pytest.raises(GPUError) as error:
        await provider.ask("grounded question")

    assert error.value.error_type == "INVALID_LITERATURE_REQUEST"
    assert error.value.retryable is False


@pytest.mark.asyncio
async def test_http_literature_worker_normalizes_non_object_errors():
    provider = HttpLiteratureProvider(
        "http://literature:8010",
        "scoped-token",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(400, json={"error": "worker exploded"})
        ),
    )

    with pytest.raises(GPUError) as error:
        await provider.health()

    assert error.value.error_type == "LITERATURE_PROVIDER_ERROR"
    assert error.value.message == "Literature worker error"


@pytest.mark.asyncio
async def test_http_literature_worker_normalizes_malformed_error_fields():
    provider = HttpLiteratureProvider(
        "http://literature:8010",
        "scoped-token",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                400,
                json={"error": {"type": None, "message": [], "retryable": "yes"}},
            )
        ),
    )

    with pytest.raises(GPUError) as error:
        await provider.health()

    assert error.value.error_type == "LITERATURE_PROVIDER_ERROR"
    assert error.value.message == "Literature worker error"
    assert error.value.retryable is False


@pytest.mark.asyncio
async def test_http_literature_worker_keeps_malformed_5xx_error_retryable():
    provider = HttpLiteratureProvider(
        "http://literature:8010",
        "scoped-token",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                502,
                json={"error": {"type": None, "message": [], "retryable": "invalid"}},
            )
        ),
    )

    with pytest.raises(GPUError) as error:
        await provider.health()

    assert error.value.error_type == "LITERATURE_PROVIDER_ERROR"
    assert error.value.retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected_message"),
    [
        (httpx.Response(200, text="not-json"), "non-JSON"),
        (httpx.Response(200, json=[]), "non-object"),
        (httpx.Response(200, json={}), "missing the result"),
    ],
)
async def test_http_literature_worker_rejects_malformed_responses(
    response, expected_message
):
    provider = HttpLiteratureProvider(
        "http://literature:8010",
        "scoped-token",
        transport=httpx.MockTransport(lambda _request: response),
    )

    with pytest.raises(GPUError) as error:
        await provider.health()

    assert error.value.error_type == "LITERATURE_PROVIDER_INVALID_RESPONSE"
    assert expected_message in error.value.message


@pytest.mark.asyncio
async def test_worker_health_is_secret_free_and_other_routes_require_auth(monkeypatch):
    from gpu_lab import literature_worker

    monkeypatch.setattr(literature_worker, "WORKER_TOKEN", "scoped-token")
    monkeypatch.setenv("OPENAI_API_KEY", "task-scoped-test-key")
    transport = httpx.ASGITransport(app=literature_worker.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        health = await client.post("/health", json={})
        unauthorized = [
            await client.post(path, json={})
            for path in (
                "/ask",
                "/ingest",
                "/search",
                "/retrieve-evidence",
                "/inspect-document",
            )
        ]
        non_ascii = await client.post(
            "/ask",
            json={"question": "q"},
            headers={b"Authorization": b"Bearer t\xc3\xa9st"},
        )

    assert health.json()["result"] == {
        "provider": "paperqa",
        "status": "ready",
        "api_key_configured": True,
        "custom_endpoint": False,
        "model": None,
        "max_retries": 2,
    }
    assert all(response.status_code == 401 for response in unauthorized)
    assert all(response.json()["error"]["type"] == "UNAUTHORIZED" for response in unauthorized)
    assert non_ascii.status_code == 401


@pytest.mark.asyncio
async def test_literature_decision_imports_candidates_then_recomputes_brain_step():
    store = FakeStore()
    decision = store.object_create(
        "project",
        "ResearchDecision",
        {"selected_action": {"action_type": "LITERATURE_SEARCH"}},
        "RESEARCH_DECISION_SELECTED",
        "SELECTED",
    )

    class FakeBrain:
        def brain_step(self, project_id):
            assert project_id == "project"
            return {"selected_action": {"action_type": "EVIDENCE_REVIEW"}}

    service = LiteratureService(store, FakeProvider())
    result = await service.resolve_decision(
        FakeBrain(),
        decision["id"],
        "What evidence exists?",
        claim_statement="The scoped effect was observed.",
        claim_scope="single paper",
    )

    assert result["decision"]["status"] == "COMPLETED"
    assert result["literature"]["evidence"][0]["status"] == "CANDIDATE"
    assert result["next_brain_step"]["selected_action"]["action_type"] == "EVIDENCE_REVIEW"

    replay = await service.resolve_decision(FakeBrain(), decision["id"], "What evidence exists?")
    assert replay["idempotent_replay"] is True
    assert replay["next_brain_step"] is None
