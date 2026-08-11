from __future__ import annotations

import math
import os
import time

import pytest

from gpu_lab.embeddings import EmbeddingService, LocalHashEmbeddingProvider
from gpu_lab.research import ResearchStore

TEST_DATABASE_URL = os.getenv("GPU_LAB_TEST_DATABASE_URL")


class EmbeddingStore:
    def __init__(self):
        self.object = {
            "id": "hypothesis-1",
            "project_id": "project-1",
            "kind": "Hypothesis",
            "status": "ACTIVE",
            "data": {
                "statement": "Anchor state transports viewpoint evidence",
                "mechanism": "decoder state propagation",
                "assumptions": ["baseline reproduced"],
                "scope": {"architectures": ["VRCNet"]},
            },
        }
        self.metadata = None
        self.vector = None

    def object_get(self, object_id):
        assert object_id == self.object["id"]
        return self.object

    def objects_list(self, project_id, limit=None):
        assert project_id == self.object["project_id"]
        assert limit is None
        return [self.object]

    def embedding_metadata_get(self, object_id):
        assert object_id == self.object["id"]
        return self.metadata

    def embedding_set(self, object_id, vector, metadata):
        assert object_id == self.object["id"]
        self.vector = vector
        self.metadata = metadata
        return {"id": object_id, "metadata": metadata}

    def semantic_search(self, project_id, vector, kind, limit):
        assert project_id == self.object["project_id"]
        assert len(vector) == 32
        return [{"id": self.object["id"], "kind": kind, "distance": 0.0}][:limit]

    def search(self, project_id, query, kind, limit):
        return [
            {
                "id": self.object["id"],
                "project_id": project_id,
                "kind": kind,
                "query": query,
            }
        ][:limit]


class FailingProvider(LocalHashEmbeddingProvider):
    async def embed_texts(self, texts):
        raise RuntimeError("provider secret must not escape")


@pytest.mark.asyncio
async def test_local_hash_embedding_is_deterministic_normalized_and_private():
    provider = LocalHashEmbeddingProvider(32)

    first, second = await provider.embed_texts(["same scientific text", "same scientific text"])

    assert first == second
    assert len(first) == 32
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)


@pytest.mark.asyncio
async def test_embedding_refresh_persists_metadata_and_recomputes_after_source_change():
    store = EmbeddingStore()
    service = EmbeddingService(store, LocalHashEmbeddingProvider(32))

    first = await service.refresh_object(store.object["id"])
    original_hash = first["metadata"]["source_text_hash"]
    replay = await service.refresh_object(store.object["id"])
    store.object["data"]["mechanism"] = "changed causal mechanism"
    changed = await service.refresh_object(store.object["id"])

    assert first["status"] == "stored"
    assert replay["status"] == "current"
    assert changed["status"] == "stored"
    assert changed["metadata"]["source_text_hash"] != original_hash
    assert changed["metadata"]["dimension"] == 32


def test_canonical_text_excludes_volatile_identity_and_timestamp_fields():
    store = EmbeddingStore()
    item = {
        **store.object,
        "created_at": "2099-01-01T00:00:00Z",
        "data": {**store.object["data"], "timestamp": "2099-01-01T00:00:00Z"},
    }

    text = EmbeddingService.canonical_text(item)

    assert "hypothesis-1" not in text
    assert "2099-01-01" not in text
    assert "decoder state propagation" in text


@pytest.mark.asyncio
async def test_provider_failure_preserves_state_and_search_falls_back_to_lexical():
    store = EmbeddingStore()
    service = EmbeddingService(store, FailingProvider(32))

    refresh = await service.refresh_object(store.object["id"])
    search = await service.search("project-1", "anchor state", "Hypothesis")

    assert refresh == {
        "object_id": "hypothesis-1",
        "status": "provider_unavailable",
        "error": {
            "type": "EMBEDDING_PROVIDER_UNAVAILABLE",
            "message": "Embedding provider failed",
        },
        "fallback": "structured_and_lexical_retrieval",
    }
    assert store.metadata is None
    assert store.vector is None
    assert search["mode"] == "lexical_fallback"
    assert search["hits"][0]["query"] == "anchor state"


@pytest.mark.asyncio
@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
async def test_postgres_embedding_recomputes_after_mutation_and_survives_store_restart():
    store = ResearchStore(TEST_DATABASE_URL)
    if not store.vector_available:
        pytest.skip("pgvector is unavailable")
    project = store.project_create(
        f"embedding-restart-{time.time_ns()}", "Can the secondary index recover?"
    )
    hypothesis = store.object_create(
        project["project_id"],
        "Hypothesis",
        {
            "statement": "Anchor state transports viewpoint evidence",
            "mechanism": "decoder propagation",
            "scope": {"architectures": ["VRCNet"]},
        },
        "HYPOTHESIS_CREATED",
    )
    service = EmbeddingService(store, LocalHashEmbeddingProvider(64))

    first = await service.refresh_object(hypothesis["id"])
    first_hash = first["metadata"]["source_text_hash"]
    semantic = await service.search(
        project["project_id"], "viewpoint evidence anchor", "Hypothesis"
    )
    store.object_update(
        hypothesis["id"],
        {**hypothesis["data"], "mechanism": "changed state routing"},
        "ACTIVE",
        "HYPOTHESIS_UPDATED",
    )
    changed = await service.refresh_object(hypothesis["id"])

    restarted = ResearchStore(TEST_DATABASE_URL)
    restarted_service = EmbeddingService(restarted, LocalHashEmbeddingProvider(64))
    status = restarted_service.project_status(project["project_id"])

    assert first["status"] == "stored"
    assert semantic["mode"] == "semantic"
    assert str(semantic["hits"][0]["id"]) == hypothesis["id"]
    assert changed["metadata"]["source_text_hash"] != first_hash
    assert status["indexed_objects"] == 1
    assert status["stale_or_missing"] == 0
