from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

from .errors import GPUError
from .research import ResearchStore

EMBEDDING_TARGETS = {
    "Hypothesis",
    "NegativeResult",
    "Claim",
    "Mechanism",
    "Anomaly",
    "Contradiction",
    "Lesson",
    "ComparativeLesson",
    "MetaLesson",
    "ResearchSituation",
    "ResearchStrategyPattern",
    "AgendaItem",
    "Paper",
}

CANONICAL_FIELDS = {
    "Hypothesis": (
        "statement",
        "mechanism",
        "variables",
        "assumptions",
        "scientific_difference",
        "prediction",
        "unique_predictions",
        "scope",
    ),
    "NegativeResult": (
        "proposal",
        "mechanism",
        "failed_assumption",
        "evidence",
        "scope",
        "revisit_condition",
    ),
    "Claim": ("statement", "scope", "status_rationale"),
    "Mechanism": ("name", "description", "attributes", "scope"),
    "Anomaly": ("statement", "description", "scope"),
    "Contradiction": ("statement", "description", "scope"),
    "Lesson": ("lesson", "statement", "conditions", "scope"),
    "ComparativeLesson": (
        "candidate_causal_difference",
        "shared_conditions",
        "metric_delta",
        "remaining_confounds",
        "scope",
    ),
    "MetaLesson": ("lesson", "pattern", "recommendation", "scope"),
    "ResearchSituation": (
        "domain",
        "research_stage",
        "phenomenon_type",
        "mechanism_status",
        "problem_signature",
        "dominant_confounds",
        "resource_constraints",
    ),
    "ResearchStrategyPattern": (
        "problem_signature",
        "research_stage",
        "conditions",
        "action_type",
        "applicability_conditions",
        "failure_conditions",
        "counterexamples",
    ),
    "AgendaItem": (
        "question",
        "scientific_scope",
        "candidate_experiments",
        "status_rationale",
    ),
    "Paper": ("title", "abstract", "card", "version"),
}


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    model: str
    model_version: str
    dimension: int

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class LocalHashEmbeddingProvider:
    """Private deterministic feature hashing; useful as a resilient local secondary index."""

    name = "local"
    model = "feature-hash"
    model_version = "1"

    def __init__(self, dimension: int = 384):
        if dimension < 32 or dimension > 4096:
            raise GPUError("INVALID_EMBEDDING_DIMENSION", str(dimension))
        self.dimension = dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            terms = re.findall(r"[a-z0-9]+", text.lower())
            features = [*terms, *(f"{left}_{right}" for left, right in pairwise(terms))]
            for feature in features:
                digest = hashlib.blake2b(feature.encode(), digest_size=16).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


class EmbeddingService:
    """Automatic, source-hashed embeddings that never own canonical scientific truth."""

    def __init__(self, store: ResearchStore, provider: EmbeddingProvider):
        self.store = store
        self.provider = provider

    @staticmethod
    def canonical_text(item: dict[str, Any]) -> str:
        fields = CANONICAL_FIELDS.get(item["kind"])
        if not fields:
            raise GPUError("EMBEDDING_KIND_UNSUPPORTED", item["kind"])
        data = item["data"]
        sections = []
        for field in fields:
            if field not in data or data[field] in (None, "", [], {}):
                continue
            value = data[field]
            rendered = (
                json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                if isinstance(value, (dict, list))
                else str(value).strip()
            )
            sections.append(f"{field}: {rendered}")
        return "\n".join(sections)

    @staticmethod
    def source_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def refresh_object(self, object_id: str) -> dict[str, Any]:
        item = self.store.object_get(object_id)
        if item["kind"] not in EMBEDDING_TARGETS:
            return {"object_id": object_id, "status": "not_targeted"}
        text = self.canonical_text(item)
        if not text:
            return {"object_id": object_id, "status": "empty_canonical_text"}
        source_hash = self.source_hash(text)
        existing = self.store.embedding_metadata_get(object_id)
        identity = {
            "provider": self.provider.name,
            "model": self.provider.model,
            "model_version": self.provider.model_version,
            "dimension": self.provider.dimension,
            "source_text_hash": source_hash,
        }
        if existing and all(existing.get(key) == value for key, value in identity.items()):
            return {"object_id": object_id, "status": "current", "metadata": existing}
        try:
            vectors = await self.provider.embed_texts([text])
            if len(vectors) != 1:
                raise GPUError(
                    "EMBEDDING_PROVIDER_INVALID_RESPONSE", "Expected exactly one vector"
                )
            vector = vectors[0]
            if len(vector) != self.provider.dimension or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in vector
            ):
                raise GPUError(
                    "EMBEDDING_PROVIDER_INVALID_RESPONSE",
                    "Provider returned an invalid dimension or non-finite value",
                )
            metadata = {**identity, "created_at": datetime.now(UTC).isoformat()}
            self.store.embedding_set(object_id, vector, metadata)
            return {"object_id": object_id, "status": "stored", "metadata": metadata}
        except GPUError as exc:
            return {
                "object_id": object_id,
                "status": "provider_unavailable",
                "error": {"type": exc.error_type, "message": exc.message},
                "fallback": "structured_and_lexical_retrieval",
            }
        except Exception:  # noqa: BLE001 - provider failure must preserve canonical state
            return {
                "object_id": object_id,
                "status": "provider_unavailable",
                "error": {
                    "type": "EMBEDDING_PROVIDER_UNAVAILABLE",
                    "message": "Embedding provider failed",
                },
                "fallback": "structured_and_lexical_retrieval",
            }

    async def refresh_project(self, project_id: str) -> dict[str, Any]:
        objects = [
            item
            for item in self.store.objects_list(project_id, limit=None)
            if item["kind"] in EMBEDDING_TARGETS
        ]
        results = [await self.refresh_object(str(item["id"])) for item in objects]
        failures = [item for item in results if item["status"] == "provider_unavailable"]
        return {
            "project_id": project_id,
            "provider": self.provider.name,
            "model": self.provider.model,
            "objects": len(results),
            "stored": sum(item["status"] == "stored" for item in results),
            "current": sum(item["status"] == "current" for item in results),
            "failures": failures,
            "fallback_active": bool(failures),
        }

    def project_status(self, project_id: str) -> dict[str, Any]:
        objects = [
            item
            for item in self.store.objects_list(project_id, limit=None)
            if item["kind"] in EMBEDDING_TARGETS
        ]
        metadata = [
            self.store.embedding_metadata_get(str(item["id"])) for item in objects
        ]
        return {
            "project_id": project_id,
            "provider": self.provider.name,
            "model": self.provider.model,
            "target_objects": len(objects),
            "indexed_objects": sum(item is not None for item in metadata),
            "stale_or_missing": sum(
                item is None
                or item.get("provider") != self.provider.name
                or item.get("model") != self.provider.model
                or item.get("model_version") != self.provider.model_version
                for item in metadata
            ),
            "canonical_truth_owner": "PostgreSQL scientific objects",
        }

    async def search(
        self, project_id: str, query: str, kind: str | None = None, limit: int = 25
    ) -> dict[str, Any]:
        try:
            vectors = await self.provider.embed_texts([query])
            hits = self.store.semantic_search(project_id, vectors[0], kind, limit)
            return {"mode": "semantic", "hits": hits}
        except Exception:  # noqa: BLE001 - any secondary-index failure falls back to lexical
            return {
                "mode": "lexical_fallback",
                "hits": self.store.search(project_id, query, kind, limit),
            }
