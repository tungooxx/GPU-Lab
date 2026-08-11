import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, Field

from .errors import GPUError
from .research import ResearchStore


class EvidenceCandidate(BaseModel):
    title: str
    paper_version: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    page: str | None = None
    section: str | None = None
    equation: str | None = None
    figure: str | None = None
    table: str | None = None
    source_excerpt: str
    extraction_method: str
    provider_locator: dict[str, Any] = Field(default_factory=dict)


class LiteratureResult(BaseModel):
    question: str
    answer: str | None = None
    evidence_candidates: list[EvidenceCandidate] = Field(default_factory=list)
    provider: str
    warning: str = "Provider output is an evidence candidate, not canonical scientific truth."


@runtime_checkable
class LiteratureProvider(Protocol):
    async def ingest(self, source: str) -> dict[str, Any]: ...

    async def search(
        self, query: str, filters: dict[str, Any] | None = None
    ) -> LiteratureResult: ...

    async def retrieve_evidence(
        self, query: str, papers: list[str] | None = None
    ) -> LiteratureResult: ...

    async def inspect_document(self, paper_id: str, request: str) -> LiteratureResult: ...

    async def ask(self, question: str, papers: list[str] | None = None) -> LiteratureResult: ...


class PaperQALiteratureProvider:
    """Optional PaperQA v5+ adapter; PaperQA's index remains a replaceable retrieval cache."""

    def __init__(
        self,
        paper_directory: Path,
        *,
        settings_factory=None,
        agent_query=None,
        docs_factory=None,
    ):
        self.paper_directory = paper_directory.resolve()
        self._settings_factory = settings_factory
        self._agent_query = agent_query
        self._docs_factory = docs_factory
        self._docs = None

    def _load(self) -> None:
        if self._settings_factory and self._agent_query and self._docs_factory:
            return
        try:
            from paperqa import Docs, Settings, agent_query
        except ImportError as exc:
            raise GPUError(
                "LITERATURE_PROVIDER_UNAVAILABLE",
                "PaperQA is optional. Install the literature worker dependency to enable it.",
            ) from exc
        self._settings_factory = Settings
        self._agent_query = agent_query
        self._docs_factory = Docs

    def _settings(self):
        self._load()
        return self._settings_factory(
            agent={"index": {"paper_directory": str(self.paper_directory)}}
        )

    async def ingest(self, source: str) -> dict[str, Any]:
        self._load()
        if self._docs is None:
            self._docs = self._docs_factory()
        if source.startswith("https://"):
            method = getattr(self._docs, "aadd_url", None)
        elif source.startswith("http://"):
            raise GPUError("INVALID_LITERATURE_SOURCE", "Remote sources must use HTTPS")
        else:
            source_path = Path(source).resolve()
            if not source_path.is_relative_to(self.paper_directory):
                raise GPUError(
                    "INVALID_LITERATURE_SOURCE",
                    "Local literature sources must be inside the worker paper directory",
                )
            if not source_path.is_file():
                raise GPUError("LITERATURE_SOURCE_NOT_FOUND", source)
            method = getattr(self._docs, "aadd", None)
            source = str(source_path)
        if method is None:
            raise GPUError("LITERATURE_PROVIDER_INCOMPATIBLE", "PaperQA ingest API is unavailable")
        result = await method(source)
        return {"provider": "paperqa", "source": source, "result": self._dump(result)}

    async def search(self, query: str, filters: dict[str, Any] | None = None) -> LiteratureResult:
        qualifier = f" Filters: {filters}." if filters else ""
        return await self.ask(f"Find the most relevant papers for: {query}.{qualifier}")

    async def retrieve_evidence(
        self, query: str, papers: list[str] | None = None
    ) -> LiteratureResult:
        return await self.ask(query, papers)

    async def inspect_document(self, paper_id: str, request: str) -> LiteratureResult:
        return await self.ask(f"For document {paper_id}, {request}", [paper_id])

    async def ask(self, question: str, papers: list[str] | None = None) -> LiteratureResult:
        self._load()
        scoped = question
        if papers:
            scoped += " Restrict the answer to these documents: " + ", ".join(papers)
        if self._docs is not None:
            answer = await self._docs.aquery(scoped, settings=self._settings())
        else:
            answer = await self._agent_query(query=scoped, settings=self._settings())
        return self._normalize(question, answer)

    def _normalize(self, question: str, answer: Any) -> LiteratureResult:
        raw = self._dump(answer)
        answer_text = self._first(raw, "formatted_answer", "answer")
        contexts = (raw.get("contexts") or raw.get("context", [])) if isinstance(raw, dict) else []
        if isinstance(contexts, dict):
            contexts = list(contexts.values())
        elif not isinstance(contexts, (list, tuple)):
            contexts = [contexts]
        candidates = []
        for context in contexts or []:
            item = self._dump(context)
            if not isinstance(item, dict):
                item = {"text": str(item)}
            text_record = self._dump(item.get("text", {}))
            if not isinstance(text_record, dict):
                text_record = {"text": str(text_record)}
            document = self._dump(text_record.get("doc", {}))
            if not isinstance(document, dict):
                document = {"citation": str(document)}
            excerpt = self._first(text_record, "text") or self._first(
                item, "context", "summary", "text"
            )
            if not excerpt:
                continue
            citation = self._dump(item.get("citation") or document.get("citation", {}))
            if not isinstance(citation, dict):
                citation = {"formatted": str(citation)}
            title = (
                self._first(item, "title", "docname", "name")
                or self._first(document, "title", "docname")
                or self._first(citation, "title", "formatted")
            )
            candidates.append(
                EvidenceCandidate(
                    title=title or "Unknown PaperQA source",
                    paper_version=self._first(item, "paper_version", "version")
                    or self._first(document, "paper_version", "version"),
                    authors=self._authors(
                        item.get("authors") or document.get("authors") or citation.get("authors")
                    ),
                    year=self._year(
                        item.get("year") or document.get("year") or citation.get("year")
                    ),
                    doi=self._first(item, "doi")
                    or self._first(document, "doi")
                    or self._first(citation, "doi"),
                    url=self._first(item, "url")
                    or self._first(document, "url", "doi_url", "pdf_url")
                    or self._first(citation, "url"),
                    page=self._first(item, "page", "pages")
                    or self._first(text_record, "name")
                    or self._first(document, "pages"),
                    section=self._first(item, "section"),
                    equation=self._first(item, "equation"),
                    figure=self._first(item, "figure"),
                    table=self._first(item, "table"),
                    source_excerpt=str(excerpt)[:12000],
                    extraction_method="paperqa",
                    provider_locator={
                        "context_id": item.get("id"),
                        "context_question": item.get("question"),
                        "context_summary": item.get("context"),
                        "relevance_score": item.get("score"),
                        "chunk_name": text_record.get("name"),
                        "document_id": document.get("doc_id") or document.get("dockey"),
                        "citation": document.get("citation"),
                    },
                )
            )
        return LiteratureResult(
            question=question,
            answer=str(answer_text) if answer_text is not None else None,
            evidence_candidates=candidates,
            provider="paperqa",
        )

    @staticmethod
    def _dump(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return value
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "__dict__"):
            return {
                name: item
                for name, item in vars(value).items()
                if not name.startswith("_") and not callable(item)
            }
        return str(value)

    @staticmethod
    def _first(value: Any, *names: str) -> Any:
        if not isinstance(value, dict):
            return None
        return next((value[name] for name in names if value.get(name) not in (None, "")), None)

    @staticmethod
    def _authors(value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(item) for item in value]

    @staticmethod
    def _year(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None


class HttpLiteratureProvider:
    """Task-scoped client for an isolated literature worker."""

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout_seconds: int = 180,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not base_url.startswith(("http://", "https://")):
            raise GPUError("INVALID_LITERATURE_WORKER_URL", base_url)
        if not token:
            raise GPUError(
                "LITERATURE_WORKER_TOKEN_REQUIRED",
                "Configure a task-scoped token for the isolated literature worker",
            )
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def _call(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
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
                "LITERATURE_PROVIDER_UNAVAILABLE",
                f"The isolated literature worker failed during {operation}: {exc}",
                retryable=True,
            ) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise GPUError(
                "LITERATURE_PROVIDER_INVALID_RESPONSE",
                f"The isolated worker returned non-JSON status {response.status_code}",
                retryable=response.status_code >= 500,
            ) from exc
        if not isinstance(data, dict):
            raise GPUError(
                "LITERATURE_PROVIDER_INVALID_RESPONSE",
                f"The isolated worker returned a non-object body with status {response.status_code}",
                retryable=response.status_code >= 500,
            )
        if "error" in data and data["error"] is not None:
            error = data["error"] if isinstance(data["error"], dict) else {}
            error_type = error.get("type")
            message = error.get("message")
            retryable = error.get("retryable")
            raise GPUError(
                error_type
                if isinstance(error_type, str) and error_type
                else "LITERATURE_PROVIDER_ERROR",
                message
                if isinstance(message, str) and message
                else "Literature worker error",
                retryable
                if isinstance(retryable, bool)
                else response.status_code >= 500,
            )
        if response.is_error:
            raise GPUError(
                "LITERATURE_PROVIDER_ERROR",
                f"The isolated worker returned HTTP {response.status_code}",
                retryable=response.status_code >= 500,
            )
        if "result" not in data:
            raise GPUError(
                "LITERATURE_PROVIDER_INVALID_RESPONSE",
                "The isolated worker response is missing the result field",
                retryable=False,
            )
        return data["result"]

    async def health(self) -> dict[str, Any]:
        return await self._call("health", {})

    async def ingest(self, source: str) -> dict[str, Any]:
        return await self._call("ingest", {"source": source})

    async def search(self, query: str, filters: dict[str, Any] | None = None) -> LiteratureResult:
        return LiteratureResult.model_validate(
            await self._call("search", {"query": query, "filters": filters})
        )

    async def retrieve_evidence(
        self, query: str, papers: list[str] | None = None
    ) -> LiteratureResult:
        return LiteratureResult.model_validate(
            await self._call("retrieve-evidence", {"query": query, "papers": papers})
        )

    async def inspect_document(self, paper_id: str, request: str) -> LiteratureResult:
        return LiteratureResult.model_validate(
            await self._call("inspect-document", {"paper_id": paper_id, "request": request})
        )

    async def ask(self, question: str, papers: list[str] | None = None) -> LiteratureResult:
        return LiteratureResult.model_validate(
            await self._call("ask", {"question": question, "papers": papers})
        )


class LiteratureService:
    """Validates provider candidates before persisting them in canonical Research OS."""

    def __init__(self, store: ResearchStore, provider: LiteratureProvider):
        self.store = store
        self.provider = provider

    async def gather(
        self,
        project_id: str,
        question: str,
        papers: list[str] | None = None,
        claim_statement: str | None = None,
        claim_scope: str | None = None,
    ) -> dict[str, Any]:
        if bool(claim_statement) != bool(claim_scope):
            raise GPUError(
                "LITERATURE_CLAIM_INCOMPLETE",
                "claim_statement and claim_scope must be provided together",
            )
        result = await self.provider.retrieve_evidence(question, papers)
        evidence = [
            self._persist_candidate(project_id, item) for item in result.evidence_candidates
        ]
        claim = None
        if claim_statement:
            if not evidence:
                raise GPUError(
                    "LITERATURE_EVIDENCE_REQUIRED",
                    "A claim cannot be created when the provider returned no evidence candidates",
                )
            claim_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "statement": claim_statement,
                        "scope": claim_scope,
                        "evidence_ids": sorted(item["id"] for item in evidence),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            existing_claims = self.store.search(project_id, claim_fingerprint, "Claim", 1)
            if existing_claims and (
                existing_claims[0]["data"].get("candidate_fingerprint") == claim_fingerprint
            ):
                claim = existing_claims[0]
            else:
                claim = self.store.object_create(
                    project_id,
                    "Claim",
                    {
                        "statement": claim_statement,
                        "scope": claim_scope,
                        "evidence_ids": [item["id"] for item in evidence],
                        "source": "literature_provider",
                        "candidate_fingerprint": claim_fingerprint,
                    },
                    "CLAIM_CREATED_FROM_LITERATURE_CANDIDATES",
                    "ACTIVE",
                )
                for item in evidence:
                    self.store.edge_create(item["id"], claim["id"], "SUPPORTS_CANDIDATE")
        return {
            "question": question,
            "provider": result.provider,
            "answer": result.answer,
            "evidence": evidence,
            "claim": claim,
            "warning": result.warning,
        }

    async def resolve_decision(
        self,
        brain: Any,
        decision_id: str,
        question: str,
        papers: list[str] | None = None,
        claim_statement: str | None = None,
        claim_scope: str | None = None,
    ) -> dict[str, Any]:
        decision = self.store.object_get(decision_id)
        if decision["kind"] != "ResearchDecision":
            raise GPUError("NOT_A_RESEARCH_DECISION", decision_id)
        if decision["status"] == "COMPLETED" and decision["data"].get("outcome", {}).get(
            "provider"
        ):
            outcome = decision["data"]["outcome"]
            return {
                "decision": decision,
                "literature": {
                    "provider": outcome["provider"],
                    "evidence": [
                        self.store.object_get(item) for item in outcome.get("evidence_ids", [])
                    ],
                    "claim": self.store.object_get(outcome["claim_id"])
                    if outcome.get("claim_id")
                    else None,
                },
                "next_brain_step": None,
                "idempotent_replay": True,
            }
        selected = decision["data"].get("selected_action", {})
        if selected.get("action_type") != "LITERATURE_SEARCH":
            raise GPUError(
                "DECISION_ACTION_MISMATCH",
                "The selected Brain action is not LITERATURE_SEARCH",
            )
        gathered = await self.gather(
            str(decision["project_id"]),
            question,
            papers,
            claim_statement,
            claim_scope,
        )
        completed = self.store.object_update(
            decision_id,
            {
                "outcome": {
                    "provider": gathered["provider"],
                    "evidence_ids": [item["id"] for item in gathered["evidence"]],
                    "claim_id": gathered["claim"]["id"] if gathered["claim"] else None,
                },
                "actual_information_gain": "PENDING_EVIDENCE_REVIEW",
                "hindsight_assessment": (
                    "Literature candidates were imported for provenance review; no scientific "
                    "status was promoted."
                ),
            },
            "COMPLETED",
            "RESEARCH_DECISION_OUTCOME_RECORDED",
        )
        next_step = brain.brain_step(str(decision["project_id"]))
        return {
            "decision": completed,
            "literature": gathered,
            "next_brain_step": next_step,
        }

    def _persist_candidate(self, project_id: str, item: EvidenceCandidate) -> dict[str, Any]:
        source_key = item.doi or item.url or item.title
        papers = self.store.search(project_id, source_key, "Paper", 10)
        paper = next(
            (
                candidate
                for candidate in papers
                if (item.doi and candidate["data"].get("doi") == item.doi)
                or (item.url and candidate["data"].get("url") == item.url)
                or candidate["data"].get("title") == item.title
            ),
            None,
        )
        if paper is None:
            paper = self.store.object_create(
                project_id,
                "Paper",
                {
                    "title": item.title,
                    "paper_version": item.paper_version,
                    "authors": item.authors,
                    "year": item.year,
                    "doi": item.doi,
                    "url": item.url,
                    "source": "literature_provider",
                },
                "PAPER_CANDIDATE_INGESTED",
                "CANDIDATE",
            )
        fingerprint = hashlib.sha256(
            (
                str(paper["id"])
                + item.source_excerpt
                + json.dumps(item.provider_locator, sort_keys=True, default=str)
            ).encode()
        ).hexdigest()
        existing = self.store.search(project_id, fingerprint, "EvidenceUnit", 1)
        if existing and existing[0]["data"].get("candidate_fingerprint") == fingerprint:
            return existing[0]
        evidence = self.store.object_create(
            project_id,
            "EvidenceUnit",
            {
                "paper_id": str(paper["id"]),
                "paper_version": item.paper_version,
                "title": item.title,
                "authors": item.authors,
                "year": item.year,
                "doi": item.doi,
                "url": item.url,
                "locator": {
                    "page": item.page,
                    "section": item.section,
                    "equation": item.equation,
                    "figure": item.figure,
                    "table": item.table,
                    "provider": item.provider_locator,
                },
                "text": item.source_excerpt,
                "extraction_method": item.extraction_method,
                "extraction_timestamp": datetime.now(UTC).isoformat(),
                "candidate_fingerprint": fingerprint,
            },
            "LITERATURE_EVIDENCE_CANDIDATE_CREATED",
            "CANDIDATE",
        )
        self.store.edge_create(str(paper["id"]), evidence["id"], "CONTAINS_EVIDENCE")
        return evidence
