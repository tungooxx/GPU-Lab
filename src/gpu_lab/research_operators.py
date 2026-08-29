from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import GPUError
from .qd import HypothesisQDService
from .research import ResearchStore

OPERATOR_NAMES = {
    "HypothesisGenerator",
    "MechanismCritic",
    "NullModelCritic",
    "ExperimentalDesignCritic",
    "ProximityCritic",
    "NoveltyCritic",
    "SearchPortfolioCritic",
}
OPERATOR_PROMPT_VERSION = "brain-v2-operators-1"
OPERATOR_SCHEMA_VERSION = "1.0"


class OperatorProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator_name: str
    provider: str
    model: str
    model_version: str | None = None
    prompt_version: str = OPERATOR_PROMPT_VERSION
    schema_version: str = OPERATOR_SCHEMA_VERSION
    context_hash: str
    timestamp: datetime


class HypothesisProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=10, max_length=20_000)
    mechanism: str = Field(min_length=10, max_length=20_000)
    state_variables: list[str] = Field(min_length=1, max_length=100)
    information_path: list[str] = Field(min_length=2, max_length=100)
    assumptions: list[str] = Field(default_factory=list, max_length=100)
    inherited_assumptions: list[str] = Field(default_factory=list, max_length=100)
    assumptions_removed: list[str] = Field(default_factory=list, max_length=100)
    scientific_difference: str = Field(min_length=5, max_length=10_000)
    niche_id: str = Field(min_length=1)
    supporting_evidence: list[str] = Field(default_factory=list, max_length=100)
    against_evidence: list[str] = Field(default_factory=list, max_length=100)
    unique_predictions: list[str] = Field(min_length=1, max_length=20)
    cheapest_kill_test: str = Field(min_length=5, max_length=10_000)
    alternative_explanations: list[str] = Field(min_length=1, max_length=30)
    expected_scope: str | dict[str, Any]
    novelty_risk: str = Field(min_length=1, max_length=5000)


class HypothesisGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypotheses: list[HypothesisProposal] = Field(min_length=3, max_length=5)
    provenance: OperatorProvenance


class AlternativeExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=1000)
    mechanism: str = Field(min_length=5, max_length=10_000)
    why_plausible: str = Field(min_length=5, max_length=10_000)
    evidence_for: list[str] = Field(default_factory=list, max_length=100)
    evidence_against: list[str] = Field(default_factory=list, max_length=100)
    discriminating_control: str = Field(min_length=5, max_length=10_000)
    estimated_cost: str | dict[str, Any]


class NullModelCritique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_claim: str = Field(min_length=5, max_length=20_000)
    alternative_explanations: list[AlternativeExplanation] = Field(min_length=1, max_length=30)
    missing_controls: list[str] = Field(default_factory=list, max_length=100)
    promotion_risk: str = Field(min_length=1, max_length=10_000)
    recommended_null_test: str = Field(min_length=5, max_length=10_000)
    provenance: OperatorProvenance


class CriticFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=200)
    severity: str = Field(pattern="^(INFO|WARNING|ERROR)$")
    description: str = Field(min_length=1, max_length=10_000)
    related_ids: list[str] = Field(default_factory=list, max_length=100)
    suggested_action: str | None = Field(default=None, max_length=10_000)


class CriticResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator_name: str
    findings: list[CriticFinding] = Field(default_factory=list, max_length=100)
    provenance: OperatorProvenance


@runtime_checkable
class ResearchOperatorProvider(Protocol):
    name: str
    model: str
    model_version: str | None

    async def run(self, operator_name: str, context: dict[str, Any]) -> dict[str, Any]: ...


class HttpResearchOperatorProvider:
    """Authenticated client for model reasoning inside the isolated literature worker."""

    name = "isolated-http"

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        model: str = "worker-configured",
        timeout_seconds: int = 180,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not base_url.startswith(("http://", "https://")):
            raise GPUError("INVALID_RESEARCH_OPERATOR_WORKER_URL", base_url)
        if not token:
            raise GPUError(
                "RESEARCH_OPERATOR_WORKER_TOKEN_REQUIRED",
                "Configure the task-scoped literature worker token",
            )
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.model = model
        self.model_version = None
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def run(self, operator_name: str, context: dict[str, Any]) -> dict[str, Any]:
        if operator_name not in OPERATOR_NAMES:
            raise GPUError("UNKNOWN_RESEARCH_OPERATOR", operator_name)
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                response = await client.post(
                    f"{self.base_url}/operator",
                    json={
                        "operator_name": operator_name,
                        "operator_context": context,
                        "prompt_version": OPERATOR_PROMPT_VERSION,
                        "schema_version": OPERATOR_SCHEMA_VERSION,
                    },
                    headers={"Authorization": f"Bearer {self.token}"},
                )
        except httpx.HTTPError as exc:
            raise GPUError(
                "RESEARCH_OPERATOR_UNAVAILABLE",
                f"The isolated operator worker failed during {operator_name}",
                retryable=True,
            ) from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise GPUError(
                "RESEARCH_OPERATOR_INVALID_RESPONSE",
                f"The operator worker returned non-JSON status {response.status_code}",
                retryable=response.status_code >= 500,
            ) from exc
        if not isinstance(body, dict):
            raise GPUError(
                "RESEARCH_OPERATOR_INVALID_RESPONSE",
                "The operator worker returned a non-object body",
                retryable=response.status_code >= 500,
            )
        if body.get("error") is not None:
            error = body["error"] if isinstance(body["error"], dict) else {}
            raise GPUError(
                error.get("type")
                if isinstance(error.get("type"), str)
                else "RESEARCH_OPERATOR_ERROR",
                error.get("message")
                if isinstance(error.get("message"), str)
                else "Research operator worker error",
                retryable=(
                    error["retryable"]
                    if isinstance(error.get("retryable"), bool)
                    else response.status_code >= 500
                ),
            )
        result = body.get("result")
        if response.is_error or not isinstance(result, dict):
            raise GPUError(
                "RESEARCH_OPERATOR_INVALID_RESPONSE",
                f"Missing operator result at status {response.status_code}",
                retryable=response.status_code >= 500,
            )
        return result


class ResearchOperatorService:
    """Typed advisory operators over bounded canonical context; never a scientific truth writer."""

    def __init__(
        self,
        store: ResearchStore,
        qd_service: HypothesisQDService,
        provider: ResearchOperatorProvider,
    ):
        self.store = store
        self.qd = qd_service
        self.provider = provider

    @staticmethod
    def _context_hash(context: dict[str, Any]) -> str:
        canonical = json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _provenance(self, operator_name: str, context: dict[str, Any]) -> OperatorProvenance:
        return OperatorProvenance(
            operator_name=operator_name,
            provider=self.provider.name,
            model=self.provider.model,
            model_version=self.provider.model_version,
            context_hash=self._context_hash(context),
            timestamp=datetime.now(UTC),
        )

    def hypothesis_context(
        self,
        project_id: str,
        agenda_item_id: str,
        discovery_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        agenda = self.store.object_get(agenda_item_id)
        if agenda["kind"] != "AgendaItem":
            raise GPUError("NOT_AN_AGENDAITEM", agenda_item_id)
        if str(agenda["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", agenda_item_id)

        def limited(kind: str, statuses: list[str] | None = None, limit: int = 25):
            return [
                {
                    "id": str(item["id"]),
                    "kind": item["kind"],
                    "status": item["status"],
                    "data": item["data"],
                }
                for item in self.store.objects_list(
                    project_id, kind, statuses=statuses, limit=limit
                )
            ]

        return {
            "agenda_item": {
                "id": str(agenda["id"]),
                "status": agenda["status"],
                "data": agenda["data"],
            },
            "world_models": limited("WorldModel", limit=2),
            "active_hypotheses": limited(
                "Hypothesis", ["ACTIVE", "SURVIVES_INITIAL_TEST", "SUPPORTED"]
            ),
            "dead_hypotheses": limited("Hypothesis", ["REFUTED", "WEAKENED"]),
            "negative_results": limited("NegativeResult"),
            "anomalies": limited("Anomaly"),
            "contradictions": limited("Contradiction"),
            "evidence": limited("EvidenceUnit", limit=30),
            "claims": limited("Claim", limit=30),
            "niches": limited("HypothesisNiche", limit=30),
            "constraints": {
                "candidate_count": "3-5",
                "advisory_only": True,
                "no_state_promotion": True,
                "untrusted_external_text": True,
            },
            "discovery_context": discovery_context or {
                "search_regime": "UNSPECIFIED",
                "required_scientific_distance": "UNSPECIFIED",
                "advisory_only": True,
            },
        }

    async def generate_hypotheses(
        self, project_id: str, agenda_item_id: str, *, persist: bool = False
    ) -> dict[str, Any]:
        context = self.hypothesis_context(project_id, agenda_item_id)
        raw = await self.provider.run("HypothesisGenerator", context)
        try:
            proposals = [HypothesisProposal.model_validate(item) for item in raw["hypotheses"]]
            if not 3 <= len(proposals) <= 5:
                raise ValueError("HypothesisGenerator must return 3 through 5 hypotheses")
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise GPUError("RESEARCH_OPERATOR_INVALID_RESPONSE", str(exc)) from exc
        provenance = self._provenance("HypothesisGenerator", context)
        niche_ids = {item["id"] for item in context["niches"]}
        results = []
        for proposal in proposals:
            if proposal.niche_id not in niche_ids:
                raise GPUError(
                    "RESEARCH_OPERATOR_UNKNOWN_NICHE",
                    f"Generated niche {proposal.niche_id} is not in the bounded context",
                )
            draft = {
                "statement": proposal.statement,
                "mechanism": proposal.mechanism,
                "prediction": proposal.unique_predictions[0],
                "unique_predictions": proposal.unique_predictions,
                "kill_condition": proposal.cheapest_kill_test,
                "cheapest_kill_test": proposal.cheapest_kill_test,
                "niche_id": proposal.niche_id,
                "assumptions": proposal.assumptions,
                "inherited_assumptions": proposal.inherited_assumptions,
                "assumptions_removed": proposal.assumptions_removed,
                "variables": proposal.state_variables,
                "state_variables": proposal.state_variables,
                "information_path": proposal.information_path,
                "scope": (
                    json.dumps(proposal.expected_scope, sort_keys=True)
                    if isinstance(proposal.expected_scope, dict)
                    else proposal.expected_scope
                ),
                "expected_scope": proposal.expected_scope,
                "scientific_difference": proposal.scientific_difference,
                # Generated proposals still traverse the same lineage gate as
                # manual/QD proposals. The operator's mechanism is the causal
                # claim; its proposed intervention is retained separately.
                "mechanistic_hypothesis": proposal.mechanism,
                "enabling_method": proposal.cheapest_kill_test,
                "supporting_evidence": proposal.supporting_evidence,
                "against_evidence": proposal.against_evidence,
                "alternative_explanations": proposal.alternative_explanations,
                "novelty_risk": proposal.novelty_risk,
                "operator_provenance": provenance.model_dump(mode="json"),
            }
            screening = self.qd.screen(project_id, draft)
            if not proposal.alternative_explanations:
                screening = {**screening, "accepted": False, "null_model_gate": "missing"}
            created = self.qd.create(project_id, draft) if persist and screening["accepted"] else None
            results.append(
                {
                    "proposal": proposal.model_dump(mode="json"),
                    "screening": screening,
                    "persisted": created,
                }
            )
        return HypothesisGeneration(
            hypotheses=proposals, provenance=provenance
        ).model_dump(mode="json") | {
            "screened_candidates": results,
            "persist_requested": persist,
            "warning": "Model output is advisory and cannot promote scientific truth.",
        }

    async def null_model_critique(
        self, project_id: str, target_claim: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        bounded = {
            "project_id": project_id,
            "target_claim": target_claim,
            "scientific_context": context,
            "required_null_families": [
                "random perturbation",
                "magnitude-matched perturbation",
                "metric or sampling artifact",
                "checkpoint or seed artifact",
                "implementation artifact",
            ],
            "advisory_only": True,
        }
        raw = await self.provider.run("NullModelCritic", bounded)
        try:
            payload = NullModelCritique.model_validate(
                {**raw, "provenance": self._provenance("NullModelCritic", bounded)}
            )
        except ValidationError as exc:
            raise GPUError("RESEARCH_OPERATOR_INVALID_RESPONSE", str(exc)) from exc
        return payload.model_dump(mode="json")

    async def critique(
        self, operator_name: str, project_id: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        if operator_name not in OPERATOR_NAMES - {"HypothesisGenerator", "NullModelCritic"}:
            raise GPUError("UNKNOWN_RESEARCH_OPERATOR", operator_name)
        bounded = {"project_id": project_id, "scientific_context": context, "advisory_only": True}
        raw = await self.provider.run(operator_name, bounded)
        try:
            payload = CriticResponse.model_validate(
                {
                    "operator_name": operator_name,
                    "findings": raw.get("findings", []),
                    "provenance": self._provenance(operator_name, bounded),
                }
            )
        except ValidationError as exc:
            raise GPUError("RESEARCH_OPERATOR_INVALID_RESPONSE", str(exc)) from exc
        return payload.model_dump(mode="json")
