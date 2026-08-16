import math
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError, field_validator

from .errors import GPUError
from .research import ResearchStore


class HypothesisDraft(BaseModel):
    statement: str | None = Field(default=None, max_length=20_000)
    mechanism: str = Field(min_length=10, max_length=20_000)
    prediction: str = Field(min_length=5, max_length=10_000)
    kill_condition: str = Field(min_length=5, max_length=10_000)
    niche_id: str
    parent_ids: list[str] = Field(default_factory=list, max_length=50)
    assumptions: list[str] = Field(default_factory=list, max_length=100)
    variables: list[str] = Field(default_factory=list, max_length=100)
    information_path: list[str] = Field(default_factory=list, max_length=100)
    scope: str = Field(min_length=1, max_length=10_000)
    scientific_difference: str | None = Field(default=None, max_length=10_000)
    state_variables: list[str] = Field(default_factory=list, max_length=100)
    inherited_assumptions: list[str] = Field(default_factory=list, max_length=100)
    assumptions_removed: list[str] = Field(default_factory=list, max_length=100)
    supporting_evidence: list[str] = Field(default_factory=list, max_length=100)
    against_evidence: list[str] = Field(default_factory=list, max_length=100)
    unique_predictions: list[str] = Field(default_factory=list, max_length=20)
    cheapest_kill_test: str | None = Field(default=None, max_length=10_000)
    alternative_explanations: list[str] = Field(default_factory=list, max_length=30)
    expected_scope: str | dict[str, Any] | None = None
    novelty_risk: str | None = Field(default=None, max_length=5000)
    operator_provenance: dict[str, Any] | None = None
    embedding: list[float] | None = Field(default=None, max_length=4096)

    @field_validator("embedding")
    @classmethod
    def finite_embedding(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and (not value or any(not math.isfinite(item) for item in value)):
            raise ValueError("embedding must contain one or more finite values")
        return value


class OperatorResult(BaseModel):
    operator: str
    advisory: bool = True
    findings: list[dict[str, Any]] = Field(default_factory=list)
    accepted: bool = True


@runtime_checkable
class ResearchOperator(Protocol):
    def run(self, context: dict[str, Any]) -> OperatorResult: ...


class HypothesisGenerator:
    """Typed candidate intake; durable creation remains owned by the QD service."""

    def run(self, context: dict[str, Any]) -> OperatorResult:
        drafts = [HypothesisDraft.model_validate(item) for item in context.get("drafts", [])]
        return OperatorResult(
            operator="HypothesisGenerator",
            findings=[{"draft": item.model_dump(mode="json")} for item in drafts],
            accepted=bool(drafts),
        )


class ScientificReflector:
    """Check falsifiability and scope without deciding whether a hypothesis is true."""

    def run(self, context: dict[str, Any]) -> OperatorResult:
        draft = HypothesisDraft.model_validate(context["draft"])
        findings = []
        if draft.prediction.lower().strip() == draft.mechanism.lower().strip():
            findings.append({"code": "PREDICTION_RESTATES_MECHANISM"})
        if draft.kill_condition.lower().strip() in draft.prediction.lower():
            findings.append({"code": "KILL_CONDITION_NOT_DISCRIMINATING"})
        if not draft.assumptions:
            findings.append({"code": "ASSUMPTIONS_UNSTATED"})
        if not draft.variables or len(draft.information_path) < 2:
            findings.append({"code": "MECHANISM_STRUCTURE_INCOMPLETE"})
        return OperatorResult(
            operator="ScientificReflector", findings=findings, accepted=not findings
        )


class ProximityCritic:
    """Interpret lexical/vector retrieval plus structured mechanism overlap."""

    def run(self, context: dict[str, Any]) -> OperatorResult:
        findings = []
        for match in context.get("matches", []):
            codes = match.get("flags", [])
            if codes:
                findings.append(
                    {
                        "related_id": match["id"],
                        "codes": codes,
                        "inherited_assumptions": match.get("inherited_assumptions", []),
                        "scientific_difference_required": any(
                            code in {"SUPERFICIAL_DUPLICATE", "DESCENDANT_OF_DEAD_IDEA"}
                            for code in codes
                        ),
                    }
                )
        return OperatorResult(
            operator="ProximityCritic", findings=findings, accepted=not findings
        )


class HypothesisQDService:
    """Native mechanistic-niche and lineage policy over canonical Research OS state."""

    def __init__(self, store: ResearchStore):
        self.store = store

    def niche_create(
        self, project_id: str, name: str, description: str, diversity_signature: dict[str, Any]
    ) -> dict[str, Any]:
        if not name.strip() or not description.strip() or not diversity_signature:
            raise GPUError(
                "INVALID_HYPOTHESIS_NICHE",
                "Name, description, and a structured diversity signature are required",
            )
        return self.store.hypothesis_niche_create(
            project_id,
            name.strip(),
            description.strip(),
            diversity_signature,
        )

    def niche_list(self, project_id: str) -> list[dict[str, Any]]:
        return self.store.objects_list(project_id, "HypothesisNiche", limit=None)

    def screen(self, project_id: str, draft_data: dict[str, Any]) -> dict[str, Any]:
        draft = self._draft(draft_data)
        niche = self._expect(draft.niche_id, project_id, "HypothesisNiche")
        lexical = self.store.related_hypotheses(project_id, draft.mechanism, 100)
        semantic: list[dict[str, Any]] = []
        semantic_unavailable = False
        if draft.embedding:
            try:
                semantic = self.store.semantic_search(project_id, draft.embedding, None, 100)
            except GPUError as exc:
                if exc.error_type != "PGVECTOR_UNAVAILABLE":
                    raise
                semantic_unavailable = True
        matches: dict[str, dict[str, Any]] = {}
        for item in lexical:
            matches[str(item["id"])] = {
                **item,
                "semantic_similarity": None,
            }
        for item in semantic:
            if item["kind"] not in {"Hypothesis", "NegativeResult"}:
                continue
            identity = str(item["id"])
            matches.setdefault(identity, {**item, "lexical_similarity": 0.0, "containment_similarity": 0.0})
            matches[identity]["semantic_similarity"] = round(
                max(-1.0, min(1.0, 1.0 - float(item["distance"]))), 4
            )
        enriched = [self._structured_match(draft, item) for item in matches.values()]
        enriched = [
            item
            for item in enriched
            if max(
                item.get("lexical_similarity") or 0,
                item.get("semantic_similarity") or 0,
                item["structured_similarity"],
            )
            >= 0.25
        ]
        enriched.sort(
            key=lambda item: max(
                item.get("lexical_similarity") or 0,
                item.get("semantic_similarity") or 0,
                item["structured_similarity"],
            ),
            reverse=True,
        )
        critic = ProximityCritic().run({"matches": enriched})
        reflection = ScientificReflector().run({"draft": draft.model_dump(mode="json")})
        blockers = [
            finding
            for finding in critic.findings
            if finding["scientific_difference_required"]
        ]
        accepted = not blockers or bool(draft.scientific_difference)
        return {
            "draft": draft.model_dump(mode="json"),
            "niche": niche,
            "matches": enriched,
            "similar_active_hypothesis_ids": [
                item["id"]
                for item in enriched
                if item["kind"] == "Hypothesis" and item["status"] != "REFUTED"
            ],
            "similar_dead_hypothesis_ids": [
                item["id"]
                for item in enriched
                if item["kind"] == "NegativeResult" or item["status"] == "REFUTED"
            ],
            "proximity_critic": critic.model_dump(mode="json"),
            "scientific_reflector": reflection.model_dump(mode="json"),
            "semantic_retrieval_unavailable": semantic_unavailable,
            "accepted": accepted,
            "warning": "Similarity affects novelty screening, never scientific truth.",
        }

    def create(self, project_id: str, draft_data: dict[str, Any]) -> dict[str, Any]:
        screened = self.screen(project_id, draft_data)
        if not screened["accepted"]:
            raise GPUError(
                "HYPOTHESIS_PROXIMITY_BLOCKED",
                "Explain the changed assumption in scientific_difference before creating this descendant",
            )
        draft = self._draft(screened["draft"])
        ancestors = self._ancestors(project_id, draft.parent_ids)
        related_ids = {
            *screened["similar_active_hypothesis_ids"],
            *screened["similar_dead_hypothesis_ids"],
        }
        edges = [(item, "PARENT_OF") for item in draft.parent_ids]
        edges.append((draft.niche_id, "CONTAINS_HYPOTHESIS"))
        edges.extend((item, "MECHANISTICALLY_RELATED_TO") for item in related_ids)
        result = self.store.hypothesis_create_with_edges(
            project_id,
            {
                **draft.model_dump(mode="json", exclude={"embedding"}),
                "niche": screened["niche"]["data"]["name"],
                "ancestor_ids": ancestors,
                "similar_active_hypothesis_ids": screened["similar_active_hypothesis_ids"],
                "similar_dead_hypothesis_ids": screened["similar_dead_hypothesis_ids"],
            },
            edges,
        )
        embedding_status = "not_requested"
        if draft.embedding:
            embedding_status = "unavailable"
            if getattr(self.store, "vector_available", False):
                try:
                    self.store.embedding_set(result["id"], draft.embedding)
                    embedding_status = "stored"
                except GPUError:
                    # Embeddings are a replaceable retrieval index, not canonical hypothesis state.
                    embedding_status = "failed_noncanonical_cache"
        return {**result, "screening": screened, "embedding_status": embedding_status}

    def niche_set_best(
        self, niche_id: str, hypothesis_id: str, rationale: str
    ) -> dict[str, Any]:
        if not rationale.strip():
            raise GPUError("NICHE_SELECTION_RATIONALE_REQUIRED", niche_id)
        return self.store.hypothesis_niche_set_best(niche_id, hypothesis_id, rationale.strip())

    def _ancestors(self, project_id: str, parent_ids: list[str]) -> list[str]:
        ancestors, queue = set(), list(parent_ids)
        while queue:
            current = queue.pop(0)
            if current in ancestors:
                continue
            if len(ancestors) >= 200:
                raise GPUError("HYPOTHESIS_LINEAGE_TOO_DEEP", "More than 200 ancestors")
            parent = self._expect(current, project_id, "Hypothesis")
            ancestors.add(current)
            queue.extend(str(item) for item in parent["data"].get("parent_ids", []))
        return sorted(ancestors)

    def _expect(self, object_id: str, project_id: str, kind: str) -> dict[str, Any]:
        item = self.store.object_get(object_id)
        if item["kind"] != kind:
            raise GPUError(f"NOT_A_{kind.upper()}", object_id)
        if str(item["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", object_id)
        return item

    @staticmethod
    def _draft(data: dict[str, Any]) -> HypothesisDraft:
        try:
            return HypothesisDraft.model_validate(data)
        except ValidationError as exc:
            raise GPUError("INVALID_HYPOTHESIS_DRAFT", str(exc)) from exc

    @classmethod
    def _structured_match(
        cls, draft: HypothesisDraft, item: dict[str, Any]
    ) -> dict[str, Any]:
        data = item["data"]
        candidate = {
            "niche_id": draft.niche_id,
            "assumptions": draft.assumptions,
            "variables": draft.variables,
            "information_path": draft.information_path,
            "scope": draft.scope,
        }
        components = {
            "same_niche": float(str(data.get("niche_id")) == draft.niche_id),
            "assumption_overlap": cls._jaccard(candidate["assumptions"], data.get("assumptions", [])),
            "variable_overlap": cls._jaccard(candidate["variables"], data.get("variables", [])),
            "information_path_overlap": cls._jaccard(
                candidate["information_path"], data.get("information_path", [])
            ),
            "same_scope": float(bool(data.get("scope")) and data.get("scope") == draft.scope),
        }
        structured = round(sum(components.values()) / len(components), 4)
        related_assumptions = list(data.get("assumptions", []))
        if data.get("failed_assumption"):
            related_assumptions.append(data["failed_assumption"])
        inherited = sorted(
            {value.lower().strip(): value for value in draft.assumptions}.keys()
            & {str(value).lower().strip() for value in related_assumptions}
        )
        dead = item["kind"] == "NegativeResult" or item["status"] == "REFUTED"
        lexical = float(item.get("containment_similarity") or 0)
        semantic = float(item.get("semantic_similarity") or 0)
        flags = []
        if max(lexical, semantic) >= 0.8 and structured >= 0.5:
            flags.append("SUPERFICIAL_DUPLICATE")
        if dead and (inherited or max(lexical, semantic) >= 0.6):
            flags.append("DESCENDANT_OF_DEAD_IDEA")
        return {
            **item,
            "id": str(item["id"]),
            "structured_components": components,
            "structured_similarity": structured,
            "inherited_assumptions": inherited,
            "flags": flags,
        }

    @staticmethod
    def _jaccard(first: list[str], second: list[str]) -> float:
        left = {str(item).lower().strip() for item in first if str(item).strip()}
        right = {str(item).lower().strip() for item in second if str(item).strip()}
        return round(len(left & right) / len(left | right), 4) if left | right else 0.0
