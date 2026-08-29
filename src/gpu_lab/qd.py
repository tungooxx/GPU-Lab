import math
import json
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError, field_validator

from .errors import GPUError
from .research import ResearchStore


_LINEAGE_KINDS = {
    "Hypothesis", "NegativeResult", "Claim", "MetaLesson", "ExperimentRun",
    "Experiment", "ResearchDecision", "EvidenceUnit", "EngineeringResult",
}
_PLACEHOLDER_VALUES = {"", "noop", "todo", "test", "tbd", "none", "null", "dummy", "n/a"}


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
    enabling_method: str | None = Field(default=None, max_length=10_000)
    mechanistic_hypothesis: str | None = Field(default=None, max_length=20_000)
    original_created_at: str | None = Field(default=None, max_length=100)
    lineage_responses: list[dict[str, Any]] = Field(default_factory=list, max_length=200)
    falsified_prerequisites: list[str] = Field(default_factory=list, max_length=100)
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

    @staticmethod
    def _words(value: Any) -> set[str]:
        return {
            word for word in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(value).lower())
            if word not in {"that", "with", "from", "this", "will", "when", "where", "into", "then"}
        }

    @staticmethod
    def _epistemic_classification(item: dict[str, Any]) -> str:
        data, kind, status = item.get("data", {}), item.get("kind"), str(item.get("status", "")).upper()
        text = " ".join(str(data.get(key, "")) for key in ("scientific_role", "scientific_verification", "execution_verification", "failure_mode", "summary")) .upper()
        if any(token in text for token in ("TECHNICAL_INVALID", "NOT_SCIENTIFIC", "SYSTEM_SMOKE", "CONTRACT_TEST")):
            return "TECHNICAL_INVALID"
        if "CONSTRUCT" in text or "IMPLEMENTATION" in text:
            return "CONSTRUCTIBILITY_ONLY"
        if kind == "NegativeResult" and ("NOOP" in text or "INVALID" in text):
            return "NOOP/INVALID"
        if kind == "NegativeResult":
            return "SCIENTIFIC_VALID"
        if kind in {"ExperimentRun", "Experiment", "EvidenceUnit"}:
            return "CAUSAL_RESULT" if any(token in text for token in ("CAUSAL", "INTERVENTION", "DECISIVE")) else "REPRESENTATION_RESULT"
        if status in {"REFUTED", "WEAKENED"}:
            return "SCIENTIFIC_VALID"
        return "TECHNICAL_ONLY" if "TECHNICAL" in text else "SCIENTIFIC_VALID"

    def discovery_context_build(self, project_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
        """Retrieve related lineage across canonical records, not merely supplied parents."""
        mechanism = str(candidate.get("mechanism") or candidate.get("mechanistic_hypothesis") or "")
        if mechanism.strip().lower() in _PLACEHOLDER_VALUES:
            raise GPUError("DISCOVERY_PLACEHOLDER_WRITE_REJECTED", "mechanism must not be a placeholder")
        candidate_words = self._words(" ".join(str(candidate.get(key, "")) for key in (
            "mechanism", "mechanistic_hypothesis", "enabling_method", "prediction", "assumptions", "variables", "scope"
        )))
        parent_ids = {str(item) for item in candidate.get("parent_ids", [])}
        if hasattr(self.store, "_connect"):
            with self.store._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                    "WHERE project_id=%s AND kind=ANY(%s) ORDER BY created_at",
                    (project_id, list(_LINEAGE_KINDS)),
                )
                records = [dict(row) for row in cur.fetchall()]
        else:  # Lightweight stores used by deterministic policy/unit tests.
            records = [
                item for kind in _LINEAGE_KINDS
                for item in self.store.objects_list(project_id, kind, limit=None)
            ]
        related = []
        for record in records:
            data = record.get("data") or {}
            record_words = self._words(" ".join(str(data.get(key, "")) for key in (
                "mechanism", "proposal", "prediction", "failed_assumption", "summary", "question", "title", "description", "assumptions", "variables"
            )))
            overlap = len(candidate_words & record_words)
            linked = str(record["id"]) in parent_ids or bool(parent_ids & {str(item) for item in data.get("parent_ids", [])})
            if linked or overlap >= 2:
                related.append({
                    "id": str(record["id"]), "kind": record["kind"], "status": record["status"],
                    "created_at": record["created_at"].isoformat() if hasattr(record.get("created_at"), "isoformat") else (str(record["created_at"]) if record.get("created_at") else None),
                    "overlap_terms": sorted(candidate_words & record_words), "explicit_lineage": linked,
                    "epistemic_classification": self._epistemic_classification(record),
                    "failed_assumption": data.get("failed_assumption") or data.get("failure_mode"),
                    "summary": data.get("summary") or data.get("mechanism") or data.get("proposal") or data.get("question"),
                })
        created_after = candidate.get("original_created_at")
        original = [item for item in related if not created_after or (item["created_at"] or "") <= created_after]
        later = [item for item in related if created_after and (item["created_at"] or "") > created_after]
        return {
            "candidate_mechanism": mechanism, "related_records": related,
            "original_discovery_basis": original,
            "current_evidence_that_should_modify_the_design": later or related,
            "retrieval_scope": {"searched_kinds": sorted(_LINEAGE_KINDS), "parent_ids": sorted(parent_ids)},
        }

    def hypothesis_lineage_audit(self, project_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
        context = self.discovery_context_build(project_id, candidate)
        responses = {str(item.get("record_id")): item for item in candidate.get("lineage_responses", []) if isinstance(item, dict)}
        important = [
            record for record in context["related_records"]
            if record["epistemic_classification"] in {"SCIENTIFIC_VALID", "REPRESENTATION_RESULT", "CAUSAL_RESULT"}
        ]
        table, missing = [], []
        for record in important:
            response = responses.get(record["id"], {})
            addressed = response.get("candidate_addresses") is True
            table.append({
                "experiment_or_record": record["id"], "result": record["summary"],
                "failed_assumption": record["failed_assumption"],
                "implication_for_candidate": response.get("implication_for_candidate"),
                "candidate_already_addresses_it": addressed,
            })
            if not addressed and not response.get("implication_for_candidate"):
                missing.append(record["id"])
        failed_prerequisites = [item["failed_assumption"] for item in important if item.get("failed_assumption")]
        declared = {str(item).strip().lower() for item in candidate.get("falsified_prerequisites", [])}
        undeclared = [item for item in failed_prerequisites if str(item).strip().lower() not in declared]
        design_fields_missing = [key for key in ("enabling_method", "mechanistic_hypothesis") if str(candidate.get(key, "")).strip().lower() in _PLACEHOLDER_VALUES]
        blockers = []
        if missing:
            blockers.append("DISCOVERY_LINEAGE_INCOMPLETE")
        if undeclared:
            blockers.append("DISCOVERY_DEAD_ASSUMPTION_UNDECLARED")
        if design_fields_missing:
            blockers.append("DISCOVERY_METHOD_MECHANISM_UNDISTINGUISHED")
        return {
            **context, "cross_lineage_synthesis": table,
            "falsified_prerequisites_still_relevant": undeclared,
            "design_fields_missing": design_fields_missing, "blockers": blockers,
            "passed": not blockers,
        }

    def discovery_adversarial_check(self, project_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
        audit = self.hypothesis_lineage_audit(project_id, candidate)
        strongest = next((item for item in audit["related_records"] if item.get("failed_assumption")), None)
        return {
            "passed": audit["passed"], "audit": audit,
            "strongest_existing_counterevidence": strongest,
            "questions": ["What existing result most strongly argues against this idea?", "What prior experiment makes it redundant?", "What assumptions have already failed?", "What evidence is merely technical?"],
        }

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
        audit = self.hypothesis_lineage_audit(project_id, draft_data)
        hard_blockers = [item for item in audit["blockers"] if item != "DISCOVERY_METHOD_MECHANISM_UNDISTINGUISHED"]
        # Lightweight in-memory stores exercise QD matching only; canonical
        # PostgreSQL paths enforce the durable cross-lineage gate.
        if hard_blockers and hasattr(self.store, "_connect"):
            raise GPUError("DISCOVERY_LINEAGE_INCOMPLETE", json.dumps({"blockers": audit["blockers"], "record_ids": [row["experiment_or_record"] for row in audit["cross_lineage_synthesis"]]}))
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
                "lineage_audit": audit,
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
