from __future__ import annotations

import json
import uuid
from collections import defaultdict
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import GPUError
from .research import ResearchStore


class EvidenceOriginType(StrEnum):
    EXPERIMENT = "EXPERIMENT"
    REPRODUCTION = "REPRODUCTION"
    PAPER = "PAPER"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    INDEPENDENT_REPLICATION = "INDEPENDENT_REPLICATION"
    OTHER = "OTHER"


ORIGIN_KINDS = {
    EvidenceOriginType.EXPERIMENT: {"ExperimentRun"},
    EvidenceOriginType.REPRODUCTION: {"Reproduction"},
    EvidenceOriginType.PAPER: {"Paper"},
    EvidenceOriginType.DATA_ANALYSIS: {"ExperimentRun", "Artifact", "EvidenceUnit"},
    EvidenceOriginType.INDEPENDENT_REPLICATION: {"ExperimentRun", "Reproduction"},
    EvidenceOriginType.OTHER: None,
}
CANONICAL_ORIGIN_TYPES = {
    "ExperimentRun": EvidenceOriginType.EXPERIMENT,
    "Reproduction": EvidenceOriginType.REPRODUCTION,
    "Paper": EvidenceOriginType.PAPER,
    "Artifact": EvidenceOriginType.DATA_ANALYSIS,
    "EvidenceUnit": EvidenceOriginType.DATA_ANALYSIS,
}

CAUSAL_SUPPORT_LEVELS = {
    "NONE",
    "SINGLE_INTERVENTION",
    "REPLICATED_WITHIN_SCOPE",
    "ROBUST_WITHIN_ARCHITECTURE",
    "CROSS_ARCHITECTURE_SUPPORT",
    "CROSS_DOMAIN_SUPPORT",
}


class ScientificScope(BaseModel):
    """Explicit bounded scope for evidence-backed causal claims."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    models: list[str] = Field(default_factory=list)
    architectures: list[str] = Field(default_factory=list)
    checkpoints: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


def normalize_scientific_scope(value: str | dict[str, Any] | None) -> dict[str, Any]:
    """Normalize legacy free text without manufacturing generalization evidence."""
    candidate: dict[str, Any]
    if value is None:
        candidate = {}
    elif isinstance(value, str):
        candidate = {"description": value.strip()}
    elif isinstance(value, dict):
        candidate = value
    else:
        raise GPUError("INVALID_SCIENTIFIC_SCOPE", "Scope must be an object or text")
    try:
        scope = ScientificScope.model_validate(candidate)
    except ValidationError as exc:
        raise GPUError("INVALID_SCIENTIFIC_SCOPE", str(exc)) from exc
    return scope.model_dump(mode="json")


def scope_is_empirically_bounded(scope: dict[str, Any]) -> bool:
    """An intervention claim needs concrete experimental dimensions, not prose alone."""
    return bool(
        scope.get("models")
        or scope.get("architectures")
        or scope.get("checkpoints")
        or scope.get("datasets")
        or scope.get("objects")
    ) and bool(scope.get("interventions") and scope.get("metrics"))


class EpistemicService:
    """Evidence independence and belief-quality services over canonical Research OS state."""

    def __init__(self, store: ResearchStore):
        self.store = store

    def evidence_family_create(
        self,
        project_id: str,
        origin_type: str,
        origin_id: str,
        description: str,
        derived_from_evidence_family_id: str | None = None,
        dependency_note: str | None = None,
    ) -> dict:
        try:
            typed_origin = EvidenceOriginType(origin_type)
        except ValueError as exc:
            raise GPUError("INVALID_EVIDENCE_ORIGIN_TYPE", origin_type) from exc
        origin = self.store.object_get(origin_id)
        if str(origin["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", origin_id)
        allowed_kinds = ORIGIN_KINDS[typed_origin]
        if allowed_kinds is not None and origin["kind"] not in allowed_kinds:
            raise GPUError(
                "INVALID_EVIDENCE_ORIGIN",
                f"{typed_origin.value} cannot originate from {origin['kind']}",
            )
        parent = None
        if derived_from_evidence_family_id:
            parent = self.store.object_get(derived_from_evidence_family_id)
            if parent["kind"] != "EvidenceFamily":
                raise GPUError("NOT_AN_EVIDENCEFAMILY", derived_from_evidence_family_id)
            if str(parent["project_id"]) != str(project_id):
                raise GPUError("RESEARCH_PROJECT_MISMATCH", derived_from_evidence_family_id)
            if not dependency_note or not dependency_note.strip():
                raise GPUError(
                    "EVIDENCE_DEPENDENCY_NOTE_REQUIRED",
                    "Dependent evidence must explain the dependency",
                )
        if not description.strip():
            raise GPUError("EVIDENCE_FAMILY_DESCRIPTION_REQUIRED", origin_id)
        canonical_origin = CANONICAL_ORIGIN_TYPES.get(origin["kind"], EvidenceOriginType.OTHER)
        independence_key = f"ORIGIN:{origin_id}"
        return self.store.evidence_family_create_atomic(
            project_id,
            {
                "origin_type": canonical_origin.value,
                "origin_roles": [typed_origin.value],
                "origin_id": origin_id,
                "independence_key": independence_key,
                "description": description.strip(),
                "derived_from_evidence_family_id": (
                    str(parent["id"]) if parent is not None else None
                ),
                "dependency_note": dependency_note.strip() if dependency_note else None,
            },
        )

    def evidence_family_link(
        self, family_id: str, entity_id: str, relationship: str = "DERIVED"
    ) -> dict:
        return self.store.evidence_family_link_atomic(
            family_id, entity_id, relationship.strip().upper()
        )

    def _families(
        self, entity_id: str, as_of: str | None = None
    ) -> tuple[dict, list[dict]]:
        temporal = {"as_of": as_of} if as_of is not None else {}
        entity = self.store.object_get(entity_id, **temporal)
        data = entity["data"]
        identifiers = list(
            dict.fromkeys(
                [
                    *data.get("evidence_family_ids", []),
                    *data.get("supporting_evidence_family_ids", []),
                    *data.get("contradicting_evidence_family_ids", []),
                ]
            )
        )
        records = self.store.references_get(identifiers, **temporal)
        pending = {
            str(item["data"].get("derived_from_evidence_family_id"))
            for item in records.values()
            if item["data"].get("derived_from_evidence_family_id")
        } - set(records)
        while pending:
            ancestors = self.store.references_get(sorted(pending), **temporal)
            if set(ancestors) != pending:
                missing = min(pending - set(ancestors))
                raise GPUError("INVALID_EVIDENCE_FAMILY_REFERENCE", missing)
            records.update(ancestors)
            pending = {
                str(item["data"].get("derived_from_evidence_family_id"))
                for item in ancestors.values()
                if item["data"].get("derived_from_evidence_family_id")
            } - set(records)
        families = []
        for identifier, family in records.items():
            if not family or family["kind"] != "EvidenceFamily":
                raise GPUError("INVALID_EVIDENCE_FAMILY_REFERENCE", str(identifier))
            if family["project_id"] != entity["project_id"]:
                raise GPUError("RESEARCH_PROJECT_MISMATCH", str(identifier))
            families.append(family)
        return entity, families

    @staticmethod
    def _direct_family_ids(entity: dict) -> set[str]:
        return {
            str(item)
            for item in [
                *entity["data"].get("evidence_family_ids", []),
                *entity["data"].get("supporting_evidence_family_ids", []),
                *entity["data"].get("contradicting_evidence_family_ids", []),
            ]
        }

    @staticmethod
    def _family_roots(families: list[dict]) -> dict[str, str]:
        by_id = {str(item["id"]): item for item in families}
        roots: dict[str, str] = {}
        for family_id in by_id:
            current = family_id
            visited: set[str] = set()
            while True:
                if current in visited:
                    raise GPUError(
                        "EVIDENCE_DEPENDENCY_CYCLE",
                        f"Cycle contains EvidenceFamily {current}",
                    )
                visited.add(current)
                parent = by_id[current]["data"].get("derived_from_evidence_family_id")
                if not parent or str(parent) not in by_id:
                    roots[family_id] = str(parent) if parent else current
                    break
                current = str(parent)
        return roots

    def independent_evidence_count(self, entity_id: str, as_of: str | None = None) -> dict:
        entity, families = self._families(entity_id, as_of)
        roots = self._family_roots(families)
        direct = self._direct_family_ids(entity)
        return {
            "entity_id": str(entity["id"]),
            "derived_record_count": sum(
                len(family["data"].get("derived_record_ids", []))
                for family in families
                if str(family["id"]) in direct
            ),
            "evidence_family_count": len(direct),
            "independent_evidence_count": len({roots[item] for item in direct}),
            "independence_roots": {item: roots[item] for item in direct},
            "as_of": as_of,
        }

    def supporting_evidence_families(
        self, entity_id: str, as_of: str | None = None
    ) -> list[dict]:
        entity, families = self._families(entity_id, as_of)
        supporting = set(entity["data"].get("supporting_evidence_family_ids", []))
        return [family for family in families if str(family["id"]) in supporting]

    def contradicting_evidence_families(
        self, entity_id: str, as_of: str | None = None
    ) -> list[dict]:
        entity, families = self._families(entity_id, as_of)
        contradicting = set(entity["data"].get("contradicting_evidence_family_ids", []))
        return [family for family in families if str(family["id"]) in contradicting]

    def group_evidence_by_origin(self, entity_id: str, as_of: str | None = None) -> dict:
        entity, families = self._families(entity_id, as_of)
        roots = self._family_roots(families)
        direct = self._direct_family_ids(entity)
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for family in families:
            if str(family["id"]) not in direct:
                continue
            groups[roots[str(family["id"])]].append(
                {
                    "family_id": str(family["id"]),
                    "origin_type": family["data"]["origin_type"],
                    "origin_id": family["data"]["origin_id"],
                    "independence_key": family["data"]["independence_key"],
                    "dependency_note": family["data"].get("dependency_note"),
                    "derived_record_ids": family["data"].get("derived_record_ids", []),
                }
            )
        return {
            "entity_id": str(entity["id"]),
            "groups": dict(groups),
            "independent_evidence_count": len(groups),
            "as_of": as_of,
        }

    @staticmethod
    def _scope_generalization(scope: dict[str, Any], field: str) -> str:
        values = {str(value) for value in scope.get(field, []) if str(value).strip()}
        if not values:
            return "NOT_TESTED"
        return "MULTIPLE_SCOPES" if len(values) > 1 else "SINGLE_SCOPE"

    def belief_audit(self, entity_id: str, as_of: str | None = None) -> dict:
        """Explain support, dependence, scope, and promotion risk without fake confidence."""
        entity, families = self._families(entity_id, as_of)
        if entity["kind"] not in {"Hypothesis", "Claim", "Mechanism", "CausalEdge"}:
            raise GPUError("BELIEF_AUDIT_KIND_UNSUPPORTED", entity["kind"])
        roots = self._family_roots(families)
        supporting_ids = {
            str(item) for item in entity["data"].get("supporting_evidence_family_ids", [])
        }
        contradicting_ids = {
            str(item) for item in entity["data"].get("contradicting_evidence_family_ids", [])
        }
        supporting = [item for item in families if str(item["id"]) in supporting_ids]
        contradicting = [item for item in families if str(item["id"]) in contradicting_ids]
        independent_support = {roots[str(item["id"])] for item in supporting}
        independent_against = {roots[str(item["id"])] for item in contradicting}
        origin_roles = [
            str(role)
            for item in supporting
            for role in item["data"].get(
                "origin_roles", [item["data"].get("origin_type")]
            )
            if role
        ]
        scope = normalize_scientific_scope(entity["data"].get("scope"))
        temporal = {"as_of": as_of} if as_of is not None else {}
        reproduction = self.store.objects_list(
            entity["project_id"], "Reproduction", limit=None, **temporal
        )
        closest_dead = []
        if entity["kind"] == "Hypothesis":
            mechanism = str(
                entity["data"].get("mechanism") or entity["data"].get("statement") or ""
            )
            closest_dead = [
                item
                for item in self.store.related_hypotheses(
                    entity["project_id"], mechanism, as_of=as_of
                )
                if item["status"] in {"REFUTED", "WEAKENED"}
                and str(item["id"]) != entity_id
            ]
        risks: list[str] = []
        if not independent_support:
            risks.append("NO_INDEPENDENT_SUPPORT")
        if supporting and len(independent_support) < len(supporting):
            risks.append("DEPENDENT_EVIDENCE_MUST_NOT_BE_DOUBLE_COUNTED")
        if entity["kind"] == "CausalEdge" and entity["data"].get(
            "edge_status"
        ) == "INTERVENTION_SUPPORTED":
            if not scope_is_empirically_bounded(scope):
                risks.append("CAUSAL_SCOPE_NOT_EMPIRICALLY_BOUNDED")
            if not any(
                origin in {"EXPERIMENT", "REPRODUCTION", "INDEPENDENT_REPLICATION"}
                for origin in origin_roles
            ):
                risks.append("NO_INTERVENTION_EVIDENCE_FAMILY")
        for field, label in (
            ("objects", "CROSS_OBJECT_UNTESTED"),
            ("checkpoints", "CROSS_CHECKPOINT_UNTESTED"),
            ("architectures", "CROSS_ARCHITECTURE_UNTESTED"),
            ("datasets", "CROSS_DATASET_UNTESTED"),
        ):
            if self._scope_generalization(scope, field) != "MULTIPLE_SCOPES":
                risks.append(label)
        recommended = []
        if "NO_INDEPENDENT_SUPPORT" in risks:
            recommended.append("Run one preregistered discriminating intervention")
        if "CROSS_ARCHITECTURE_UNTESTED" in risks:
            recommended.append("Replicate on a distinct architecture before generalizing")
        if independent_against:
            recommended.append("Resolve contradicting evidence before promotion")
        return {
            "entity": entity,
            "status": entity["data"].get("edge_status", entity["status"]),
            "scope": scope,
            "support": {
                "independent_evidence_families": sorted(independent_support),
                "derived_records": sum(
                    len(item["data"].get("derived_record_ids", [])) for item in supporting
                ),
                "interventions": sum(origin == "EXPERIMENT" for origin in origin_roles),
                "replications": sum(
                    origin in {"REPRODUCTION", "INDEPENDENT_REPLICATION"}
                    for origin in origin_roles
                ),
                "matched_controls": int(entity["data"].get("matched_control_count", 0)),
                "counter_interventions": int(
                    entity["data"].get("counter_intervention_count", 0)
                ),
            },
            "against": {
                "evidence_families": sorted(independent_against),
                "contradictions": entity["data"].get("contradiction_ids", []),
            },
            "reproduction_status": [
                {"id": str(item["id"]), "status": item["status"]} for item in reproduction
            ],
            "closest_dead_ideas": closest_dead,
            "untested_predictions": entity["data"].get(
                "unresolved_prediction_ids", entity["data"].get("prediction_ids", [])
            ),
            "unresolved_assumptions": entity["data"].get("assumptions", []),
            "generalization": {
                "cross_object": self._scope_generalization(scope, "objects"),
                "cross_checkpoint": self._scope_generalization(scope, "checkpoints"),
                "cross_architecture": self._scope_generalization(scope, "architectures"),
                "cross_dataset": self._scope_generalization(scope, "datasets"),
            },
            "evidence_dependencies": self.group_evidence_by_origin(entity_id, as_of)["groups"],
            "promotion_risks": risks,
            "recommended_next_evidence": recommended,
            "as_of": as_of,
        }

    @staticmethod
    def _issue(
        issue_type: str,
        severity: str,
        entities: list[str],
        description: str,
        evidence: list[str],
        suggested_action: str,
    ) -> dict:
        fingerprint = "|".join([issue_type, *sorted(entities), description])
        return {
            "issue_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"gpu-lab:{fingerprint}")),
            "severity": severity,
            "type": issue_type,
            "entities": entities,
            "description": description,
            "evidence": evidence,
            "suggested_action": suggested_action,
        }

    def world_model_consistency_check(
        self, project_id: str, as_of: str | None = None
    ) -> dict:
        """Detect scientific graph inconsistencies without mutating or deleting truth."""
        temporal = {"as_of": as_of} if as_of is not None else {}
        models = self.store.objects_list(project_id, "WorldModel", limit=None, **temporal)
        hypotheses = self.store.objects_list(project_id, "Hypothesis", limit=None, **temporal)
        issues: list[dict] = []
        all_edges: list[dict] = []
        for model in models:
            records = self.store.references_get(model["data"].get("edge_ids", []), **temporal)
            edges = [records[str(item)] for item in model["data"].get("edge_ids", []) if str(item) in records]
            all_edges.extend(edges)
            node_ids = {str(item) for item in model["data"].get("node_ids", [])}
            adjacency: dict[str, set[str]] = defaultdict(set)
            for edge in edges:
                edge_id, data = str(edge["id"]), edge["data"]
                source, target = str(data.get("source_id")), str(data.get("target_id"))
                if source not in node_ids or target not in node_ids:
                    issues.append(
                        self._issue(
                            "ACTIVE_EDGE_REFERENCES_MISSING_NODE",
                            "ERROR",
                            [edge_id, source, target],
                            "A causal edge endpoint is not present in its WorldModel",
                            [],
                            "Restore the versioned node reference or retire the edge",
                        )
                    )
                audit = self.belief_audit(edge_id, as_of)
                support = audit["support"]["independent_evidence_families"]
                against = audit["against"]["evidence_families"]
                status = data.get("edge_status")
                legacy_evidence = [
                    *data.get("supporting_ids", []),
                    *data.get("against_ids", []),
                ]
                if status not in {"UNKNOWN", "HYPOTHESIZED_CAUSAL"} and not (
                    support or against or legacy_evidence
                ):
                    issues.append(
                        self._issue(
                            "CAUSAL_EDGE_HAS_NO_EVIDENCE",
                            "ERROR",
                            [edge_id],
                            "A scientific causal-edge status has no evidence provenance",
                            [],
                            "Attach an EvidenceFamily or downgrade the edge to UNKNOWN",
                        )
                    )
                if status == "INTERVENTION_SUPPORTED" and not support:
                    issues.append(
                        self._issue(
                            "INTERVENTION_SUPPORTED_WITHOUT_INTERVENTION_EVIDENCE",
                            "ERROR",
                            [edge_id],
                            "An intervention-supported edge has no independent support family",
                            [],
                            "Block promotion and attach the originating inspected experiment",
                        )
                    )
                if status == "INTERVENTION_SUPPORTED" and not scope_is_empirically_bounded(
                    audit["scope"]
                ):
                    issues.append(
                        self._issue(
                            "CAUSAL_EDGE_SCOPE_MISMATCH",
                            "ERROR",
                            [edge_id],
                            "Intervention support lacks concrete experimental scope",
                            support,
                            "Record model/data/intervention/metric scope before promotion",
                        )
                    )
                level = data.get("support_level", "NONE")
                if level not in CAUSAL_SUPPORT_LEVELS:
                    issues.append(
                        self._issue(
                            "INVALID_CAUSAL_SUPPORT_LEVEL",
                            "ERROR",
                            [edge_id],
                            f"Unknown causal support level {level}",
                            support,
                            "Recalculate support from independent evidence families",
                        )
                    )
                if level in CAUSAL_SUPPORT_LEVELS - {"NONE", "SINGLE_INTERVENTION"} and len(support) < 2:
                    issues.append(
                        self._issue(
                            "REPLICATED_CAUSAL_EDGE_HAS_ONE_EVIDENCE_FAMILY",
                            "ERROR",
                            [edge_id],
                            "Replicated support requires at least two independent origins",
                            support,
                            "Downgrade support or add an independent replication",
                        )
                    )
                if level == "CROSS_ARCHITECTURE_SUPPORT" and len(
                    set(audit["scope"].get("architectures", []))
                ) < 2:
                    issues.append(
                        self._issue(
                            "UNSUPPORTED_CAUSAL_UNIVERSALIZATION",
                            "ERROR",
                            [edge_id],
                            "Cross-architecture support is claimed from fewer than two architectures",
                            support,
                            "Downgrade support or add a distinct architecture replication",
                        )
                    )
                if status == "INTERVENTION_SUPPORTED" and against:
                    issues.append(
                        self._issue(
                            "CONTRADICTING_EVIDENCE_NOT_REFLECTED_IN_EDGE_STATUS",
                            "ERROR",
                            [edge_id],
                            "The supported edge has unresolved contradicting evidence",
                            against,
                            "Reassess and weaken, scope, or resolve the contradiction",
                        )
                    )
                if data.get("relation") != "ASSOCIATED_WITH" and status not in {"REFUTED", "UNKNOWN"}:
                    adjacency[source].add(target)

            visiting: set[str] = set()
            visited: set[str] = set()

            def visit(
                node: str,
                path: list[str],
                adjacency: dict[str, set[str]] = adjacency,
                visiting: set[str] = visiting,
                visited: set[str] = visited,
            ) -> None:
                if node in visiting:
                    cycle = path[path.index(node) :] + [node]
                    issues.append(
                        self._issue(
                            "UNEXPECTED_CAUSAL_CYCLE",
                            "WARNING",
                            cycle,
                            "The active causal graph contains a directed cycle",
                            [],
                            "Confirm feedback is intended or revise the edge set",
                        )
                    )
                    return
                if node in visited:
                    return
                visiting.add(node)
                for child in adjacency.get(node, set()):
                    visit(child, [*path, child])
                visiting.remove(node)
                visited.add(node)

            for node in sorted(node_ids):
                visit(node, [node])

        by_id = {str(edge["id"]): edge for edge in all_edges}
        scoped_edges: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
        for edge in all_edges:
            data = edge["data"]
            signature = (
                str(data.get("source_id")),
                str(data.get("target_id")),
                str(data.get("relation")),
                json.dumps(
                    normalize_scientific_scope(data.get("scope")),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            scoped_edges[signature].append(edge)
        for equivalent in scoped_edges.values():
            statuses = {item["data"].get("edge_status") for item in equivalent}
            if "REFUTED" in statuses and statuses & {
                "OBSERVED_ASSOCIATION",
                "HYPOTHESIZED_CAUSAL",
                "INTERVENTION_SUPPORTED",
            }:
                issues.append(
                    self._issue(
                        "IDENTICAL_SCOPE_CONTRADICTORY_EDGES",
                        "ERROR",
                        [str(item["id"]) for item in equivalent],
                        "Equivalent causal edges carry contradictory active statuses",
                        [],
                        "Resolve the contradiction or encode the missing scope difference",
                    )
                )
        for hypothesis in hypotheses:
            if hypothesis["status"] not in {"ACTIVE", "SURVIVES_INITIAL_TEST", "SUPPORTED"}:
                continue
            required = {
                str(item)
                for item in [
                    *hypothesis["data"].get("causal_edge_ids", []),
                    *hypothesis["data"].get("required_edge_ids", []),
                ]
            }
            for edge_id in required:
                edge = by_id.get(edge_id)
                if edge and edge["data"].get("edge_status") == "REFUTED":
                    issues.append(
                        self._issue(
                            "REFUTED_EDGE_REQUIRED_BY_ACTIVE_HYPOTHESIS",
                            "ERROR",
                            [str(hypothesis["id"]), edge_id],
                            "An active hypothesis still requires a refuted causal edge",
                            edge["data"].get("against_ids", []),
                            "Reassess the hypothesis before further promotion",
                        )
                    )
            for ref in [
                *hypothesis["data"].get("mechanism_ids", []),
                *hypothesis["data"].get("mechanism_state_ids", []),
            ]:
                try:
                    self.store.object_get(str(ref), **temporal)
                except GPUError:
                    issues.append(
                        self._issue(
                            "ACTIVE_HYPOTHESIS_REFERENCES_MISSING_MECHANISM",
                            "ERROR",
                            [str(hypothesis["id"]), str(ref)],
                            "An active hypothesis references missing mechanistic state",
                            [],
                            "Restore the referenced object or revise the hypothesis",
                        )
                    )
        return {
            "project_id": project_id,
            "as_of": as_of,
            "issues": issues,
            "error_count": sum(item["severity"] == "ERROR" for item in issues),
            "warning_count": sum(item["severity"] == "WARNING" for item in issues),
            "consistent": not any(item["severity"] == "ERROR" for item in issues),
        }
