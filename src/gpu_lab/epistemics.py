from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from typing import Any

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
        independence_key = f"{typed_origin.value}:{origin_id}"
        return self.store.evidence_family_create_atomic(
            project_id,
            {
                "origin_type": typed_origin.value,
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
        families = []
        for identifier in identifiers:
            family = records.get(str(identifier))
            if not family or family["kind"] != "EvidenceFamily":
                raise GPUError("INVALID_EVIDENCE_FAMILY_REFERENCE", str(identifier))
            if family["project_id"] != entity["project_id"]:
                raise GPUError("RESEARCH_PROJECT_MISMATCH", str(identifier))
            families.append(family)
        return entity, families

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
        return {
            "entity_id": str(entity["id"]),
            "derived_record_count": sum(
                len(family["data"].get("derived_record_ids", [])) for family in families
            ),
            "evidence_family_count": len(families),
            "independent_evidence_count": len(set(roots.values())),
            "independence_roots": roots,
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
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for family in families:
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
