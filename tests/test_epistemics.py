from __future__ import annotations

import os
import time
import uuid

import pytest

from gpu_lab.epistemics import EpistemicService
from gpu_lab.errors import GPUError
from gpu_lab.research import ResearchStore

TEST_DATABASE_URL = os.getenv("GPU_LAB_TEST_DATABASE_URL")


class EvidenceStore:
    def __init__(self):
        self.records = {}

    def add(self, kind, data=None, project_id="project", status="ACTIVE"):
        identifier = str(uuid.uuid4())
        record = {
            "id": identifier,
            "project_id": project_id,
            "kind": kind,
            "status": status,
            "data": data or {},
        }
        self.records[identifier] = record
        return record

    def object_get(self, identifier, **_kwargs):
        if identifier not in self.records:
            raise GPUError("RESEARCH_OBJECT_NOT_FOUND", identifier)
        return self.records[identifier]

    def references_get(self, identifiers, **_kwargs):
        return {identifier: self.records[identifier] for identifier in identifiers}

    def evidence_family_create_atomic(self, project_id, data):
        existing = next(
            (
                item
                for item in self.records.values()
                if item["kind"] == "EvidenceFamily"
                and item["project_id"] == project_id
                and item["data"]["independence_key"] == data["independence_key"]
            ),
            None,
        )
        if existing:
            return {**existing, "idempotent_replay": True}
        return self.add(
            "EvidenceFamily",
            {
                **data,
                "supporting_entity_ids": [],
                "contradicting_entity_ids": [],
                "derived_record_ids": [],
            },
            project_id,
        )

    def evidence_family_link_atomic(self, family_id, entity_id, relationship):
        family, entity = self.records[family_id], self.records[entity_id]
        family["data"]["derived_record_ids"] = list(
            dict.fromkeys([*family["data"]["derived_record_ids"], entity_id])
        )
        entity["data"]["evidence_family_ids"] = list(
            dict.fromkeys([*entity["data"].get("evidence_family_ids", []), family_id])
        )
        if relationship == "SUPPORTS":
            family["data"]["supporting_entity_ids"] = list(
                dict.fromkeys([*family["data"]["supporting_entity_ids"], entity_id])
            )
            entity["data"]["supporting_evidence_family_ids"] = list(
                dict.fromkeys(
                    [*entity["data"].get("supporting_evidence_family_ids", []), family_id]
                )
            )
        return {"family": family, "entity": entity, "relationship": relationship}


def test_five_derived_records_from_one_experiment_count_as_one_origin():
    store = EvidenceStore()
    service = EpistemicService(store)
    run = store.add("ExperimentRun")
    family = service.evidence_family_create(
        "project", "EXPERIMENT", run["id"], "One frozen intervention"
    )
    records = [
        store.add("EvidenceUnit"),
        store.add("Prediction"),
        store.add("Claim"),
        store.add("CausalEdge"),
        store.add("Lesson"),
    ]
    for record in records:
        service.evidence_family_link(family["id"], record["id"], "DERIVED")
    service.evidence_family_link(family["id"], records[2]["id"], "SUPPORTS")

    count = service.independent_evidence_count(records[2]["id"])

    assert count["derived_record_count"] == 5
    assert count["evidence_family_count"] == 1
    assert count["independent_evidence_count"] == 1


def test_dependent_paper_families_share_one_independence_root():
    store = EvidenceStore()
    service = EpistemicService(store)
    paper_a, paper_b = store.add("Paper"), store.add("Paper")
    family_a = service.evidence_family_create(
        "project", "PAPER", paper_a["id"], "Original study"
    )
    family_b = service.evidence_family_create(
        "project",
        "PAPER",
        paper_b["id"],
        "Paper repeating the original study",
        family_a["id"],
        "Uses the same reported experiment rather than an independent replication",
    )
    claim = store.add("Claim")
    service.evidence_family_link(family_a["id"], claim["id"], "SUPPORTS")
    service.evidence_family_link(family_b["id"], claim["id"], "SUPPORTS")

    grouped = service.group_evidence_by_origin(claim["id"])

    assert grouped["independent_evidence_count"] == 1
    assert list(grouped["groups"]) == [family_a["id"]]
    assert len(grouped["groups"][family_a["id"]]) == 2


def test_dependent_family_requires_dependency_note():
    store = EvidenceStore()
    service = EpistemicService(store)
    paper_a, paper_b = store.add("Paper"), store.add("Paper")
    family_a = service.evidence_family_create(
        "project", "PAPER", paper_a["id"], "Original study"
    )

    with pytest.raises(GPUError) as error:
        service.evidence_family_create(
            "project", "PAPER", paper_b["id"], "Dependent study", family_a["id"]
        )

    assert error.value.error_type == "EVIDENCE_DEPENDENCY_NOTE_REQUIRED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_postgres_evidence_family_link_is_idempotent_and_preserves_one_origin():
    store = ResearchStore(TEST_DATABASE_URL)
    service = EpistemicService(store)
    project = store.project_create(f"evidence-family-{time.time_ns()}", "How many origins?")
    run = store.object_create(
        project["project_id"], "ExperimentRun", {"job_id": "fixture"}, "EXPERIMENT_STARTED"
    )
    family = service.evidence_family_create(
        project["project_id"], "EXPERIMENT", run["id"], "One intervention"
    )
    records = [
        store.object_create(
            project["project_id"], kind, {"fixture": True}, f"{kind.upper()}_CREATED"
        )
        for kind in ("EvidenceUnit", "Prediction", "Claim", "CausalEdge", "Lesson")
    ]
    for record in records:
        service.evidence_family_link(family["id"], record["id"])
    service.evidence_family_link(family["id"], records[2]["id"], "SUPPORTS")
    replay = service.evidence_family_link(family["id"], records[2]["id"], "SUPPORTS")

    count = service.independent_evidence_count(records[2]["id"])
    persisted_family = store.object_get(family["id"])

    assert replay["idempotent_replay"] is True
    assert len(persisted_family["data"]["derived_record_ids"]) == 5
    assert count["independent_evidence_count"] == 1
