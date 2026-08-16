from __future__ import annotations

import os
import threading
import time
import uuid

import pytest

from gpu_lab.brain import ResearchBrain
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

    def objects_list(
        self, project_id, kind=None, statuses=None, limit=500, data_filters=None, **_kwargs
    ):
        records = [
            item
            for item in self.records.values()
            if item["project_id"] == project_id
            and (kind is None or item["kind"] == kind)
            and (not statuses or item["status"] in statuses)
            and (
                not data_filters
                or all(item["data"].get(key) == value for key, value in data_filters.items())
            )
        ]
        return records if limit is None else records[:limit]

    def related_hypotheses(self, project_id, _mechanism, **_kwargs):
        return [
            item
            for item in self.records.values()
            if item["project_id"] == project_id
            and item["kind"] in {"Hypothesis", "NegativeResult"}
        ]

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


def test_one_run_cannot_be_double_counted_by_changing_its_origin_role():
    store = EvidenceStore()
    service = EpistemicService(store)
    run = store.add("ExperimentRun")

    experiment = service.evidence_family_create(
        "project", "EXPERIMENT", run["id"], "Original intervention"
    )
    replication_alias = service.evidence_family_create(
        "project",
        "INDEPENDENT_REPLICATION",
        run["id"],
        "The same run presented under another role",
    )
    claim = store.add("Claim")
    service.evidence_family_link(experiment["id"], claim["id"], "SUPPORTS")

    assert replication_alias["id"] == experiment["id"]
    assert service.independent_evidence_count(claim["id"])[
        "independent_evidence_count"
    ] == 1


def test_dependency_root_walks_through_unlinked_intermediate_families():
    store = EvidenceStore()
    service = EpistemicService(store)
    papers = [store.add("Paper") for _ in range(5)]
    root = service.evidence_family_create(
        "project", "PAPER", papers[0]["id"], "Original empirical result"
    )
    left = service.evidence_family_create(
        "project", "PAPER", papers[1]["id"], "Left summary", root["id"], "Reuses root"
    )
    right = service.evidence_family_create(
        "project", "PAPER", papers[2]["id"], "Right summary", root["id"], "Reuses root"
    )
    left_child = service.evidence_family_create(
        "project",
        "PAPER",
        papers[3]["id"],
        "Left descendant",
        left["id"],
        "Reuses left summary",
    )
    right_child = service.evidence_family_create(
        "project",
        "PAPER",
        papers[4]["id"],
        "Right descendant",
        right["id"],
        "Reuses right summary",
    )
    claim = store.add("Claim")
    service.evidence_family_link(left_child["id"], claim["id"], "SUPPORTS")
    service.evidence_family_link(right_child["id"], claim["id"], "SUPPORTS")

    grouped = service.group_evidence_by_origin(claim["id"])

    assert grouped["independent_evidence_count"] == 1
    assert set(grouped["groups"]) == {root["id"]}
    assert {item["family_id"] for item in grouped["groups"][root["id"]]} == {
        left_child["id"],
        right_child["id"],
    }


def test_hasi_belief_audit_does_not_universalize_one_intervention():
    store = EvidenceStore()
    service = EpistemicService(store)
    run = store.add("ExperimentRun")
    family = service.evidence_family_create(
        "project", "EXPERIMENT", run["id"], "One VRCNet intervention"
    )
    edge = store.add(
        "CausalEdge",
        {
            "edge_status": "INTERVENTION_SUPPORTED",
            "support_level": "SINGLE_INTERVENTION",
            "scope": {
                "description": "HASI VRCNet intervention",
                "models": ["VRCNet"],
                "architectures": ["VRCNet"],
                "checkpoints": ["checkpoint-a"],
                "datasets": ["Completion3D"],
                "objects": ["object-a"],
                "interventions": ["anchor-state substitution"],
                "metrics": ["Chamfer distance"],
            },
        },
    )
    service.evidence_family_link(family["id"], edge["id"], "SUPPORTS")

    audit = service.belief_audit(edge["id"])

    assert audit["support"]["independent_evidence_families"] == [family["id"]]
    assert audit["generalization"]["cross_architecture"] == "SINGLE_SCOPE"
    assert "CROSS_ARCHITECTURE_UNTESTED" in audit["promotion_risks"]
    assert audit["status"] == "INTERVENTION_SUPPORTED"


def test_world_model_consistency_flags_invalid_promotion_and_refuted_dependency():
    store = EvidenceStore()
    service = EpistemicService(store)
    source, target = store.add("MechanismState"), store.add("MechanismState")
    unsupported = store.add(
        "CausalEdge",
        {
            "source_id": source["id"],
            "target_id": target["id"],
            "relation": "CAUSES",
            "edge_status": "INTERVENTION_SUPPORTED",
            "support_level": "SINGLE_INTERVENTION",
            "scope": {},
        },
    )
    refuted = store.add(
        "CausalEdge",
        {
            "source_id": target["id"],
            "target_id": source["id"],
            "relation": "INFLUENCES",
            "edge_status": "REFUTED",
            "support_level": "NONE",
            "scope": {},
        },
        status="RESULT_INSPECTED",
    )
    store.add(
        "WorldModel",
        {
            "node_ids": [source["id"], target["id"]],
            "edge_ids": [unsupported["id"], refuted["id"]],
        },
    )
    store.add("Hypothesis", {"required_edge_ids": [refuted["id"]]})

    result = service.world_model_consistency_check("project")
    issue_types = {item["type"] for item in result["issues"]}

    assert "INTERVENTION_SUPPORTED_WITHOUT_INTERVENTION_EVIDENCE" in issue_types
    assert "REFUTED_EDGE_REQUIRED_BY_ACTIVE_HYPOTHESIS" in issue_types
    assert result["consistent"] is False


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


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_postgres_result_assessment_atomically_creates_scoped_causal_support():
    store = ResearchStore(TEST_DATABASE_URL)
    brain = ResearchBrain(store)
    service = EpistemicService(store)
    project = store.project_create(
        f"scoped-assessment-{time.time_ns()}", "Does one intervention support this edge?"
    )
    project_id = project["project_id"]
    model = brain.world_model_create(project_id, "Scoped model", "VRC fixture")
    model_id = model["world_model"]["id"]
    source = brain.world_entity_create(model_id, "MechanismState", "A", "source")
    target = brain.world_entity_create(model_id, "MechanismState", "B", "target")
    prediction = store.object_create(
        project_id, "Prediction", {"statement": "A changes B"}, "PREDICTION_CREATED"
    )
    edge = brain.causal_edge_create(
        model_id,
        source["entity"]["id"],
        target["entity"]["id"],
        "CAUSES",
        "HYPOTHESIZED_CAUSAL",
        unresolved_prediction_ids=[prediction["id"]],
    )["edge"]
    hypothesis = store.object_create(
        project_id, "Hypothesis", {"mechanism": "A causes B"}, "HYPOTHESIS_CREATED"
    )
    agenda = store.object_create(
        project_id, "AgendaItem", {"question": "Does A cause B?"}, "AGENDA_ITEM_CREATED"
    )
    decision = store.object_create(
        project_id,
        "ResearchDecision",
        {"agenda_item_id": agenda["id"], "hypotheses_affected": [hypothesis["id"]]},
        "RESEARCH_DECISION_SELECTED",
        "SELECTED",
    )
    run = store.object_create(
        project_id,
        "ExperimentRun",
        {"decision_id": decision["id"], "experiment_id": str(uuid.uuid4())},
        "EXPERIMENT_FINISHED",
        "completed",
    )
    scope = {
        "description": "One VRCNet checkpoint intervention",
        "models": ["VRCNet"],
        "architectures": ["VRCNet"],
        "checkpoints": ["checkpoint-a"],
        "datasets": ["Completion3D"],
        "objects": ["object-a"],
        "interventions": ["state substitution"],
        "metrics": ["Chamfer distance"],
    }

    result = store.result_assessment_apply(
        run_id=run["id"],
        decision_id=decision["id"],
        hypothesis_id=hypothesis["id"],
        agenda_item_id=agenda["id"],
        evidence_data={
            "prediction_outcome": "The frozen prediction passed",
            "scope": scope,
            "matched_control_preregistered": True,
            "matched_control_passed": True,
        },
        hypothesis_transition="SURVIVES_INITIAL_TEST",
        rationale="One bounded intervention passed",
        inspection={"scope": scope},
        agenda_status="RESOLVED",
        actual_information_gain="HIGH",
        causal_edge_id=edge["id"],
        causal_edge_status="INTERVENTION_SUPPORTED",
    )

    persisted_edge = store.object_get(edge["id"])
    family_ids = result["evidence"]["data"]["evidence_family_ids"]
    assert len(family_ids) == 1
    assert service.independent_evidence_count(hypothesis["id"])[
        "independent_evidence_count"
    ] == 1
    assert persisted_edge["data"]["scope"] == scope
    assert persisted_edge["data"]["support_level"] == "SINGLE_INTERVENTION"
    assert persisted_edge["data"]["supporting_evidence_family_ids"] == family_ids
    assert persisted_edge["data"]["matched_control_count"] == 1


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_postgres_concurrent_direct_edge_updates_detect_stale_writer_without_lost_evidence():
    store = ResearchStore(TEST_DATABASE_URL)
    brain = ResearchBrain(store)
    project = store.project_create(
        f"causal-concurrency-{time.time_ns()}", "Can concurrent evidence be lost?"
    )
    project_id = project["project_id"]
    model = brain.world_model_create(project_id, "Concurrent model", "fixture")
    model_id = model["world_model"]["id"]
    source = brain.world_entity_create(model_id, "MechanismState", "A", "source")
    target = brain.world_entity_create(model_id, "MechanismState", "B", "target")
    prediction = store.object_create(
        project_id, "Prediction", {"statement": "A changes B"}, "PREDICTION_CREATED"
    )
    edge = brain.causal_edge_create(
        model_id,
        source["entity"]["id"],
        target["entity"]["id"],
        "CAUSES",
        "HYPOTHESIZED_CAUSAL",
        unresolved_prediction_ids=[prediction["id"]],
    )["edge"]
    evidence = [
        store.object_create(
            project_id,
            "EvidenceUnit",
            {"statement": f"observation {index}"},
            "EVIDENCE_RECORDED",
        )
        for index in range(2)
    ]
    barrier = threading.Barrier(2)
    results, errors = [], []

    def update(evidence_id):
        barrier.wait()
        try:
            results.append(
                store.causal_edge_update_atomic(
                    edge["id"],
                    {
                        "edge_status": "OBSERVED_ASSOCIATION",
                        "supporting_ids": [evidence_id],
                    },
                    "RESULT_INSPECTED",
                    {},
                    [evidence_id],
                    None,
                    expected_edge_status="HYPOTHESIZED_CAUSAL",
                )
            )
        except GPUError as exc:
            errors.append(exc)

    workers = [threading.Thread(target=update, args=(item["id"],)) for item in evidence]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert all(not worker.is_alive() for worker in workers)
    assert len(results) == 1
    assert [error.error_type for error in errors] == ["CAUSAL_EDGE_CONCURRENT_UPDATE"]
    persisted = store.object_get(edge["id"])
    assert persisted["data"]["supporting_ids"] in ([evidence[0]["id"]], [evidence[1]["id"]])


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_postgres_generic_family_link_cannot_mutate_causal_support():
    store = ResearchStore(TEST_DATABASE_URL)
    service = EpistemicService(store)
    project = store.project_create(f"causal-link-{time.time_ns()}", "Can links bypass assessment?")
    run = store.object_create(
        project["project_id"], "ExperimentRun", {"job_id": "fixture"}, "EXPERIMENT_STARTED"
    )
    family = service.evidence_family_create(
        project["project_id"], "EXPERIMENT", run["id"], "One experiment"
    )
    edge = store.object_create(
        project["project_id"],
        "CausalEdge",
        {"edge_status": "HYPOTHESIZED_CAUSAL"},
        "CAUSAL_EDGE_CREATED",
    )

    with pytest.raises(GPUError) as error:
        service.evidence_family_link(family["id"], edge["id"], "SUPPORTS")

    assert error.value.error_type == "CAUSAL_EVIDENCE_FAMILY_REQUIRES_RESULT_ASSESSMENT"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_postgres_result_assessment_rejects_precreated_dependent_run_family():
    store = ResearchStore(TEST_DATABASE_URL)
    service = EpistemicService(store)
    project = store.project_create(
        f"dependent-run-family-{time.time_ns()}", "Can a dependent run look independent?"
    )
    project_id = project["project_id"]
    paper = store.object_create(project_id, "Paper", {"title": "root"}, "PAPER_INGESTED")
    root = service.evidence_family_create(project_id, "PAPER", paper["id"], "Root evidence")
    decision = store.object_create(
        project_id, "ResearchDecision", {}, "RESEARCH_DECISION_SELECTED", "SELECTED"
    )
    hypothesis = store.object_create(
        project_id, "Hypothesis", {"mechanism": "fixture"}, "HYPOTHESIS_CREATED"
    )
    agenda = store.object_create(
        project_id, "AgendaItem", {"question": "fixture"}, "AGENDA_ITEM_CREATED"
    )
    run = store.object_create(
        project_id,
        "ExperimentRun",
        {"decision_id": decision["id"], "experiment_id": str(uuid.uuid4())},
        "EXPERIMENT_FINISHED",
        "completed",
    )
    service.evidence_family_create(
        project_id,
        "EXPERIMENT",
        run["id"],
        "Dependent run alias",
        root["id"],
        "Reuses the paper's empirical origin",
    )

    with pytest.raises(GPUError) as error:
        store.result_assessment_apply(
            run_id=run["id"],
            decision_id=decision["id"],
            hypothesis_id=hypothesis["id"],
            agenda_item_id=agenda["id"],
            evidence_data={"prediction_outcome": "fixture", "scope": {}},
            hypothesis_transition="INCONCLUSIVE",
            rationale="fixture",
            inspection={},
            agenda_status="ACTIVE",
            actual_information_gain="LOW",
        )

    assert error.value.error_type == "EVIDENCE_FAMILY_KEY_CONFLICT"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_postgres_cannot_recreate_identical_scope_positive_edge_after_refutation():
    store = ResearchStore(TEST_DATABASE_URL)
    brain = ResearchBrain(store)
    project = store.project_create(
        f"refuted-edge-create-{time.time_ns()}", "Can refuted causality be recreated?"
    )
    project_id = project["project_id"]
    model = brain.world_model_create(project_id, "Model", "fixture")
    model_id = model["world_model"]["id"]
    source = brain.world_entity_create(model_id, "MechanismState", "A", "source")
    target = brain.world_entity_create(model_id, "MechanismState", "B", "target")
    evidence = store.object_create(
        project_id, "EvidenceUnit", {"statement": "failed intervention"}, "EVIDENCE_RECORDED"
    )
    scope = {"architectures": ["VRCNet"], "interventions": ["state swap"], "metrics": ["CD"]}
    brain.causal_edge_create(
        model_id,
        source["entity"]["id"],
        target["entity"]["id"],
        "CAUSES",
        "REFUTED",
        against_ids=[evidence["id"]],
        scope=scope,
    )
    prediction = store.object_create(
        project_id, "Prediction", {"statement": "A changes B"}, "PREDICTION_CREATED"
    )

    with pytest.raises(GPUError) as error:
        brain.causal_edge_create(
            model_id,
            source["entity"]["id"],
            target["entity"]["id"],
            "CAUSES",
            "HYPOTHESIZED_CAUSAL",
            unresolved_prediction_ids=[prediction["id"]],
            scope=scope,
        )

    assert error.value.error_type == "WORLD_MODEL_CONSISTENCY_ERROR"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_postgres_generic_positive_assessment_rejects_refuted_required_edge():
    store = ResearchStore(TEST_DATABASE_URL)
    service = EpistemicService(store)
    project = store.project_create(
        f"refuted-hypothesis-assess-{time.time_ns()}", "Can assessment bypass consistency?"
    )
    edge = store.object_create(
        project["project_id"],
        "CausalEdge",
        {"edge_status": "REFUTED"},
        "CAUSAL_EDGE_CREATED",
        "RESULT_INSPECTED",
    )
    run = store.object_create(
        project["project_id"], "ExperimentRun", {"job_id": "fixture"}, "EXPERIMENT_STARTED"
    )
    family = service.evidence_family_create(
        project["project_id"], "EXPERIMENT", run["id"], "Supporting family"
    )
    hypothesis = store.object_create(
        project["project_id"],
        "Hypothesis",
        {
            "mechanism": "requires a refuted edge",
            "required_edge_ids": [edge["id"]],
            "supporting_evidence_family_ids": [family["id"]],
        },
        "HYPOTHESIS_CREATED",
    )

    with pytest.raises(GPUError) as error:
        store.assess(hypothesis["id"], "SUPPORTED", "generic promotion attempt")

    assert error.value.error_type == "WORLD_MODEL_CONSISTENCY_ERROR"
