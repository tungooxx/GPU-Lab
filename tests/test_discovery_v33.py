import os
import time

import pytest

from gpu_lab.discovery_v33 import DistributedDiscoveryService
from gpu_lab.errors import GPUError
from gpu_lab.lab import LabController
from gpu_lab.research import ResearchStore

TEST_DATABASE_URL = os.getenv("GPU_LAB_TEST_DATABASE_URL")


def _candidate(title: str, dimensions: dict, niche: str | None = None) -> dict:
    return {
        "title": title,
        "mechanism": f"Mechanistic account for {title}",
        "predictions": [f"A discriminating outcome for {title}"],
        "falsifier": f"A result inconsistent with {title}",
        "diversity_signature": dimensions,
        "mechanistic_niche": niche,
        "quality_components": {"discriminating_value": 3, "option_value": 2},
    }


def test_shadow_signature_accepts_existing_brain_candidate_shape_without_writes():
    assert DistributedDiscoveryService._signature({
        "payload": {"scientific_dimensions": {"representation": "token-grid"}},
    }) == {"representation": "token-grid"}


def test_structural_niche_ignores_untrusted_label_and_requires_serious_candidate_fields():
    signature = {"representation": "token-grid"}
    assert DistributedDiscoveryService._niche(signature, "TOTALLY_NEW_NICHE") == "REPRESENTATION::TOKEN-GRID"
    with pytest.raises(GPUError) as exc:
        DistributedDiscoveryService._validate_candidate({
            "predictions": ["P"], "falsifier": "F", "diversity_signature": signature,
        })
    assert exc.value.error_type == "DISCOVERY_CANDIDATE_MECHANISM_REQUIRED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_round_isolates_batches_then_archives_distinct_candidates():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-v33-{time.time_ns()}", "Independent discovery")['project_id']
    first = lab.join(None, "dde-a", "CODEX", project_id)
    second = lab.join(None, "dde-b", "LOCAL_AGENT", project_id)
    round_ = dde.create_round(project_id, None, "DIVERGENT_SEARCH")
    batch_a = dde.join_round(round_["id"], first["worker"]["id"], first["session_id"], "REPRESENTATION_RESET", "FAR")
    batch_b = dde.join_round(round_["id"], second["worker"]["id"], second["session_id"], "STRONG_NULL_CONSTRUCTION", "ORTHOGONAL")
    candidate_a = dde.submit_candidate(round_["id"], batch_a["id"], first["worker"]["id"], first["session_id"], _candidate("representation reset", {"representation": "point tokens"}))
    with pytest.raises(GPUError, match="PEER_ISOLATION_ACTIVE"):
        dde.batch_get(round_["id"], batch_a["id"], second["session_id"])
    dde.batch_freeze(round_["id"], batch_a["id"], first["worker"]["id"], first["session_id"])
    with pytest.raises(GPUError, match="PEER_ISOLATION_ACTIVE"):
        dde.batch_get(round_["id"], batch_a["id"], second["session_id"])
    candidate_b = dde.submit_candidate(round_["id"], batch_b["id"], second["worker"]["id"], second["session_id"], _candidate("strong null", {"causal_object": "null ontology"}))
    dde.batch_freeze(round_["id"], batch_b["id"], second["worker"]["id"], second["session_id"])
    assert dde.round_get(round_["id"])["data"]["peer_visibility"] == "VISIBLE_FOR_SYNTHESIS"
    synthesis_work = [
        item for item in lab.work_list(project_id, limit=100)
        if item["kind"] == "DISCOVERY_SYNTHESIS"
    ]
    assert len(synthesis_work) == 1 and synthesis_work[0]["status"] == "READY"
    archive = dde.synthesize(round_["id"], literature_available=False)
    assert archive["data"]["coverage"]["effective_niche_count"] == 2
    assert archive["data"]["coverage"]["literature_status"] == "UNAVAILABLE_NOVELTY_UNVERIFIED"
    assert dde.synthesize(round_["id"])["id"] == archive["id"]
    with pytest.raises(GPUError, match="NOT_GENERATING"):
        dde.submit_candidate(round_["id"], batch_a["id"], first["worker"]["id"], first["session_id"], _candidate("late", {"representation": "late"}))
    assert dde.outcome_get(candidate_a["id"])["resolution_status"] == "UNKNOWN"
    assert candidate_b["data"]["scientific_distance"] == "ORTHOGONAL"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_empty_batch_requires_explicit_honest_abstention():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-abstain-{time.time_ns()}", "Honest distant abstention")['project_id']
    worker = lab.join(None, "dde-abstain", "CODEX", project_id)
    round_ = dde.create_round(project_id, None, "PARADIGM_RESET")
    batch = dde.join_round(round_["id"], worker["worker"]["id"], worker["session_id"], "ONTOLOGY_CHALLENGE", "ORTHOGONAL")
    with pytest.raises(GPUError, match="ABSTENTION_REASON_REQUIRED"):
        dde.batch_freeze(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"])
    frozen = dde.batch_freeze(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"], "No coherent ontology challenge survived canonical constraints")
    assert frozen["status"] == "ABSTAINED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_distance_reservations_prevent_near_slots_from_consuming_open_search():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-reserved-{time.time_ns()}", "Distance reservation")['project_id']
    workers = [lab.join(None, f"reserved-{index}", "CODEX", project_id) for index in range(4)]
    round_ = dde.create_round(project_id, None, "DIVERGENT_SEARCH")
    dde.join_round(round_["id"], workers[0]["worker"]["id"], workers[0]["session_id"], "LOCAL_CAUSAL_REPAIR", "NEAR")
    with pytest.raises(GPUError, match="DISTANCE_SLOT_RESERVED"):
        dde.join_round(round_["id"], workers[1]["worker"]["id"], workers[1]["session_id"], "LOCAL_CAUSAL_REPAIR", "NEAR")


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_isolated_round_blocks_peer_finding_messages_and_detects_stale_state():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-isolation-{time.time_ns()}", "Message isolation")['project_id']
    first = lab.join(None, "isolation-a", "CODEX", project_id)
    second = lab.join(None, "isolation-b", "LOCAL_AGENT", project_id)
    round_ = dde.create_round(project_id, None, "MECHANISM_SEARCH")
    batch = dde.join_round(round_["id"], first["worker"]["id"], first["session_id"], "CAUSAL_INVERSION", "FAR")
    with pytest.raises(GPUError, match="PEER_MESSAGE_BLOCKED"):
        lab.message_send(
            project_id, first["worker"]["id"], first["session_id"], "SHARE_FINDING",
            "candidate", "A current-round proposal", to_worker_id=second["worker"]["id"],
        )
    store.object_create(project_id, "Claim", {"statement": "state changed", "scope": "test"}, "CLAIM_CREATED")
    assert dde.stale_check(round_["id"])["stale"] is True
    assert dde.stale_check(round_["id"], mark_stale=True)["round"]["status"] == "STALE"
    assert batch["data"]["worker_session_id"] == first["session_id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_stale_round_cannot_be_synthesized_into_a_new_execution_portfolio():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, dde = LabController(store), DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-stale-{time.time_ns()}", "Stale round")['project_id']
    worker = lab.join(None, "stale-worker", "CODEX", project_id)
    round_ = dde.create_round(project_id, None, "MECHANISM_SEARCH")
    batch = dde.join_round(round_["id"], worker["worker"]["id"], worker["session_id"], "CAUSAL_INVERSION", "FAR")
    dde.submit_candidate(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"], _candidate("candidate", {"causal_object": "new"}))
    dde.batch_freeze(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"])
    store.object_create(project_id, "Claim", {"statement": "decisive update", "scope": "test"}, "CLAIM_CREATED")
    with pytest.raises(GPUError) as exc:
        dde.synthesize(round_["id"])
    assert exc.value.error_type == "DISCOVERY_ROUND_STALE"
    assert dde.round_get(round_["id"])["status"] == "STALE"
