"""Exercise v3.3 independent discovery against a disposable PostgreSQL project."""

from __future__ import annotations

import os
import time

from gpu_lab.discovery_v33 import DistributedDiscoveryService
from gpu_lab.errors import GPUError
from gpu_lab.lab import LabController
from gpu_lab.research import ResearchStore


def candidate(title: str, signature: dict[str, str]) -> dict:
    return {
        "title": title,
        "mechanism": f"Mechanism for {title}",
        "predictions": [f"Prediction for {title}"],
        "falsifier": f"Falsifier for {title}",
        "diversity_signature": signature,
        "quality_components": {"discriminating_value": 3, "option_value": 2},
    }


def main() -> None:
    store = ResearchStore(os.environ["GPU_LAB_TEST_DATABASE_URL"])
    lab, dde = LabController(store), DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-v33-smoke-{time.time_ns()}", "Disposable DDE 3.3 smoke")["project_id"]
    timeline: list[str] = []

    def event(text: str) -> None:
        timeline.append(text)
        print(f"  {len(timeline):02d}. {text}")

    model = store.object_create(project_id, "WorldModel", {"mechanisms": ["frozen-mechanism"]}, "SMOKE_MODEL")
    agenda = store.object_create(project_id, "ResearchAgenda", {"question": "What mechanism remains?"}, "SMOKE_AGENDA")
    item = store.object_create(project_id, "AgendaItem", {"agenda_id": str(agenda["id"]), "question": "Search distant explanations", "importance": 1, "uncertainty": 1}, "SMOKE_ITEM", "OPEN")
    store.object_create(project_id, "FrontierGap", {"severity": "HIGH", "lineage": "mature-model"}, "SMOKE_FRONTIER")
    store.object_create(project_id, "StagnationState", {"local_search_saturation": "HIGH"}, "SMOKE_STAGNATION")
    store.object_create(project_id, "ArchitectureLineage", {"name": "saturated-lineage"}, "SMOKE_LINEAGE")
    store.object_create(project_id, "NegativeResult", {"scientific_dimensions": {"representation": "retired-token"}}, "SMOKE_NEGATIVE")
    store.object_create(project_id, "BreakthroughSignal", {"type": "PARTIAL", "discovery_value": "partial"}, "SMOKE_BREAKTHROUGH")
    workers = [lab.join(None, f"dde-smoke-{name}", "CODEX", project_id) for name in "ABCD"]
    event("created frozen model/agenda/frontier/negative-memory state and four durable workers")
    round_ = dde.create_round(project_id, item["id"], "PARADIGM_RESET")
    assert round_["data"]["frozen_state"]["frontier_gap_ids"]
    assert round_["data"]["frozen_state"]["negative_result_ids"]
    assert round_["data"]["frozen_state"]["architecture_lineage_ids"]
    assignments = [
        ("REPRESENTATION_RESET", "FAR", {"representation": "point tokens"}),
        ("CAUSAL_INVERSION", "MID", {"information_path": "inverse causal probe"}),
        ("STRONG_NULL_CONSTRUCTION", "ORTHOGONAL", {"causal_object": "strong null ontology"}),
        ("ONTOLOGY_CHALLENGE", "NEAR", {"architecture_family": "local control"}),
    ]
    batches = []
    for index, (operator, distance, signature) in enumerate(assignments):
        worker = workers[index]
        batch = dde.join_round(round_["id"], worker["worker"]["id"], worker["session_id"], operator, distance)
        batches.append(batch)
        dde.submit_candidate(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"], candidate(operator, signature))
    assert len(dde.lab_work_items(round_["id"])) == 4
    event("assigned four distinct generation operators and submitted independent batches")
    try:
        dde.batch_get(round_["id"], batches[0]["id"], workers[1]["session_id"])
        raise AssertionError("peer candidate leakage was not rejected")
    except GPUError as exc:
        assert exc.error_type == "DISCOVERY_PEER_ISOLATION_ACTIVE"
    try:
        lab.message_send(
            project_id, workers[0]["worker"]["id"], workers[0]["session_id"], "SHARE_FINDING",
            "candidate", "Do not leak this current-round idea", to_worker_id=workers[1]["worker"]["id"],
        )
        raise AssertionError("peer candidate message leakage was not rejected")
    except GPUError as exc:
        assert exc.error_type == "DISCOVERY_PEER_MESSAGE_BLOCKED"
    event("verified peer candidate batches stay hidden during independent generation")
    for batch, worker in zip(batches, workers, strict=True):
        dde.batch_freeze(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"])
    assert dde.round_get(round_["id"])["data"]["peer_visibility"] == "VISIBLE_FOR_SYNTHESIS"
    event("froze every batch; peer visibility became available only for synthesis")
    archive = dde.synthesize(round_["id"], literature_available=False)
    coverage = archive["data"]["coverage"]
    assert coverage["effective_niche_count"] >= 3
    assert coverage["literature_status"] == "UNAVAILABLE_NOVELTY_UNVERIFIED"
    assert len(archive["data"]["mechanistic_niche_ids"]) >= 3
    assert dde.synthesize(round_["id"])["id"] == archive["id"]
    restarted = DistributedDiscoveryService(ResearchStore(os.environ["GPU_LAB_TEST_DATABASE_URL"]))
    assert restarted.archive_get(archive["id"])["id"] == archive["id"]
    store.object_create(project_id, "Claim", {"statement": "new post-round state", "scope": "smoke"}, "CLAIM_CREATED")
    assert restarted.stale_check(round_["id"])["stale"] is True
    event("created one durable cross-worker QD archive; literature absence did not stop generation")
    print("DISCOVERY_V33_SMOKE_OK")
    print(f"project={project_id} timeline_events={len(timeline)}")


if __name__ == "__main__":
    main()
