import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from gpu_lab.discovery_v33 import DistributedDiscoveryService
from gpu_lab.discovery import classify_scientific_distance
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


def test_hyperparameter_only_variants_share_a_scientific_equivalence_key():
    first = {"data": {"diversity_signature": {"representation": "token-grid", "learning_rate": 0.001}}}
    second = {"data": {"diversity_signature": {"representation": "token-grid", "learning_rate": 0.01}}}
    assert DistributedDiscoveryService._scientific_equivalence_key(first) == DistributedDiscoveryService._scientific_equivalence_key(second)


def test_scientific_snapshot_records_have_a_stable_canonical_order():
    service = DistributedDiscoveryService.__new__(DistributedDiscoveryService)
    service.store = type("Store", (), {"state_get": staticmethod(lambda _project: {
        "state_freshness": {"research_state_version": 2, "world_model_version": None},
        "canonical_state": {"negative_results": []},
        "objects": [
            {"id": "b", "kind": "Claim", "status": "ACTIVE", "data": {"v": 2}},
            {"id": "a", "kind": "Claim", "status": "ACTIVE", "data": {"v": 1}},
        ],
    })})()

    assert [item["id"] for item in service._scientific_snapshot("project")["records"]] == ["a", "b"]


def test_explicit_round_baseline_allows_all_distance_classes():
    baseline = {"representation": "point-tokens", "state_variable": "anchor-state"}

    near = classify_scientific_distance(
        {"scientific_dimensions": baseline}, {"scientific_dimensions": baseline}
    )
    mid = classify_scientific_distance(
        {"scientific_dimensions": {**baseline, "state_variable": "global-state"}},
        {"scientific_dimensions": baseline},
    )
    far = classify_scientific_distance(
        {"scientific_dimensions": {**baseline, "representation": "voxel-grid"}},
        {"scientific_dimensions": baseline},
    )
    orthogonal = classify_scientific_distance(
        {"scientific_dimensions": {**baseline, "causal_object": "joint-objective"}},
        {"scientific_dimensions": baseline},
    )

    assert [near["scientific_distance"], mid["scientific_distance"], far["scientific_distance"], orthogonal["scientific_distance"]] == ["NEAR", "MID", "FAR", "ORTHOGONAL"]


def test_snapshot_baseline_requires_exactly_one_selected_active_niche():
    signature, source = DistributedDiscoveryService._baseline_from_snapshot(
        {
            "selected_niche_baselines": [
                {
                    "id": "niche-1",
                    "active_best_hypothesis_id": "hypothesis-1",
                    "diversity_signature": {"representation": "point-tokens"},
                    "data_hash": "frozen-hash",
                }
            ]
        }
    )

    assert signature == {"representation": "point-tokens"}
    assert source["kind"] == "HypothesisNiche"

    with pytest.raises(GPUError) as missing:
        DistributedDiscoveryService._baseline_from_snapshot({"selected_niche_baselines": []})
    assert missing.value.error_type == "DISCOVERY_BASELINE_REQUIRED"

    with pytest.raises(GPUError) as ambiguous:
        DistributedDiscoveryService._baseline_from_snapshot(
            {"selected_niche_baselines": [{"id": "a"}, {"id": "b"}]}
        )
    assert ambiguous.value.error_type == "DISCOVERY_BASELINE_AMBIGUOUS"


def test_recovery_wave_idempotence_normalizes_uuid_identity():
    """Persisted JSON IDs are text; database adapters may return UUID objects."""
    parent_id = uuid.uuid4()
    archive_id = uuid.uuid4()
    existing_id = uuid.uuid4()
    parent = {
        "id": parent_id,
        "project_id": "project",
        "kind": "DiscoveryRound",
        "data": {
            "archive_id": str(archive_id), "generation_wave": 1,
            "generation_budget": {"max_generation_waves": 2},
            "required_distance_coverage": {"NEAR": 1, "MID": 1},
            "frozen_state_hash": "frozen",
        },
    }
    archive = {
        "id": archive_id,
        "kind": "CrossWorkerQDArchive",
        "data": {"coverage": {"filled_distance_slots": {"NEAR": 1, "MID": 0}}},
    }
    existing = {
        "id": existing_id,
        "kind": "DiscoveryRound",
        # Deliberately model a driver-returned UUID instead of a JSON string.
        "data": {"recovery_of_round_id": parent_id},
    }

    class Store:
        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, *_args, **_kwargs):
                return None

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def cursor():
                return Store.Cursor()

        @staticmethod
        def _connect():
            return Store.Connection()

        @staticmethod
        def object_get(identifier):
            return parent if str(identifier) == str(parent_id) else archive

        @staticmethod
        def objects_list(_project_id, _kind, limit=None):
            return [existing]

        @staticmethod
        def _event(*_args, **_kwargs):
            return None

    service = DistributedDiscoveryService.__new__(DistributedDiscoveryService)
    service.store = Store()
    service._event = lambda *_args, **_kwargs: None

    recovered = service.recovery_wave_create(str(parent_id), "RECOVER_DISCOVERY_WAVE")
    assert str(recovered["id"]) == str(existing_id)


def test_incomplete_synthesis_automatically_opens_only_the_remaining_frozen_wave():
    parent = {
        "id": "round", "project_id": "project",
        "data": {
            "generation_wave": 1,
            "generation_budget": {"max_generation_waves": 2},
            "required_distance_coverage": {"NEAR": 1, "MID": 1, "FAR": 1},
        },
    }
    archive = {
        "id": "archive",
        "data": {"coverage": {"filled_distance_slots": {"NEAR": 1, "MID": 0, "FAR": 0}}},
    }
    service = DistributedDiscoveryService.__new__(DistributedDiscoveryService)
    service._round = lambda _round_id: parent
    service.archive_get = lambda _archive_id: archive
    calls: list[tuple[str, str, bool]] = []
    service.coverage_recover = lambda round_id, confirmation, automatic=False: calls.append(
        (round_id, confirmation, automatic)
    ) or {"id": "recovery"}

    assert service._recovery_wave_after_synthesis("round", "archive") == {"id": "recovery"}
    assert calls == [("round", "RECOVER_DISCOVERY_WAVE", True)]


def test_coverage_recovery_preserves_source_round_lineage_and_hidden_isolation():
    parent = {
        "id": "root", "project_id": "project", "kind": "DiscoveryRound", "status": "COMPLETED",
        "data": {
            "archive_id": "archive", "generation_wave": 1,
            "generation_budget": {"max_generation_waves": 2},
            "required_distance_coverage": {"NEAR": 1, "MID": 1, "FAR": 1},
            "frozen_state_hash": "frozen", "peer_visibility": "VISIBLE_FOR_SYNTHESIS",
        },
    }
    archive = {"id": "archive", "kind": "CrossWorkerQDArchive", "data": {
        "coverage": {"filled_distance_slots": {"NEAR": 1, "MID": 0, "FAR": 0}},
    }}
    created: list[dict] = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args, **_kwargs):
            return None

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def cursor():
            return Cursor()

    class Store:
        @staticmethod
        def _connect():
            return Connection()

        @staticmethod
        def object_get(identifier):
            return parent if identifier == "root" else archive

        @staticmethod
        def objects_list(*_args, **_kwargs):
            return []

        @staticmethod
        def object_create(_project_id, _kind, data, _event, _status):
            created.append(data)
            return {"id": "recovery", "data": data}

        @staticmethod
        def _event(*_args, **_kwargs):
            return None

    service = DistributedDiscoveryService.__new__(DistributedDiscoveryService)
    service.store = Store()
    service._event = lambda *_args, **_kwargs: None

    recovered = service.coverage_recover("root", "RECOVER_DISCOVERY_WAVE")
    assert recovered["id"] == "recovery"
    assert created[0]["source_round_ids"] == ["root"]
    assert created[0]["required_distance_coverage"] == {"NEAR": 0, "MID": 1, "FAR": 1}
    assert created[0]["peer_visibility"] == "HIDDEN"


def test_coverage_recovery_normalizes_an_existing_legacy_wave():
    parent_id = uuid.uuid4()
    recovery_id = uuid.uuid4()
    parent = {
        "id": parent_id, "project_id": "project", "kind": "DiscoveryRound",
        "data": {
            "archive_id": "archive", "generation_wave": 1,
            "generation_budget": {"max_generation_waves": 2},
            "required_distance_coverage": {"MID": 1},
        },
    }
    archive = {"id": "archive", "kind": "CrossWorkerQDArchive", "data": {
        "coverage": {"filled_distance_slots": {"MID": 0}},
    }}
    legacy = {"id": recovery_id, "kind": "DiscoveryRound", "data": {
        "recovery_of_round_id": str(parent_id), "peer_visibility": "VISIBLE_FOR_SYNTHESIS",
    }}
    updates: list[tuple] = []

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, *args, **_kwargs): updates.append(args)

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        @staticmethod
        def cursor(): return Cursor()

    class Store:
        @staticmethod
        def _connect(): return Connection()
        @staticmethod
        def object_get(identifier): return parent if str(identifier) == str(parent_id) else archive
        @staticmethod
        def objects_list(*_args, **_kwargs): return [legacy]
        @staticmethod
        def _event(*_args, **_kwargs): return None

    service = DistributedDiscoveryService.__new__(DistributedDiscoveryService)
    service.store = Store()
    service._event = lambda *_args, **_kwargs: None
    recovered = service.coverage_recover(str(parent_id), "RECOVER_DISCOVERY_WAVE")
    assert recovered["data"]["source_round_ids"] == [str(parent_id)]
    assert recovered["data"]["peer_visibility"] == "HIDDEN"
    assert recovered["data"]["coverage_recover"] == "COVERAGE_CONTINUED"
    assert any("UPDATE research_objects SET data" in statement[0] for statement in updates)


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_round_isolates_batches_then_archives_distinct_candidates():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-v33-{time.time_ns()}", "Independent discovery")['project_id']
    first = lab.join(None, "dde-a", "CODEX", project_id)
    second = lab.join(None, "dde-b", "LOCAL_AGENT", project_id)
    round_ = dde.create_round(project_id, None, "DIVERGENT_SEARCH", baseline_signature={"causal_object": "baseline"})
    assert all(
        "data_hash" in record and "data" not in record
        for record in round_["data"]["frozen_state"]["records"]
    )
    assignments = dde.recommended_assignments(round_["id"])
    assert {item["requested_distance"] for item in assignments} >= {"NEAR", "MID", "FAR", "ORTHOGONAL"}
    assert len({item["generation_operator"] for item in assignments}) == len(assignments)
    batch_a = dde.join_round(round_["id"], first["worker"]["id"], first["session_id"], "REPRESENTATION_RESET", "FAR")
    batch_b = dde.join_round(round_["id"], second["worker"]["id"], second["session_id"], "STRONG_NULL_CONSTRUCTION", "ORTHOGONAL")
    candidate_a = dde.submit_candidate(round_["id"], batch_a["id"], first["worker"]["id"], first["session_id"], _candidate("representation reset", {"representation": "point tokens"}))
    with pytest.raises(GPUError) as exc:
        dde.batch_get(round_["id"], batch_a["id"], second["session_id"])
    assert exc.value.error_type == "DISCOVERY_PEER_ISOLATION_ACTIVE"
    overridden = dde.peer_isolation_override(
        round_["id"], batch_b["id"], second["worker"]["id"], second["session_id"], "User requested comparison with A",
    )
    assert overridden["data"]["independent_generation"] is False
    assert dde.batch_get(round_["id"], batch_a["id"], second["session_id"])["id"] == batch_a["id"]
    assert lab.message_send(
        project_id, second["worker"]["id"], second["session_id"], "SHARE_FINDING",
        "override", "User explicitly authorized this non-independent comparison.", to_worker_id=first["worker"]["id"],
    )["message_type"] == "SHARE_FINDING"
    dde.batch_freeze(round_["id"], batch_a["id"], first["worker"]["id"], first["session_id"])
    # The explicit audited override persists for this round; freezing A must
    # not silently revoke the comparison authority granted to B.
    assert dde.batch_get(round_["id"], batch_a["id"], second["session_id"])["id"] == batch_a["id"]
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
    with pytest.raises(GPUError) as exc:
        dde.submit_candidate(round_["id"], batch_a["id"], first["worker"]["id"], first["session_id"], _candidate("late", {"representation": "late"}))
    assert exc.value.error_type == "DISCOVERY_ROUND_NOT_GENERATING"
    assert dde.outcome_get(candidate_a["id"])["resolution_status"] == "UNKNOWN"
    assert candidate_b["data"]["scientific_distance"] == "ORTHOGONAL"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_legacy_full_discovery_snapshot_keeps_compatible_staleness_hashing():
    store = ResearchStore(TEST_DATABASE_URL)
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-legacy-snapshot-{time.time_ns()}", "Legacy snapshot")['project_id']
    claim = store.object_create(project_id, "Claim", {"statement": "frozen", "scope": "fixture"}, "CLAIM_CREATED")
    legacy_snapshot = dde._scientific_snapshot(project_id, legacy_full_data=True)
    legacy_round = store.object_create(project_id, "DiscoveryRound", {
        "frozen_state": legacy_snapshot,
        "scientific_snapshot_hash": dde._hash(legacy_snapshot["records"]),
    }, "LEGACY_DISCOVERY_ROUND", "ACTIVE")
    assert dde.stale_check(legacy_round["id"])["stale"] is False
    store.object_update(claim["id"], {"scope": "changed"}, "ACTIVE", "CLAIM_CHANGED")
    assert dde.stale_check(legacy_round["id"])["stale"] is True


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_empty_batch_requires_explicit_honest_abstention():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-abstain-{time.time_ns()}", "Honest distant abstention")['project_id']
    worker = lab.join(None, "dde-abstain", "CODEX", project_id)
    round_ = dde.create_round(project_id, None, "PARADIGM_RESET", baseline_signature={"causal_object": "baseline"})
    batch = dde.join_round(round_["id"], worker["worker"]["id"], worker["session_id"], "ONTOLOGY_CHALLENGE", "ORTHOGONAL")
    with pytest.raises(GPUError) as exc:
        dde.batch_freeze(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"])
    assert exc.value.error_type == "DISCOVERY_ABSTENTION_REASON_REQUIRED"
    frozen = dde.batch_freeze(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"], "No coherent ontology challenge survived canonical constraints")
    assert frozen["status"] == "ABSTAINED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_distance_reservations_prevent_near_slots_from_consuming_open_search():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-reserved-{time.time_ns()}", "Distance reservation")['project_id']
    workers = [lab.join(None, f"reserved-{index}", "CODEX", project_id) for index in range(4)]
    round_ = dde.create_round(project_id, None, "DIVERGENT_SEARCH", baseline_signature={"causal_object": "baseline"})
    dde.join_round(round_["id"], workers[0]["worker"]["id"], workers[0]["session_id"], "LOCAL_CAUSAL_REPAIR", "NEAR")
    with pytest.raises(GPUError) as exc:
        dde.join_round(round_["id"], workers[1]["worker"]["id"], workers[1]["session_id"], "LOCAL_CAUSAL_REPAIR", "NEAR")
    assert exc.value.error_type == "DISCOVERY_DISTANCE_SLOT_RESERVED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_isolated_round_blocks_peer_finding_messages_and_detects_stale_state():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-isolation-{time.time_ns()}", "Message isolation")['project_id']
    first = lab.join(None, "isolation-a", "CODEX", project_id)
    second = lab.join(None, "isolation-b", "LOCAL_AGENT", project_id)
    round_ = dde.create_round(
        project_id, None, "MECHANISM_SEARCH",
        baseline_signature={"causal_object": "baseline"},
    )
    batch = dde.join_round(round_["id"], first["worker"]["id"], first["session_id"], "CAUSAL_INVERSION", "FAR")
    with pytest.raises(GPUError) as exc:
        lab.message_send(
            project_id, first["worker"]["id"], first["session_id"], "SHARE_FINDING",
            "candidate", "A current-round proposal", to_worker_id=second["worker"]["id"],
        )
    assert exc.value.error_type == "DISCOVERY_PEER_MESSAGE_BLOCKED"
    store.object_create(project_id, "Claim", {"statement": "state changed", "scope": "test"}, "CLAIM_CREATED")
    assert dde.stale_check(round_["id"])["stale"] is True
    assert dde.stale_check(round_["id"], mark_stale=True)["round"]["status"] == "STALE"
    assert batch["data"]["worker_session_id"] == first["session_id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_engineering_bookkeeping_does_not_self_stale_a_discovery_round():
    store = ResearchStore(TEST_DATABASE_URL)
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-engineering-bookkeeping-{time.time_ns()}", "Snapshot scope")['project_id']
    round_ = dde.create_round(project_id, None, "MECHANISM_SEARCH", baseline_signature={"causal_object": "baseline"})

    store.object_create(project_id, "EngineeringTask", {"purpose": "non-scientific review"}, "ENGINEERING_DIFF_REVIEWED")

    result = dde.stale_check(round_["id"])
    assert result["stale"] is False
    assert result["changed_scientific_record_count"] == 0


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_brain_portfolio_bookkeeping_does_not_invalidate_the_archive_it_consumes():
    store = ResearchStore(TEST_DATABASE_URL)
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-control-plane-{time.time_ns()}", "Archive freshness")['project_id']
    round_ = dde.create_round(project_id, None, "MECHANISM_SEARCH", baseline_signature={"causal_object": "baseline"})

    for kind in ("HypothesisPortfolio", "ResearchSituation", "CandidatePortfolio", "ResearchActionCandidate", "ResearchDecision"):
        store.object_create(project_id, kind, {"control_plane": True}, "BRAIN_BOOKKEEPING")

    result = dde.stale_check(round_["id"])
    assert result["stale"] is False
    assert result["changed_scientific_record_count"] == 0


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_stale_transition_expires_open_batches_and_makes_them_visibly_non_writeable():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, dde = LabController(store), DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-stale-open-{time.time_ns()}", "Stale open batch")['project_id']
    worker = lab.join(None, "stale-open", "CODEX", project_id)
    round_ = dde.create_round(project_id, None, "MECHANISM_SEARCH", baseline_signature={"causal_object": "baseline"})
    batch = dde.join_round(round_["id"], worker["worker"]["id"], worker["session_id"], "CAUSAL_INVERSION", "FAR")
    store.object_create(project_id, "Claim", {"statement": "external scientific change", "scope": "test"}, "CLAIM_CREATED")

    stale = dde.stale_check(round_["id"], mark_stale=True)
    assert stale["round"]["status"] == "STALE"
    view = dde.batch_get(round_["id"], batch["id"], worker["session_id"])
    assert view["status"] == "EXPIRED"
    assert view["effective_status"] == "EXPIRED"
    assert view["writeable"] is False
    assert view["recovery_action"] == "CREATE_FRESH_ROUND"
    assert next(
        member for member in dde._members(round_["id"])
        if str(member["candidate_batch_id"]) == str(batch["id"])
    )["status"] == "EXPIRED"

    with pytest.raises(GPUError) as exc:
        dde.submit_candidate(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"], _candidate("late", {"causal_object": "late"}))
    assert exc.value.error_type == "DISCOVERY_ROUND_NOT_GENERATING"
    assert exc.value.response()["error"] == {
        "type": "DISCOVERY_ROUND_NOT_GENERATING",
        "message": "Discovery round is not accepting candidate submissions.",
        "retryable": False,
        "actual_round_phase": "STALE",
        "stale": True,
        "batch_effective_status": "EXPIRED",
        "recovery_action": "CREATE_FRESH_ROUND",
    }


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_expired_independent_generation_work_claim_creates_fresh_batch_lane():
    """A new claimant must never inherit an expired generator session's batch."""
    store = ResearchStore(TEST_DATABASE_URL)
    lab, dde = LabController(store, lease_seconds=30), DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-reassigned-lane-{time.time_ns()}", "Fresh replacement lane")['project_id']
    original = lab.join(None, f"dde-original-generator-{time.time_ns()}", "CODEX", project_id)
    replacement = lab.join(None, f"dde-replacement-generator-{time.time_ns()}", "LOCAL_AGENT", project_id)
    round_ = dde.create_round(
        project_id, None, "MECHANISM_SEARCH",
        baseline_signature={"causal_object": "baseline"},
    )
    original_batch = dde.join_round(
        round_["id"], original["worker"]["id"], original["session_id"], "CAUSAL_INVERSION", "FAR"
    )
    work = dde.lab_work_items(round_["id"])[0]
    claimed = lab.claim_work(work["id"], original["worker"]["id"], original["session_id"])
    lab.start_work(claimed["id"], original["worker"]["id"], original["session_id"])
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE lab_work_leases SET expires_at=NOW() - INTERVAL '1 second' WHERE id=%s",
            (claimed["lease_id"],),
        )
    assert lab.recover_stale_leases(project_id)["recovered"] == 1
    assert lab.work_get(work["id"])["status"] == "READY"

    rebound = lab.claim_work(work["id"], replacement["worker"]["id"], replacement["session_id"])
    replacement_batch_id = rebound["related_refs"]["candidate_batch_id"]
    assert replacement_batch_id != original_batch["id"]
    assert rebound["related_refs"]["replaced_candidate_batch_id"] == str(original_batch["id"])
    assert rebound["related_refs"]["discovery_batch_recovery"] == "FRESH_INDEPENDENT_REPLACEMENT"
    assert dde.batch_get(round_["id"], original_batch["id"], original["session_id"])["effective_status"] == "EXPIRED"
    replacement_batch = dde.batch_get(round_["id"], replacement_batch_id, replacement["session_id"])
    assert replacement_batch["writeable"] is True
    assert replacement_batch["data"]["worker_session_id"] == replacement["session_id"]
    assert replacement_batch["data"]["replacement_for_batch_id"] == str(original_batch["id"])

    candidate = dde.submit_candidate(
        round_["id"], replacement_batch_id, replacement["worker"]["id"], replacement["session_id"],
        _candidate("fresh replacement", {"causal_object": "replacement"}),
    )
    assert candidate["data"]["candidate_batch_id"] == replacement_batch_id
    with pytest.raises(GPUError) as exc:
        dde.submit_candidate(
            round_["id"], original_batch["id"], original["worker"]["id"], original["session_id"],
            _candidate("expired lane", {"causal_object": "expired"}),
        )
    assert exc.value.error_type == "DISCOVERY_BATCH_NOT_WRITABLE"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_allocator_claims_ready_discovery_lane_after_stale_worker_projection():
    """A fresh independent lane must not be hidden by SUPPORTING classification.

    This is the production path used by ``portfolio_assign_existing`` after a
    previous generator expires.  A stale EXECUTING worker projection must also
    be reconciled before the allocator rejects an otherwise free worker.
    """
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, feature_flags={"BRANCH_AWARE_ASSIGNMENT": True})
    dde = DistributedDiscoveryService(store)
    project_id = store.project_create(
        f"dde-allocator-lane-{time.time_ns()}", "Allocator claims discovery lane"
    )["project_id"]
    worker = lab.join(None, f"dde-allocator-generator-{time.time_ns()}", "CODEX", project_id)
    round_ = dde.create_round(
        project_id, None, "MECHANISM_SEARCH", baseline_signature={"causal_object": "baseline"}
    )
    batch = dde.join_round(
        round_["id"], worker["worker"]["id"], worker["session_id"], "ONTOLOGY_CHALLENGE", "ORTHOGONAL"
    )
    work = dde.lab_work_items(round_["id"])[0]
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE research_workers SET availability_state='EXECUTING' WHERE id=%s",
            (worker["worker"]["id"],),
        )

    assigned = lab.portfolio_assign_existing(project_id, worker["worker"]["id"], worker["session_id"])
    assert assigned["status"] == "ASSIGNED"
    assert assigned["work_item"]["id"] == work["id"]
    assert assigned["work_item"]["related_refs"]["candidate_batch_id"] == str(batch["id"])
    assert lab.work_get(work["id"])["status"] == "CLAIMED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_stale_transition_and_candidate_submission_leave_no_joined_active_batch():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, dde = LabController(store), DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-stale-race-{time.time_ns()}", "Stale race")['project_id']
    worker = lab.join(None, "stale-race", "CODEX", project_id)
    round_ = dde.create_round(project_id, None, "MECHANISM_SEARCH", baseline_signature={"causal_object": "baseline"})
    batch = dde.join_round(round_["id"], worker["worker"]["id"], worker["session_id"], "CAUSAL_INVERSION", "FAR")
    store.object_create(project_id, "Claim", {"statement": "external scientific change", "scope": "test"}, "CLAIM_CREATED")
    barrier = Barrier(2)

    def mark_stale():
        barrier.wait()
        return dde.stale_check(round_["id"], mark_stale=True)

    def submit():
        barrier.wait()
        try:
            return dde.submit_candidate(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"], _candidate("racing", {"causal_object": "race"}))
        except GPUError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        stale_future = pool.submit(mark_stale)
        submit_future = pool.submit(submit)
        stale_result = stale_future.result()
        submit_result = submit_future.result()

    # The two state transitions share the parent round lock. A submission may
    # win before expiry, but after staleness commits the batch can never remain
    # ACTIVE/JOINED or advertise write authority.
    assert stale_result["stale"] is True
    assert not isinstance(submit_result, Exception) or submit_result.error_type == "DISCOVERY_ROUND_NOT_GENERATING"
    view = dde.batch_get(round_["id"], batch["id"], worker["session_id"])
    assert (view["status"], view["effective_status"], view["writeable"]) == ("EXPIRED", "EXPIRED", False)


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_stale_round_cannot_be_synthesized_into_a_new_execution_portfolio():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, dde = LabController(store), DistributedDiscoveryService(store)
    project_id = store.project_create(f"dde-stale-{time.time_ns()}", "Stale round")['project_id']
    worker = lab.join(None, "stale-worker", "CODEX", project_id)
    round_ = dde.create_round(project_id, None, "MECHANISM_SEARCH", baseline_signature={"causal_object": "baseline"})
    batch = dde.join_round(round_["id"], worker["worker"]["id"], worker["session_id"], "CAUSAL_INVERSION", "FAR")
    dde.submit_candidate(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"], _candidate("candidate", {"causal_object": "new"}))
    dde.batch_freeze(round_["id"], batch["id"], worker["worker"]["id"], worker["session_id"])
    store.object_create(project_id, "Claim", {"statement": "decisive update", "scope": "test"}, "CLAIM_CREATED")
    with pytest.raises(GPUError) as exc:
        dde.synthesize(round_["id"])
    assert exc.value.error_type == "DISCOVERY_ROUND_STALE"
    assert dde.round_get(round_["id"])["status"] == "STALE"
