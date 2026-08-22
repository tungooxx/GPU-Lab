import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpu_lab.cockpit import CockpitController
from gpu_lab.errors import GPUError
from gpu_lab.lab import ACTIVE_WORK_STATUSES, LabController
from gpu_lab.research import ResearchStore

TEST_DATABASE_URL = os.getenv("GPU_LAB_TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_two_workers_claim_distinct_work_and_messages_are_not_evidence():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project = store.project_create(f"lab-two-workers-{time.time_ns()}", "Shared coordination")
    project_id = project["project_id"]
    first = lab.join(None, "worker-a", "CHATGPT_WEB", project_id)
    second = lab.join(None, "worker-b", "CODEX", project_id)
    first_worker, second_worker = first["worker"]["id"], second["worker"]["id"]
    work_a = lab.create_work(project_id, "LITERATURE", "Read evidence", "Independent retrieval", "LITERATURE_RESEARCHER", first_worker, created_session_id=first["session_id"])
    work_b = lab.create_work(project_id, "REVIEW", "Review result", "Independent review", "ADVERSARIAL_REVIEWER", second_worker, created_session_id=second["session_id"])

    lab.claim_work(work_a["id"], first_worker, first["session_id"])
    assert lab.start_work(work_a["id"], first_worker, first["session_id"])["status"] == "RUNNING"
    state_b = lab.state_get(project_id, second["session_id"])
    assert {item["id"] for item in state_b["running_work_items"]} == {work_a["id"]}
    with pytest.raises(GPUError, match="LAB_WORK_NOT_CLAIMABLE"):
        lab.claim_work(work_a["id"], second_worker, second["session_id"])
    assert lab.claim_work(work_b["id"], second_worker, second["session_id"])["id"] == work_b["id"]


    lab.message_send(project_id, first_worker, first["session_id"], "SHARE_FINDING", "Opinion", "H1 is definitely correct")
    assert lab.message_list(project_id, second_worker)
    assert store.objects_list(project_id, "Hypothesis", limit=None) == []


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_browser_runtime_marks_first_successful_connection_attached():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    cockpit = CockpitController(store, lab)
    project = store.project_create(f"cockpit-runtime-{time.time_ns()}", "Runtime attachment")
    joined = lab.join(None, "browser-worker", "CHATGPT_WEB", project["project_id"])

    runtime = cockpit.runtime_attach(
        project["project_id"],
        joined["worker"]["id"],
        joined["session_id"],
        "https://chatgpt.com/c/demo",
    )
    assert runtime["attached_at"] is None

    connected = cockpit.runtime_status(runtime["id"], "READY")

    assert connected["attached_at"] is not None


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_cockpit_groups_live_workers_by_project_with_identifiers():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    cockpit = CockpitController(store, lab)
    first = store.project_create(f"cockpit-live-a-{time.time_ns()}", "First project")
    second = store.project_create(f"cockpit-live-b-{time.time_ns()}", "Second project")
    first_worker = lab.join(None, "live-a", "CHATGPT_WEB", first["project_id"])
    second_worker = lab.join(None, "live-b", "CODEX", second["project_id"])

    by_project = {item["project_id"]: item for item in cockpit.live_workers_by_project()}

    assert by_project[first["project_id"]]["live_worker_count"] == 1
    assert by_project[first["project_id"]]["workers"][0]["display_name"] == "live-a"
    assert by_project[first["project_id"]]["workers"][0]["worker_id"] == first_worker["worker"]["id"]
    assert by_project[second["project_id"]]["workers"][0]["worker_id"] == second_worker["worker"]["id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_ready_work_wake_is_deduplicated():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    cockpit = CockpitController(store, lab)
    project_id = store.project_create(f"cockpit-wake-{time.time_ns()}", "Wake deduplication")["project_id"]
    joined = lab.join(None, "wake-worker", "CHATGPT_WEB", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    cockpit.controls_set(project_id, worker_id, session_id, autopilot_enabled=True, auto_continue_enabled=True)
    runtime = cockpit.runtime_attach(project_id, worker_id, session_id, "https://chatgpt.com/c/wake-test")
    cockpit.runtime_status(runtime["id"], "READY")
    work = lab.create_work(project_id, "REVIEW", "Ready review", "Wake once", "RESULT_INSPECTOR", worker_id, created_session_id=session_id)

    assert cockpit.wake_ready_work(project_id, [work["id"]]) == {"queued": 1}
    assert cockpit.wake_ready_work(project_id, [work["id"]]) == {"queued": 0}
    assert len(cockpit.state_get(project_id)["pending_wake_requests"]) == 1


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_atomic_claim_and_dependency_reactivation_survive_store_restart():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project = store.project_create(f"lab-atomic-{time.time_ns()}", "Atomic claims")
    project_id = project["project_id"]
    first = lab.join(None, "atomic-a", "CHATGPT_WEB", project_id)
    second = lab.join(None, "atomic-b", "CODEX", project_id)
    work = lab.create_work(project_id, "REVIEW", "One canonical task", "Must only claim once", "ADVERSARIAL_REVIEWER", first["worker"]["id"], created_session_id=first["session_id"])
    barrier = threading.Barrier(2)

    def claim(joined):
        barrier.wait()
        controller = LabController(ResearchStore(TEST_DATABASE_URL))
        try:
            return controller.claim_work(work["id"], joined["worker"]["id"], joined["session_id"])["lease_id"]
        except GPUError as error:
            return error.error_type

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, (first, second)))
    assert sum(isinstance(result, str) and result != "LAB_WORK_NOT_CLAIMABLE" for result in results) == 1
    assert results.count("LAB_WORK_NOT_CLAIMABLE") == 1

    run = store.object_create(project_id, "ExperimentRun", {"label": "simulated"}, "EXPERIMENT_STARTED", "running")
    waiting = lab.create_work(project_id, "INSPECT_RESULT", "Inspect simulated run", "Wait for result", "RESULT_INSPECTOR", first["worker"]["id"], dependencies=[{"target_type": "EXPERIMENT_RUN", "target_id": run["id"], "required_statuses": ["completed"]}], created_session_id=first["session_id"])
    assert waiting["status"] == "WAITING_DEPENDENCY"
    store.object_update(run["id"], {}, "completed", "EXPERIMENT_COMPLETED")
    assert lab.resolve_dependencies(project_id)["ready"] == 1
    assert LabController(ResearchStore(TEST_DATABASE_URL)).work_get(waiting["id"])["status"] == "READY"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_unsatisfied_dependency_demotes_ready_work_and_blocks_claim():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-dependency-invariant-{time.time_ns()}", "Dependency invariant")["project_id"]
    worker = lab.join(None, "dependency-worker", "CODEX", project_id)
    prerequisite = lab.create_work(
        project_id, "ENGINEERING", "Recover service", "Repair upstream service", "ENGINEER",
        worker["worker"]["id"], created_session_id=worker["session_id"],
    )
    dependent = lab.create_work(
        project_id, "EXPERIMENT_DESIGN", "Preregister", "Must wait", "SCIENTIST",
        worker["worker"]["id"], created_session_id=worker["session_id"],
        dependencies=[{"target_type": "WORK_ITEM", "target_id": prerequisite["id"], "required_statuses": ["COMPLETED"]}],
    )
    assert dependent["status"] == "WAITING_DEPENDENCY"
    # Simulate a legacy/partial write that attached a dependency to READY work.
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE lab_work_items SET status='READY',blocked_reason=NULL WHERE id=%s", (dependent["id"],))
    assert lab.resolve_dependencies(project_id)["waiting"] == 1
    assert lab.work_get(dependent["id"])["status"] == "WAITING_DEPENDENCY"
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE lab_work_items SET status='READY',blocked_reason=NULL WHERE id=%s", (dependent["id"],))
    with pytest.raises(GPUError) as exc:
        lab.claim_work(dependent["id"], worker["worker"]["id"], worker["session_id"])
    assert exc.value.error_type == "LAB_WORK_DEPENDENCY_UNSATISFIED"
    assert lab.work_get(dependent["id"])["status"] == "WAITING_DEPENDENCY"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_dependency_reconciliation_supersedes_duplicate_equivalent_dormant_work():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-equivalence-reconcile-{time.time_ns()}", "Equivalent work") ["project_id"]
    worker = lab.join(None, "equivalence-worker", "CODEX", project_id)
    prerequisite = lab.create_work(
        project_id, "ENGINEERING", "Implement", "Complete first", "ENGINEER",
        worker["worker"]["id"], created_session_id=worker["session_id"],
    )
    dependency = [{"target_type": "WORK_ITEM", "target_id": prerequisite["id"], "required_statuses": ["COMPLETED"]}]
    first = lab.create_work(
        project_id, "REVIEW", "Canonical review", "One review", "ADVERSARIAL_REVIEWER",
        worker["worker"]["id"], created_session_id=worker["session_id"], dependencies=dependency,
        equivalence_key="same-review", dormant_until_dependencies=True,
    )
    duplicate = lab.create_work(
        project_id, "REVIEW", "Duplicate review", "Same review", "ADVERSARIAL_REVIEWER",
        worker["worker"]["id"], created_session_id=worker["session_id"], dependencies=dependency,
        equivalence_key="same-review", dormant_until_dependencies=True,
    )
    claimed = lab.claim_work(prerequisite["id"], worker["worker"]["id"], worker["session_id"])
    lab.complete_work(claimed["id"], worker["worker"]["id"], worker["session_id"], summary="Implemented")
    # complete_work invokes dependency reconciliation itself; the explicit
    # second pass must be idempotent rather than recreating either work item.
    assert lab.resolve_dependencies(project_id) == {"ready": 0, "waiting": 0, "invalidated": 0}
    assert lab.work_get(first["id"])["status"] == "READY"
    reconciled = lab.work_get(duplicate["id"])
    assert reconciled["status"] == "SUPERSEDED"
    assert reconciled["superseded_by"] == first["id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_lab_state_summary_does_not_return_historical_large_object_data():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-compact-state-{time.time_ns()}", "Compact state") ["project_id"]
    store.object_create(project_id, "Artifact", {"content": "x" * 200_000}, "ARTIFACT_CREATED", "COMPLETED")
    summary = lab.state_get(project_id)
    assert summary["research_state_version"] == 1
    assert "content" not in str(summary)
    assert len(str(summary)) < 20_000

@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_budget_and_expired_lease_release_work_without_touching_experiment():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, lease_seconds=30)
    project = store.project_create(f"lab-budget-{time.time_ns()}", "Budget and lease")
    project_id = project["project_id"]
    first = lab.join(None, "budget-a", "CHATGPT_WEB", project_id)
    second = lab.join(None, "budget-b", "CODEX", project_id)
    lab.budget_set(project_id, first["worker"]["id"], first["session_id"], {"max_active_workers": 1})
    first_work = lab.create_work(project_id, "REVIEW", "First", "First", "ADVERSARIAL_REVIEWER", first["worker"]["id"], created_session_id=first["session_id"])
    second_work = lab.create_work(project_id, "REVIEW", "Second", "Second", "ADVERSARIAL_REVIEWER", second["worker"]["id"], created_session_id=second["session_id"])
    lab.claim_work(first_work["id"], first["worker"]["id"], first["session_id"])
    with pytest.raises(GPUError, match="LAB_WORKER_BUDGET_EXCEEDED"):
        lab.claim_work(second_work["id"], second["worker"]["id"], second["session_id"])

    # Lease recovery releases only coordination ownership; the external run remains canonical and running.
    run = store.object_create(project_id, "ExperimentRun", {"label": "must-not-cancel"}, "EXPERIMENT_STARTED", "running")
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE lab_work_leases SET expires_at=NOW() - INTERVAL '1 second' WHERE work_item_id=%s", (first_work["id"],))
    assert lab.recover_stale_leases(project_id)["recovered"] == 1
    assert lab.work_get(first_work["id"])["status"] == "READY"
    assert store.object_get(run["id"])["status"] == "running"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_orphaned_running_work_without_a_lease_is_recovered():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, lease_seconds=30)
    project_id = store.project_create(f"lab-orphan-{time.time_ns()}", "Orphan recovery")["project_id"]
    joined = lab.join(None, "orphaned-worker", "CODEX", project_id)
    work = lab.create_work(
        project_id, "REVIEW", "Recover me", "No owner may run forever", "REVIEWER",
        joined["worker"]["id"], created_session_id=joined["session_id"],
    )
    lab.claim_work(work["id"], joined["worker"]["id"], joined["session_id"])
    lab.start_work(work["id"], joined["worker"]["id"], joined["session_id"])
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM lab_work_leases WHERE work_item_id=%s", (work["id"],))
        cur.execute(
            "UPDATE research_worker_sessions SET status='EXPIRED',last_heartbeat_at="
            "NOW() - INTERVAL '1 hour' WHERE id=%s",
            (joined["session_id"],),
        )

    assert lab.recover_stale_leases(project_id) == {"recovered": 1}
    assert lab.work_get(work["id"])["status"] == "READY"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_attached_execution_cannot_return_to_ready_after_worker_disconnect():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, lease_seconds=30)
    project_id = store.project_create(f"lab-attachment-{time.time_ns()}", "Attached execution")["project_id"]
    joined = lab.join(None, "attached-worker", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    run = store.object_create(project_id, "ExperimentRun", {"label": "canonical"}, "EXPERIMENT_STARTED", "running")
    work = lab.create_work(project_id, "TRAINING_RUN", "Run canonical", "Launch once", "EXECUTION", worker_id, created_session_id=session_id)
    lab.claim_work(work["id"], worker_id, session_id)
    attached = lab.attach_experiment_run(work["id"], worker_id, session_id, run["id"])
    assert attached["status"] == "RUNNING_DETACHED"
    assert attached["related_refs"]["experiment_run_id"] == run["id"]
    assert lab.work_list(project_id, ["READY"]) == []

    assert lab.experiment_run_terminal(run["id"], "completed") == {"result_ready": 1}
    assert lab.work_get(work["id"])["status"] == "RESULT_READY"
    assert lab.work_list(project_id, ["READY"]) == []


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_scientific_gate_authority_preflight_unlock_and_supersession():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, lease_seconds=30)
    project_id = store.project_create(f"lab-gates-{time.time_ns()}", "Gate coordination")["project_id"]
    worker = lab.join(None, "gate-worker", "CODEX", project_id)
    worker_id, session_id = worker["worker"]["id"], worker["session_id"]

    gate = lab.gate_ensure(project_id, "RESULT_ASSESSMENT", "experiment-v1", "v1", worker_id, session_id)
    assert lab.gate_ensure(project_id, "RESULT_ASSESSMENT", "experiment-v1", "v1", worker_id, session_id)["id"] == gate["id"]
    first = lab.gate_work_ensure(gate["id"], "REVIEW", "Canonical review", "Review E1", "RESULT_INSPECTOR", worker_id, session_id)
    second = lab.gate_work_ensure(gate["id"], "REVIEW", "Duplicate review", "Must reuse", "RESULT_INSPECTOR", worker_id, session_id)
    assert first["id"] == second["id"]
    assert first["authority_status"] == "AUTHORITATIVE"

    dependent = lab.create_work(
        project_id, "GENERALIZATION", "Conditional next work", "Wait for gate", "SCIENTIST",
        worker_id, dependencies=[{"target_type": "SCIENTIFIC_GATE", "target_id": gate["id"], "required_statuses": ["PASS"]}],
        created_session_id=session_id, dormant_until_dependencies=True,
    )
    assert dependent["status"] == "DORMANT"
    failed = lab.preflight_run(gate["id"], worker_id, session_id, {"checkpoint": False, "tokenizer": True})
    assert failed["preflight"]["status"] == "FAIL"
    with pytest.raises(GPUError, match="PREFLIGHT_NOT_PASS"):
        lab.gate_resolve(gate["id"], worker_id, session_id, "PASS")

    passed = lab.preflight_run(gate["id"], worker_id, session_id, {"checkpoint": True, "tokenizer": True})
    assert passed["preflight"]["status"] == "PASS"
    with pytest.raises(GPUError, match="SEMANTIC_REVIEW_REQUIRED"):
        lab.gate_resolve(gate["id"], worker_id, session_id, "PASS")
    claimed_review = lab.claim_work(first["id"], worker_id, session_id)
    lab.complete_work(claimed_review["id"], worker_id, session_id, summary="Semantic review passed")
    resolved = lab.gate_resolve(gate["id"], worker_id, session_id, "PASS", first["id"], rationale="Semantic review passed")
    assert resolved["gate"]["status"] == "PASS"
    assert lab.work_get(dependent["id"])["status"] == "READY"
    repeated = lab.gate_resolve(gate["id"], worker_id, session_id, "PASS", first["id"])
    assert repeated["idempotent"] is True
    with pytest.raises(GPUError, match="ALREADY_RESOLVED"):
        lab.gate_resolve(gate["id"], worker_id, session_id, "FAIL", first["id"])

    obsolete = lab.create_work(
        project_id, "REVIEW", "Obsolete E1 follow-up", "Bound to E1", "RESULT_INSPECTOR",
        worker_id, related_refs={"experiment_id": "experiment-v1"}, created_session_id=session_id,
    )
    claimed = lab.claim_work(obsolete["id"], worker_id, session_id)
    successor = lab.gate_ensure(project_id, "RESULT_ASSESSMENT", "experiment-v2", "v2", worker_id, session_id)
    summary = lab.supersede_subject(project_id, "experiment-v1", "experiment-v2", "Corrected canonical experiment", worker_id, session_id, successor["id"])
    assert summary["work_items_superseded"] == 1
    assert lab.work_get(first["id"])["status"] == "COMPLETED"
    assert lab.work_get(obsolete["id"])["status"] == "SUPERSEDED"
    synced = lab.sync(session_id, project_id, current_work_item_id=claimed["id"], expected_work_version=claimed["work_version"])
    assert synced["lease_state"] == "LEASE_LOST"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_concurrent_gate_authority_creation_reuses_one_work_item():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-gate-concurrent-{time.time_ns()}", "Concurrent gate authority")["project_id"]
    first = lab.join(None, "concurrent-gate-a", "CODEX", project_id)
    second = lab.join(None, "concurrent-gate-b", "LOCAL_AGENT", project_id)
    gate = lab.gate_ensure(project_id, "RESULT_ASSESSMENT", "experiment", "v1", first["worker"]["id"], first["session_id"])
    barrier = threading.Barrier(2)

    def ensure(joined):
        barrier.wait()
        controller = LabController(ResearchStore(TEST_DATABASE_URL))
        return controller.gate_work_ensure(
            gate["id"], "REVIEW", "Canonical review", "One authority", "RESULT_INSPECTOR",
            joined["worker"]["id"], joined["session_id"],
        )["id"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(ensure, (first, second)))
    assert result[0] == result[1]
    authorities = lab.work_list(project_id, list(ACTIVE_WORK_STATUSES))
    assert [item["id"] for item in authorities if item["authority_status"] == "AUTHORITATIVE"] == [result[0]]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_project_scope_and_message_acknowledgement_require_active_session():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    one = store.project_create(f"lab-scope-one-{time.time_ns()}", "One")["project_id"]
    two = store.project_create(f"lab-scope-two-{time.time_ns()}", "Two")["project_id"]
    sender = lab.join(None, "scope-sender", "CODEX", one)
    recipient = lab.join(None, "scope-recipient", "CHATGPT_WEB", one)
    outsider = lab.join(None, "scope-outsider", "OTHER", two)
    message = lab.message_send(one, sender["worker"]["id"], sender["session_id"], "REQUEST_REVIEW", "Scope", "Project one only", to_worker_id=recipient["worker"]["id"])
    assert lab.message_list(two, outsider["worker"]["id"]) == []
    assert lab.message_mark_read(one, recipient["worker"]["id"], recipient["session_id"], [message["id"]]) == {"marked_read": 1}
    assert lab.message_list(one, recipient["worker"]["id"], unread_only=True) == []
    with pytest.raises(GPUError, match="LAB_MESSAGE_RECIPIENT_NOT_IN_PROJECT"):
        lab.message_send(one, sender["worker"]["id"], sender["session_id"], "REQUEST_REVIEW", "Bad", "No cross-project recipient", to_worker_id=outsider["worker"]["id"])
