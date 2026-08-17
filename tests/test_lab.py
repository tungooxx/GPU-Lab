import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpu_lab.errors import GPUError
from gpu_lab.lab import LabController
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
    work_a = lab.create_work(project_id, "LITERATURE", "Read evidence", "Independent retrieval", "LITERATURE_RESEARCHER", first_worker)
    work_b = lab.create_work(project_id, "REVIEW", "Review result", "Independent review", "ADVERSARIAL_REVIEWER", second_worker)

    lab.claim_work(work_a["id"], first_worker, first["session_id"])
    state_b = lab.state_get(project_id, second["session_id"])
    assert {item["id"] for item in state_b["running_work_items"]} == {work_a["id"]}
    with pytest.raises(GPUError, match="LAB_WORK_NOT_CLAIMABLE"):
        lab.claim_work(work_a["id"], second_worker, second["session_id"])
    assert lab.claim_work(work_b["id"], second_worker, second["session_id"])["id"] == work_b["id"]

    lab.message_send(project_id, first_worker, "SHARE_FINDING", "Opinion", "H1 is definitely correct")
    assert lab.message_list(project_id, second_worker)
    assert store.objects_list(project_id, "Hypothesis", limit=None) == []


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_atomic_claim_and_dependency_reactivation_survive_store_restart():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project = store.project_create(f"lab-atomic-{time.time_ns()}", "Atomic claims")
    project_id = project["project_id"]
    first = lab.join(None, "atomic-a", "CHATGPT_WEB", project_id)
    second = lab.join(None, "atomic-b", "CODEX", project_id)
    work = lab.create_work(project_id, "REVIEW", "One canonical task", "Must only claim once", "ADVERSARIAL_REVIEWER")
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
    waiting = lab.create_work(project_id, "INSPECT_RESULT", "Inspect simulated run", "Wait for result", "RESULT_INSPECTOR", dependencies=[{"target_type": "EXPERIMENT_RUN", "target_id": run["id"], "required_statuses": ["completed"]}])
    assert waiting["status"] == "WAITING_DEPENDENCY"
    store.object_update(run["id"], {}, "completed", "EXPERIMENT_COMPLETED")
    assert lab.resolve_dependencies(project_id)["ready"] == 1
    assert LabController(ResearchStore(TEST_DATABASE_URL)).work_get(waiting["id"])["status"] == "READY"
