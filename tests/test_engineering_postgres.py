import os
import time

import pytest

from gpu_lab.engineering import EngineeringService
from gpu_lab.research import ResearchStore

TEST_DATABASE_URL = os.getenv("GPU_LAB_TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_engineering_task_result_and_events_survive_store_restart():
    store = ResearchStore(TEST_DATABASE_URL)
    project = store.project_create(
        f"engineering-restart-{time.time_ns()}", "Engineering persistence smoke"
    )
    service = EngineeringService(store)
    task = service.task_create(
        project["project_id"], "Verify a bounded implementation", "BUG_FIX",
        repository="gpu-lab", base_commit="test-base",
        implementation_guards=[{"name": "smoke", "type": "BOOLEAN"}],
    )
    service.task_start(
        task["id"], {"files_read": ["src/gpu_lab/engineering.py"]},
        {"commands_run": ["pytest tests/test_engineering.py"], "passed": True},
    )
    service.diff_review(task["id"], {
        "files_changed": ["src/gpu_lab/engineering.py"],
        "diff_summary": "Bounded persistence smoke.",
        "unrelated_changes": False,
        "scientific_variable_drift": False,
    })
    result = service.result_record(task["id"], {
        "implementation_verification": "VERIFIED_INTEGRATION",
        "implementation_guard_results": [{"name": "smoke", "passed": True}],
    })

    restarted = ResearchStore(TEST_DATABASE_URL)
    restarted_service = EngineeringService(restarted)
    persisted_task = restarted_service.task_get(task["id"])
    persisted_result = restarted_service.result_get(result["id"])
    readiness = restarted_service.task_verify(task["id"])
    events = restarted.events(project["project_id"], 50)

    assert persisted_task["status"] == "COMPLETED"
    assert persisted_result["data"]["scientific_result"] == "NOT_ASSESSED"
    assert readiness["ready_for_scientific_execution"] is True
    assert {event["event_type"] for event in events} >= {
        "ENGINEERING_TASK_CREATED", "BASELINE_VERIFIED", "ENGINEERING_DIFF_REVIEWED",
        "ENGINEERING_RESULT_RECORDED", "ENGINEERING_VERIFIED",
    }
