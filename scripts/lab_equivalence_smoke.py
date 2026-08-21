"""Verify dependency reconciliation safely resolves dormant equivalent WorkItems."""

from __future__ import annotations

import os
import time

from gpu_lab.lab import LabController
from gpu_lab.research import ResearchStore


def main() -> None:
    store = ResearchStore(os.environ["GPU_LAB_TEST_DATABASE_URL"])
    lab = LabController(store)
    project_id = store.project_create(f"lab-equivalence-smoke-{time.time_ns()}", "Disposable equivalence smoke")["project_id"]
    worker = lab.join(None, "equivalence-smoke", "CODEX", project_id)
    worker_id, session_id = worker["worker"]["id"], worker["session_id"]
    prerequisite = lab.create_work(project_id, "ENGINEERING", "Implement", "Finish first", "ENGINEER", worker_id, created_session_id=session_id)
    dependency = [{"target_type": "WORK_ITEM", "target_id": prerequisite["id"], "required_statuses": ["COMPLETED"]}]
    first = lab.create_work(project_id, "REVIEW", "First", "Canonical", "ADVERSARIAL_REVIEWER", worker_id, created_session_id=session_id, dependencies=dependency, equivalence_key="smoke-equivalence", dormant_until_dependencies=True)
    duplicate = lab.create_work(project_id, "REVIEW", "Second", "Duplicate", "ADVERSARIAL_REVIEWER", worker_id, created_session_id=session_id, dependencies=dependency, equivalence_key="smoke-equivalence", dormant_until_dependencies=True)
    lab.complete_work(lab.claim_work(prerequisite["id"], worker_id, session_id)["id"], worker_id, session_id, summary="Done")
    assert lab.resolve_dependencies(project_id) == {"ready": 0, "waiting": 0, "invalidated": 0}
    assert lab.work_get(first["id"])["status"] == "READY"
    assert lab.work_get(duplicate["id"])["status"] == "SUPERSEDED"
    print("LAB_EQUIVALENCE_SMOKE_OK")


if __name__ == "__main__":
    main()
