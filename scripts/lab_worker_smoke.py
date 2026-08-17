"""Exercise durable Lab Worker Mode against an isolated PostgreSQL database."""

from __future__ import annotations

import os
import time

from gpu_lab.lab import LabController
from gpu_lab.research import ResearchStore


def main() -> None:
    url = os.environ["GPU_LAB_TEST_DATABASE_URL"]
    store = ResearchStore(url)
    lab = LabController(store, lease_seconds=60)
    project = store.project_create(f"lab-worker-smoke-{time.time_ns()}", "Lab worker smoke")
    project_id = project["project_id"]
    first = lab.join(None, "lab-smoke-a", "CHATGPT_WEB", project_id)
    second = lab.join(None, "lab-smoke-b", "CODEX", project_id)
    work_a = lab.create_work(project_id, "LITERATURE", "Independent literature", "Find evidence", "LITERATURE_RESEARCHER")
    work_b = lab.create_work(project_id, "REVIEW", "Independent review", "Review a branch", "ADVERSARIAL_REVIEWER")
    lab.claim_work(work_a["id"], first["worker"]["id"], first["session_id"])
    lab.claim_work(work_b["id"], second["worker"]["id"], second["session_id"])
    lab.message_send(project_id, first["worker"]["id"], "REQUEST_REVIEW", "Review", "Please review the branch.", to_worker_id=second["worker"]["id"])
    run = store.object_create(project_id, "ExperimentRun", {"mode": "smoke"}, "EXPERIMENT_STARTED", "running")
    inspection = lab.create_work(project_id, "INSPECT_RESULT", "Inspect E1", "Wait for E1", "RESULT_INSPECTOR", dependencies=[{"target_type": "EXPERIMENT_RUN", "target_id": run["id"], "required_statuses": ["completed"]}])
    store.object_update(run["id"], {}, "completed", "EXPERIMENT_COMPLETED")
    lab.resolve_dependencies(project_id)
    restarted = LabController(ResearchStore(url))
    assert restarted.work_get(inspection["id"])["status"] == "READY"
    print("LAB_WORKER_SMOKE_OK")
    print(f"project={project_id} workers=2 claimed=2 dependent_ready={inspection['id']}")


if __name__ == "__main__":
    main()
