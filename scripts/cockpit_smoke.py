"""Exercise the durable cockpit substrate against an explicit PostgreSQL URL."""

from __future__ import annotations

import os
import time

from gpu_lab.cockpit import CockpitController
from gpu_lab.lab import LabController
from gpu_lab.research import ResearchStore


def main() -> None:
    store = ResearchStore(os.environ["GPU_LAB_TEST_DATABASE_URL"])
    lab = LabController(store)
    cockpit = CockpitController(store, lab)
    project = store.project_create(f"cockpit-smoke-{time.time_ns()}", "Cockpit smoke")
    project_id = project["project_id"]
    joined = lab.join(None, "cockpit-smoke-worker", "CHATGPT_WEB", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    work = lab.create_work(
        project_id, "REVIEW", "Bounded review", "Smoke work", "ADVERSARIAL_REVIEWER",
        worker_id, created_session_id=session_id,
    )
    lab.claim_work(work["id"], worker_id, session_id)
    controls = cockpit.controls_set(
        project_id, worker_id, session_id, autopilot_enabled=True, auto_continue_enabled=True,
    )
    report = cockpit.turn_report(project_id, worker_id, session_id, "CONTINUE", "Smoke continuation")
    assert report["wake_request_id"]
    restarted = CockpitController(ResearchStore(os.environ["GPU_LAB_TEST_DATABASE_URL"]))
    state = restarted.state_get(project_id, session_id)
    assert state["controls"]["autopilot_enabled"] is True
    assert state["pending_wake_requests"][0]["id"] == report["wake_request_id"]
    print("COCKPIT_SMOKE_OK")
    print(f"project={project_id} worker={worker_id} wake={report['wake_request_id']} paused={controls['paused']}")


if __name__ == "__main__":
    main()
