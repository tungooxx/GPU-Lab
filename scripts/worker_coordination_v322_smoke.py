"""Exercise v3.2.2 deterministic coordination against a disposable PostgreSQL project."""

from __future__ import annotations

import os
import time

from gpu_lab.cockpit import CockpitController
from gpu_lab.lab import LabController
from gpu_lab.research import ResearchStore


def event(timeline: list[str], text: str) -> None:
    timeline.append(text)
    print(f"  {len(timeline):02d}. {text}")


def main() -> None:
    store = ResearchStore(os.environ["GPU_LAB_TEST_DATABASE_URL"])
    lab = LabController(store, lease_seconds=30)
    cockpit = CockpitController(store, lab)
    project_id = store.project_create(f"v322-coordination-{time.time_ns()}", "Disposable v3.2.2 smoke")["project_id"]
    timeline: list[str] = []
    first = lab.join(None, "v322-worker-a", "CHATGPT_WEB", project_id)
    second = lab.join(None, "v322-worker-b", "CODEX", project_id)
    third = lab.join(None, "v322-worker-c", "LOCAL_AGENT", project_id)
    cockpit.controls_set(project_id, third["worker"]["id"], third["session_id"], autopilot_enabled=True, auto_continue_enabled=True)
    runtime = cockpit.runtime_attach(project_id, third["worker"]["id"], third["session_id"], "https://chatgpt.com/c/v322-smoke")
    cockpit.runtime_status(runtime["id"], "READY")
    event(timeline, "created project and three durable worker sessions")

    gate_one = lab.gate_ensure(project_id, "RESULT_ASSESSMENT", "E1", "v1", first["worker"]["id"], first["session_id"])
    review_one = lab.gate_work_ensure(gate_one["id"], "REVIEW", "E1 canonical review", "Review E1", "RESULT_INSPECTOR", first["worker"]["id"], first["session_id"])
    assert lab.gate_work_ensure(gate_one["id"], "REVIEW", "duplicate", "must reuse", "RESULT_INSPECTOR", first["worker"]["id"], first["session_id"])["id"] == review_one["id"]
    event(timeline, "created one canonical E1 gate authority; duplicate creation reused it")

    claim = lab.claim_work(review_one["id"], first["worker"]["id"], first["session_id"])
    gate_two = lab.gate_ensure(project_id, "RESULT_ASSESSMENT", "E2", "v2", second["worker"]["id"], second["session_id"])
    assert lab.supersede_subject(project_id, "E1", "E2", "E2 is canonical corrected experiment", second["worker"]["id"], second["session_id"], gate_two["id"])["work_items_superseded"] == 1
    assert lab.sync(first["session_id"], project_id, current_work_item_id=claim["id"], expected_work_version=claim["work_version"])["lease_state"] == "LEASE_LOST"
    event(timeline, "superseded E1; its review is historical and worker A lost its stale lease")

    review_two = lab.gate_work_ensure(gate_two["id"], "REVIEW", "E2 canonical review", "Review E2", "RESULT_INSPECTOR", second["worker"]["id"], second["session_id"])
    claim_two = lab.claim_work(review_two["id"], second["worker"]["id"], second["session_id"])
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE lab_work_leases SET expires_at=NOW()-INTERVAL '1 second' WHERE id=%s", (claim_two["lease_id"],))
    assert lab.recover_stale_leases(project_id)["recovered"] == 1
    reclaimed = lab.claim_work(review_two["id"], third["worker"]["id"], third["session_id"])
    assert reclaimed["id"] == review_two["id"]
    event(timeline, "expired worker B lease; worker C reclaimed the same E2 canonical WorkItem")

    downstream = lab.create_work(project_id, "GENERALIZATION", "E2 conditional probe", "Await gate PASS", "SCIENTIST", third["worker"]["id"], dependencies=[{"target_type": "SCIENTIFIC_GATE", "target_id": gate_two["id"], "required_statuses": ["PASS"]}], created_session_id=third["session_id"], dormant_until_dependencies=True)
    assert downstream["status"] == "DORMANT"
    assert lab.preflight_run(gate_two["id"], third["worker"]["id"], third["session_id"], {"repo": True, "checkpoint": True, "tokenizer": True})["preflight"]["status"] == "PASS"
    lab.complete_work(reclaimed["id"], third["worker"]["id"], third["session_id"], summary="Semantic review passed")
    assert lab.gate_resolve(gate_two["id"], third["worker"]["id"], third["session_id"], "PASS", reclaimed["id"], rationale="semantic review passed")["gate"]["status"] == "PASS"
    assert lab.work_get(downstream["id"])["status"] == "READY"
    assert cockpit.wake_ready_work(project_id, [downstream["id"]]) == {"queued": 1}
    assert cockpit.wake_ready_work(project_id, [downstream["id"]]) == {"queued": 0}
    assert len(cockpit.state_get(project_id)["pending_wake_requests"]) == 1
    assert LabController(ResearchStore(os.environ["GPU_LAB_TEST_DATABASE_URL"])).work_get(downstream["id"])["status"] == "READY"
    event(timeline, "preflight and semantic PASS unlocked dormant downstream work with one deduplicated worker wake")

    print("V322_COORDINATION_SMOKE_OK")
    print(f"project={project_id} timeline_events={len(timeline)}")


if __name__ == "__main__":
    main()
