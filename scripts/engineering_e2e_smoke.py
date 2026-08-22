"""Durable engineering-only smoke; requires an explicit PostgreSQL URL."""

from __future__ import annotations

import argparse
import json
import os
import time

from gpu_lab.engineering import EngineeringService
from gpu_lab.research import ResearchStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=os.getenv("GPU_LAB_TEST_DATABASE_URL"))
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("Set GPU_LAB_TEST_DATABASE_URL or pass --database-url explicitly")
    store = ResearchStore(args.database_url)
    project = store.project_create(f"engineering-smoke-{time.time_ns()}", "Engineering smoke")
    service = EngineeringService(store)
    task = service.task_create(
        project["project_id"], "Exercise the scientist-coder lifecycle", "BUG_REPRODUCTION",
        repository="gpu-lab", relevant_files=["src/gpu_lab/engineering.py"],
        implementation_guards=[{"name": "smoke", "type": "BOOLEAN"}],
    )
    service.task_start(
        task["id"], {"files_read": ["src/gpu_lab/engineering.py"]},
        {"commands_run": ["python -m compileall src/gpu_lab"], "passed": True},
    )
    service.diff_review(task["id"], {
        "files_changed": [], "diff_summary": "No production change in smoke.",
        "unrelated_changes": False, "scientific_variable_drift": False,
    })
    result = service.result_record(task["id"], {
        "implementation_verification": "VERIFIED_INTEGRATION",
        "implementation_guard_results": [{"name": "smoke", "passed": True}],
    })
    restarted = EngineeringService(ResearchStore(args.database_url))
    readiness = restarted.task_verify(task["id"])
    print(json.dumps({
        "verification": result["data"]["implementation_verification"],
        "scientific_result": result["data"]["scientific_result"],
        "project_id": project["project_id"], "task_id": task["id"],
        "result_id": result["id"], "ready_for_scientific_execution": readiness["ready_for_scientific_execution"],
        "scientific_execution": "NOT_RUN",
    }))
    if not readiness["ready_for_scientific_execution"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
