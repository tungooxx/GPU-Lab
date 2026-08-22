"""Measure the bounded Lab state projection for one existing project."""

from __future__ import annotations

import argparse
import json
import os
import time

from gpu_lab.lab import LabController
from gpu_lab.research import ResearchStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    args = parser.parse_args()
    database_url = os.environ.get("GPU_LAB_RESEARCH_DATABASE_URL")
    if not database_url:
        raise SystemExit("Set GPU_LAB_RESEARCH_DATABASE_URL before running this benchmark.")
    store = ResearchStore(database_url)
    lab = LabController(store)
    started = time.perf_counter()
    lab.recover_stale_leases(args.project_id)
    leases_ms = round((time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    lab.resolve_dependencies(args.project_id)
    dependencies_ms = round((time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    state = lab.state_get(args.project_id)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    print(json.dumps({
        "project_id": args.project_id,
        "elapsed_ms": elapsed_ms,
        "recover_stale_leases_ms": leases_ms,
        "resolve_dependencies_ms": dependencies_ms,
        "response_bytes": len(json.dumps(state, default=str).encode()),
        "research_state_version": state["research_state_version"],
        "recent_events": len(state["recent_events"]),
        "gpu_activity": len(state["gpu_activity"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
