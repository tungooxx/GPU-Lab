"""Run the durable Lab dependency/equivalence reconciler for one project."""

from __future__ import annotations

import argparse
import os

from gpu_lab.lab import LabController
from gpu_lab.research import ResearchStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    args = parser.parse_args()
    database_url = os.environ.get("GPU_LAB_RESEARCH_DATABASE_URL")
    if not database_url:
        raise SystemExit("Set GPU_LAB_RESEARCH_DATABASE_URL before running reconciliation.")
    print(LabController(ResearchStore(database_url)).resolve_dependencies(args.project_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
