"""Read-only v3.4 correction planning against an existing project and target."""

from __future__ import annotations

import argparse
import json
import os

from gpu_lab.correction_v34 import DistributedCorrectionService
from gpu_lab.research import ResearchStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    parser.add_argument("target_id")
    args = parser.parse_args()
    database_url = os.environ.get("GPU_LAB_RESEARCH_DATABASE_URL")
    if not database_url:
        raise SystemExit("Set GPU_LAB_RESEARCH_DATABASE_URL before running this read-only comparison.")
    store = ResearchStore(database_url)
    before = store.state_get(args.project_id)["state_freshness"]["research_state_version"]
    preview = DistributedCorrectionService(store, migrate=False).shadow_preview(args.project_id, args.target_id)
    after = store.state_get(args.project_id)["state_freshness"]["research_state_version"]
    print(json.dumps({**preview, "state_version_unchanged": before == after, "before_research_state_version": before, "after_research_state_version": after}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
