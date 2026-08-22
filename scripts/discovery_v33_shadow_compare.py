"""Read-only comparison of current Brain candidates with DDE v3.3 characterization.

This intentionally creates no DiscoveryRound, Lab work item, decision, or GPU job.
Use it before opting a live project into a durable distributed round.
"""

from __future__ import annotations

import argparse
import json
import os

from gpu_lab.brain import ResearchBrain
from gpu_lab.discovery_v33 import DistributedDiscoveryService
from gpu_lab.research import ResearchStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    parser.add_argument(
        "--regime",
        default="DIVERGENT_SEARCH",
        choices=("EXPLOIT", "MECHANISM_SEARCH", "DIVERGENT_SEARCH", "PARADIGM_RESET"),
    )
    args = parser.parse_args()
    database_url = os.environ.get("GPU_LAB_RESEARCH_DATABASE_URL")
    if not database_url:
        raise SystemExit("Set GPU_LAB_RESEARCH_DATABASE_URL before running this read-only comparison.")
    store = ResearchStore(database_url)
    before = store.state_get(args.project_id)["state_freshness"]["research_state_version"]
    preview = ResearchBrain(store).brain_step(args.project_id, persist=False)
    dde = DistributedDiscoveryService(store, migrate=False)
    comparison = dde.shadow_preview(args.project_id, preview["candidate_actions"], args.regime)
    after = store.state_get(args.project_id)["state_freshness"]["research_state_version"]
    print(json.dumps({
        "read_only": True,
        "state_version_unchanged": before == after,
        "before_research_state_version": before,
        "after_research_state_version": after,
        "existing_brain_candidate_count": len(preview["candidate_actions"]),
        "existing_brain_search_regime": preview["search_regime"],
        "dde_characterization": comparison,
        "executed_experiments": 0,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
