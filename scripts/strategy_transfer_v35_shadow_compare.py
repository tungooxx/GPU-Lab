"""Read-only comparison of existing decisions with v3.5 strategy retrieval.

It intentionally never creates a transfer candidate or changes a ResearchDecision.
"""
from __future__ import annotations

import argparse
import json
import os

from gpu_lab.research import ResearchStore
from gpu_lab.strategy_transfer_v35 import StrategyTransferService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    url = os.environ.get("GPU_LAB_RESEARCH_DATABASE_URL")
    if not url:
        raise SystemExit("Set GPU_LAB_RESEARCH_DATABASE_URL.")
    store, transfer = ResearchStore(url), None
    transfer = StrategyTransferService(store)
    decisions = store.objects_list(args.project_id, "ResearchDecision", limit=args.limit)
    rows = []
    for decision in decisions:
        data = decision["data"]
        context = data.get("strategy_transfer_context")
        if not isinstance(context, dict):
            rows.append({"decision_id": str(decision["id"]), "shadow": "NOT_EVALUABLE", "reason": "No frozen structural transfer context"})
            continue
        # Avoid service.search because shadow mode must be read-only.
        candidates = []
        for pattern in store.objects_global_list("ResearchStrategyPattern", {"ACTIVE", "WEAKENED"}, limit=None):
            if str(pattern["project_id"]) == args.project_id:
                continue
            required = pattern["data"].get("applicability", {}).get("required_conditions", {})
            mismatch = [key for key, value in required.items() if key in context and context[key] != value]
            candidates.append({"strategy_id": str(pattern["id"]), "applicability": "PARTIAL_MATCH" if mismatch else "STRONG_MATCH", "mismatches": mismatch})
        rows.append({"decision_id": str(decision["id"]), "shadow": "EVALUATED", "retrieved": candidates[:5], "actually_applied": data.get("applied_strategy_ids", []), "potential_anchoring": bool(candidates and data.get("discovery_mode") != "STATE_ONLY_GENERATION")})
    print(json.dumps({"mode": "READ_ONLY_SHADOW", "project_id": args.project_id, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
