"""Read-only registry audit for real projects; makes no mutations."""
from __future__ import annotations

import json
import os

from gpu_lab.research import ResearchStore
from gpu_lab.strategy_transfer_v35 import StrategyTransferService


def main() -> None:
    url = os.environ.get("GPU_LAB_RESEARCH_DATABASE_URL")
    if not url:
        raise SystemExit("Set GPU_LAB_RESEARCH_DATABASE_URL.")
    store = ResearchStore(url)
    summary = StrategyTransferService(store).registry_summary()
    outcomes = store.objects_global_list("StrategyTransferOutcome", limit=None)
    transfers = store.objects_global_list("StrategyTransferCandidate", limit=None)
    summary["audit"] = {
        "retrieved_uses": sum(item["transfer_counts"].get("retrieved", 0) for item in summary["patterns"]),
        "considered_or_applied_transfers": len(transfers),
        "positive_transfers": sum(item["status"] == "POSITIVE_TRANSFER" for item in outcomes),
        "negative_transfers": sum(item["status"] == "NEGATIVE_TRANSFER" for item in outcomes),
        "unresolved_transfers": [str(item["id"]) for item in transfers if item["status"] in {"PROPOSED", "ELIGIBLE", "APPLIED"}],
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
