"""Prove freeze and synthesis races remain durable and single-archive on PostgreSQL."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

from gpu_lab.discovery_v33 import DistributedDiscoveryService
from gpu_lab.lab import LabController
from gpu_lab.research import ResearchStore


def candidate(name: str, signature: dict[str, str]) -> dict:
    return {
        "title": name, "mechanism": name, "predictions": [f"{name} predicts X"],
        "falsifier": f"not {name}", "diversity_signature": signature,
    }


def main() -> None:
    url = os.environ["GPU_LAB_TEST_DATABASE_URL"]
    store, lab = ResearchStore(url), LabController(ResearchStore(url))
    project_id = store.project_create(f"dde-v33-concurrency-{time.time_ns()}", "DDE race smoke")["project_id"]
    workers = [lab.join(None, f"race-{index}", "CODEX", project_id) for index in range(2)]
    dde = DistributedDiscoveryService(ResearchStore(url))
    round_ = dde.create_round(project_id, None, "MECHANISM_SEARCH")
    batches = [
        dde.join_round(round_["id"], workers[0]["worker"]["id"], workers[0]["session_id"], "CAUSAL_INVERSION", "FAR"),
        dde.join_round(round_["id"], workers[1]["worker"]["id"], workers[1]["session_id"], "STRONG_NULL_CONSTRUCTION", "ORTHOGONAL"),
    ]
    dde.submit_candidate(round_["id"], batches[0]["id"], workers[0]["worker"]["id"], workers[0]["session_id"], candidate("causal", {"causal_object": "inverse"}))
    dde.submit_candidate(round_["id"], batches[1]["id"], workers[1]["worker"]["id"], workers[1]["session_id"], candidate("null", {"causal_object": "null"}))

    def freeze(index: int) -> str:
        service = DistributedDiscoveryService(ResearchStore(url))
        return str(service.batch_freeze(round_["id"], batches[index]["id"], workers[index]["worker"]["id"], workers[index]["session_id"])["id"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(freeze, range(2))) == {str(batch["id"]) for batch in batches}

    def synthesize() -> str:
        return str(DistributedDiscoveryService(ResearchStore(url)).synthesize(round_["id"])["id"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        archives = list(pool.map(lambda _: synthesize(), range(2)))
    assert len(set(archives)) == 1
    print("DISCOVERY_V33_CONCURRENCY_SMOKE_OK")


if __name__ == "__main__":
    main()
