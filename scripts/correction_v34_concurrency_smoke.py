"""Prove duplicate case creation and adjudication remain single-writer under races."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

from gpu_lab.correction_v34 import DistributedCorrectionService
from gpu_lab.lab import LabController
from gpu_lab.research import ResearchStore


def main() -> None:
    url = os.environ["GPU_LAB_TEST_DATABASE_URL"]
    store, lab = ResearchStore(url), LabController(ResearchStore(url))
    project_id = store.project_create(f"correction-v34-race-{time.time_ns()}", "Correction race")["project_id"]
    target = store.object_create(project_id, "Claim", {"statement": "fixture", "scope": "race"}, "SMOKE_CLAIM", "ACTIVE")
    evidence = store.object_create(project_id, "EvidenceUnit", {"statement": "fixture evidence"}, "SMOKE_EVIDENCE", "COMPLETED")

    def create() -> str:
        return str(DistributedCorrectionService(store).create_case(project_id, target["id"])["id"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        case_ids = list(pool.map(lambda _: create(), range(4)))
    assert len(set(case_ids)) == 1
    case_id = case_ids[0]
    workers = [lab.join(None, f"correction-race-{index}", "CODEX", project_id) for index in range(2)]
    service = DistributedCorrectionService(ResearchStore(url))
    challenges = []
    for worker, operator, issue in zip(workers, ("CAUSAL_LOGIC", "STRONGEST_NULL"), ("CAUSAL_OVERREACH", "MISSING_NULL"), strict=True):
        service.join_case(case_id, worker["worker"]["id"], worker["session_id"], operator)
        challenges.append(service.submit_challenge(case_id, worker["worker"]["id"], worker["session_id"], {"issue_type": issue, "issue_statement": issue}))
    for worker in workers:
        service.freeze_challenge(case_id, worker["worker"]["id"], worker["session_id"])
    service.verify(challenges[0]["id"], workers[1]["worker"]["id"], {"verification_status": "VERIFIED", "evidence_refs": [evidence["id"]]})

    def adjudicate() -> str:
        return str(DistributedCorrectionService(store).adjudicate(case_id, "NARROW_SCOPE", "race-safe grounded resolution")["id"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        record_ids = list(pool.map(lambda _: adjudicate(), range(4)))
    assert len(set(record_ids)) == 1
    print("CORRECTION_V34_CONCURRENCY_SMOKE_OK")


if __name__ == "__main__":
    main()
