"""Exercise v3.4 correction phases against a disposable PostgreSQL project.

This is intentionally non-scientific: it proves durable coordination and never
submits a GPU job or changes the target Claim.
"""

from __future__ import annotations

import os
import time

from gpu_lab.correction_v34 import DistributedCorrectionService
from gpu_lab.errors import GPUError
from gpu_lab.lab import LabController
from gpu_lab.research import ResearchStore


def main() -> None:
    store = ResearchStore(os.environ["GPU_LAB_TEST_DATABASE_URL"])
    lab, correction = LabController(store), DistributedCorrectionService(store)
    project_id = store.project_create(f"correction-v34-smoke-{time.time_ns()}", "Disposable correction smoke")["project_id"]
    target = store.object_create(project_id, "Claim", {"statement": "X causes Y", "scope": "fixture"}, "SMOKE_CLAIM", "ACTIVE")
    evidence = store.object_create(project_id, "EvidenceUnit", {"statement": "control rules out universal claim"}, "SMOKE_EVIDENCE", "COMPLETED")
    workers = [lab.join(None, f"correction-smoke-{name}", "CODEX", project_id) for name in "AB"]
    case = correction.create_case(project_id, target["id"])
    assert str(correction.create_case(project_id, target["id"])["id"]) == str(case["id"])
    operators = ("CAUSAL_LOGIC", "STRONGEST_NULL")
    challenges = []
    for worker, operator, issue in zip(workers, operators, ("CAUSAL_OVERREACH", "MISSING_NULL"), strict=True):
        correction.join_case(case["id"], worker["worker"]["id"], worker["session_id"], operator)
        challenges.append(correction.submit_challenge(case["id"], worker["worker"]["id"], worker["session_id"], {"issue_type": issue, "issue_statement": f"{issue} survives the current target."}))
    try:
        correction.challenge_get(case["id"], challenges[0]["id"], workers[1]["session_id"])
        raise AssertionError("peer challenge leaked before freeze")
    except GPUError as exc:
        assert exc.error_type == "CORRECTION_PEER_ISOLATION_ACTIVE"
    for worker in workers:
        correction.freeze_challenge(case["id"], worker["worker"]["id"], worker["session_id"])
    assert correction.case_get(case["id"])["status"] == "VERIFICATION"
    correction.verify(challenges[0]["id"], workers[1]["worker"]["id"], {"verification_status": "VERIFIED", "evidence_refs": [evidence["id"]], "verification_reason": "Fixture evidence independently supports the criticism."})
    record = correction.adjudicate(case["id"], "NARROW_SCOPE", "Evidence supports only a bounded claim.")
    assert record["data"]["target_mutated"] is False
    assert store.object_get(target["id"])["status"] == "ACTIVE"
    restarted = DistributedCorrectionService(ResearchStore(os.environ["GPU_LAB_TEST_DATABASE_URL"]))
    assert restarted.adjudicate(case["id"], "NARROW_SCOPE", "restart replay")["id"] == record["id"]
    assert store.objects_list(project_id, "ExperimentRun", limit=None) == []
    print("CORRECTION_V34_SMOKE_OK")


if __name__ == "__main__":
    main()
