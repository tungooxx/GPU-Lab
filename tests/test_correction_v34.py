"""v3.4 correction-engine contract and PostgreSQL integration coverage."""

import os
import time

import pytest

from gpu_lab.correction_v34 import DistributedCorrectionService
from gpu_lab.errors import GPUError
from gpu_lab.lab import LabController
from gpu_lab.research import RESEARCH_OBJECT_KINDS, RESEARCH_OBJECT_STATUSES, ResearchStore

TEST_DATABASE_URL = os.getenv("GPU_LAB_TEST_DATABASE_URL")


def test_correction_kinds_and_terminal_statuses_are_registered():
    assert {"CorrectionCase", "CorrectionChallenge", "CorrectionVerification", "ScientificDisagreement", "CorrectionRecord", "CorrectionHindsight"} <= set(RESEARCH_OBJECT_KINDS)
    assert {"CRITIQUE_GENERATION", "VERIFICATION", "NEEDS_EXPERIMENT", "RESOLVED_KEEP", "RESOLVED_REVISE", "RESOLVED_REJECT", "RESOLVED_NARROW_SCOPE"} <= RESEARCH_OBJECT_STATUSES


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_correction_isolates_then_verifies_and_records_single_non_mutating_adjudication():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, correction = LabController(store), DistributedCorrectionService(store)
    project_id = store.project_create(f"v34-correction-{time.time_ns()}", "Correction v3.4") ["project_id"]
    target = store.object_create(project_id, "Claim", {"statement": "A causes B", "scope": "fixture"}, "CLAIM_CREATED", "ACTIVE")
    evidence = store.object_create(project_id, "EvidenceUnit", {"statement": "control disagrees", "source": "fixture"}, "EVIDENCE_RECORDED", "COMPLETED")
    first = lab.join(None, "correction-a", "CODEX", project_id)
    second = lab.join(None, "correction-b", "LOCAL_AGENT", project_id)
    case = correction.create_case(project_id, target["id"])
    duplicate = correction.create_case(project_id, target["id"])
    assert duplicate["id"] == case["id"]
    correction.join_case(case["id"], first["worker"]["id"], first["session_id"], "CAUSAL_LOGIC")
    correction.join_case(case["id"], second["worker"]["id"], second["session_id"], "STRONGEST_NULL")
    challenge_a = correction.submit_challenge(case["id"], first["worker"]["id"], first["session_id"], {
        "issue_type": "CAUSAL_OVERREACH", "issue_statement": "Observed association lacks intervention.",
        "proposed_null": "A shared driver explains both.",
    })
    challenge_b = correction.submit_challenge(case["id"], second["worker"]["id"], second["session_id"], {
        "issue_type": "MISSING_NULL", "issue_statement": "A null model was not ruled out.",
    })
    with pytest.raises(GPUError) as exc:
        correction.challenge_get(case["id"], challenge_a["id"], second["session_id"])
    assert exc.value.error_type == "CORRECTION_PEER_ISOLATION_ACTIVE"
    correction.freeze_challenge(case["id"], first["worker"]["id"], first["session_id"])
    correction.freeze_challenge(case["id"], second["worker"]["id"], second["session_id"])
    assert correction.case_get(case["id"])["status"] == "VERIFICATION"
    verification = correction.verify(challenge_a["id"], second["worker"]["id"], {
        "verification_status": "VERIFIED", "evidence_refs": [evidence["id"]],
        "verification_reason": "The recorded control is incompatible with the claim.",
    })
    record = correction.adjudicate(case["id"], "NARROW_SCOPE", "Restrict the causal claim to the observed setting.")
    assert record["data"]["target_mutated"] is False
    assert record["data"]["grounded_verification_ids"] == [verification["id"]]
    assert store.object_get(target["id"])["status"] == "ACTIVE"
    assert correction.adjudicate(case["id"], "NARROW_SCOPE", "retry")["id"] == record["id"]
    assert correction.hindsight_record(record["id"], {"outcome": "USEFUL", "prevented_invalid_compute": True})["kind"] == "CorrectionHindsight"
    assert challenge_b["status"] == "PROPOSED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_correction_requires_grounding_and_creates_disagreement_without_executing_experiment():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, correction = LabController(store), DistributedCorrectionService(store)
    project_id = store.project_create(f"v34-disagreement-{time.time_ns()}", "Disagreement") ["project_id"]
    target = store.object_create(project_id, "Hypothesis", {"statement": "mechanism", "scope": "fixture"}, "HYPOTHESIS_CREATED", "ACTIVE")
    worker = lab.join(None, "correction-c", "CODEX", project_id)
    case = correction.create_case(project_id, target["id"])
    correction.join_case(case["id"], worker["worker"]["id"], worker["session_id"], "EXPERIMENT_VALIDITY")
    challenge = correction.submit_challenge(case["id"], worker["worker"]["id"], worker["session_id"], {
        "issue_type": "MISSING_CONTROL", "issue_statement": "No discriminating control exists.",
    })
    correction.freeze_challenge(case["id"], worker["worker"]["id"], worker["session_id"])
    with pytest.raises(GPUError) as exc:
        correction.adjudicate(case["id"], "EXPERIMENT_REQUIRED", "Need a discriminating test.")
    assert exc.value.error_type == "CORRECTION_DISCRIMINATING_TEST_REQUIRED"
    record = correction.adjudicate(case["id"], "EXPERIMENT_REQUIRED", "Need a discriminating test.", discriminating_test="Intervene on the candidate mediator while holding input fixed.")
    disagreement = store.object_get(record["data"]["scientific_disagreement_id"])
    assert disagreement["kind"] == "ScientificDisagreement"
    assert disagreement["data"]["not_evidence"] is True
    assert store.objects_list(project_id, "ExperimentRun", limit=None) == []
    with pytest.raises(GPUError) as exc:
        correction.verify(challenge["id"], worker["worker"]["id"], {"verification_status": "VERIFIED"})
    assert exc.value.error_type == "CORRECTION_VERIFICATION_GROUNDING_REQUIRED"
