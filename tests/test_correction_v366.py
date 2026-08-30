"""v3.6.6 correction layers: attack ideas, test code, attack interpretation."""

import os
import time

import pytest

from gpu_lab.correction_v366 import ScientificSelfCorrectionService
from gpu_lab.errors import GPUError
from gpu_lab.lab import LabController
from gpu_lab.research import RESEARCH_OBJECT_KINDS, RESEARCH_OBJECT_STATUSES, ResearchStore

TEST_DATABASE_URL = os.getenv("GPU_LAB_TEST_DATABASE_URL")


def test_v366_kinds_and_statuses_are_registered():
    assert {
        "ScientificDesign", "DesignAssumption", "ScientificDesignAttack", "DesignChallenge",
        "DesignAttackSynthesis", "RequiredDiscriminator", "ImplementationFidelityCheck",
        "ResultInterpretationAttack", "ReviewTargetClassification",
    } <= set(RESEARCH_OBJECT_KINDS)
    assert {"UNDER_ATTACK", "ACCEPTED_FOR_TEST", "REPAIR_REQUIRED", "FIDELITY_CONFIRMED"} <= RESEARCH_OBJECT_STATUSES


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v366_fidelity_pass_never_completes_idea_attack_and_peer_isolation_holds():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, correction = LabController(store, feature_flags={"LEAN_ENGINEERING_REVIEW_V366": True}), ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-{time.time_ns()}", "idea correction") ["project_id"]
    one = lab.join(None, "attack-a", "CODEX", project_id)
    two = lab.join(None, "attack-b", "LOCAL_AGENT", project_id)
    with pytest.raises(GPUError) as exc:
        lab.create_work(project_id, "REVIEW", "Read code again", "Generic code review", "GENERIC_CODE_REVIEW",
                        created_by=one["worker"]["id"], created_session_id=one["session_id"])
    assert exc.value.error_type == "LAB_GENERIC_REVIEW_SUPPRESSED"
    design = correction.create_design(project_id, "V5", {"carrier": "GRU-256", "binding": "G"}, scientific_risk="HIGH")
    recurrence = correction.assumption_create(design["id"], "Persistent plan state materially benefits from recurrence.", "NECESSITY", centrality="CENTRAL")
    binding = correction.assumption_create(design["id"], "G causally binds the carrier to interventions.", "CAUSALITY", centrality="CENTRAL")

    fidelity = correction.fidelity_record(design["id"], implementation_ref="commit:fixture", science_to_code_map={"G": "binding.py"}, deterministic_checks=["test_binding"], verdict="FIDELITY_CONFIRMED", rationale="Exact mapping matches.")
    assert fidelity["status"] == "FIDELITY_CONFIRMED"
    assert correction.design_coverage(design["id"])["implementation_fidelity_is_not_design_correction"] is True
    assert correction.design_coverage(design["id"])["synthesis"] is None

    batch = correction.attack_batch_create(design["id"], [recurrence["id"], binding["id"]])
    first = correction.attack_join(batch["id"], one["worker"]["id"], one["session_id"], "TEMPORAL_CONFOUND")
    second = correction.attack_join(batch["id"], two["worker"]["id"], two["session_id"], "STRONGEST_NULL_ATTACKER")
    hidden = correction.attack_batch_get(batch["id"], one["session_id"])
    assert hidden["peer_attacks_visible"] is False
    assert hidden["members"][1].get("attack_id") is None

    correction.attack_submit(batch["id"], one["worker"]["id"], one["session_id"], verdict="SURVIVES_CURRENT_ATTACK", challenges=[])
    correction.attack_submit(batch["id"], two["worker"]["id"], two["session_id"], verdict="MATERIAL_ASSUMPTION_CHALLENGED", challenges=[{
        "target_assumption_id": binding["id"], "challenge_type": "ALTERNATIVE_MECHANISM",
        "statement": "Static state plus stronger binding could explain gains.", "severity": "HIGH",
        "competing_explanation": "H_STATIC_BINDING", "proposed_discriminating_test": "Hold binding capacity fixed while ablating recurrence.",
    }])
    correction.attack_freeze(batch["id"], one["worker"]["id"], one["session_id"])
    correction.attack_freeze(batch["id"], two["worker"]["id"], two["session_id"])
    assert correction.attack_batch_get(batch["id"])["peer_attacks_visible"] is True
    with pytest.raises(GPUError) as exc:
        correction.synthesize(batch["id"], "RETAIN_FOR_TEST", "Two attackers were favorable.")
    assert exc.value.error_type == "DESIGN_ATTACK_MATERIAL_CHALLENGE_UNRESOLVED"

    synthesis = correction.synthesize(batch["id"], "DISCRIMINATING_EXPERIMENT_REQUIRED", "The static-binding alternative is material.")
    assert synthesis["data"]["majority_vote_used"] is False
    assert synthesis["data"]["discriminating_experiment_ids"]
    assert store.object_get(design["id"])["status"] == "REPAIR_REQUIRED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v366_result_interpretation_is_separate_from_evidence_admission():
    store = ResearchStore(TEST_DATABASE_URL)
    correction = ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-result-{time.time_ns()}", "result correction") ["project_id"]
    result = store.object_create(project_id, "ExperimentRun", {"metric": 0.9}, "RUN_RECORDED", "RESULT_INSPECTED")
    attack = correction.interpretation_attack_record(project_id, result["id"], claim="Mechanism improves planning.", verdict="MECHANISM_NONSPECIFIC", strongest_alternative="Strongest null improves equally.", rationale="The result does not distinguish mechanisms.")
    assert attack["kind"] == "ResultInterpretationAttack"
    assert attack["data"]["does_not_admit_evidence"] is True


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v366_attack_without_code_preserves_independent_duplicate_origins():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, correction = LabController(store), ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-isolation-{time.time_ns()}", "attack before code") ["project_id"]
    first = lab.join(None, "idea-a", "CODEX", project_id)
    second = lab.join(None, "idea-b", "LOCAL_AGENT", project_id)
    design = correction.create_design(project_id, "No-code design", {"mechanism": "recurrent carrier"}, scientific_risk="HIGH")
    assumption = correction.assumption_create(design["id"], "The carrier is distinguishable from transition memory.", "IDENTIFIABILITY", centrality="CENTRAL")
    batch = correction.attack_batch_create(design["id"])
    correction.attack_join(batch["id"], first["worker"]["id"], first["session_id"], "CAUSAL_IDENTIFIABILITY_ATTACKER")
    correction.attack_join(batch["id"], second["worker"]["id"], second["session_id"], "DEGENERATE_SOLUTION_ATTACKER")
    duplicate = {
        "target_assumption_id": assumption["id"], "challenge_type": "TEMPORAL_CONFOUND",
        "statement": "Transition memory can explain the same behavior.", "severity": "MEDIUM",
        "competing_explanation": "H_TRANSITION_MEMORY",
        "proposed_discriminating_test": "Permute transition order while preserving marginal action frequencies.",
    }
    for worker in (first, second):
        correction.attack_submit(batch["id"], worker["worker"]["id"], worker["session_id"],
                                 verdict="DISCRIMINATING_EXPERIMENT_REQUIRED", challenges=[duplicate])
        correction.attack_freeze(batch["id"], worker["worker"]["id"], worker["session_id"])
    synthesis = correction.synthesize(batch["id"], "DISCRIMINATING_EXPERIMENT_REQUIRED", "The independent attacks converge on one unresolved discriminator.")
    assert len(synthesis["data"]["attack_ids"]) == 2
    assert len(synthesis["data"]["unique_challenge_ids"]) == 1
    assert len(synthesis["data"]["duplicate_challenge_ids"]) == 1
    assert store.objects_list(project_id, "EngineeringResult", limit=None) == []
