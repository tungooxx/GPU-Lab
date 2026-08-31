"""v3.6.6 correction layers: attack ideas, test code, attack interpretation."""

import os
import time

import pytest

from gpu_lab.correction_v366 import ScientificSelfCorrectionService
from gpu_lab.cockpit import CockpitController
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
    assert batch["recommended_attack_intensity"]["minimum_independent_attacks"] == 3
    assert "DEGENERATE_SOLUTION_ATTACKER" in batch["recommended_attack_intensity"]["operators"]
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
    assert synthesis["transition_work_item"]["kind"] == "SCIENTIFIC_DESIGN_REPAIR"
    assert synthesis["transition_work_item"]["related_refs"]["no_execution_authority"] is True
    assert store.object_get(design["id"])["status"] == "REPAIR_REQUIRED"
    hindsight = correction.challenge_hindsight_record(
        synthesis["data"]["unique_challenge_ids"][0], outcome="STILL_UNRESOLVED",
        notes="The required discriminator has not run; no later evidence settles the challenge.",
    )
    assert hindsight["kind"] == "DesignChallengeHindsight"
    regression = correction.regression_proposal_create(
        project_id, fidelity["id"], test_reference="tests/test_binding.py::test_mapping",
        invariant="The bound intervention map preserves the frozen carrier interface.",
        rationale="A repeatable fidelity finding should become a deterministic guard.",
    )
    assert regression["kind"] == "DeterministicRegressionProposal"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v366_result_interpretation_is_separate_from_evidence_admission():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, correction = LabController(store), ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-result-{time.time_ns()}", "result correction") ["project_id"]
    reviewer = lab.join(None, f"v366-result-reviewer-{time.time_ns()}", "CODEX", project_id)
    result = store.object_create(project_id, "ExperimentRun", {"metric": 0.9}, "RUN_RECORDED", "RESULT_INSPECTED")
    attack = correction.interpretation_attack_record(
        project_id, result["id"], claim="Mechanism improves planning.", verdict="MECHANISM_NONSPECIFIC",
        strongest_alternative="Strongest null improves equally.", rationale="The result does not distinguish mechanisms.",
        attack_operator="STRONGEST_NULL", reviewer_worker_id=reviewer["worker"]["id"],
        reviewer_session_id=reviewer["session_id"],
    )
    assert attack["kind"] == "ResultInterpretationAttack"
    assert attack["data"]["does_not_admit_evidence"] is True
    assert attack["data"]["target_layer"] == "SCIENTIFIC_INTERPRETATION"
    assert attack["data"]["reviewer_identity_status"] == "VERIFIED"
    unchallenged = store.object_create(
        project_id, "ExperimentRun", {"metric": 0.7}, "RUN_RECORDED", "RESULT_INSPECTED"
    )
    preview = correction.interpretation_attack_work_preview(project_id)
    assert preview["read_only"] is True
    assert preview["mutated"] is False
    assert len(preview["bounded_work_proposals"]) == 1
    proposal = preview["bounded_work_proposals"][0]
    assert proposal["kind"] == "RESULT_INTERPRETATION_ATTACK"
    assert proposal["related_refs"]["result_id"] == unchallenged["id"]
    assert proposal["related_refs"]["no_execution_authority"] is True
    repeat = correction.interpretation_attack_record(
        project_id, result["id"], claim="Mechanism improves planning.", verdict="MECHANISM_NONSPECIFIC",
        strongest_alternative="Same strongest null.", rationale="Same inference question.",
        attack_operator="STRONGEST_NULL", reviewer_worker_id=reviewer["worker"]["id"],
        reviewer_session_id=reviewer["session_id"],
    )
    assert str(repeat["id"]) == str(attack["id"])
    with pytest.raises(GPUError) as exc:
        correction.interpretation_attack_record(
            project_id, result["id"], claim="A second claim", verdict="MECHANISM_NONSPECIFIC",
            strongest_alternative="", rationale="An attack needs an explicit competing explanation.",
        )
    assert exc.value.error_type == "RESULT_INTERPRETATION_ATTACK_INVALID"


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


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v366_high_risk_acceptance_requires_distinct_workers_and_orthogonal_operators():
    """High-risk design acceptance cannot be inflated by session restarts."""
    store = ResearchStore(TEST_DATABASE_URL)
    lab, correction = LabController(store), ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-independence-{time.time_ns()}", "independence") ["project_id"]
    worker = lab.join(None, "independence-a", "CODEX", project_id)
    second_session = lab.join(worker["worker"]["id"], None, "CODEX", project_id)
    design = correction.create_design(
        project_id, "major new representation", {"representation": "novel recurrent carrier"}, scientific_risk="HIGH"
    )
    assumption = correction.assumption_create(design["id"], "The carrier is identifiable.", "IDENTIFIABILITY")
    batch = correction.attack_batch_create(design["id"], [assumption["id"]])
    correction.attack_join(batch["id"], worker["worker"]["id"], worker["session_id"], "NECESSITY_ATTACKER")
    with pytest.raises(GPUError) as exc:
        correction.attack_join(batch["id"], worker["worker"]["id"], second_session["session_id"], "CAUSAL_IDENTIFIABILITY_ATTACKER")
    assert exc.value.error_type == "DESIGN_ATTACK_WORKER_ALREADY_JOINED"
    correction.attack_submit(
        batch["id"], worker["worker"]["id"], worker["session_id"],
        verdict="SURVIVES_CURRENT_ATTACK", challenges=[], considered_assumption_ids=[assumption["id"]],
    )
    correction.attack_freeze(batch["id"], worker["worker"]["id"], worker["session_id"])
    with pytest.raises(GPUError) as exc:
        correction.synthesize(batch["id"], "RETAIN_FOR_TEST", "One critic cannot satisfy high-risk independence.")
    assert exc.value.error_type == "DESIGN_ATTACK_INDEPENDENT_COVERAGE_INCOMPLETE"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v366_coverage_uses_assumptions_actually_considered_not_batch_membership():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, correction = LabController(store), ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-coverage-{time.time_ns()}", "coverage") ["project_id"]
    worker = lab.join(None, "coverage-attacker", "CODEX", project_id)
    design = correction.create_design(project_id, "scoped design", {"mechanism": "test"}, scientific_risk="LOW")
    first = correction.assumption_create(design["id"], "First assumption.", "NECESSITY")
    second = correction.assumption_create(design["id"], "Second assumption.", "CAUSALITY")
    batch = correction.attack_batch_create(design["id"], [first["id"], second["id"]])
    correction.attack_join(batch["id"], worker["worker"]["id"], worker["session_id"], "NECESSITY_ATTACKER")
    correction.attack_submit(
        batch["id"], worker["worker"]["id"], worker["session_id"],
        verdict="SURVIVES_CURRENT_ATTACK", challenges=[], considered_assumption_ids=[first["id"]],
    )
    correction.attack_freeze(batch["id"], worker["worker"]["id"], worker["session_id"])
    coverage = correction.design_coverage(design["id"])
    assert coverage["covered_assumption_ids"] == [first["id"]]
    assert coverage["missing_assumption_ids"] == [second["id"]]
    with pytest.raises(GPUError) as exc:
        correction.synthesize(batch["id"], "RETAIN_FOR_TEST", "The second assumption remains unexamined.")
    assert exc.value.error_type == "DESIGN_ATTACK_COVERAGE_INCOMPLETE"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v366_attack_batch_freezes_current_lineage_without_counting_technical_failures():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, correction = LabController(store), ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-lineage-{time.time_ns()}", "lineage") ["project_id"]
    worker = lab.join(None, "lineage-attacker", "CODEX", project_id)
    design = correction.create_design(project_id, "current lineage design", {"mechanism": "test"}, scientific_risk="LOW")
    assumption = correction.assumption_create(design["id"], "The mechanism is falsifiable.", "NECESSITY")
    scientific = store.object_create(project_id, "NegativeResult", {"classification": "SCIENTIFIC_VALID"}, "NEGATIVE_RESULT_RECORDED", "COMPLETED")
    technical = store.object_create(project_id, "NegativeResult", {"scientific_result": "TECHNICAL_ONLY"}, "NEGATIVE_RESULT_RECORDED", "COMPLETED")

    batch = correction.attack_batch_create(design["id"], [assumption["id"]])
    assert batch["frozen_lineage_context"]["scientific_negative_result_ids"] == [scientific["id"]]
    assert batch["frozen_lineage_context"]["technical_only_negative_result_ids"] == [technical["id"]]
    attack = correction.attack_join(batch["id"], worker["worker"]["id"], worker["session_id"], "NECESSITY_ATTACKER")
    assert attack["data"]["frozen_lineage_context"] == batch["frozen_lineage_context"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v366_cockpit_projects_design_fidelity_and_interpretation_separately():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, correction = LabController(store), ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-cockpit-{time.time_ns()}", "cockpit") ["project_id"]
    design = correction.create_design(project_id, "visible design", {"mechanism": "test"})
    fidelity = correction.fidelity_record(
        design["id"], implementation_ref="commit:fixture", science_to_code_map={"m": "model.py"},
        deterministic_checks=["test_model"], verdict="FIDELITY_CONFIRMED", rationale="Exact mapping checked.",
    )
    result = store.object_create(project_id, "ExperimentRun", {"metric": 1.0}, "RUN_RECORDED", "RESULT_INSPECTED")
    interpretation = correction.interpretation_attack_record(
        project_id, result["id"], claim="The mechanism worked.", verdict="MECHANISM_NONSPECIFIC",
        strongest_alternative="A strongest null matches the result.", rationale="The result does not discriminate.",
    )
    state = CockpitController(store, lab).state_get(project_id)
    assert state["scientific_designs"][0]["id"] == design["id"]
    assert state["implementation_fidelity_checks"][0]["id"] == fidelity["id"]
    assert state["result_interpretation_attacks"][0]["id"] == interpretation["id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v366_legacy_design_attack_backfill_preserves_source_without_claiming_independence():
    store = ResearchStore(TEST_DATABASE_URL)
    correction = ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-legacy-{time.time_ns()}", "legacy") ["project_id"]
    design = correction.create_design(project_id, "V5 legacy", {"carrier": "GRU-256", "binding": "learned G"})
    assumption = correction.assumption_create(design["id"], "The recurrent carrier is necessary.", "NECESSITY")
    source = store.object_create(
        project_id, "EngineeringResult", {"summary": "Historical reviewer questioned recurrence."},
        "ENGINEERING_RESULT_RECORDED", "COMPLETED",
    )
    extracted = correction.legacy_design_attack_extract(
        project_id, source["id"], design["id"], "The source explicitly challenges recurrence necessity.",
        [assumption["id"]], [{
            "target_assumption_id": assumption["id"], "challenge_type": "NECESSITY_CHALLENGE",
            "statement": "A static carrier may be sufficient.", "severity": "MEDIUM",
        }],
    )
    assert extracted["status"] == "FROZEN"
    assert extracted["data"]["source_review_id"] == source["id"]
    assert extracted["data"]["independence"]["does_not_count_toward_live_independent_batch_quorum"] is True
    assert correction.design_coverage(design["id"])["covered_assumption_ids"] == [assumption["id"]]
    assert str(correction.legacy_design_attack_extract(
        project_id, source["id"], design["id"], "same source", [assumption["id"]]
    )["id"]) == str(extracted["id"])


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v5_regression_fidelity_is_not_full_scientific_design_self_correction():
    store = ResearchStore(TEST_DATABASE_URL)
    correction = ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-v5-{time.time_ns()}", "V5 correction regression") ["project_id"]
    design = correction.create_design(project_id, "V5 GRU-256 + learned G", {
        "recurrent_state": "GRU-256", "objective": "predictive-state", "binding": "learned G",
    }, scientific_risk="HIGH", engineering_risk="SCIENCE_CRITICAL")
    statements = [
        ("Persistent plan state requires or materially benefits from recurrence.", "NECESSITY"),
        ("A 256D recurrent state preserves relevant persistent information.", "CAPACITY"),
        ("Predictive training identifies behaviorally relevant latent state.", "IDENTIFIABILITY"),
        ("Learned G binds recurrent state to intervention space causally.", "CAUSALITY"),
        ("GRU effects are not merely transition/action-history statistics.", "TEMPORAL"),
        ("Temporal controls discriminate persistence from nuisance memory.", "CONTROL_VALIDITY"),
        ("Static binding/capacity cannot equally explain gains.", "ALTERNATIVE_MECHANISM"),
    ]
    # A7 is a competing-explanation question but is registered under the
    # closest first-class assumption type rather than hidden as prose.
    ids = [correction.assumption_create(design["id"], text, "CAPACITY" if kind == "ALTERNATIVE_MECHANISM" else kind, centrality="CENTRAL")["id"] for text, kind in statements]
    fidelity = correction.fidelity_record(design["id"], implementation_ref="commit:v5", science_to_code_map={"gru": "model.py", "G": "binding.py"}, deterministic_checks=["test_v5_wiring"], verdict="FIDELITY_CONFIRMED", rationale="Exact code mapping verified.")
    coverage = correction.design_coverage(design["id"])
    assert fidelity["status"] == "FIDELITY_CONFIRMED"
    assert coverage["assumption_count"] == 7
    assert set(coverage["missing_assumption_ids"]) == set(ids)
    assert coverage["synthesis"] is None
    preview = correction.v5_missing_attack_work_preview(project_id)
    assert preview["read_only"] is True
    assert preview["mutated"] is False
    assert len(preview["bounded_work_proposals"]) == 1
    proposal = preview["bounded_work_proposals"][0]
    assert proposal["kind"] == "V5_SCIENTIFIC_DESIGN_ATTACK"
    assert proposal["missing_assumption_ids"] == coverage["missing_assumption_ids"]
    assert len(proposal["related_refs"]["required_focus"]) == 8


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v366_read_only_audit_classifies_review_like_lab_workitems():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, correction = LabController(store), ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-audit-{time.time_ns()}", "audit") ["project_id"]
    worker = lab.join(None, "audit-worker", "CODEX", project_id)
    work = lab.create_work(
        project_id, "REVIEW", "Fidelity map", "Check the science-to-code fidelity surface.",
        "IMPLEMENTATION_FIDELITY_CHECK", created_by=worker["worker"]["id"],
        created_session_id=worker["session_id"],
    )
    audit = correction.production_audit(project_id)
    assert audit["read_only"] is True
    assert audit["review_like_work_item_count"] == 1
    assert f"work:{work['id']}" in audit["review_like_work_by_type"]["IMPLEMENTATION_FIDELITY"]
    assert audit["historical_classification_persisted"] is False
    assert audit["major_designs_lacking_assumption_maps"] == []
    assert audit["result_interpretation_correction_coverage"]["inspected_result_ids"] == []
    backfill = correction.review_target_classification_persist(project_id, work["id"])
    assert backfill["kind"] == "ReviewTargetClassification"
    assert backfill["data"]["target_type"] == "WORK_ITEM"
    assert backfill["data"]["inferred_classification"] == "IMPLEMENTATION"
    assert backfill["data"]["provenance"] == "BACKFILL_INFERRED_CLASSIFICATION"
    assert backfill["data"]["preserves_original_artifact"] is True
    assert str(correction.review_target_classification_persist(project_id, work["id"])["id"]) == backfill["id"]
    persisted_audit = correction.production_audit(project_id)
    assert persisted_audit["historical_classification_persisted"] is True
    assert persisted_audit["persisted_review_classification_count"] == 1


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v366_synthesis_atomically_materializes_only_its_bounded_transition_work():
    store = ResearchStore(TEST_DATABASE_URL)
    lab, correction = LabController(store), ScientificSelfCorrectionService(store)
    project_id = store.project_create(f"v366-transition-{time.time_ns()}", "transition") ["project_id"]
    worker = lab.join(None, f"v366-transition-worker-{time.time_ns()}", "CODEX", project_id)
    design = correction.create_design(
        project_id, "bounded transition", {"mechanism": "frozen testable mechanism"}, scientific_risk="LOW"
    )
    assumption = correction.assumption_create(
        design["id"], "The mechanism has a prospective falsifier.", "NECESSITY"
    )
    batch = correction.attack_batch_create(design["id"], [assumption["id"]])
    correction.attack_join(batch["id"], worker["worker"]["id"], worker["session_id"], "NECESSITY_ATTACKER")
    correction.attack_submit(
        batch["id"], worker["worker"]["id"], worker["session_id"],
        verdict="SURVIVES_CURRENT_ATTACK", challenges=[],
    )
    correction.attack_freeze(batch["id"], worker["worker"]["id"], worker["session_id"])

    synthesis = correction.synthesize(batch["id"], "RETAIN_FOR_TEST", "The required assumption was independently attacked.")
    transition = synthesis["transition_work_item"]
    assert transition["kind"] == "PRE_SCIENCE_DESIGN_FINALIZATION"
    assert transition["status"] == "READY"
    assert transition["related_refs"]["scientific_design_id"] == design["id"]
    assert transition["related_refs"]["design_attack_synthesis_id"] == synthesis["id"]
    assert transition["related_refs"]["no_execution_authority"] is True
    assert store.object_get(design["id"])["data"]["authority"] == "PASS_FOR_TEST"

    # The same frozen synthesis is idempotent and cannot multiply successor work.
    repeat = correction.synthesize(batch["id"], "RETAIN_FOR_TEST", "same request")
    assert str(repeat["id"]) == str(synthesis["id"])
    work = lab.work_get(transition["id"])
    assert work["equivalence_key"].startswith(f"v366-design-transition:{synthesis['id']}")
