from gpu_lab.discovery import (
    FrontierGapState,
    SearchRegime,
    breakthrough_signal,
    choose_regime,
    classify_scientific_distance,
    frontier_gap,
    local_search_collapse_diagnosis,
    portfolio_critique,
    stagnation_state,
)


def candidate(dimensions=None, **payload):
    return {"payload": {"scientific_dimensions": dimensions or {}, **payload}, "available": True}


def test_hyperparameter_variants_are_near_not_far():
    result = classify_scientific_distance(
        candidate({"architecture_family": "decoder-a"}, tuning_dimensions=["width", "epochs"]),
        candidate({"architecture_family": "decoder-a"}),
    )
    assert result["scientific_distance"] == "NEAR"
    assert result["changed_scientific_dimensions"] == []


def test_representation_change_is_far_and_object_change_is_orthogonal():
    baseline = candidate({"representation": "voxels", "causal_object": "reconstruction"})
    assert classify_scientific_distance(candidate({"representation": "point_tokens"}), baseline)["scientific_distance"] == "FAR"
    assert classify_scientific_distance(candidate({"causal_object": "joint_stability_objective"}), baseline)["scientific_distance"] == "ORTHOGONAL"


def test_portfolio_critic_rejects_fake_diversity_but_allows_prerequisite():
    variants = [candidate({"architecture_family": "decoder-a"}, tuning_dimensions=["width"]) for _ in range(4)]
    for item in variants:
        item.update(classify_scientific_distance(item, variants[0]))
    assert not portfolio_critique(variants, prerequisite=False)["adequate"]
    assert portfolio_critique(variants[:1], prerequisite=True)["adequate"]


def test_frontier_requires_matched_comparable_evaluator():
    unknown = frontier_gap([{"metric_name": "CD", "current_value": 2, "matched_reference_value": 1, "information_matching_status": "UNMATCHED", "comparability_status": "COMPARABLE"}])
    assert unknown["state"] == FrontierGapState.UNKNOWN
    severe = frontier_gap([{"metric_name": "CD", "direction": "LOWER_IS_BETTER", "current_value": 3, "matched_reference_value": 1, "information_matching_status": "MATCHED", "comparability_status": "COMPARABLE", "thresholds": {"material_ratio": 1.2, "severe_ratio": 2}}])
    assert severe["state"] == FrontierGapState.SEVERE_GAP


def test_stagnation_excludes_administrative_actions_and_changes_regime():
    system = {"data": {"decision_role": "SYSTEM_SMOKE", "scientific_role": "SYSTEM_SMOKE", "selected_action": {"action_type": "ARTIFACT_ANALYSIS"}}}
    assert not stagnation_state([system] * 10, [])["meaningful"]
    scientific = {"data": {"scientific_role": "SCIENTIFIC_ACTION", "selected_action": {"action_type": "TRAINING_RUN", "payload": {"scientific_dimensions": {"architecture_family": "decoder-a"}}}}}
    state = stagnation_state([scientific] * 3, [])
    assert state["meaningful"]
    regime = choose_regime(prerequisite=False, mechanism_unknown=False, frontier={"state": "SEVERE_GAP"}, stagnation=state)
    assert regime["search_regime"] == SearchRegime.PARADIGM_RESET


def test_partial_breakthrough_never_changes_refuted_hypothesis_truth():
    signal = breakthrough_signal(
        hypothesis_status="REFUTED",
        improved_dimensions=["reconstruction"],
        regressed_dimensions=["stability"],
        evidence_family_ids=["family-1"],
    )
    assert signal["hypothesis_status"] == "REFUTED"
    assert signal["discovery_value"] == "HIGH"
    assert len(signal["branch_recommendations"]) >= 3


def test_improve_diagnosis_flags_only_open_ended_local_portfolio_collapse():
    diagnosis = local_search_collapse_diagnosis(
        [{"data": {"portfolio_type": "OPEN_ENDED_DISCOVERY", "valid_candidate_indexes": [0], "distance_coverage": {"NEAR": 1}}}],
        [],
    )
    assert diagnosis["diagnosis"] == "LOCAL_SEARCH_COLLAPSE"
    assert diagnosis["advisory_only"]
