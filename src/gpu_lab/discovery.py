"""Deterministic v3.1 discovery-search policy.

This module deliberately does not decide scientific truth or authorize an
experiment.  It turns canonical, inspectable research state into a diverse
candidate portfolio that the existing Brain can score and persist.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Any

BRAIN_POLICY_VERSION = "brain-v3.1-discovery-search-v1"
SCIENTIFIC_DIMENSIONS = (
    "causal_object",
    "representation",
    "information_path",
    "architecture_family",
    "state_variable",
    "decoder_or_generator_formulation",
    "optimization_objective",
    "inference_process",
    "temporal_or_iterative_structure",
    "correspondence_structure",
    "geometric_primitive",
    "generative_assumption",
)
TUNING_DIMENSIONS = {
    "learning_rate", "seed", "width", "depth", "residual_cap", "threshold",
    "loss_coefficient", "top_k", "temperature", "epochs", "batch_size",
}


class SearchRegime(StrEnum):
    EXPLOIT = "EXPLOIT"
    MECHANISM_SEARCH = "MECHANISM_SEARCH"
    DIVERGENT_SEARCH = "DIVERGENT_SEARCH"
    PARADIGM_RESET = "PARADIGM_RESET"


class ScientificDistance(StrEnum):
    NEAR = "NEAR"
    MID = "MID"
    FAR = "FAR"
    ORTHOGONAL = "ORTHOGONAL"


class FrontierGapState(StrEnum):
    COMPETITIVE = "COMPETITIVE"
    MATERIAL_GAP = "MATERIAL_GAP"
    SEVERE_GAP = "SEVERE_GAP"
    UNKNOWN = "UNKNOWN"


def scientific_dimensions(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return only explicit scientific dimensions; hyperparameters never count."""
    payload = candidate.get("payload", {}) if isinstance(candidate, dict) else {}
    source = payload.get("scientific_dimensions", candidate.get("scientific_dimensions", {}))
    if not isinstance(source, dict):
        return {}
    return {key: value for key, value in source.items() if key in SCIENTIFIC_DIMENSIONS and value not in (None, "", [], {})}


def classify_scientific_distance(candidate: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    dimensions = scientific_dimensions(candidate)
    baseline_dimensions = scientific_dimensions(baseline or {})
    changed = [key for key, value in dimensions.items() if baseline_dimensions.get(key) != value]
    tuning = sorted(set((candidate.get("payload", {}) or {}).get("tuning_dimensions", [])) & TUNING_DIMENSIONS)
    if not changed:
        distance = ScientificDistance.NEAR
        reason = "No changed scientific dimension; parameter or implementation variation is NEAR."
    elif any(key in changed for key in ("causal_object", "generative_assumption")):
        distance = ScientificDistance.ORTHOGONAL
        reason = "Changes the causal object or generative assumption."
    elif any(key in changed for key in ("representation", "architecture_family", "information_path", "decoder_or_generator_formulation")):
        distance = ScientificDistance.FAR
        reason = "Changes a primary representation, architecture, information-flow, or formulation dimension."
    elif len(changed) >= 2:
        distance = ScientificDistance.FAR
        reason = "Changes multiple scientific dimensions together."
    else:
        distance = ScientificDistance.MID
        reason = "Changes one bounded scientific mechanism/subsystem while retaining the main framing."
    return {"scientific_distance": distance.value, "changed_scientific_dimensions": changed, "tuning_dimensions": tuning, "reason": reason}


def frontier_gap(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess only matched, comparable metrics; never fabricate a frontier gap."""
    assessed: list[dict[str, Any]] = []
    states: list[FrontierGapState] = []
    for metric in metrics:
        comparable = metric.get("information_matching_status") in {"MATCHED", True} and metric.get("comparability_status") in {"COMPARABLE", True}
        current, reference = metric.get("current_value"), metric.get("matched_reference_value")
        if not comparable or not isinstance(current, (int, float)) or not isinstance(reference, (int, float)) or reference == 0:
            assessed.append({**metric, "strategic_state": FrontierGapState.UNKNOWN.value})
            continue
        direction = metric.get("direction", "LOWER_IS_BETTER")
        ratio = current / reference if direction == "LOWER_IS_BETTER" else reference / current if current else float("inf")
        thresholds = metric.get("thresholds", {})
        material = float(thresholds.get("material_ratio", 1.25))
        severe = float(thresholds.get("severe_ratio", 2.0))
        state = FrontierGapState.SEVERE_GAP if ratio >= severe else FrontierGapState.MATERIAL_GAP if ratio >= material else FrontierGapState.COMPETITIVE
        states.append(state)
        assessed.append({**metric, "ratio_or_difference": ratio, "strategic_state": state.value})
    overall = max(states, key=lambda state: [FrontierGapState.COMPETITIVE, FrontierGapState.MATERIAL_GAP, FrontierGapState.SEVERE_GAP].index(state)) if states else FrontierGapState.UNKNOWN
    return {"state": overall.value, "metrics": assessed, "matched_metric_count": len(states), "unknown_metric_count": len(assessed) - len(states)}


def stagnation_state(decisions: list[dict[str, Any]], negative_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Count scientific choices only: smoke/recovery/administrative actions are excluded."""
    scientific = []
    for decision in decisions:
        data = decision.get("data", decision)
        if data.get("scientific_role") not in {None, "SCIENTIFIC_ACTION"}:
            continue
        if data.get("decision_role") in {"SYSTEM_SMOKE", "CONTRACT_TEST", "NOT_SCIENTIFIC", "ADMINISTRATIVE"}:
            continue
        scientific.append(data)
    families = [str(item.get("selected_action", {}).get("payload", {}).get("scientific_dimensions", {}).get("architecture_family", item.get("selected_action", {}).get("action_type", "UNKNOWN"))) for item in scientific]
    family_count = Counter(families).most_common(1)[0][1] if families else 0
    negative_scientific = [item for item in negative_results if item.get("data", item).get("scientific_role", "SCIENTIFIC_ACTION") == "SCIENTIFIC_ACTION"]
    saturated = family_count >= 3 or len(negative_scientific) >= 4
    return {
        "meaningful": saturated,
        "repeated_action_family_count": family_count,
        "number_of_recent_scientific_actions": len(scientific),
        "failed_descendant_count": len(negative_scientific),
        "local_search_saturation": saturated,
        "required_search_radius": ScientificDistance.FAR.value if saturated else ScientificDistance.NEAR.value,
    }


def choose_regime(*, prerequisite: bool, mechanism_unknown: bool, frontier: dict[str, Any], stagnation: dict[str, Any]) -> dict[str, Any]:
    evidence: list[str] = []
    if prerequisite:
        return {"search_regime": SearchRegime.MECHANISM_SEARCH.value, "reason": ["SINGLE_PATH_PREREQUISITE: mandatory inspection/reproduction cannot be replaced by speculative search."]}
    if mechanism_unknown and not stagnation["meaningful"]:
        return {"search_regime": SearchRegime.MECHANISM_SEARCH.value, "reason": ["Unresolved causal uncertainty needs a discriminating mechanism test."]}
    if stagnation["meaningful"]:
        evidence.append("scientific local-lineage stagnation is present")
        if frontier["state"] == FrontierGapState.SEVERE_GAP.value:
            evidence.append("matched frontier evidence shows a severe gap")
            return {"search_regime": SearchRegime.PARADIGM_RESET.value, "reason": evidence}
        return {"search_regime": SearchRegime.DIVERGENT_SEARCH.value, "reason": evidence}
    return {"search_regime": SearchRegime.EXPLOIT.value, "reason": ["No prerequisite, meaningful stagnation, or severe comparable frontier gap requires a broader regime."]}


def portfolio_critique(candidates: list[dict[str, Any]], prerequisite: bool) -> dict[str, Any]:
    distances = [item.get("scientific_distance", ScientificDistance.NEAR.value) for item in candidates if item.get("available", True)]
    coverage = {distance: distances.count(distance) for distance in ScientificDistance}
    signatures = [tuple(sorted(scientific_dimensions(item).items())) for item in candidates if item.get("available", True)]
    unique = len(set(signatures))
    adequate = prerequisite or (len(candidates) > 1 and unique > 1 and bool(coverage[ScientificDistance.FAR] or coverage[ScientificDistance.ORTHOGONAL]))
    return {"adequate": adequate, "reason": "SINGLE_PATH_PREREQUISITE" if prerequisite else ("Multi-distance scientific coverage is present." if adequate else "Insufficient scientific diversity; parameter variants do not form a discovery portfolio."), "distance_coverage": coverage, "unique_scientific_signatures": unique}


def fallback_candidates(question: str, hypothesis_ids: list[str], baseline: dict[str, Any] | None, regime: str) -> list[dict[str, Any]]:
    """Safe, non-executing alternatives used when an LLM/operator is unavailable."""
    base = scientific_dimensions(baseline or {})
    variants = [
        ("FROZEN_DIAGNOSTIC", {**base, "state_variable": "mechanism-specific diagnostic"}, "MID"),
        ("LITERATURE_SEARCH", {**base, "representation": "alternative representation family"}, "FAR"),
        ("LITERATURE_SEARCH", {**base, "causal_object": "reformulated objective"}, "ORTHOGONAL"),
    ]
    if regime == SearchRegime.PARADIGM_RESET.value:
        variants.insert(1, ("LITERATURE_SEARCH", {**base, "architecture_family": "different architecture family"}, "FAR"))
    result = []
    for action_type, dimensions, expected_distance in variants:
        result.append({"action_type": action_type, "question_addressed": question, "hypotheses_discriminated": hypothesis_ids, "predicted_outcomes": ["Generate a preregistration-ready, scientifically distinct option; do not execute it automatically."], "required_resources": ["existing evidence", "human review"], "score": {"scientific_importance": 3.5, "expected_discrimination": 3.0, "expected_information_gain": 3.0, "feasibility": 4.0, "compute_cost": 0.2, "engineering_cost": 0.5, "execution_risk": 0.2, "decision_relevance": 3.0}, "payload": {"scientific_dimensions": dimensions, "generation_source": "DETERMINISTIC_DISCOVERY_FALLBACK", "requires_preregistration": True, "non_executing_discovery_candidate": True, "expected_distance": expected_distance}, "available": True})
    return result


def breakthrough_signal(
    *, hypothesis_status: str, improved_dimensions: list[str], regressed_dimensions: list[str], evidence_family_ids: list[str] | None = None
) -> dict[str, Any]:
    """Strategic discovery metadata, intentionally independent of hypothesis truth."""
    discovery_value = "HIGH" if improved_dimensions and regressed_dimensions else "MEDIUM" if improved_dimensions else "NONE"
    signal_type = "PARTIAL_METRIC_BREAKTHROUGH" if improved_dimensions and regressed_dimensions else "FRONTIER_CLOSURE" if improved_dimensions else "OTHER"
    branches = []
    if discovery_value == "HIGH":
        branches = [
            "preserve_breakthrough_repair_tradeoff",
            "extract_mechanism_redesign_architecture",
            "search_different_architecture_or_representation",
        ]
        if len(regressed_dimensions) > 1:
            branches.append("reformulate_joint_objective")
    return {
        "hypothesis_status": hypothesis_status,
        "discovery_value": discovery_value,
        "type": signal_type,
        "improved_dimensions": improved_dimensions,
        "regressed_dimensions": regressed_dimensions,
        "branch_recommendations": branches,
        "evidence_family_ids": evidence_family_ids or [],
        "scientific_truth_independent": True,
    }


def local_search_collapse_diagnosis(
    portfolios: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Advisory /improve signal; it never changes policy or scientific state."""
    open_ended = [
        item.get("data", item)
        for item in portfolios
        if item.get("data", item).get("portfolio_type") == "OPEN_ENDED_DISCOVERY"
    ]
    collapsed = [
        item for item in open_ended if len(item.get("valid_candidate_indexes", [])) <= 1
    ]
    local_only = [
        item
        for item in open_ended
        if not (item.get("distance_coverage", {}).get("FAR") or item.get("distance_coverage", {}).get("ORTHOGONAL"))
    ]
    legacy = [item.get("data", item) for item in decisions if item.get("data", item).get("brain_policy_version") != BRAIN_POLICY_VERSION]
    detected = bool(collapsed or local_only)
    evidence = {
        "open_ended_portfolio_count": len(open_ended),
        "single_candidate_open_decisions": len(collapsed),
        "missing_far_or_orthogonal_coverage": len(local_only),
        "legacy_policy_decisions_seen": len(legacy),
    }
    return {
        "diagnosis": "LOCAL_SEARCH_COLLAPSE" if detected else "NO_LOCAL_SEARCH_COLLAPSE_DETECTED",
        "detected": detected,
        "evidence": evidence,
        "recommendation": (
            "Require a v3.1 CandidatePortfolio and inspect distance/frontier/stagnation provenance before selecting another local descendant."
            if detected
            else "No advisory policy change recommended from the observed portfolio history."
        ),
        "advisory_only": True,
    }
