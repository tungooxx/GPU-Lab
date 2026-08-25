"""Prospective PostgreSQL smoke for the v3.5 strategy-transfer lifecycle.

Fixture projects are deliberately meta-scientific and do not create EvidenceUnits
or claims.  The script is safe to run repeatedly because each project name has a
nanosecond suffix.
"""
from __future__ import annotations

import os
import time

from gpu_lab.research import ResearchStore
from gpu_lab.strategy_transfer_v35 import (
    ApplicabilityState, StrategyApplicabilityAssessment, StrategyApplicabilityModel,
    StrategyPatternCreate, StrategyTransferApply, StrategyTransferOutcomeRecord,
    StrategyTransferPropose, StrategyTransferService,
)


def proposal(strategy_id: str, context: dict) -> StrategyTransferPropose:
    return StrategyTransferPropose(strategy_id=strategy_id, target_context=context, target_problem_structure=["causal manipulation", "matched nuisance"], selection_mechanism="Prospective structural-match smoke", predicted_benefit="Reduce invalid causal experiments.", predicted_failure="The target cannot preserve nuisance structure.", planned_use="Use adapted matched-state intervention.")


def main() -> None:
    url = os.environ.get("GPU_LAB_RESEARCH_DATABASE_URL")
    if not url:
        raise SystemExit("Set GPU_LAB_RESEARCH_DATABASE_URL.")
    store = ResearchStore(url)
    service = StrategyTransferService(store)
    suffix = str(time.time_ns())
    source = store.project_create(f"v35-source-{suffix}", "Can a methodological strategy transfer prospectively?")
    target = store.project_create(f"v35-target-{suffix}", "Can a target adapt the method without source evidence leakage?")
    incompatible = store.project_create(f"v35-incompatible-{suffix}", "Does the screen reject incompatible structure?")
    negative = store.project_create(f"v35-negative-{suffix}", "Can a failed transfer narrow applicability?")
    pattern = service.pattern_create(source["project_id"], StrategyPatternCreate(name="matched-state causal control", strategy_type="CAUSAL_DESIGN", principle="Vary target identity while retaining matched nuisance state.", mechanism_of_value="Prevents correlated carrier confounding.", source_implementation_example="Frozen source-side state substitution.", applicability=StrategyApplicabilityModel(required_conditions={"identity_manipulable": True}, contraindications={"structure_unavoidably_changes": True}, structural_features=["causal intervention", "matched nuisance"], rationale="Identity must be separable from nuisance structure.")))
    found = service.search(target["project_id"], {"identity_manipulable": True})
    assert found["strategies"] and found["strategies"][0]["id"] == pattern["id"]
    transfer = service.propose(target["project_id"], proposal(pattern["id"], {"identity_manipulable": True}))
    service.applicability_assess(transfer["id"], StrategyApplicabilityAssessment(state=ApplicabilityState.STRONG_MATCH, matched_conditions=["identity_manipulable"], rationale="Target has separable identity and matched state."))
    service.apply(transfer["id"], StrategyTransferApply(applied_method="Matched anchor intervention", target_implementation_realization="Target-specific geometry-preserving substitution"))
    positive = service.outcome_record(transfer["id"], StrategyTransferOutcomeRecord(kind="POSITIVE_TRANSFER", rationale="Observed process improvement: one invalid branch avoided.", observed_process_effects={"invalid_experiments_reduced": 1}, independence_factors={"domain": "smoke-domain-a", "codebase": "target"}))
    rejected = service.propose(incompatible["project_id"], proposal(pattern["id"], {"structure_unavoidably_changes": True}))
    screen = service.applicability_assess(rejected["id"], StrategyApplicabilityAssessment(state=ApplicabilityState.CONTRAINDICATED, mismatch_conditions=["structure_unavoidably_changes"], rationale="Adaptation would alter the nuisance structure."))
    assert screen["status"] == "SCREENED_OUT"
    failed = service.propose(negative["project_id"], proposal(pattern["id"], {"identity_manipulable": True}))
    service.applicability_assess(failed["id"], StrategyApplicabilityAssessment(state=ApplicabilityState.PARTIAL_MATCH, rationale="Only the main prerequisite is verified."))
    service.apply(failed["id"], StrategyTransferApply(applied_method="Adapted intervention", target_implementation_realization="Known-risk adaptation"))
    negative_outcome = service.outcome_record(failed["id"], StrategyTransferOutcomeRecord(kind="NEGATIVE_TRANSFER", rationale="Target adaptation changed the nuisance structure and did not isolate the method.", independence_factors={"domain": "smoke-domain-b", "codebase": "negative"}, applicability_update=StrategyApplicabilityModel(required_conditions={"identity_manipulable": True, "nuisance_preserved": True}, contraindications={"structure_unavoidably_changes": True}, structural_features=["causal intervention"], rationale="Preserved nuisance structure is now an explicit prerequisite.")))
    # A new store models a fresh process and exercises migration/read durability.
    restarted = ResearchStore(url)
    assert restarted.object_get(positive["outcome"]["id"])["status"] == "POSITIVE_TRANSFER"
    assert restarted.object_get(negative_outcome["outcome"]["id"])["status"] == "NEGATIVE_TRANSFER"
    print({"verification": "VERIFIED_INTEGRATION", "pattern_id": pattern["id"], "positive_transfer": positive["outcome"]["id"], "screened_out_transfer": rejected["id"], "negative_transfer": negative_outcome["outcome"]["id"]})


if __name__ == "__main__":
    main()
