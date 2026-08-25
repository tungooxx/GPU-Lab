from __future__ import annotations

import copy
import uuid

import pytest

from gpu_lab.errors import GPUError
from gpu_lab.strategy_transfer_v35 import (
    ApplicabilityState,
    StrategyApplicabilityAssessment,
    StrategyApplicabilityModel,
    StrategyPatternCreate,
    StrategyScope,
    StrategyTransferApply,
    StrategyTransferOutcomeRecord,
    StrategyTransferHindsightRecord,
    StrategyTransferPropose,
    StrategyTransferService,
    TransferOutcomeKind,
)
from gpu_lab.strategy_transfer_bench_v35 import contract_summary


class Store:
    def __init__(self):
        self.items = {}
        self.projects = {"a", "b", "c"}

    def project_get(self, project_id):
        if project_id not in self.projects:
            raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
        return {"id": project_id}

    def object_create(self, project_id, kind, data, event_type, status="ACTIVE"):
        ident = str(uuid.uuid4())
        item = {"id": ident, "project_id": project_id, "kind": kind, "status": status, "data": copy.deepcopy(data)}
        self.items[ident] = item
        return copy.deepcopy(item)

    def object_get(self, ident):
        if ident not in self.items:
            raise GPUError("RESEARCH_OBJECT_NOT_FOUND", ident)
        return copy.deepcopy(self.items[ident])

    def object_update(self, ident, data_update, status, event_type):
        item = self.items[ident]
        item["data"].update(copy.deepcopy(data_update))
        item["status"] = status
        return copy.deepcopy(item)

    def objects_global_list(self, kind, statuses=None, limit=None):
        return [copy.deepcopy(x) for x in self.items.values() if x["kind"] == kind and (statuses is None or x["status"] in statuses)]

    def objects_list(self, project_id, kind, limit=None):
        return [copy.deepcopy(x) for x in self.items.values() if x["project_id"] == project_id and x["kind"] == kind]


def _draft():
    return StrategyPatternCreate(
        name="Matched nuisance intervention",
        strategy_type="CAUSAL_DESIGN",
        principle="Hold nuisance state fixed while varying the hypothesized identity.",
        mechanism_of_value="Separates an intervention from a correlated carrier.",
        source_implementation_example="Frozen-state substitution.",
        applicability=StrategyApplicabilityModel(
            required_conditions={"identity_manipulable": True},
            contraindications={"geometry_changes": True},
            structural_features=["causal intervention"],
            rationale="The target must preserve the nuisance state.",
        ),
    )


def test_transfer_is_prospective_and_does_not_expose_source_science():
    store, service = Store(), None
    service = StrategyTransferService(store)
    pattern = service.pattern_create("a", _draft())
    found = service.search("b", {"identity_manipulable": True})
    assert found["strategies"][0]["id"] == pattern["id"]
    assert "source_implementation_example" not in found["strategies"][0]
    proposal = service.propose("b", StrategyTransferPropose(
        strategy_id=pattern["id"], target_context={"identity_manipulable": True},
        target_problem_structure=["causal intervention"], selection_mechanism="Structured causal match.",
        predicted_benefit="Higher causal discrimination.", predicted_failure="Nuisance geometry changes.",
        planned_use="Use adapted anchor-state substitution.",
    ))
    screened = service.applicability_assess(proposal["id"], StrategyApplicabilityAssessment(
        state=ApplicabilityState.STRONG_MATCH, matched_conditions=["identity_manipulable"], rationale="Required state is available."
    ))
    assert screened["status"] == "ELIGIBLE"
    applied = service.apply(proposal["id"], StrategyTransferApply(applied_method="Adapted substitution", target_implementation_realization="Geometry-preserving anchor substitution"))
    assert applied["transfer_hypothesis"]["data"]["prediction"] == "Higher causal discrimination."
    outcome = service.outcome_record(proposal["id"], StrategyTransferOutcomeRecord(
        kind=TransferOutcomeKind.POSITIVE_TRANSFER, rationale="The method reduced an invalid-experiment branch.",
        observed_process_effects={"invalid_experiments_reduced": 1}, independence_factors={"domain": "point-cloud", "codebase": "b"},
    ))
    assert outcome["outcome"]["status"] == "POSITIVE_TRANSFER"
    assert store.object_get(pattern["id"])["data"]["transfer_counts"]["positive"] == 1
    hindsight = service.hindsight_record(outcome["outcome"]["id"], StrategyTransferHindsightRecord(observed_generalization="Not yet assessed.", rationale="Awaiting an independent target context."))
    assert hindsight["kind"] == "StrategyTransferHindsight"


def test_negative_transfer_requires_applicability_refinement():
    store, service = Store(), StrategyTransferService(Store())
    # Use the service's own backing store consistently.
    store = service.store
    pattern = service.pattern_create("a", _draft())
    proposal = service.propose("b", StrategyTransferPropose(strategy_id=pattern["id"], target_context={}, target_problem_structure=[], selection_mechanism="Test", predicted_benefit="Benefit", predicted_failure="Failure", planned_use="Use"))
    service.applicability_assess(proposal["id"], StrategyApplicabilityAssessment(state="PARTIAL_MATCH", rationale="Partial."))
    service.apply(proposal["id"], StrategyTransferApply(applied_method="Use", target_implementation_realization="Adaptation"))
    with pytest.raises(GPUError) as exc_info:
        service.outcome_record(proposal["id"], StrategyTransferOutcomeRecord(kind="NEGATIVE_TRANSFER", rationale="Failed."))
    assert exc_info.value.error_type == "NEGATIVE_TRANSFER_APPLICABILITY_UPDATE_REQUIRED"


def test_state_only_discovery_is_not_anchored():
    service = StrategyTransferService(Store())
    assert service.search("b", {}, discovery_mode="STATE_ONLY_GENERATION")["strategies"] == []


def test_promotion_fails_closed_without_independent_prospective_support():
    store = Store()
    service = StrategyTransferService(store)
    pattern = service.pattern_create("a", _draft())
    with pytest.raises(GPUError) as exc_info:
        service.promotion_decide(pattern["id"], StrategyScope.DOMAIN, "Too early.")
    assert exc_info.value.error_type == "STRATEGY_PROMOTION_EVIDENCE_INSUFFICIENT"


def test_v35_contract_bench_has_all_required_cases():
    summary = contract_summary()
    assert summary["case_count"] == 12
    assert {item["id"][0] for item in summary["cases"]} == set("ABCDEFGHIJKL")
