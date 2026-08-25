"""Small, explicit v3.5 contract bench for transfer invariants.

These cases complement (rather than alter) the blinded scientific-action Brain
bench: they test meta-scientific transfer behavior and contain no project truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyTransferBenchCase:
    case_id: str
    expected: str


V35_CASES = (
    StrategyTransferBenchCase("A_SCIENTIFIC_RESULT_LEAKAGE", "BLOCK_SOURCE_SCIENCE_AS_TARGET_EVIDENCE"),
    StrategyTransferBenchCase("B_STRATEGY_PRINCIPLE_TRANSFER", "ALLOW_METHOD_CANDIDATE"),
    StrategyTransferBenchCase("C_RETRIEVAL_NOT_APPLICATION", "PERSIST_DISTINCT_STAGES"),
    StrategyTransferBenchCase("D_POSITIVE_TRANSFER", "RECORD_PROSPECTIVE_OUTCOME"),
    StrategyTransferBenchCase("E_NEGATIVE_TRANSFER", "REFINE_APPLICABILITY"),
    StrategyTransferBenchCase("F_INVALID_TRANSFER", "DO_NOT_COUNT_AS_NEGATIVE"),
    StrategyTransferBenchCase("G_SAME_DOMAIN_PSEUDOINDEPENDENCE", "BLOCK_GLOBAL_PROMOTION"),
    StrategyTransferBenchCase("H_CROSS_DOMAIN_SUPPORT", "ALLOW_PROMOTION_REVIEW"),
    StrategyTransferBenchCase("I_DISCOVERY_COLLAPSE", "PRESERVE_STATE_ONLY_GENERATION"),
    StrategyTransferBenchCase("J_CORE_INVARIANT", "KEEP_MANDATORY_AND_NON_TRANSFERABLE"),
    StrategyTransferBenchCase("K_DOMAIN_HEURISTIC", "DO_NOT_OVERGENERALIZE"),
    StrategyTransferBenchCase("L_LITERATURE_ANALOGY", "ALLOW_CANDIDATE_NOT_EVIDENCE"),
)


def contract_summary() -> dict[str, object]:
    return {"version": "ResearchBrainBench-v3.5-strategy-transfer", "case_count": len(V35_CASES), "cases": [{"id": case.case_id, "expected": case.expected} for case in V35_CASES]}
