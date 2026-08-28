"""Deterministic ResearchPortfolioScheduler benchmark cases for v3.6.

These are scheduler-policy fixtures, not scientific experiments. They prove
allocation invariants without asserting any counterfactual scientific outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .research_portfolio_v36_shadow import ResearchPortfolioV36Shadow


@dataclass(frozen=True)
class PortfolioBenchCase:
    case_id: str
    description: str
    workers: int
    ready_work: list[dict[str, Any]]
    branch_coverage: dict[str, dict[str, Any]]
    global_blocks: list[dict[str, Any]]
    expected_assignment_count: int
    expected_idle_count: int
    expected_planner_action: str


class ResearchPortfolioBenchV36:
    """Small, reproducible benchmark matrix for v3.6 shadow allocation."""

    VERSION = "research-portfolio-bench-v3.6"

    @classmethod
    def cases(cls) -> list[PortfolioBenchCase]:
        def work(work_id: str, branch_id: str, priority: int = 1) -> dict[str, Any]:
            return {"id": work_id, "branch_id": branch_id, "priority": priority}

        uncovered = {"hb1": {"active_worker_count": 0}, "hb2": {"active_worker_count": 0}, "hb3": {"active_worker_count": 0}}
        return [
            PortfolioBenchCase("A_WAITING_BRANCH_INDEPENDENT_READY", "A branch wait releases capacity to another existing branch.", 1, [work("hb2-ready", "hb2")], uncovered, [], 1, 0, "CLAIM_EXISTING_ONLY"),
            PortfolioBenchCase("B_WAITING_BRANCH_NO_USEFUL_WORK", "No filler work is invented when no existing work exists.", 1, [], uncovered, [], 0, 1, "DO_NOT_CREATE_WORK"),
            PortfolioBenchCase("C_TEN_WORKERS_THREE_BRANCHES", "Ten workers allocate three independent branches and leave seven idle.", 10, [work("w1", "hb1"), work("w2", "hb2"), work("w3", "hb3")], uncovered, [], 3, 7, "CLAIM_EXISTING_ONLY"),
            PortfolioBenchCase("D_GLOBAL_REPRODUCTION_BLOCK", "Global prerequisite blocks all allocation without hiding ready work.", 3, [work("w1", "hb1")], uncovered, [{"id": "global-reproduction"}], 0, 3, "OBJECTIVE_GLOBAL_BLOCK"),
            PortfolioBenchCase("E_BRANCH_LOCAL_BLOCK", "A locally blocked branch does not prevent another branch from being assigned.", 1, [work("hb2-ready", "hb2")], {**uncovered, "hb1": {"active_worker_count": 0, "next_actionability": "WAITING_DEPENDENCY"}}, [], 1, 0, "CLAIM_EXISTING_ONLY"),
            PortfolioBenchCase("F_NO_SAME_BRANCH_CONCENTRATION", "Two ready descendants on one branch consume one worker only.", 2, [work("w1", "hb1", 2), work("w2", "hb1", 1)], uncovered, [], 1, 1, "CLAIM_EXISTING_ONLY"),
            PortfolioBenchCase("G_ACTIVE_OWNER_NOT_STOLEN", "Ready work in a healthy active branch is not scheduled again.", 1, [work("w1", "hb1")], {**uncovered, "hb1": {"active_worker_count": 1}}, [], 0, 1, "CLAIM_EXISTING_ONLY"),
            PortfolioBenchCase("H_EXISTING_READY_BEFORE_PLANNING", "Existing canonical work is assigned; shadow never materializes new work.", 1, [work("w1", "hb3")], uncovered, [], 1, 0, "CLAIM_EXISTING_ONLY"),
        ]

    @classmethod
    def run_case(cls, case: PortfolioBenchCase) -> dict[str, Any]:
        workers = [{"id": f"worker-{index}", "availability_state": "AVAILABLE"} for index in range(case.workers)]
        projection = ResearchPortfolioV36Shadow.project(workers, case.ready_work, case.branch_coverage, case.global_blocks)
        assignment_count = sum(item["suggested_work_item_id"] is not None for item in projection["assignments"])
        idle_count = sum(item["suggested_work_item_id"] is None for item in projection["assignments"])
        passed = (
            assignment_count == case.expected_assignment_count
            and idle_count == case.expected_idle_count
            and projection["planner_action"] == case.expected_planner_action
            and projection["mutated"] is False
        )
        return {
            "case_id": case.case_id,
            "description": case.description,
            "passed": passed,
            "expected": {"assignments": case.expected_assignment_count, "idle": case.expected_idle_count, "planner_action": case.expected_planner_action},
            "actual": {"assignments": assignment_count, "idle": idle_count, "planner_action": projection["planner_action"]},
        }

    @classmethod
    def run_all(cls) -> dict[str, Any]:
        results = [cls.run_case(case) for case in cls.cases()]
        return {"benchmark_version": cls.VERSION, "results": results, "passed": all(result["passed"] for result in results)}
