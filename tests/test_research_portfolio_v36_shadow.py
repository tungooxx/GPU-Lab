from gpu_lab.research_portfolio_v36_shadow import ResearchPortfolioV36Shadow


def test_shadow_prefers_distinct_uncovered_branches_and_never_mutates():
    projected = ResearchPortfolioV36Shadow.project(
        workers=[{"id": "A", "availability_state": "AVAILABLE"}, {"id": "B", "availability_state": "AVAILABLE"}],
        ready_work=[
            {"id": "w1", "branch_id": "hb1", "priority": 10},
            {"id": "w2", "branch_id": "hb1", "priority": 9},
            {"id": "w3", "branch_id": "hb2", "priority": 1},
        ],
        branch_coverage={"hb1": {"active_worker_count": 0}, "hb2": {"active_worker_count": 0}},
        objective_global_blocks=[],
    )
    assert [item["suggested_work_item_id"] for item in projected["assignments"]] == ["w1", "w3"]
    assert projected["unassigned_ready_work_item_ids"] == ["w2"]
    assert projected["mutated"] is False


def test_shadow_global_block_prevents_assignment_without_hiding_ready_work():
    projected = ResearchPortfolioV36Shadow.project(
        workers=[{"id": "A", "availability_state": "AVAILABLE"}],
        ready_work=[{"id": "w1", "branch_id": "hb1", "priority": 1}],
        branch_coverage={"hb1": {"active_worker_count": 0}},
        objective_global_blocks=[{"id": "global-wait"}],
    )
    assert projected["assignments"][0]["reason"] == "OBJECTIVE_GLOBAL_DEPENDENCY_BLOCK"
    assert projected["unassigned_ready_work_item_ids"] == ["w1"]
    assert projected["mutated"] is False
