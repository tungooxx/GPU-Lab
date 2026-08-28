from gpu_lab.research_portfolio_bench_v36 import ResearchPortfolioBenchV36
from gpu_lab import server


def test_research_portfolio_bench_v36_named_scheduler_cases_pass():
    result = ResearchPortfolioBenchV36.run_all()
    assert result["benchmark_version"] == "research-portfolio-bench-v3.6"
    assert result["passed"] is True
    assert {item["case_id"] for item in result["results"]} == {
        "A_WAITING_BRANCH_INDEPENDENT_READY",
        "B_WAITING_BRANCH_NO_USEFUL_WORK",
        "C_TEN_WORKERS_THREE_BRANCHES",
        "D_GLOBAL_REPRODUCTION_BLOCK",
        "E_BRANCH_LOCAL_BLOCK",
        "F_NO_SAME_BRANCH_CONCENTRATION",
        "G_ACTIVE_OWNER_NOT_STOLEN",
        "H_EXISTING_READY_BEFORE_PLANNING",
    }


async def test_portfolio_bench_mcp_surface_is_read_only():
    result = await server.research_portfolio_bench_v36()
    assert result["passed"] is True
