from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gpu_lab.brain_bench import (
    BenchmarkDecision,
    BenchmarkEpisode,
    BenchmarkPolicy,
    ProvenanceKind,
    ResearchBrainBench,
)
from gpu_lab.errors import GPUError

BENCH_ROOT = Path(__file__).parents[1] / "research_bench"


def test_all_historical_episodes_validate_and_have_sourced_provenance():
    episodes = ResearchBrainBench(BENCH_ROOT).load_all()

    assert len(episodes) >= 3
    assert len({episode.episode_id for episode in episodes}) == len(episodes)
    assert all(episode.cutoff_timestamp.utcoffset() is not None for episode in episodes)
    assert all(episode.source_provenance for episode in episodes)
    assert {
        provenance.kind
        for episode in episodes
        for provenance in episode.source_provenance
    } == {ProvenanceKind.HISTORICAL_FACT, ProvenanceKind.RECONSTRUCTED_INFERENCE}


def test_visible_payload_excludes_hidden_future_and_benchmark_answers():
    episode = ResearchBrainBench(BENCH_ROOT).load_episode("hasi_before_intervention.json")

    payload = episode.visible_payload()
    serialized = json.dumps(payload)

    assert "hidden_future_state" not in payload
    assert "forbidden_future_records" not in payload
    assert "strong_next_actions" not in payload
    assert "HASI was later killed" not in serialized
    assert "SCOPE_VIOLATION" not in serialized
    assert "PROMOTE_WITHOUT_EVIDENCE" not in serialized
    assert "expected_information_gain" not in serialized
    assert "compute_cost" not in serialized


def test_episode_rejects_naive_cutoff_timestamp():
    raw = json.loads((BENCH_ROOT / "after_stage4.json").read_text(encoding="utf-8"))
    raw["cutoff_timestamp"] = "2026-08-09T11:12:40"

    with pytest.raises(ValidationError, match="UTC offset"):
        BenchmarkEpisode.model_validate(raw)


def test_algorithmic_policy_baselines_are_deterministic_and_distinct():
    episode = ResearchBrainBench(BENCH_ROOT).load_episode("after_stage4.json")

    cheapest = ResearchBrainBench.baseline_decision(
        episode, BenchmarkPolicy.CHEAPEST_FEASIBLE_ACTION
    )
    maximum_information = ResearchBrainBench.baseline_decision(
        episode, BenchmarkPolicy.MAX_EXPECTED_INFORMATION_ACTION
    )
    random_one = ResearchBrainBench.baseline_decision(
        episode, BenchmarkPolicy.RANDOM_VALID_ACTION
    )
    random_two = ResearchBrainBench.baseline_decision(
        episode, BenchmarkPolicy.RANDOM_VALID_ACTION
    )

    assert cheapest.selected_action_id == "declare-universal-latent-defect"
    assert maximum_information.selected_action_id == "derive-competing-mechanisms"
    assert random_one == random_two


def test_structured_policy_runner_never_receives_hidden_future_state():
    episode = ResearchBrainBench(BENCH_ROOT).load_episode("before_ejc.json")
    seen = {}

    def runner(payload):
        seen.update(payload)
        return BenchmarkDecision(selected_action_id="frozen-boundary-diagnostic")

    decision = ResearchBrainBench.baseline_decision(
        episode, BenchmarkPolicy.BRAIN_V1_5, runner
    )

    assert decision.selected_action_id == "frozen-boundary-diagnostic"
    assert "hidden_future_state" not in seen
    assert "strong_next_actions" not in seen


def test_non_algorithmic_policy_requires_explicit_runner():
    episode = ResearchBrainBench(BENCH_ROOT).load_episode("before_ejc.json")

    with pytest.raises(GPUError) as error:
        ResearchBrainBench.baseline_decision(episode, BenchmarkPolicy.CURRENT_BRAIN_V1)

    assert error.value.error_type == "BRAIN_BENCH_POLICY_RUNNER_REQUIRED"


def test_scorecard_exposes_leakage_dead_idea_and_gate_failures_separately():
    episode = ResearchBrainBench(BENCH_ROOT).load_episode("hasi_before_intervention.json")
    decision = BenchmarkDecision(
        selected_action_id="train-anchor-model",
        retrieved_record_ids=["hasi-substitution-result", "ers-latent-box"],
        considered_hypothesis_ids=["state-propagation:hasi"],
        realized_information_gain=0,
    )

    scorecard = ResearchBrainBench.score(
        episode, BenchmarkPolicy.LLM_DIRECT_WITHOUT_STRUCTURED_MEMORY, decision
    )

    assert scorecard.metrics["future_information_leakage_rate"].value == 1
    assert scorecard.metrics["future_information_leakage_rate"].passed is False
    assert scorecard.metrics["reproduction_gate_compliance"].passed is False
    assert scorecard.metrics["architecture_too_early_rate"].passed is False
    assert scorecard.metrics["negative_memory_reuse"].value == 1
    assert not hasattr(scorecard, "total_score")


def test_unknown_policy_action_is_rejected():
    episode = ResearchBrainBench(BENCH_ROOT).load_episode("after_stage4.json")

    with pytest.raises(GPUError) as error:
        ResearchBrainBench.score(
            episode,
            BenchmarkPolicy.BRAIN_V2_STRATEGY_AUGMENTED,
            BenchmarkDecision(selected_action_id="future-action"),
        )

    assert error.value.error_type == "BRAIN_BENCH_UNKNOWN_ACTION"


def test_aggregate_keeps_every_metric_separate():
    bench = ResearchBrainBench(BENCH_ROOT)
    cards = []
    for episode in bench.load_all():
        decision = bench.baseline_decision(
            episode, BenchmarkPolicy.MAX_EXPECTED_INFORMATION_ACTION
        )
        cards.append(
            bench.score(episode, BenchmarkPolicy.MAX_EXPECTED_INFORMATION_ACTION, decision)
        )

    aggregate = bench.aggregate(cards)

    assert aggregate.scorecards == len(cards)
    assert aggregate.metrics["future_information_leakage_rate"].mean == 0
    assert aggregate.metrics["strong_next_action_recall"].observations == len(cards)
    assert aggregate.metrics["realized_information_gain"].mean is None
    assert aggregate.metrics["realized_information_gain"].observations == 0
    assert not hasattr(aggregate, "total_score")


@pytest.mark.asyncio
async def test_mcp_benchmark_tools_do_not_expose_hidden_state(monkeypatch):
    from gpu_lab import server

    async def direct_call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except GPUError as exc:
            return exc.response()

    monkeypatch.setattr(server, "call", direct_call)
    monkeypatch.setattr(server, "brain_bench_service", ResearchBrainBench(BENCH_ROOT))

    listed = await server.research_benchmark_list()
    visible = await server.research_benchmark_episode_get("pointcloud-after-stage4")
    baseline = await server.research_benchmark_policy_run(
        "pointcloud-after-stage4", "MAX_EXPECTED_INFORMATION_ACTION"
    )

    assert len(listed) >= 3
    assert "hidden_future_state" not in json.dumps(listed)
    assert "expected_information_gain" not in json.dumps(visible)
    assert baseline["selected_action_id"] == "derive-competing-mechanisms"
    assert "metrics" not in baseline
