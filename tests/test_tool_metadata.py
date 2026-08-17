import uuid

import pytest

from gpu_lab import server
from gpu_lab.server import (
    _compact_brain_step,
    _compact_research_state,
    _prioritise_monitor_jobs,
    _safe_request_id,
    mcp,
)


def test_vast_status_tool_has_provider_specific_name():
    names = set(mcp._tool_manager._tools)

    assert "vast_gpu_status" in names
    assert "gpu_status" not in names


@pytest.mark.asyncio
async def test_improve_start_search_uses_resolved_literature_provider(monkeypatch):
    class Candidate:
        source_excerpt = "candidate evidence"

    class Result:
        def __init__(self):
            self.answer = "provider answer"
            self.evidence_candidates = [Candidate()]

    class Provider:
        async def search(self, query, filters=None):
            assert "measured policy weakness" in query
            assert filters is None
            return Result()

    class Literature:
        provider = Provider()

    class PolicyLab:
        def improve(self, project_id, **kwargs):
            return {"project_id": project_id, "paper": kwargs["paper"]}

    async def direct_call(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return await result if hasattr(result, "__await__") else result

    monkeypatch.setattr(server.settings, "gpu_lab_literature_provider", "paperqa-http")
    monkeypatch.setattr(server, "literature", lambda: Literature())
    monkeypatch.setattr(server, "policy_lab", lambda: PolicyLab())
    monkeypatch.setattr(server, "call", direct_call)

    assert await server.improve_start("project-1", idea="better selection", search=True) == {
        "project_id": "project-1", "paper": "provider answer"
    }


def test_monitor_prioritises_active_jobs_over_newer_history():
    class Job:
        def __init__(self, job_id):
            self.job_id = job_id

    active = Job("running-current")
    result = _prioritise_monitor_jobs(
        [active], [], [Job("history-newer"), active, Job("history-older")], limit=2
    )

    assert [job.job_id for job in result] == ["running-current", "history-newer"]


def test_research_decision_creation_is_explicitly_discoverable():
    tool = mcp._tool_manager._tools["research_decision_create"]

    assert tool.fn_metadata.arg_model.model_json_schema()["required"] == [
        "project_id",
        "experiment_id",
        "command",
    ]
    assert "research_experiment_execute" in tool.description


def test_every_mcp_tool_has_chatgpt_metadata():
    for tool in mcp._tool_manager._tools.values():
        assert tool.title
        assert tool.description
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is not None
        assert tool.annotations.destructiveHint is not None
        assert tool.annotations.openWorldHint is not None
        assert tool.output_schema is not None
        assert tool.fn_metadata.output_model is not None


def test_every_mcp_tool_can_convert_a_structured_result():
    """Guard FastMCP's output-schema/output-model contract for every tool."""
    for tool in mcp._tool_manager._tools.values():
        converted = tool.fn_metadata.convert_result({"smoke": "ok"})

        assert isinstance(converted, tuple)
        _, structured_content = converted
        assert structured_content == {"result": {"smoke": "ok"}}


def test_strategy_writes_are_not_advertised_as_read_only():
    for name in (
        "research_null_model_create",
        "research_null_model_test",
        "research_decision_outcome_assess",
    ):
        annotations = mcp._tool_manager._tools[name].annotations
        assert annotations.readOnlyHint is False
        assert annotations.destructiveHint is True

    for name in ("research_strategy_list", "research_strategy_dataset_export"):
        annotations = mcp._tool_manager._tools[name].annotations
        assert annotations.readOnlyHint is True
        assert annotations.destructiveHint is False


def test_research_state_summary_excludes_large_object_payloads():
    state = {
        "name": "Fixture",
        "question": "What happened?",
        "state": {"research_question": "What happened?"},
        "canonical_state": {
            "active_hypotheses": [
                {
                    "id": "hypothesis-1",
                    "kind": "Hypothesis",
                    "status": "ACTIVE",
                    "data": {"mechanism": "A mechanism", "large": "x" * 100_000},
                }
            ],
            "active_anomalies": [],
        },
        "objects": [
            {"id": "hypothesis-1", "kind": "Hypothesis", "status": "ACTIVE", "data": {}},
            {"id": "decision-1", "kind": "ResearchDecision", "status": "SELECTED", "data": {}},
        ],
    }

    compact = _compact_research_state(state, limit=10)

    assert "objects" not in compact
    assert compact["object_counts"] == {"Hypothesis": 1, "ResearchDecision": 1}
    assert compact["canonical_state"]["active_hypotheses"][0]["data"] == {
        "mechanism": "A mechanism"
    }


def test_brain_step_summary_excludes_large_scientific_state_payloads():
    result = {
        "brain_step_id": "step-1",
        "decision_id": "decision-1",
        "agenda_item": {"id": "agenda-1", "kind": "AgendaItem", "status": "ACTIVE", "data": {}},
        "question": "What next?",
        "scientific_state": {
            "active_hypotheses": [
                {"id": "hypothesis-1", "kind": "Hypothesis", "status": "ACTIVE", "data": {"large": "x" * 100_000}}
            ]
        },
        "world_model": {"id": "model-1", "kind": "WorldModel", "status": "ACTIVE", "data": {}},
        "competing_hypotheses": [],
        "dead_ideas_retrieved": [],
        "candidate_actions": [],
        "selected_action": {"action_type": "ARTIFACT_ANALYSIS"},
        "brain_policy_version": "brain-v3.1-discovery-search-v1",
        "search_regime": "DIVERGENT_SEARCH",
    }

    compact = _compact_brain_step(result)

    assert compact["scientific_state"]["active_hypotheses"][0]["data"] == {}
    assert "large" not in str(compact)
    assert compact["brain_policy_version"] == "brain-v3.1-discovery-search-v1"
    assert compact["search_regime"] == "DIVERGENT_SEARCH"


def test_brain_step_summary_preserves_scalar_canonical_entries():
    result = {
        "brain_step_id": "step-1",
        "decision_id": "decision-1",
        "agenda_item": {"id": "agenda-1", "kind": "AgendaItem", "status": "ACTIVE", "data": {}},
        "question": "What next?",
        "scientific_state": {"established_facts": ["fact-a", "fact-b"]},
        "world_model": {"id": "model-1", "kind": "WorldModel", "status": "ACTIVE", "data": {}},
        "competing_hypotheses": [],
        "dead_ideas_retrieved": [],
        "candidate_actions": [],
        "selected_action": {"action_type": "ARTIFACT_ANALYSIS"},
    }

    compact = _compact_brain_step(result)

    assert compact["scientific_state"]["established_facts"] == ["fact-a", "fact-b"]


def test_request_ids_are_safe_to_reflect_in_mcp_logs_and_headers():
    assert _safe_request_id("trace-42.request") == "trace-42.request"
    assert _safe_request_id("bad\r\nlog-forgery") != "bad\r\nlog-forgery"
    assert _safe_request_id("x" * 129) != "x" * 129


def test_research_map_payload_contains_only_graph_rendering_fields(monkeypatch):
    class FakeStore:
        def objects_list(self, *_args, **_kwargs):
            return [{"id": "model-1"}]

    class FakeBrain:
        def world_model_get(self, _model_id):
            return {
                "world_model": {
                    "id": "model-1",
                    "project_id": "project-1",
                    "kind": "WorldModel",
                    "status": "ACTIVE",
                    "created_at": "now",
                    "data": {"name": "Mechanism map", "secret": "never render"},
                },
                "nodes": [
                    {
                        "id": "node-1",
                        "kind": "Mechanism",
                        "status": "ACTIVE",
                        "data": {
                            "name": "Anchor state",
                            "description": "Carrier",
                            "attributes": {"layer": 8, "evidence_id": uuid.UUID(int=1)},
                            "secret": "never render",
                        },
                    }
                ],
                "edges": [
                    {
                        "id": "edge-1",
                        "kind": "CausalEdge",
                        "status": "ACTIVE",
                        "data": {
                            "source_id": "node-1",
                            "target_id": "node-1",
                            "relation": "CAUSES",
                            "edge_status": "HYPOTHESIZED_CAUSAL",
                            "secret": "never render",
                        },
                    }
                ],
                "versions": [],
            }

    monkeypatch.setattr(server, "research", lambda: FakeStore())
    monkeypatch.setattr(server, "brain", lambda: FakeBrain())

    payload = server._research_map_payload("project-1")

    assert payload["world_model"]["data"] == {"name": "Mechanism map"}
    assert payload["nodes"][0]["data"] == {
        "name": "Anchor state",
        "description": "Carrier",
        "attributes": {"layer": 8, "evidence_id": "00000000-0000-0000-0000-000000000001"},
    }
    assert payload["edges"][0]["data"] == {
        "source_id": "node-1",
        "target_id": "node-1",
        "relation": "CAUSES",
        "edge_status": "HYPOTHESIZED_CAUSAL",
    }


def test_research_runtime_is_initialized_before_mcp_accepts_requests(monkeypatch):
    created = []

    class InitializedStore:
        def __init__(self, url):
            created.append(url)

    monkeypatch.setattr(server.settings, "gpu_lab_research_database_url", "postgresql://fixture")
    monkeypatch.setattr(server, "research_store", None)
    monkeypatch.setattr(server, "ResearchStore", InitializedStore)

    initialized = server.initialize_research_runtime()
    replay = server.research()

    assert initialized is replay
    assert created == ["postgresql://fixture"]
