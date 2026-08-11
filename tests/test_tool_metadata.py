from gpu_lab.server import _compact_research_state, mcp


def test_vast_status_tool_has_provider_specific_name():
    names = set(mcp._tool_manager._tools)

    assert "vast_gpu_status" in names
    assert "gpu_status" not in names


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
