from gpu_lab.server import mcp


def test_vast_status_tool_has_provider_specific_name():
    names = set(mcp._tool_manager._tools)

    assert "vast_gpu_status" in names
    assert "gpu_status" not in names


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
