from gpu_lab.server import mcp


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
