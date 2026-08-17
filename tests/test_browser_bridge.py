import pytest

from gpu_lab.browser_bridge import ChatGPTWebPlaywrightRuntime
from gpu_lab.errors import GPUError


@pytest.mark.asyncio
async def test_browser_runtime_reports_disconnected_without_starting_browser(tmp_path):
    runtime = ChatGPTWebPlaywrightRuntime(tmp_path / "private-profile")

    assert await runtime.health() == {"status": "DISCONNECTED"}
    await runtime.pause()
    with pytest.raises(GPUError) as error:
        await runtime.submit_turn("continue")
    assert error.value.error_type == "BROWSER_RUNTIME_PAUSED"
