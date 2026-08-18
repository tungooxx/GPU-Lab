import pytest

from gpu_lab.browser_bridge import ChatGPTWebPlaywrightRuntime
from gpu_lab.browser_scheduler import BrowserWakeDispatcher
from gpu_lab.errors import GPUError


@pytest.mark.asyncio
async def test_browser_runtime_reports_disconnected_without_starting_browser(tmp_path):
    runtime = ChatGPTWebPlaywrightRuntime(tmp_path / "private-profile")

    assert runtime.headless is True
    assert await runtime.health() == {"status": "DISCONNECTED"}
    await runtime.pause()
    with pytest.raises(GPUError) as error:
        await runtime.submit_turn("continue")
    assert error.value.error_type == "BROWSER_RUNTIME_PAUSED"


@pytest.mark.asyncio
async def test_wake_dispatcher_resyncs_before_one_bounded_browser_turn(tmp_path):
    class Cockpit:
        def __init__(self):
            self.statuses = []
            self.finished = []

        def wake_claim_next(self, project_id=None):
            return {"id": "wake-1", "worker_session_id": "session-1", "work_item_id": "work-1"}

        def runtime_status(self, runtime_id, status, error=None):
            self.statuses.append((runtime_id, status, error))

        def wake_finish(self, wake_id, failure_reason=None):
            self.finished.append((wake_id, failure_reason))

    class Runtime:
        def __init__(self):
            self.prompt = None

        async def attach(self, conversation_url):
            assert conversation_url == "https://chatgpt.com/c/demo"

        async def health(self):
            return {"status": "READY"}

        async def submit_turn(self, prompt):
            self.prompt = prompt

    cockpit, runtime = Cockpit(), Runtime()
    dispatcher = BrowserWakeDispatcher(cockpit, tmp_path, runtime_factory=lambda _: runtime)
    dispatcher._runtime_for_session = lambda _: {"id": "runtime-1", "worker_id": "worker-1", "conversation_url": "https://chatgpt.com/c/demo"}

    result = await dispatcher.dispatch_one()

    assert result == {"dispatched": True, "wake_id": "wake-1", "runtime_id": "runtime-1"}
    assert "Re-sync LabState first" in runtime.prompt
    assert cockpit.statuses == [("runtime-1", "PROMPT_SUBMITTED", None), ("runtime-1", "RESPONSE_IN_PROGRESS", None)]
    assert cockpit.finished == [("wake-1", None)]


@pytest.mark.asyncio
async def test_wake_dispatcher_records_non_gpu_browser_failure(tmp_path):
    class Cockpit:
        def __init__(self):
            self.statuses = []
            self.finished = []

        def wake_claim_next(self, project_id=None):
            return {"id": "wake-1", "worker_session_id": "session-1", "work_item_id": "work-1"}

        def runtime_status(self, runtime_id, status, error=None):
            self.statuses.append((runtime_id, status, error))

        def wake_finish(self, wake_id, failure_reason=None):
            self.finished.append((wake_id, failure_reason))

    class Runtime:
        async def attach(self, conversation_url):
            raise RuntimeError("TargetClosedError")

        async def close(self):
            pass

    cockpit = Cockpit()
    dispatcher = BrowserWakeDispatcher(cockpit, tmp_path, runtime_factory=lambda _: Runtime())
    dispatcher._runtime_for_session = lambda _: {"id": "runtime-1", "worker_id": "worker-1"}

    result = await dispatcher.dispatch_one()

    assert result == {"dispatched": False, "wake_id": "wake-1", "error": "BROWSER_DISPATCH_FAILED"}
    assert cockpit.statuses == [("runtime-1", "ERROR", "BROWSER_DISPATCH_FAILED")]
    assert cockpit.finished == [("wake-1", "BROWSER_DISPATCH_FAILED")]
