"""Bounded dispatcher from durable WorkerWakeRequests to browser runtimes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .browser_bridge import ChatGPTWebPlaywrightRuntime, ResearchWorkerRuntime
from .cockpit import CockpitController
from .errors import GPUError


class BrowserWakeDispatcher:
    """Dispatch at most one persisted wake at a time; never infer science from the page."""

    def __init__(self, cockpit: CockpitController, profile_root: Path,
                 runtime_factory: Callable[[Path], ResearchWorkerRuntime] = ChatGPTWebPlaywrightRuntime):
        self.cockpit = cockpit
        self.profile_root = profile_root
        self.runtime_factory = runtime_factory
        self._runtimes: dict[str, ResearchWorkerRuntime] = {}

    async def dispatch_one(self, project_id: str | None = None) -> dict:
        wake = self.cockpit.wake_claim_next(project_id)
        if not wake:
            return {"dispatched": False, "reason": "NO_ELIGIBLE_WAKE"}
        wake_id = wake["id"]
        try:
            runtime_row = self._runtime_for_session(wake["worker_session_id"])
            runtime = self._runtimes.get(runtime_row["id"])
            if not runtime:
                runtime = self.runtime_factory(self.profile_root / runtime_row["worker_id"])
                self._runtimes[runtime_row["id"]] = runtime
            await runtime.attach(runtime_row.get("conversation_url"))
            health = await runtime.health()
            if health["status"] == "LOGIN_REQUIRED":
                self.cockpit.runtime_status(runtime_row["id"], "LOGIN_REQUIRED")
                raise GPUError("CHATGPT_WEB_LOGIN_REQUIRED", "Operator login is required")
            if health["status"] != "READY":
                self.cockpit.runtime_status(runtime_row["id"], "ERROR", health["status"])
                raise GPUError("CHATGPT_WEB_RUNTIME_NOT_READY", health["status"])
            self.cockpit.runtime_status(runtime_row["id"], "PROMPT_SUBMITTED")
            prompt = self._continuation_prompt(wake)
            await runtime.submit_turn(prompt)
            self.cockpit.runtime_status(runtime_row["id"], "RESPONSE_IN_PROGRESS")
            self.cockpit.wake_finish(wake_id)
            return {"dispatched": True, "wake_id": wake_id, "runtime_id": runtime_row["id"]}
        except GPUError as exc:
            self.cockpit.wake_finish(wake_id, failure_reason=exc.error_type)
            return {"dispatched": False, "wake_id": wake_id, "error": exc.response()["error"]}

    def _runtime_for_session(self, session_id: str) -> dict:
        with self.cockpit.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_worker_runtimes WHERE worker_session_id=%s", (session_id,))
            runtime = cur.fetchone()
            if not runtime:
                raise GPUError("BROWSER_RUNTIME_NOT_ATTACHED", session_id)
            return self.cockpit.lab._record(runtime) or {}

    @staticmethod
    def _continuation_prompt(wake: dict) -> str:
        return (
            "Re-sync LabState first with lab_sync. Verify your WorkLease and WorkItem are still valid. "
            "Do only bounded work, persist any canonical outputs through MCP, then call lab_turn_report "
            "with exactly one outcome: CONTINUE, WAITING_DEPENDENCY, IDLE, BLOCKED, REPLAN, HUMAN_REQUIRED, or ERROR. "
            f"Your current work item is {wake.get('work_item_id') or 'not assigned'}; do not rely on prior chat text."
        )
