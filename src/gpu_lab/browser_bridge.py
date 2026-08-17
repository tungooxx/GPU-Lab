"""Provider-neutral browser worker runtime interfaces.

Selectors for ChatGPT Web are intentionally contained here. The bridge reports
operational state only; it cannot persist scientific results or infer work
completion from page text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .errors import GPUError


class ResearchWorkerRuntime(Protocol):
    async def attach(self, conversation_url: str | None) -> None: ...
    async def health(self) -> dict: ...
    async def activate(self) -> None: ...
    async def submit_turn(self, prompt: str) -> None: ...
    async def focus(self) -> None: ...
    async def pause(self) -> None: ...
    async def resume(self) -> None: ...
    async def close(self) -> None: ...


class ChatGPTWebPlaywrightRuntime:
    """Server-side persistent-profile adapter for one worker conversation."""

    def __init__(self, profile_dir: Path, *, headless: bool = False):
        self.profile_dir = profile_dir
        self.headless = headless
        self._playwright = None
        self._context = None
        self._page = None
        self._paused = False

    @staticmethod
    def _imports():
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover
            raise GPUError(
                "CHATGPT_WEB_BRIDGE_UNAVAILABLE",
                "Install the optional browser runtime and Chromium before enabling the bridge.",
            ) from exc
        return async_playwright

    async def _ensure_context(self) -> None:
        if self._context:
            return
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        factory = self._imports()
        self._playwright = await factory().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            str(self.profile_dir), headless=self.headless
        )
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()

    async def attach(self, conversation_url: str | None) -> None:
        await self._ensure_context()
        if conversation_url:
            await self._page.goto(conversation_url, wait_until="domcontentloaded")

    async def health(self) -> dict:
        if not self._page or self._page.is_closed():
            return {"status": "DISCONNECTED"}
        url = self._page.url
        if "login" in url or "auth" in url:
            return {"status": "LOGIN_REQUIRED"}
        if self._paused:
            return {"status": "IDLE", "paused": True}
        return {"status": "READY", "conversation_url": url}

    async def activate(self) -> None:
        await self._ensure_context()
        await self._page.bring_to_front()

    async def focus(self) -> None:
        await self.activate()

    async def submit_turn(self, prompt: str) -> None:
        if self._paused:
            raise GPUError("BROWSER_RUNTIME_PAUSED", "Resume the worker before submitting a turn")
        await self.activate()
        composer = self._page.locator("textarea[data-id='root'], #prompt-textarea").first
        if await composer.count() == 0:
            raise GPUError("CHATGPT_WEB_COMPOSER_NOT_FOUND", "Login or attach a usable conversation")
        await composer.fill(prompt)
        await composer.press("Enter")

    async def pause(self) -> None:
        self._paused = True

    async def resume(self) -> None:
        self._paused = False

    async def close(self) -> None:
        if self._context:
            await self._context.close()
        if self._playwright:
            await self._playwright.stop()
        self._context = self._page = self._playwright = None
