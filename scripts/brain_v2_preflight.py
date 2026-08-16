"""Shared, non-scientific preflight helpers for Brain v2 smoke entrypoints."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

MCP_URL = os.environ.get("GPU_LAB_MCP_URL", "http://127.0.0.1:8000/mcp")


def rpc(method: str, params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(
        MCP_URL,
        data=json.dumps(
            {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "gpu-lab-brain-v2-preflight/1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        envelope = json.loads(response.read())
    if "error" in envelope:
        raise RuntimeError(envelope["error"])
    return envelope["result"]


def call(name: str, arguments: dict[str, Any], timeout: int = 60) -> Any:
    result = rpc("tools/call", {"name": name, "arguments": arguments}, timeout)
    value = result.get("structuredContent", {}).get("result")
    if value is None:
        value = json.loads(result["content"][0]["text"])
    return value


def wait_for_gateway() -> dict[str, Any]:
    deadline = time.monotonic() + 30
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            listed = rpc("tools/list", {}, timeout=5)
            return {"status": "ready", "tool_count": len(listed.get("tools", []))}
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            last_error = str(exc)
            time.sleep(1)
    return {"status": "unavailable", "error": last_error}


def gate(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def report(script: str, status: str, **details: Any) -> None:
    print(json.dumps({"script": script, "verification": status, **details}, indent=2))
