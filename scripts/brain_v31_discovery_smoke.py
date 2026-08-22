"""Read-only production v3.1 smoke against an explicitly supplied project.

It never executes a GPU action.  It validates that the live MCP preview sees
canonical state and exposes the portfolio/regime provenance required by v3.1.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
import uuid


def call(url: str, tool: str, arguments: dict) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "tools/call", "params": {"name": tool, "arguments": arguments}}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    with urllib.request.urlopen(request, timeout=60) as response:
        envelope = json.loads(response.read())
    if "error" in envelope:
        raise RuntimeError(envelope["error"])
    result = envelope["result"].get("structuredContent", {}).get("result")
    return result if result is not None else json.loads(envelope["result"]["content"][0]["text"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id", help="Existing project; this smoke is read-only.")
    parser.add_argument("--mcp-url", default=os.environ.get("GPU_LAB_MCP_URL", "http://127.0.0.1:8000/mcp"))
    arguments = parser.parse_args()
    preview = call(arguments.mcp_url, "brain_preview", {"project_id": arguments.project_id})
    required = {"candidate_portfolio", "frontier_gap", "stagnation_state", "state_freshness"}
    missing = sorted(key for key in required if not preview.get(key))
    if missing:
        raise AssertionError(f"v3.1 preview missing {missing}: {preview}")
    print(json.dumps({"status": "VERIFIED_INTEGRATION", "decision_id": preview.get("decision_id"), "search_regime": preview["candidate_portfolio"].get("data", {}).get("search_regime", preview["candidate_portfolio"].get("search_regime")), "candidate_count": len(preview.get("candidate_actions", [])), "state_freshness": preview["state_freshness"]}, indent=2))


if __name__ == "__main__":
    main()
