"""Live QD smoke: MCP discovery -> PostgreSQL dead memory -> screened lineage -> restart."""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

MCP_URL = os.environ.get("GPU_LAB_MCP_URL", "http://127.0.0.1:8000/mcp")


def rpc(method: str, params: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        MCP_URL,
        data=json.dumps(
            {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "gpu-lab-qd-e2e/1",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        envelope = json.loads(response.read())
    if "error" in envelope:
        raise RuntimeError(envelope["error"])
    return envelope["result"]


def call(name: str, arguments: dict[str, Any]) -> Any:
    result = rpc("tools/call", {"name": name, "arguments": arguments})
    value = result.get("structuredContent", {}).get("result")
    if value is None:
        value = json.loads(result["content"][0]["text"])
    if isinstance(value, dict) and value.get("error"):
        raise RuntimeError({"tool": name, "error": value["error"]})
    return value


def wait_for_gateway() -> None:
    health_url = MCP_URL.removesuffix("/mcp") + "/health"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError("GPU-Lab did not recover after restart")


def main() -> None:
    wait_for_gateway()
    tools = {item["name"] for item in rpc("tools/list", {})["tools"]}
    required = {
        "hypothesis_niche_create",
        "hypothesis_niche_list",
        "hypothesis_qd_screen",
        "hypothesis_qd_create",
        "hypothesis_niche_set_best",
    }
    assert required <= tools

    nonce = uuid.uuid4().hex[:10]
    project = call(
        "research_project_create",
        {"name": f"QD live smoke {nonce}", "question": "Which mechanism carries viewpoint failure?"},
    )
    project_id = project["project_id"]
    niche = call(
        "hypothesis_niche_create",
        {
            "project_id": project_id,
            "name": "state propagation",
            "description": "Intermediate states that transmit failure information",
            "diversity_signature": {"family": "state propagation", "stage": "decoder entry"},
        },
    )
    dead = call(
        "negative_result_create",
        {
            "project_id": project_id,
            "proposal": "Anchor state transmits viewpoint evidence into the decoder carrier",
            "prediction": "Static anchor correlation predicts the carrier",
            "result": "Correlation did not survive intervention",
            "failed_assumption": "anchor state is causally upstream",
            "revisit_condition": "A direct intervention changes the carrier",
        },
    )
    draft = {
        "mechanism": "Anchor state transmits viewpoint evidence into the decoder carrier",
        "prediction": "Replacing anchor state changes the downstream carrier",
        "kill_condition": "Replacement leaves the carrier unchanged under fixed decoder state",
        "niche_id": niche["id"],
        "assumptions": ["anchor state is causally upstream"],
        "variables": ["anchor_state", "carrier"],
        "information_path": ["viewpoint", "anchor_state", "carrier"],
        "scope": "VRCNet frozen inference",
    }
    screened = call("hypothesis_qd_screen", {"project_id": project_id, "draft": draft})
    assert screened["accepted"] is False
    assert dead["id"] in screened["similar_dead_hypothesis_ids"]

    draft["scientific_difference"] = (
        "Tests a frozen internal intervention instead of relying on static correlation"
    )
    hypothesis = call("hypothesis_qd_create", {"project_id": project_id, "draft": draft})
    call(
        "hypothesis_niche_set_best",
        {
            "niche_id": niche["id"],
            "hypothesis_id": hypothesis["id"],
            "rationale": "Cheapest discriminating intervention",
        },
    )

    subprocess.run(["docker", "compose", "restart", "postgres", "gpu-lab"], check=True)
    wait_for_gateway()
    niches = call("hypothesis_niche_list", {"project_id": project_id})
    recovered = next(item for item in niches if item["id"] == niche["id"])
    assert recovered["data"]["active_best_hypothesis_id"] == hypothesis["id"]
    print(
        json.dumps(
            {
                "project_id": project_id,
                "niche_id": niche["id"],
                "hypothesis_id": hypothesis["id"],
                "dead_idea_id": dead["id"],
                "restart_recovered": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
