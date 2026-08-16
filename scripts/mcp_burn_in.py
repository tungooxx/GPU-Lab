"""Measure public MCP read reliability without mutating research state or local jobs."""

import argparse
import json
import sys
import time
import uuid

import httpx


def invoke(client: httpx.Client, url: str, name: str, arguments: dict) -> dict:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = client.post(
            url,
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        payload = response.json()
        result = payload.get("result", {}).get("structuredContent", {}).get("result", {})
        return {
            "tool": name,
            "request_id": response.headers.get("X-Request-ID", request_id),
            "http_status": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "error": result.get("error"),
            "ok": response.status_code == 200 and "error" not in result,
        }
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "tool": name,
            "request_id": request_id,
            "http_status": None,
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "error": str(exc),
            "ok": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://chucky-lab.com/mcp")
    parser.add_argument("--project-id")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--interval-seconds", type=float, default=3)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    calls = [("local_status", {}), ("literature_provider_status", {})]
    if args.project_id:
        calls.append(("research_state_get", {"project_id": args.project_id}))

    results: list[dict] = []
    with httpx.Client(timeout=45) as client:
        for iteration in range(args.iterations):
            for name, arguments in calls:
                result = invoke(client, args.url, name, arguments)
                result["iteration"] = iteration + 1
                results.append(result)
                print(json.dumps(result), flush=True)
            if iteration + 1 < args.iterations:
                time.sleep(args.interval_seconds)

    failures = [result for result in results if not result["ok"]]
    print(
        json.dumps(
            {
                "summary": {
                    "url": args.url,
                    "requests": len(results),
                    "failures": len(failures),
                    "success_rate": (len(results) - len(failures)) / len(results),
                }
            }
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
