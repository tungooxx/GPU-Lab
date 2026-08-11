"""Verify optional workers retain public HTTPS but cannot reach the MCP gateway."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
import urllib.request
import urllib.error

checks = [
    ("public", "https://github.com/", 200),
    ("service_mcp", "http://gpu-lab:8000/mcp", 403),
    ("host_mcp", "http://host.docker.internal:8000/mcp", 403),
]
for name, target, expected_status in checks:
    try:
        with urllib.request.urlopen(target, timeout=10) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception as exc:
        raise SystemExit(f"{name} failed without an HTTP policy response: {type(exc).__name__}") from exc
    print(f"{name}=HTTP_{status}")
    if status != expected_status:
        raise SystemExit(f"unexpected worker route status: {name}={status}")
'''


def main() -> None:
    for service in ("literature", "paper2agent"):
        completed = subprocess.run(
            ["docker", "compose", "exec", "-T", service, "python", "-"],
            cwd=REPO_ROOT,
            input=PROBE,
            text=True,
            check=True,
            capture_output=True,
        )
        print(f"[{service}]\n{completed.stdout.strip()}")


if __name__ == "__main__":
    main()
