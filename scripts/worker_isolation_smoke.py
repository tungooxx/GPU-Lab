"""Verify optional workers retain public HTTPS but cannot reach the MCP gateway."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
import urllib.request

checks = [
    ("public", "https://github.com/", True),
    ("service_mcp", "http://gpu-lab:8000/mcp", False),
    ("host_mcp", "http://host.docker.internal:8000/mcp", False),
]
for name, target, should_succeed in checks:
    try:
        with urllib.request.urlopen(target, timeout=10) as response:
            succeeded = response.status == 200
    except Exception:
        succeeded = False
    print(f"{name}={'REACHABLE' if succeeded else 'BLOCKED'}")
    if succeeded != should_succeed:
        raise SystemExit(f"unexpected worker route: {name}")
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
