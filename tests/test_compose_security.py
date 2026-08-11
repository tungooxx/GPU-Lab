import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_gateway_blanks_worker_only_credentials_inherited_from_env_file():
    """Keep task-scoped provider credentials out of the gateway service."""
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    gateway = compose.split("  literature:\n", maxsplit=1)[0]

    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CROSSREF_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
    ):
        assert f'{name}: ""' in gateway


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker Compose is unavailable")
def test_resolved_compose_keeps_sentinel_worker_credentials_out_of_gateway():
    environment = {
        **os.environ,
        "OPENAI_API_KEY": "sentinel-openai",
        "ANTHROPIC_API_KEY": "sentinel-anthropic",
        "CROSSREF_API_KEY": "sentinel-crossref",
        "SEMANTIC_SCHOLAR_API_KEY": "sentinel-semantic",
    }
    result = subprocess.run(
        ["docker", "compose", "--profile", "literature", "config", "--format", "json"],
        cwd=Path(__file__).parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    gateway = json.loads(result.stdout)["services"]["gpu-lab"]["environment"]

    assert {name: gateway[name] for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CROSSREF_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
    )} == {
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "CROSSREF_API_KEY": "",
        "SEMANTIC_SCHOLAR_API_KEY": "",
    }
