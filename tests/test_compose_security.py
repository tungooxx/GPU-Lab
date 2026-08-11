from pathlib import Path


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
