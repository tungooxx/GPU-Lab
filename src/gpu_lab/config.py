from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    vast_api_key: str | None = None
    gpu_lab_ssh_private_key_path: Path | None = None
    gpu_lab_data_dir: Path = Path("./data")
    gpu_lab_database_url: str = "sqlite:///./data/gpu_lab.db"
    gpu_lab_remote_root: str = "/workspace/gpu-lab"
    gpu_lab_enable_remote_exec: bool = False
    gpu_lab_max_log_lines: int = 1000
    gpu_lab_max_text_artifact_bytes: int = 1_048_576
    gpu_lab_ssh_timeout: int = 20
    gpu_lab_ssh_known_hosts: Path | None = None
    gpu_lab_ssh_allow_unverified_hosts: bool = False
    gpu_lab_allowed_hosts: str = "127.0.0.1:*,localhost:*"
    gpu_lab_enable_local_runner: bool = False
    gpu_lab_local_workspace: Path = Path("/workspace/local-vlm")
    gpu_lab_local_env_root: Path = Path("/opt/gpu-lab/envs")
    gpu_lab_canonical_vrc_env: str = "vrc-py313-torch260-cu124"
    gpu_lab_terminal_password: str | None = None
    gpu_lab_research_database_url: str | None = None
    gpu_lab_literature_provider: str = "disabled"
    gpu_lab_literature_worker_url: str = "http://literature:8010"
    gpu_lab_literature_worker_token: str | None = None
    gpu_lab_paperqa_directory: Path = Path("/opt/gpu-lab/papers")
    gpu_lab_executable_paper_provider: str = "disabled"
    gpu_lab_executable_paper_worker_url: str = "http://paper2agent:8020"
    gpu_lab_executable_paper_worker_token: str | None = None
    gpu_lab_approval_secret: str | None = None
    gpu_lab_denied_mcp_client_cidrs: str = ""
    fastmcp_host: str = "127.0.0.1"
    fastmcp_port: int = 8000

    @field_validator("gpu_lab_ssh_private_key_path", "gpu_lab_ssh_known_hosts", mode="before")
    @classmethod
    def blank_paths_are_none(cls, value):
        return None if value is None or (isinstance(value, str) and not value.strip()) else value

    @field_validator("gpu_lab_literature_provider")
    @classmethod
    def valid_literature_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"disabled", "paperqa-http"}:
            raise ValueError("GPU_LAB_LITERATURE_PROVIDER must be disabled or paperqa-http")
        return normalized

    @field_validator("gpu_lab_executable_paper_provider")
    @classmethod
    def valid_executable_paper_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"disabled", "paper2agent-http"}:
            raise ValueError(
                "GPU_LAB_EXECUTABLE_PAPER_PROVIDER must be disabled or paper2agent-http"
            )
        return normalized

    @property
    def db_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.gpu_lab_database_url.startswith(prefix):
            raise ValueError("Only sqlite database URLs are supported in the MVP")
        return Path(self.gpu_lab_database_url.removeprefix(prefix))
