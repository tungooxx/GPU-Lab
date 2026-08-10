from pathlib import Path

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

    @property
    def db_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.gpu_lab_database_url.startswith(prefix):
            raise ValueError("Only sqlite database URLs are supported in the MVP")
        return Path(self.gpu_lab_database_url.removeprefix(prefix))
