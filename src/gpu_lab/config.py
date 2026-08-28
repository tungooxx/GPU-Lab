from pathlib import Path

from pydantic import Field, field_validator
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
    gpu_lab_ssh_connection_idle_seconds: int = Field(default=60, ge=1, le=900)
    gpu_lab_ssh_known_hosts: Path | None = None
    gpu_lab_ssh_allow_unverified_hosts: bool = False
    gpu_lab_allowed_hosts: str = "127.0.0.1:*,localhost:*"
    gpu_lab_enable_local_runner: bool = False
    gpu_lab_local_workspace: Path = Path("/workspace/local-vlm")
    gpu_lab_source_checkout: Path = Path("/workspace/gpu-lab-source")
    gpu_lab_local_env_root: Path = Path("/opt/gpu-lab/envs")
    gpu_lab_canonical_vrc_env: str = "vrc-py313-torch260-cu124"
    gpu_lab_terminal_password: str | None = None
    gpu_lab_research_database_url: str | None = None
    gpu_lab_research_bench_dir: Path = Path("./research_bench")
    gpu_lab_policy_auto_evaluate: bool = True
    gpu_lab_policy_auto_reject: bool = True
    gpu_lab_policy_auto_revise: bool = True
    gpu_lab_policy_auto_promote_production: bool = False
    gpu_lab_policy_max_revisions: int = Field(default=1, ge=0, le=3)
    # v3 defaults to narrowly-scoped autonomy; domain/global still require an
    # explicit operator choice and every run remains budgeted.
    gpu_lab_policy_autonomy_mode: str = "AUTO_PROJECT"
    gpu_lab_policy_meta_candidate_budget: int = Field(default=3, ge=1, le=10)
    gpu_lab_policy_meta_benchmark_budget: int = Field(default=6, ge=1, le=30)
    gpu_lab_policy_meta_literature_budget: int = Field(default=1, ge=0, le=5)
    gpu_lab_embedding_provider: str = "local-hash"
    gpu_lab_embedding_dimension: int = Field(default=384, ge=32, le=4096)
    gpu_lab_research_operator_provider: str = "disabled"
    gpu_lab_literature_provider: str = "disabled"
    gpu_lab_literature_worker_url: str = "http://literature:8010"
    gpu_lab_literature_worker_token: str | None = None
    gpu_lab_paperqa_directory: Path = Path("/opt/gpu-lab/papers")
    gpu_lab_executable_paper_provider: str = "disabled"
    gpu_lab_executable_paper_worker_url: str = "http://paper2agent:8020"
    gpu_lab_executable_paper_worker_token: str | None = None
    gpu_lab_approval_secret: str | None = None
    gpu_lab_denied_mcp_client_cidrs: str = ""
    lab_ui_enabled: bool = False
    chatgpt_web_bridge_enabled: bool = False
    autopilot_enabled: bool = False
    auto_continue_enabled: bool = False
    live_browser_preview_enabled: bool = False
    chatgpt_web_profile_root: Path = Path("/var/lib/gpu-lab/chatgpt-web")
    gpu_lab_worker_max_turns_per_work_item: int = Field(default=20, ge=1, le=100)
    gpu_lab_worker_max_consecutive_continues: int = Field(default=3, ge=1, le=20)
    gpu_lab_browser_wake_poll_seconds: int = Field(default=5, ge=1, le=60)
    gpu_lab_lease_reconciliation_poll_seconds: int = Field(default=30, ge=5, le=300)
    # v3.5.5 rollout is deliberately staged; defaults preserve existing live
    # behavior until an operator enables each enforcement phase.
    canonical_authority_v355: bool = False
    atomic_work_dedupe: bool = False
    versioned_worker_writes: bool = False
    canonical_sync_context: bool = False
    supersession_propagation: bool = False
    dependency_reconciliation: bool = True
    work_proposal_mode: bool = False
    portfolio_scheduler_v36: bool = False
    waiting_work_release: bool = False
    branch_aware_assignment: bool = False
    agenda_coverage: bool = False
    planner_on_idle: bool = False
    gpu_worker_detach: bool = False
    speculative_work_policy: bool = False
    gpu_lab_dashboard_monitor_enabled: bool = True
    gpu_lab_cockpit_password: str | None = None
    gpu_lab_cockpit_session_secret: str | None = None
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

    @field_validator("gpu_lab_policy_autonomy_mode")
    @classmethod
    def valid_policy_autonomy_mode(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"ADVISORY", "AUTO_PROJECT", "AUTO_DOMAIN", "AUTO_GLOBAL"}:
            raise ValueError("GPU_LAB_POLICY_AUTONOMY_MODE is invalid")
        return normalized

    @field_validator("gpu_lab_embedding_provider")
    @classmethod
    def valid_embedding_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"disabled", "local-hash"}:
            raise ValueError("GPU_LAB_EMBEDDING_PROVIDER must be disabled or local-hash")
        return normalized

    @field_validator("gpu_lab_research_operator_provider")
    @classmethod
    def valid_research_operator_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"disabled", "literature-http"}:
            raise ValueError(
                "GPU_LAB_RESEARCH_OPERATOR_PROVIDER must be disabled or literature-http"
            )
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
