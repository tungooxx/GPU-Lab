from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled", "unknown"]


class Instance(BaseModel):
    id: str
    provider: str = "vast"
    provider_instance_id: str
    status: str = "unknown"
    label: str | None = None
    hostname: str | None = None
    ssh_port: int | None = None
    ssh_user: str = "root"
    gpu_model: str | None = None
    gpu_count: int | None = None
    hourly_price: float | None = None
    created_at: datetime | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
    job_id: str
    instance_id: str
    name: str | None = None
    repo_url: str | None = None
    repo_path: str
    git_commit: str | None = None
    git_dirty: bool = False
    command: str
    status: JobStatus = "queued"
    remote_pid: int | None = None
    remote_session: str | None = None
    exit_code: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
