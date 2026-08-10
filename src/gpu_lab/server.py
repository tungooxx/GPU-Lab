import argparse
import os

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .config import Settings
from .errors import GPUError
from .service import GPUService

settings, service = Settings(), None
allowed_hosts = [
    host.strip()
    for host in os.getenv("GPU_LAB_ALLOWED_HOSTS", "127.0.0.1:*,localhost:*").split(",")
    if host.strip()
]
mcp = FastMCP(
    "GPU Lab",
    json_response=True,
    instructions="Safe, structured remote GPU experiment control plane. Credentials are never returned.",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=[
            f"https://{host.removesuffix(':*')}" for host in allowed_hosts if ":*" not in host
        ],
    ),
)


def svc() -> GPUService:
    global service
    if service is None:
        service = GPUService(settings)
    return service


async def call(fn, *args, **kwargs):
    try:
        return await fn(*args, **kwargs)
    except GPUError as exc:
        return exc.response()


@mcp.tool()
async def gpu_list():
    """List known and provider-visible GPU instances."""
    return await call(svc().gpu_list)


@mcp.tool()
async def gpu_status(instance_id: str):
    """Return provider and nvidia-smi runtime state."""
    return await call(svc().gpu_status, instance_id)


@mcp.tool()
async def gpu_search(
    gpu_name: str | None = None,
    min_vram_gb: int | None = None,
    max_hourly_price: float | None = None,
):
    """Return safely ranked Vast offers; create requires an explicit offer ID."""
    return await call(svc().gpu_search, gpu_name, min_vram_gb, max_hourly_price)


@mcp.tool()
async def gpu_create(
    offer_id: str, disk_gb: int = 100, image: str | None = None, label: str | None = None
):
    """Create a Vast instance from a caller-selected offer."""
    return await call(svc().gpu_create, offer_id, disk_gb, image, label)


@mcp.tool()
async def gpu_stop(instance_id: str):
    """Stop an instance while preserving data when Vast supports it."""
    return await call(svc().gpu_stop, instance_id)


@mcp.tool()
async def gpu_destroy(instance_id: str, confirmation: str):
    """Destroy only with confirmation exactly DESTROY."""
    return await call(svc().gpu_destroy, instance_id, confirmation)


@mcp.tool()
async def repo_checkout(
    instance_id: str,
    repo_url: str,
    commit: str | None = None,
    branch: str | None = None,
    name: str | None = None,
):
    return await call(svc().repo_checkout, instance_id, repo_url, commit, branch, name)


@mcp.tool()
async def env_prepare(instance_id: str, repo_path: str, strategy: str = "auto"):
    return await call(svc().env_prepare, instance_id, repo_path, strategy)


@mcp.tool()
async def experiment_submit(
    instance_id: str,
    repo_path: str,
    command: str,
    name: str | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
    artifact_patterns: list[str] | None = None,
    metadata: dict | None = None,
):
    return await call(
        svc().experiment_submit,
        instance_id,
        repo_path,
        command,
        name,
        env,
        timeout_seconds,
        artifact_patterns,
        metadata,
    )


@mcp.tool()
async def experiment_status(job_id: str):
    return await call(svc().experiment_status, job_id)


@mcp.tool()
async def experiment_logs(job_id: str, tail: int = 200, stream: str = "combined"):
    return await call(svc().experiment_logs, job_id, tail, stream)


@mcp.tool()
async def experiment_cancel(job_id: str):
    return await call(svc().experiment_cancel, job_id)


@mcp.tool()
async def experiment_list(
    instance_id: str | None = None, status: str | None = None, limit: int = 50
):
    return await call(svc().experiment_list, instance_id, status, limit)


@mcp.tool()
async def artifact_list(job_id: str):
    return await call(svc().artifact_list, job_id)


@mcp.tool()
async def artifact_read(job_id: str, path: str, max_bytes: int | None = None):
    return await call(svc().artifact_read, job_id, path, max_bytes)


@mcp.tool()
async def remote_exec(instance_id: str, command: str, timeout_seconds: int = 60):
    """Dangerous, disabled by default, bounded non-interactive debug command."""
    return await call(svc().remote_exec, instance_id, command, timeout_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    args = parser.parse_args()
    mcp.run(transport=args.transport)
