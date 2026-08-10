import shlex

import asyncssh

from .config import Settings
from .errors import GPUError
from .models import Instance


def q(value: str) -> str:
    return shlex.quote(value)


class SSHClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def run(
        self, instance: Instance, command: str, timeout: int | None = None
    ) -> tuple[str, str, int]:
        if (
            not instance.hostname
            or not instance.ssh_port
            or not self.settings.gpu_lab_ssh_private_key_path
        ):
            raise GPUError(
                "SSH_CONNECTION_FAILED",
                "Instance connection metadata or SSH key is unavailable",
                True,
            )
        if (
            not self.settings.gpu_lab_ssh_known_hosts
            and not self.settings.gpu_lab_ssh_allow_unverified_hosts
        ):
            raise GPUError(
                "SSH_HOST_KEY_POLICY_REQUIRED",
                "Configure GPU_LAB_SSH_KNOWN_HOSTS or explicitly set GPU_LAB_SSH_ALLOW_UNVERIFIED_HOSTS=true for dynamic hosts",
            )
        try:
            async with asyncssh.connect(
                instance.hostname,
                port=instance.ssh_port,
                username=instance.ssh_user,
                client_keys=[str(self.settings.gpu_lab_ssh_private_key_path)],
                known_hosts=(
                    str(self.settings.gpu_lab_ssh_known_hosts)
                    if self.settings.gpu_lab_ssh_known_hosts
                    else None
                ),
                connect_timeout=self.settings.gpu_lab_ssh_timeout,
            ) as conn:
                result = await conn.run(
                    command, check=False, timeout=timeout or self.settings.gpu_lab_ssh_timeout
                )
                return result.stdout, result.stderr, result.exit_status
        except (asyncssh.Error, OSError, TimeoutError) as exc:
            raise GPUError("SSH_CONNECTION_FAILED", str(exc), True) from exc
