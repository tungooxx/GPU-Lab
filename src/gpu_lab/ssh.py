import asyncio
import shlex
import time

import asyncssh

from .config import Settings
from .errors import GPUError
from .models import Instance


def q(value: str) -> str:
    return shlex.quote(value)


class SSHClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        # This cache is deliberately process-local. It improves repeated
        # status/log/debug calls, while the database remains the source of truth
        # for jobs and no credential material is persisted here.
        self._connections: dict[tuple[str, int, str, str], tuple[asyncssh.SSHClientConnection, float]] = {}
        self._connection_locks: dict[tuple[str, int, str, str], asyncio.Lock] = {}

    def _connection_key(self, instance: Instance) -> tuple[str, int, str, str]:
        return (
            instance.hostname or "",
            int(instance.ssh_port or 0),
            instance.ssh_user,
            str(self.settings.gpu_lab_ssh_private_key_path),
        )

    @staticmethod
    def _usable(connection: asyncssh.SSHClientConnection) -> bool:
        # AsyncSSH exposes ``is_closed()``, unlike asyncio transports' common
        # ``is_closing()`` spelling. Keeping the check on the public AsyncSSH
        # API avoids turning a cache hit into an internal error.
        return not connection.is_closed()

    def _discard(self, key: tuple[str, int, str, str], connection: asyncssh.SSHClientConnection | None = None) -> None:
        cached = self._connections.get(key)
        if not cached or (connection is not None and cached[0] is not connection):
            return
        self._connections.pop(key, None)
        cached[0].close()

    async def _connection(self, instance: Instance) -> tuple[tuple[str, int, str, str], asyncssh.SSHClientConnection]:
        key = self._connection_key(instance)
        now = time.monotonic()
        cached = self._connections.get(key)
        if cached and self._usable(cached[0]) and now - cached[1] < self.settings.gpu_lab_ssh_connection_idle_seconds:
            return key, cached[0]
        if cached:
            self._discard(key)
        lock = self._connection_locks.setdefault(key, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            cached = self._connections.get(key)
            if cached and self._usable(cached[0]) and now - cached[1] < self.settings.gpu_lab_ssh_connection_idle_seconds:
                return key, cached[0]
            if cached:
                self._discard(key)
            connection = await asyncssh.connect(
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
            )
            self._connections[key] = (connection, time.monotonic())
            return key, connection

    async def close(self) -> None:
        """Close cached SSH connections during an orderly service shutdown."""
        for key in list(self._connections):
            self._discard(key)
        self._connection_locks.clear()

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
            key, conn = await self._connection(instance)
            result = await conn.run(
                command, check=False, timeout=timeout or self.settings.gpu_lab_ssh_timeout
            )
            # Mark activity only after the remote command completed. A dropped
            # connection is discarded below and never reused for a later call.
            self._connections[key] = (conn, time.monotonic())
            return result.stdout, result.stderr, result.exit_status
        except (asyncssh.Error, OSError, TimeoutError) as exc:
            # Never retry a command here: a remote write may have succeeded even
            # when its response was lost. The next explicitly requested call
            # reconnects safely.
            if "key" in locals():
                self._discard(key, locals().get("conn"))
            raise GPUError("SSH_CONNECTION_FAILED", str(exc), True) from exc
