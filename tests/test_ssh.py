import pytest
from types import SimpleNamespace

from gpu_lab.config import Settings
from gpu_lab.errors import GPUError
from gpu_lab.models import Instance
from gpu_lab.ssh import SSHClient


@pytest.mark.asyncio
async def test_ssh_requires_explicit_host_key_policy(tmp_path):
    client = SSHClient(
        Settings(
            gpu_lab_database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
            gpu_lab_ssh_private_key_path=tmp_path / "key",
            gpu_lab_ssh_allow_unverified_hosts=False,
        )
    )
    instance = Instance(id="vast_1", provider_instance_id="1", hostname="host", ssh_port=22)
    with pytest.raises(GPUError, match="KNOWN_HOSTS"):
        await client.run(instance, "true")


@pytest.mark.asyncio
async def test_ssh_reuses_live_connection_for_same_instance(monkeypatch, tmp_path):
    opened = []

    class FakeConnection:
        def __init__(self):
            self.closed = False

        def is_closing(self):
            return self.closed

        def close(self):
            self.closed = True

        async def run(self, command, check, timeout):
            return SimpleNamespace(stdout=f"ran:{command}", stderr="", exit_status=0)

    async def connect(*args, **kwargs):
        connection = FakeConnection()
        opened.append(connection)
        return connection

    monkeypatch.setattr("gpu_lab.ssh.asyncssh.connect", connect)
    client = SSHClient(
        Settings(
            gpu_lab_database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
            gpu_lab_ssh_private_key_path=tmp_path / "key",
            gpu_lab_ssh_allow_unverified_hosts=True,
        )
    )
    instance = Instance(id="vast_1", provider_instance_id="1", hostname="host", ssh_port=22)
    assert (await client.run(instance, "one"))[0] == "ran:one"
    assert (await client.run(instance, "two"))[0] == "ran:two"
    assert len(opened) == 1
    await client.close()
    assert opened[0].closed is True
