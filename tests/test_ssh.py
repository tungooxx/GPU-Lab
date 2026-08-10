import pytest

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
