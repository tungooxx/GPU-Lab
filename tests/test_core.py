import pytest

from gpu_lab.config import Settings
from gpu_lab.db import Repository
from gpu_lab.errors import GPUError
from gpu_lab.models import Instance, Job
from gpu_lab.providers import VastProvider
from gpu_lab.service import GPUService
from gpu_lab.ssh import q


def test_quote_is_shell_safe():
    assert q("a; rm -rf /") == "'a; rm -rf /'"


def test_provider_normalizes_connection_metadata():
    item = VastProvider.normalize(
        {
            "id": 12,
            "actual_status": "running",
            "ssh_host": "host",
            "ssh_port": 123,
            "gpu_name": "A100",
            "num_gpus": 1,
            "dph_total": 1.2,
        }
    )
    assert item.id == "vast_12" and item.hostname == "host" and item.ssh_port == 123


def test_provider_prefers_direct_public_ssh_mapping():
    item = VastProvider.normalize(
        {
            "id": 12,
            "actual_status": "running",
            "ssh_host": "ssh3.vast.ai",
            "ssh_port": 30072,
            "public_ipaddr": "191.223.212.127",
            "ports": {"22/tcp": [{"HostPort": "31708"}]},
        }
    )
    assert item.hostname == "191.223.212.127"
    assert item.ssh_port == 31708


@pytest.mark.asyncio
async def test_offer_search_uses_model_substring(monkeypatch):
    provider = VastProvider("not-a-real-key")

    async def fake_request(*args, **kwargs):
        return {
            "offers": [
                {
                    "gpu_name": "NVIDIA A100-SXM4-80GB",
                    "gpu_ram": 81920,
                    "dph_total": 1.0,
                    "reliability": 0.99,
                    "verification": "verified",
                }
            ]
        }

    monkeypatch.setattr(provider, "_request", fake_request)
    offers = await provider.search_offers(gpu_name="A100", min_vram_gb=40, max_hourly_price=1.5)
    await provider.client.aclose()
    assert len(offers) == 1


@pytest.mark.asyncio
async def test_list_instances_uses_non_deprecated_path(monkeypatch):
    provider = VastProvider("not-a-real-key")
    captured = {}

    async def fake_request(method, path, **kwargs):
        captured["path"] = path
        return {"instances": []}

    monkeypatch.setattr(provider, "_request_v1", fake_request)
    assert await provider.list_instances() == []
    await provider.client.aclose()
    assert captured["path"] == "/instances"


def test_database_persists_models(tmp_path):
    db = Repository(tmp_path / "state.db")
    db.save_instance(Instance(id="vast_1", provider_instance_id="1"))
    db.save_job(
        Job(
            job_id="exp_a",
            instance_id="vast_1",
            repo_path="/workspace/gpu-lab/repos/a",
            command="python x.py",
        )
    )
    assert db.get_instance("vast_1") and db.get_job("exp_a")


@pytest.mark.asyncio
async def test_destroy_requires_confirmation(tmp_path):
    service = GPUService(Settings(gpu_lab_database_url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    with pytest.raises(GPUError, match="DESTROY"):
        await service.gpu_destroy("vast_1", "no")


@pytest.mark.asyncio
async def test_artifact_path_traversal_is_rejected(tmp_path):
    service = GPUService(Settings(gpu_lab_database_url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    service.repo.save_job(
        Job(
            job_id="exp_a",
            instance_id="vast_1",
            repo_path="/workspace/gpu-lab/repos/a",
            command="x",
        )
    )
    with pytest.raises(GPUError, match="escapes"):
        await service.artifact_read("exp_a", "../../secret")


@pytest.mark.asyncio
async def test_remote_exec_is_disabled(tmp_path):
    service = GPUService(
        Settings(
            gpu_lab_database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
            gpu_lab_enable_remote_exec=False,
        )
    )
    with pytest.raises(GPUError, match="REMOTE_EXEC"):
        await service.remote_exec("vast_1", "id")
