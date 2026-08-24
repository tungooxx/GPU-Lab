import pytest
import shlex

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
async def test_gpu_list_marks_provider_missing_instances_without_deleting_history(tmp_path):
    service = GPUService(Settings(gpu_lab_database_url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    stale = Instance(id="vast_old", provider_instance_id="old", status="running")
    current = Instance(id="vast_current", provider_instance_id="current", status="running")
    service.repo.save_instance(stale)

    class Provider:
        async def list_instances(self):
            return [current]

    service.provider = Provider()
    items = await service.gpu_list()

    assert {item["id"] for item in items} == {"vast_old", "vast_current"}
    assert next(item for item in items if item["id"] == "vast_old")["status"] == "provider_missing"
    assert service.repo.get_instance("vast_old").metadata["provider_visible"] is False


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


@pytest.mark.asyncio
async def test_missing_vast_instance_returns_a_domain_error(monkeypatch):
    provider = VastProvider("not-a-real-key")

    async def fake_request(*_args, **_kwargs):
        return {"instances": None}

    monkeypatch.setattr(provider, "_request", fake_request)
    with pytest.raises(GPUError, match="Vast instance 123 was not found"):
        await provider.get_instance("vast_123")
    await provider.client.aclose()


@pytest.mark.asyncio
async def test_vast_creation_requests_direct_ssh(monkeypatch):
    provider = VastProvider("not-a-real-key")
    captured = {}

    async def fake_request(method, path, **kwargs):
        captured.update(method=method, path=path, payload=kwargs["json"])
        return {"new_contract": 123}

    async def fake_get_instance(_instance_id):
        return Instance(id="vast_123", provider_instance_id="123")

    monkeypatch.setattr(provider, "_request", fake_request)
    monkeypatch.setattr(provider, "get_instance", fake_get_instance)
    await provider.create_instance("offer-1")
    await provider.client.aclose()
    assert captured["method"] == "PUT"
    assert captured["path"] == "/asks/offer-1"
    assert captured["payload"]["runtype"] == "ssh_direct"


@pytest.mark.asyncio
async def test_vast_start_requests_running_state_and_refreshes_instance(monkeypatch):
    provider = VastProvider("not-a-real-key")
    captured = {}

    async def fake_request(method, path, **kwargs):
        captured.update(method=method, path=path, payload=kwargs["json"])
        return {"success": True}

    async def fake_get_instance(instance_id):
        assert instance_id == "123"
        return Instance(id="vast_123", provider_instance_id="123", status="scheduling")

    monkeypatch.setattr(provider, "_request", fake_request)
    monkeypatch.setattr(provider, "get_instance", fake_get_instance)
    item = await provider.start_instance("vast_123")
    await provider.client.aclose()

    assert captured == {"method": "PUT", "path": "/instances/123", "payload": {"state": "running"}}
    assert item.status == "scheduling"


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
async def test_destroy_treats_provider_404_as_already_deleted_and_keeps_history(tmp_path):
    service = GPUService(Settings(gpu_lab_database_url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    service.repo.save_instance(Instance(id="vast_deleted", provider_instance_id="deleted", status="running"))

    class Provider:
        async def destroy_instance(self, _instance_id):
            raise GPUError("PROVIDER_NOT_FOUND", "Vast returned 404", details={"status_code": 404})

    service.provider = Provider()
    result = await service.gpu_destroy("vast_deleted", "DESTROY")

    assert result == {
        "instance_id": "vast_deleted",
        "status": "ALREADY_DELETED",
        "historical_record_retained": True,
    }
    saved = service.repo.get_instance("vast_deleted")
    assert saved.status == "destroyed"
    assert saved.metadata["provider_already_deleted"] is True


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


@pytest.mark.asyncio
async def test_remote_experiment_submission_quotes_the_tmux_pane_once(tmp_path):
    service = GPUService(Settings(gpu_lab_database_url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    instance = Instance(id="vast_1", provider_instance_id="1", gpu_model="RTX")
    service.repo.save_instance(instance)
    captured = {}

    class SSH:
        async def run(self, _instance, command, _timeout):
            captured["command"] = command
            return "12345\n", "", 0

    service.ssh = SSH()
    await service.experiment_submit(
        "vast_1",
        f"{service.settings.gpu_lab_remote_root}/repos/repo",
        "python run.py --label 'causal development'",
        env={"MODE": "causal development"},
        job_id="exp_quote_regression",
    )

    tmux_command = captured["command"].split(" && tmux display-message", 1)[0].rsplit(" && ", 1)[1]
    tokens = shlex.split(tmux_command)
    assert tokens[:5] == ["tmux", "new-session", "-d", "-s", "exp_quote_regression"]
    pane_command = tokens[5]
    assert pane_command.startswith("setsid sh -c ")
    assert "exit $code" in pane_command
    assert "stdout.log" in pane_command and "stderr.log" in pane_command
