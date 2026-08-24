import json
from typing import Any, Protocol

import httpx

from .errors import GPUError
from .models import Instance


class GPUProvider(Protocol):
    async def search_offers(self, **filters: Any) -> list[dict]: ...
    async def create_instance(self, offer_id: str, **options: Any) -> Instance: ...
    async def get_instance(self, instance_id: str) -> Instance: ...
    async def list_instances(self) -> list[Instance]: ...
    async def start_instance(self, instance_id: str) -> Instance: ...
    async def stop_instance(self, instance_id: str) -> None: ...
    async def destroy_instance(self, instance_id: str) -> None: ...


class VastProvider:
    """Vast API v0 adapter. Raw responses are retained in Instance.metadata."""

    base_url = "https://console.vast.ai/api/v0"
    v1_base_url = "https://console.vast.ai/api/v1"

    def __init__(self, api_key: str):
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=30, follow_redirects=True
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self.client.request(method, f"{self.base_url}{path}", **kwargs)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise GPUError(
                "PROVIDER_API_ERROR", str(exc), isinstance(exc, httpx.TimeoutException)
            ) from exc
        if isinstance(data, dict) and data.get("success") is False:
            raise GPUError(
                "PROVIDER_API_ERROR", data.get("msg", data.get("error", "Vast request failed"))
            )
        return data

    async def _request_v1(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self.client.request(method, f"{self.v1_base_url}{path}", **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise GPUError(
                "PROVIDER_API_ERROR", str(exc), isinstance(exc, httpx.TimeoutException)
            ) from exc

    @staticmethod
    def normalize(raw: dict) -> Instance:
        ident = str(raw.get("id", raw.get("contract_id")))
        ssh_port = raw.get("ssh_port")
        ports = raw.get("ports") or {}
        if isinstance(ports, dict):
            mapped = ports.get("22/tcp") or []
            if mapped and isinstance(mapped[0], dict):
                ssh_port = mapped[0].get("HostPort") or ssh_port
        hostname = raw.get("public_ipaddr") or raw.get("ssh_host")
        return Instance(
            id=f"vast_{ident}",
            provider_instance_id=ident,
            status=str(raw.get("actual_status", raw.get("status", "unknown"))),
            label=raw.get("label"),
            hostname=hostname,
            ssh_port=int(ssh_port) if ssh_port is not None else None,
            ssh_user=raw.get("ssh_user", "root"),
            gpu_model=raw.get("gpu_name"),
            gpu_count=raw.get("num_gpus"),
            hourly_price=raw.get("dph_total") or raw.get("dph_base"),
            metadata={"vast": raw},
        )

    async def search_offers(self, **filters: Any) -> list[dict]:
        # Vast's documented search endpoint accepts a JSON filter in q.
        query: dict[str, Any] = {"rentable": {"eq": True}}
        if filters.get("min_vram_gb"):
            query["gpu_ram"] = {"gte": int(filters["min_vram_gb"]) * 1000}
        data = await self._request("GET", "/bundles/", params={"q": json.dumps(query)})
        offers = data.get("offers", data if isinstance(data, list) else [])
        maximum = filters.get("max_hourly_price")
        candidates = [
            o
            for o in offers
            if maximum is None or float(o.get("dph_total", o.get("dph_base", 999))) <= maximum
        ]
        if gpu_name := filters.get("gpu_name"):
            candidates = [
                o
                for o in candidates
                if gpu_name.casefold() in str(o.get("gpu_name", "")).casefold()
            ]
        return sorted(
            candidates,
            key=lambda o: (
                str(o.get("verification", "")).casefold() not in {"verified", "verified_user"},
                -float(o.get("reliability", 0)),
                float(o.get("dph_total", 999)),
            ),
        )

    async def create_instance(self, offer_id: str, **options: Any) -> Instance:
        payload = {
            "image": options.get("image") or "vastai/base-image:@vastai-automatic-tag",
            "disk": options.get("disk_gb", 100),
            # Vast documents ``ssh_direct`` for a directly reachable SSH
            # instance. The remote executor requires that connectivity.
            "runtype": "ssh_direct",
            "label": options.get("label"),
        }
        data = await self._request(
            "PUT", f"/asks/{offer_id}", json={k: v for k, v in payload.items() if v is not None}
        )
        return await self.get_instance(str(data["new_contract"]))

    async def get_instance(self, instance_id: str) -> Instance:
        ident = instance_id.removeprefix("vast_")
        data = await self._request("GET", f"/instances/{ident}")
        raw = data.get("instances", data) if isinstance(data, dict) else data
        if not isinstance(raw, dict):
            raise GPUError("INSTANCE_NOT_FOUND", f"Vast instance {ident} was not found")
        return self.normalize(raw)

    async def list_instances(self) -> list[Instance]:
        data = await self._request_v1("GET", "/instances")
        return [
            self.normalize(i) for i in data.get("instances", data if isinstance(data, list) else [])
        ]

    async def start_instance(self, instance_id: str) -> Instance:
        ident = instance_id.removeprefix("vast_")
        # Vast manages both stop and start through its documented instance
        # lifecycle endpoint; the subsequent read returns fresh connection
        # metadata if the provider has already assigned it.
        await self._request("PUT", f"/instances/{ident}", json={"state": "running"})
        return await self.get_instance(ident)

    async def stop_instance(self, instance_id: str) -> None:
        await self._request(
            "PUT", f"/instances/{instance_id.removeprefix('vast_')}", json={"state": "stopped"}
        )

    async def destroy_instance(self, instance_id: str) -> None:
        await self._request("DELETE", f"/instances/{instance_id.removeprefix('vast_')}")
