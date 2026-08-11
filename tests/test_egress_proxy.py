import asyncio

import pytest

import gpu_lab.egress_proxy as proxy


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
async def test_egress_proxy_rejects_loopback_destinations(host):
    with pytest.raises(ValueError, match="non-global"):
        await proxy._public_target(host, 443)


@pytest.mark.asyncio
async def test_egress_proxy_rejects_non_http_ports_before_connecting():
    with pytest.raises(ValueError, match="HTTP"):
        await proxy._public_target("example.com", 8000)


@pytest.mark.asyncio
async def test_egress_proxy_bounds_dns_resolution(monkeypatch):
    class SlowLoop:
        async def getaddrinfo(self, *_args, **_kwargs):
            await asyncio.sleep(1)

    monkeypatch.setattr(proxy, "UPSTREAM_CONNECT_TIMEOUT", 0.001)
    monkeypatch.setattr(proxy.asyncio, "get_running_loop", lambda: SlowLoop())

    with pytest.raises(TimeoutError):
        await proxy._public_target("example.com", 443)
