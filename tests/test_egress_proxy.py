import pytest

from gpu_lab.egress_proxy import _public_target


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
async def test_egress_proxy_rejects_loopback_destinations(host):
    with pytest.raises(ValueError, match="non-global"):
        await _public_target(host, 443)


@pytest.mark.asyncio
async def test_egress_proxy_rejects_non_http_ports_before_connecting():
    with pytest.raises(ValueError, match="HTTP"):
        await _public_target("example.com", 8000)
