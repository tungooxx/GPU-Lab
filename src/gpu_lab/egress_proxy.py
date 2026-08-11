"""Small public-address-only HTTP CONNECT proxy for isolated external workers."""

import asyncio
import ipaddress
import os
import socket
from urllib.parse import urlsplit

MAX_HEADER = 65_536
LISTEN_PORT = int(os.environ.get("EGRESS_PROXY_PORT", "3128"))
UPSTREAM_CONNECT_TIMEOUT = float(os.environ.get("EGRESS_PROXY_CONNECT_TIMEOUT", "15"))


async def _public_target(host: str, port: int) -> tuple[str, int]:
    if port not in {80, 443}:
        raise ValueError("only public HTTP(S) egress is allowed")
    infos = await asyncio.wait_for(
        asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM),
        timeout=UPSTREAM_CONNECT_TIMEOUT,
    )
    addresses = {item[4][0] for item in infos}
    if not addresses or any(not ipaddress.ip_address(item).is_global for item in addresses):
        raise ValueError("private, loopback, and non-global destinations are blocked")
    return min(addresses), port


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65_536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    upstream_writer = None
    try:
        header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=15)
        if len(header) > MAX_HEADER:
            raise ValueError("proxy request header is too large")
        lines = header.decode("iso-8859-1").split("\r\n")
        method, target, version = lines[0].split(" ", 2)
        if method.upper() == "CONNECT":
            host, separator, port_text = target.rpartition(":")
            if not separator or not host:
                raise ValueError("CONNECT requires host:port")
            address, port = await _public_target(host.strip("[]"), int(port_text))
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(address, port), timeout=UPSTREAM_CONNECT_TIMEOUT
            )
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
        else:
            parsed = urlsplit(target)
            if method.upper() not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}:
                raise ValueError("unsupported HTTP proxy method")
            if parsed.scheme != "http" or not parsed.hostname:
                raise ValueError("plain proxy requests require an http URL")
            address, port = await _public_target(parsed.hostname, parsed.port or 80)
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(address, port), timeout=UPSTREAM_CONNECT_TIMEOUT
            )
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            filtered = [
                line
                for line in lines[1:]
                if line and not line.lower().startswith(("proxy-connection:", "connection:"))
            ]
            if not any(line.lower().startswith("host:") for line in filtered):
                filtered.append(f"Host: {parsed.netloc}")
            upstream_writer.write(
                (f"{method} {path} {version}\r\n" + "\r\n".join(filtered) + "\r\n\r\n").encode(
                    "iso-8859-1"
                )
            )
            await upstream_writer.drain()
        await asyncio.gather(
            _pipe(reader, upstream_writer),
            _pipe(upstream_reader, writer),
        )
    except (
        ValueError,
        OSError,
        UnicodeDecodeError,
        asyncio.IncompleteReadError,
        asyncio.LimitOverrunError,
        TimeoutError,
    ):
        if not writer.is_closing():
            writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()
    finally:
        if upstream_writer and not upstream_writer.is_closing():
            upstream_writer.close()
            await upstream_writer.wait_closed()


async def main() -> None:
    server = await asyncio.start_server(handle, "0.0.0.0", LISTEN_PORT, limit=MAX_HEADER)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
