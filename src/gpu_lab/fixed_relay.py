"""Fixed-target HTTP relay used to keep untrusted workers off the MCP network."""

import http.client
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TARGET_HOST = os.environ.get("RELAY_TARGET_HOST", "")
TARGET_PORT = int(os.environ.get("RELAY_TARGET_PORT", "0"))
LISTEN_PORT = int(os.environ.get("RELAY_LISTEN_PORT", "0"))
MAX_REQUEST_BODY = 1_048_576
CLIENT_READ_TIMEOUT = 30
_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(CLIENT_READ_TIMEOUT)

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def _forward(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "invalid Content-Length")
            return
        if length < 0:
            self.send_error(400, "invalid Content-Length")
            return
        if length > MAX_REQUEST_BODY:
            self.send_error(413, "request body exceeds 1 MiB relay limit")
            return
        try:
            body = self.rfile.read(length) if length else None
        except TimeoutError:
            self.send_error(408, "request body timeout")
            return
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in _HOP_HEADERS and name.lower() != "host"
        }
        connection = http.client.HTTPConnection(TARGET_HOST, TARGET_PORT, timeout=14_400)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() not in _HOP_HEADERS and name.lower() != "content-length":
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (OSError, http.client.HTTPException):
            self.send_error(502, "worker unavailable")
        finally:
            connection.close()

    def log_message(self, format_string: str, *args) -> None:
        print(f"relay {self.client_address[0]} {format_string % args}", flush=True)


if __name__ == "__main__":
    if not TARGET_HOST or TARGET_PORT <= 0 or LISTEN_PORT <= 0:
        raise RuntimeError("relay target and listen environment variables are required")
    ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), RelayHandler).serve_forever()
