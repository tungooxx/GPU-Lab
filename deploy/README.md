# Production reverse proxy

Run the GPU Lab container with `docker compose up -d --build`. It exposes the
MCP server only at `127.0.0.1:8000` on the host.

Install Caddy on the host and place `Caddyfile` at `/etc/caddy/Caddyfile`.
With `chucky-lab.com` DNS pointing at that host and ports 80/443 reachable,
Caddy obtains and renews TLS certificates and forwards `/mcp` unchanged to the
container. The public endpoint is `https://chucky-lab.com/mcp`.

Before making the endpoint available outside a private network, configure
authentication. This MCP server performs GPU lifecycle actions and must not be
made publicly callable without an authorization layer.
