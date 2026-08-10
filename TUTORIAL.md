# Deploy GPU Lab MCP on Vast.ai with Codex

This guide gives a Codex session on a Vast.ai instance a precise, safe deployment task. It assumes the repository is in a private Git remote and that you have a domain or secure tunnel for the final MCP endpoint.

## Architecture

For a first test, it is fine to run the gateway on the existing GPU instance. For normal use, run it on a separate always-on low-cost VM: the gateway can then safely stop or destroy GPU workers without terminating itself.

```text
ChatGPT -- HTTPS MCP --> GPU Lab Gateway -- Vast API / SSH --> GPU workers
```

The gateway—not ChatGPT—stores the Vast API key and SSH private key.

## Before starting Codex

Have these ready, but never paste them into chat, Git, logs, or source code:

- A newly created Vast API key.
- An SSH private key that can access Vast instances.
- The private Git repository URL for this project.
- A public HTTPS domain, or a Secure MCP Tunnel, for ChatGPT access.

Rotate any secret that has been shared in a chat transcript.

## Prompt for Codex on Vast

Copy this into Codex after connecting to the Vast instance:

```text
Deploy the GPU Lab MCP repository as a persistent Streamable HTTP MCP server.

1. Clone <PRIVATE_GIT_REPOSITORY_URL> into /opt/gpu-lab-mcp. Do not put credentials in Git URLs.
2. Install uv if absent, then run `uv sync --no-dev` in /opt/gpu-lab-mcp.
3. Create /opt/gpu-lab-mcp/.env with values supplied through environment variables or an interactive secret mechanism. Never print any secret.
4. Set GPU_LAB_DATA_DIR=/opt/gpu-lab-mcp/data and GPU_LAB_DATABASE_URL=sqlite:////opt/gpu-lab-mcp/data/gpu_lab.db.
5. Set GPU_LAB_SSH_PRIVATE_KEY_PATH to the existing private-key file and set its permissions to 600.
6. Prefer GPU_LAB_SSH_KNOWN_HOSTS. Only if this is a development test and the Vast host key cannot be pinned, explicitly set GPU_LAB_SSH_ALLOW_UNVERIFIED_HOSTS=true.
7. Run `uv run pytest` and `uv run ruff check .`; fix any failures.
8. Create and enable a systemd service that runs `uv run gpu-lab-mcp --transport streamable-http` as an unprivileged gpu-lab user, with WorkingDirectory=/opt/gpu-lab-mcp and FASTMCP_HOST=127.0.0.1 and FASTMCP_PORT=8000.
9. Put an HTTPS reverse proxy or Secure MCP Tunnel in front of it. Do not expose port 8000 directly to the Internet.
10. Verify `curl http://127.0.0.1:8000/mcp` reaches the MCP server. Report the public HTTPS /mcp URL, service status, and test results, but no secrets.

Do not create or destroy any Vast instance. Do not enable remote_exec. Do not weaken SSH host-key validation except through the explicitly named development setting above.
```

Replace `<PRIVATE_GIT_REPOSITORY_URL>` before submitting the prompt.

## Configuration file

Codex should create `.env` from `.env.example`; it must remain untracked. A production-shaped configuration is:

```env
VAST_API_KEY=replace-with-secret
GPU_LAB_SSH_PRIVATE_KEY_PATH=/home/gpu-lab/.ssh/id_ed25519
GPU_LAB_DATA_DIR=/opt/gpu-lab-mcp/data
GPU_LAB_DATABASE_URL=sqlite:////opt/gpu-lab-mcp/data/gpu_lab.db
GPU_LAB_REMOTE_ROOT=/workspace/gpu-lab
GPU_LAB_ENABLE_REMOTE_EXEC=false
GPU_LAB_MAX_LOG_LINES=1000
GPU_LAB_MAX_TEXT_ARTIFACT_BYTES=1048576
GPU_LAB_SSH_TIMEOUT=20
GPU_LAB_SSH_KNOWN_HOSTS=/home/gpu-lab/.ssh/known_hosts
GPU_LAB_SSH_ALLOW_UNVERIFIED_HOSTS=false
GPU_LAB_ALLOWED_HOSTS=127.0.0.1:*,localhost:*,chucky-lab.com
```

Do not commit this file.

## systemd service

Use a dedicated service account. After Codex has installed the app, create `/etc/systemd/system/gpu-lab-mcp.service`:

```ini
[Unit]
Description=GPU Lab MCP Gateway
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=gpu-lab
Group=gpu-lab
WorkingDirectory=/opt/gpu-lab-mcp
Environment=PATH=/home/gpu-lab/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=FASTMCP_HOST=127.0.0.1
Environment=FASTMCP_PORT=8000
ExecStart=/home/gpu-lab/.local/bin/uv run gpu-lab-mcp --transport streamable-http
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-lab-mcp
sudo systemctl status gpu-lab-mcp
```

## HTTPS endpoint

ChatGPT needs a remote MCP server; a local or raw public HTTP endpoint is not appropriate. Choose one:

- Recommended for a private/testing deployment: use Secure MCP Tunnel.
- Production: use an HTTPS reverse proxy with a real domain and TLS certificate.

The public endpoint must end in `/mcp`, for example:

```text
https://gpu-lab.example.com/mcp
```

Do not expose the gateway without authentication and HTTPS. The current MVP does not include its own user authentication, so a private tunnel or an authenticated reverse proxy is strongly recommended.

## Connect ChatGPT

In ChatGPT web, enable Developer Mode, then go to **Settings or Workspace Settings → Apps → Create**. Enter the public HTTPS endpoint, use **Scan Tools**, and create the app. Start a new chat, select the GPU Lab app, and first call `gpu_list` or `gpu_status`.

Custom write-capable MCP apps currently require an eligible ChatGPT workspace plan and appropriate administrator/developer-mode permissions. See [OpenAI’s Developer Mode and MCP apps guide](https://help.openai.com/en/articles/12584461-developer-mode-and-full-mcp-connectors-in-chatgpt).

## Initial validation

Use these low-risk checks first:

```text
gpu_list()
gpu_status(instance_id="vast_47350073")
```

Only after they work should you use `repo_checkout`, `env_prepare`, and `experiment_submit`. Always require `confirmation="DESTROY"` for `gpu_destroy`.
