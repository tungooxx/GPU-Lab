# GPU Lab MCP

GPU Lab is a minimal, safe MCP experiment control plane for Vast.ai GPUs. It deliberately exposes structured lifecycle, repository, environment, job, log, and artifact operations rather than a general SSH shell.

## Run

```bash
cp .env.example .env
# set VAST_API_KEY and GPU_LAB_SSH_PRIVATE_KEY_PATH in .env
uv sync
uv run gpu-lab-mcp                         # stdio for local MCP clients
uv run gpu-lab-mcp --transport streamable-http # HTTP at /mcp (port 8000)
uv run gpu-lab-mcp --transport sse              # legacy SSE clients at /sse (port 8000)
```

For development, connect a local MCP client to the stdio process. For production, run the streamable HTTP transport behind authenticated HTTPS/private networking. The MCP gateway, not ChatGPT, holds provider and SSH credentials.

## Tools

`gpu_list`, `gpu_status`, `gpu_search`, `gpu_create`, `gpu_stop`, `gpu_destroy`, `repo_checkout`, `env_prepare`, `experiment_submit`, `experiment_status`, `experiment_logs`, `experiment_cancel`, `experiment_list`, `artifact_list`, `artifact_read`, and disabled-by-default `remote_exec`.

Use `gpu_search` before `gpu_create`; creation requires a specific offer rather than making an unbounded cost decision. `gpu_destroy` requires `confirmation="DESTROY"`.

## Local CLI

```bash
uv run gpu-lab gpu list
uv run gpu-lab gpu status vast_123
uv run gpu-lab experiment status exp_abc
uv run gpu-lab experiment logs exp_abc --tail 100

cd /workspace/GPU-Lab

GPU_LAB_ALLOWED_HOSTS='127.0.0.1:*,localhost:*,chucky-lab.com' \
uv run gpu-lab-mcp --transport streamable-http
```
```bash
cd ~/.ssh

ssh-keygen -t ed25519 \
  -f /root/.ssh/gpu_lab_ed25519 \
  -N "" \
  -C "gpu-lab-self-ssh"

cat /root/.ssh/gpu_lab_ed25519.pub >> /root/.ssh/authorized_keys

chmod 600 /root/.ssh/gpu_lab_ed25519
chmod 600 /root/.ssh/authorized_keys
```
## Security

The server never returns API keys or private keys, only allows repository and artifact paths under `GPU_LAB_REMOTE_ROOT`, refuses dirty checkouts, bounds logs/artifact reads, and does not enable `remote_exec` unless `GPU_LAB_ENABLE_REMOTE_EXEC=true`. SSH requires `GPU_LAB_SSH_KNOWN_HOSTS` by default. Vast hosts can change, so the explicitly named `GPU_LAB_SSH_ALLOW_UNVERIFIED_HOSTS=true` override is available for development only; production should pin host keys or use a trusted gateway network.

## Limitations

The MVP currently has one provider (Vast), SQLite state, tmux-based detached jobs, and a compact CLI. `artifact_download` and `experiment_summary` are intentionally not exposed yet; large files remain on the remote host. Vast endpoint shapes are normalized defensively but should be integration-tested against the account before production use.
