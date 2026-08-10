#!/usr/bin/env sh
set -eu
command -v git >/dev/null || { apt-get update && apt-get install -y --no-install-recommends git curl tmux rsync python3-venv; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p /workspace/gpu-lab/{repos,jobs,artifacts,cache,bootstrap}
nvidia-smi || true
python3 --version
df -h /workspace || true
free -h || true
