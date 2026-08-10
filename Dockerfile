FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml README.md ./
RUN uv sync --no-dev
COPY src ./src
ENV GPU_LAB_DATA_DIR=/data
CMD ["uv", "run", "gpu-lab-mcp", "--transport", "streamable-http"]
