FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev
COPY src ./src
ENV GPU_LAB_DATA_DIR=/data
ENV FASTMCP_HOST=0.0.0.0
ENV FASTMCP_PORT=8000
EXPOSE 8000
CMD ["uv", "run", "gpu-lab-mcp", "--transport", "streamable-http"]
