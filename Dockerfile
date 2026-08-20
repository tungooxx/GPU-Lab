FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.13 \
    && ln -sf "$(uv python find 3.13)" /usr/local/bin/python3.13
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra browser --no-dev --no-install-project \
    && uv run --no-sync playwright install --with-deps chromium
COPY src ./src
COPY research_bench ./research_bench
RUN uv sync --locked --extra browser --no-dev
ENV GPU_LAB_DATA_DIR=/data
ENV FASTMCP_HOST=0.0.0.0
ENV FASTMCP_PORT=8000
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "gpu-lab-mcp", "--transport", "streamable-http"]
