FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra literature --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --extra literature
EXPOSE 8010
CMD ["uv", "run", "--no-sync", "python", "-m", "gpu_lab.literature_worker"]
