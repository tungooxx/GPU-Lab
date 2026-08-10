.PHONY: dev test lint run
dev:
	uv sync
test:
	uv run pytest
lint:
	uv run ruff check .
run:
	uv run gpu-lab-mcp
