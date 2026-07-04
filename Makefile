.PHONY: check lint format typecheck test install ingest serve eval

# `uv run --no-sync` uses the existing .venv without re-resolving deps.
# Run `make install` once after clone to populate it.

install:
	uv pip install -e ".[dev,eval]"

lint:
	uv run --no-sync ruff check src tests

format:
	uv run --no-sync ruff format src tests

typecheck:
	uv run --no-sync mypy src

test:
	uv run --no-sync pytest

check: lint format typecheck test

ingest:
	uv run --no-sync python -m rag_harness ingest

serve:
	uv run --no-sync uvicorn rag_harness.api.server:app --reload --host 0.0.0.0 --port 8000

eval:
	uv run --no-sync python -m rag_harness eval
