.PHONY: check lint format typecheck test install ingest serve eval

install:
	pip install -e ".[dev,eval]"

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

test:
	pytest

check: lint format typecheck test

ingest:
	python -m rag_harness ingest

serve:
	uvicorn rag_harness.api.server:app --reload --host 0.0.0.0 --port 8000

eval:
	python -m rag_harness eval
