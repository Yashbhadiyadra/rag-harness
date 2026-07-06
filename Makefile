.PHONY: check lint format typecheck test install ingest serve eval docker-build docker-run metrics-page

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

# --- Docker (ADR-0010) --------------------------------------------------
# chroma_db/ is baked into the image; the target runs `make ingest` first
# only if the index directory is missing, so repeat builds are fast.

docker-build:
	@test -d chroma_db || (echo "chroma_db/ missing — running make ingest first" && $(MAKE) ingest)
	docker build -t rag-harness:local .

docker-run:
	docker run --rm -p 8080:8080 -e PORT=8080 --env-file .env rag-harness:local

# --- Metrics page (ADR-0010) --------------------------------------------
# Regenerates docs/metrics/index.html from evals/history/runs.jsonl.
# Called by CI on every nightly eval and every release.

metrics-page:
	uv run --no-sync python -m scripts.render_metrics_page
