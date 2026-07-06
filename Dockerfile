# Stage 1 — build: install runtime deps into an isolated prefix.
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml .
COPY src/ src/

# `[eval]` is included so /query can be judged post-hoc if evaluation
# ever needs to target the deployed instance. The [rerank] and
# [observability] extras are deliberately omitted — the deployed service
# uses the default 'dense' strategy and does not ship Phoenix alongside
# (see ADR-0009 / ADR-0010). Skipping them keeps the runtime image slim
# (avoids ~500 MB of PyTorch and ~200 MB of Phoenix).
RUN pip install --no-cache-dir --prefix=/install ".[eval]"


# Stage 2 — runtime: slim image with the Chroma index baked in.
#
# chroma_db/ is a BUILD INPUT. Produce it with `make ingest` before
# building this image, or `make docker-build` which does both. The image
# build fails fast if chroma_db/ is missing — that is the correct
# failure mode; a runtime image without the index is broken. See
# ADR-0010 for the bake-into-image rationale.
FROM python:3.12-slim

# Non-root user for defense in depth.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ src/
COPY chroma_db/ /app/chroma_db/

# Own everything under /app so the non-root user can read the index and
# write the process's own working files at runtime.
RUN chown -R app:app /app

USER app

ENV PYTHONUNBUFFERED=1

# Cloud Run injects PORT at container start; 8000 is the local-dev default.
# Shell form of CMD is required so ${PORT} interpolates at runtime.
# `exec` replaces the shell with uvicorn so SIGTERM (used by Cloud Run for
# graceful shutdown) reaches uvicorn as PID 1.
EXPOSE 8000
CMD exec uvicorn rag_harness.api.server:app --host 0.0.0.0 --port ${PORT:-8000}
