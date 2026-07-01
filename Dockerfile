# Stage 1 — build: install dependencies into an isolated prefix
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install ".[eval]"


# Stage 2 — runtime: copy only what's needed
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ src/

# Chroma and embedding cache live on a mounted volume at runtime
VOLUME ["/app/chroma_db", "/app/embedding_cache.db"]

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "rag_harness.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
