"""SQLite-backed cache for embedding vectors, keyed by content hash.

Prevents redundant OpenAI API calls when re-ingesting an unchanged corpus.
The cache is model-aware: a key encodes both the model name and the chunk text,
so switching models automatically produces cache misses for all entries.
"""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS embeddings (
    key  TEXT PRIMARY KEY,
    vec  TEXT NOT NULL
)
"""


class EmbeddingCache:
    """Persistent embedding cache backed by a local SQLite database."""

    def __init__(self, path: Path) -> None:
        """Open (or create) the cache database at *path*."""
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        logger.debug("embedding cache opened at %s", path)

    def get(self, key: str) -> list[float] | None:
        """Return the cached vector for *key*, or None on a miss."""
        row = self._conn.execute("SELECT vec FROM embeddings WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, vector: list[float]) -> None:
        """Store *vector* under *key*, overwriting any existing entry."""
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, vec) VALUES (?, ?)",
            (key, json.dumps(vector)),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    @staticmethod
    def make_key(model: str, text: str) -> str:
        """Return a deterministic cache key for a (model, text) pair."""
        return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()
