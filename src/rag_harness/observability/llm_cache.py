"""SQLite-backed cache for LLM-judge responses.

Wraps the four LLM-as-judge calls (``faithfulness``, ``correctness``,
``answer_relevancy``, ``context_precision``). Never wraps ``generate()`` or
the corrective critic — those affect user-facing behaviour and returning a
stale generation answer to a live user is a footgun. Judges score existing
answers against reference material and are called only from the eval path.

**Rationale for caching judge calls specifically.** Two properties combine:

1. Judges are **near-deterministic** at ``temperature=0``. LLM inference is
   not strictly deterministic — same prompt can produce marginally different
   outputs across API-side changes — but for scoring purposes the drift is
   small and predictable.
2. A stale judge score is **negligible-cost** to the operation of the system.
   The score is a signal for the developer, not a live prediction; being one
   evaluator revision behind on a scoring rubric does not hurt an ablation
   comparison as long as every configuration in the same run sees the same
   cache. Cache lifetime is bounded by ``llm_cache.db`` on disk — delete the
   file to invalidate.

Key schema mirrors the embedding cache (SHA-256 keyed) so cache misses on
model or prompt changes are automatic without an explicit invalidation step.
"""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS llm_responses (
    key      TEXT PRIMARY KEY,
    response TEXT NOT NULL
)
"""


class LLMResponseCache:
    """Persistent LLM-judge response cache backed by a local SQLite database."""

    def __init__(self, path: Path) -> None:
        """Open (or create) the cache database at *path*."""
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        logger.debug("llm response cache opened at %s", path)

    def get(self, key: str) -> str | None:
        """Return the cached response for *key*, or None on a miss."""
        row = self._conn.execute(
            "SELECT response FROM llm_responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        try:
            return str(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError):
            # Corrupted row — treat as miss rather than crash
            logger.warning("llm cache row for key=%s was corrupted; ignoring", key[:12])
            return None

    def set(self, key: str, response: str) -> None:
        """Store *response* under *key*, overwriting any existing entry."""
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_responses (key, response) VALUES (?, ?)",
            (key, json.dumps(response)),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    @staticmethod
    def make_key(model: str, system_prompt: str, user_message: str) -> str:
        """Return a deterministic cache key for a (model, system, user) triple."""
        payload = f"{model}\n---SYSTEM---\n{system_prompt}\n---USER---\n{user_message}"
        return hashlib.sha256(payload.encode()).hexdigest()
