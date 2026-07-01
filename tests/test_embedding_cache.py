"""Unit tests for the SQLite-backed embedding cache."""

import hashlib
from pathlib import Path

import pytest

from rag_harness.ingest.embedding_cache import EmbeddingCache


@pytest.fixture()
def cache(tmp_path: Path) -> EmbeddingCache:
    c = EmbeddingCache(tmp_path / "test_cache.db")
    yield c
    c.close()


def test_cache_miss_returns_none(cache: EmbeddingCache) -> None:
    assert cache.get("nonexistent-key") is None


def test_set_and_get_roundtrip(cache: EmbeddingCache) -> None:
    vector = [0.1, 0.2, 0.3]
    cache.set("key1", vector)
    assert cache.get("key1") == vector


def test_set_overwrites_existing_entry(cache: EmbeddingCache) -> None:
    cache.set("key1", [0.1, 0.2])
    cache.set("key1", [0.9, 0.8])
    assert cache.get("key1") == [0.9, 0.8]


def test_multiple_keys_are_independent(cache: EmbeddingCache) -> None:
    cache.set("a", [1.0, 0.0])
    cache.set("b", [0.0, 1.0])
    assert cache.get("a") == [1.0, 0.0]
    assert cache.get("b") == [0.0, 1.0]


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "persist.db"
    c1 = EmbeddingCache(db_path)
    c1.set("key1", [0.5, 0.5])
    c1.close()

    c2 = EmbeddingCache(db_path)
    assert c2.get("key1") == [0.5, 0.5]
    c2.close()


def test_make_key_is_deterministic() -> None:
    k1 = EmbeddingCache.make_key("text-embedding-3-small", "hello world")
    k2 = EmbeddingCache.make_key("text-embedding-3-small", "hello world")
    assert k1 == k2


def test_make_key_is_model_aware() -> None:
    k1 = EmbeddingCache.make_key("model-a", "same text")
    k2 = EmbeddingCache.make_key("model-b", "same text")
    assert k1 != k2


def test_make_key_is_text_aware() -> None:
    k1 = EmbeddingCache.make_key("same-model", "text one")
    k2 = EmbeddingCache.make_key("same-model", "text two")
    assert k1 != k2


def test_make_key_is_sha256_hex() -> None:
    key = EmbeddingCache.make_key("model", "text")
    expected = hashlib.sha256(b"model\ntext").hexdigest()
    assert key == expected
    assert len(key) == 64
