"""Unit tests for the SQLite LLM response cache and its metrics wiring."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_harness.observability.llm_cache import LLMResponseCache


@pytest.fixture()
def cache(tmp_path: Path):
    c = LLMResponseCache(tmp_path / "test_llm.db")
    yield c
    c.close()


# --- LLMResponseCache basics ---


def test_cache_miss_returns_none(cache: LLMResponseCache) -> None:
    assert cache.get("nonexistent") is None


def test_set_and_get_roundtrip(cache: LLMResponseCache) -> None:
    cache.set("k1", "0.85")
    assert cache.get("k1") == "0.85"


def test_set_overwrites_existing_entry(cache: LLMResponseCache) -> None:
    cache.set("k1", "0.50")
    cache.set("k1", "0.95")
    assert cache.get("k1") == "0.95"


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    p = tmp_path / "persist.db"
    c1 = LLMResponseCache(p)
    c1.set("k1", "0.75")
    c1.close()

    c2 = LLMResponseCache(p)
    assert c2.get("k1") == "0.75"
    c2.close()


def test_make_key_includes_model_system_and_user() -> None:
    k1 = LLMResponseCache.make_key("gpt-4o-mini", "sys", "usr")
    k2 = LLMResponseCache.make_key("gpt-4o-mini", "sys", "usr-different")
    k3 = LLMResponseCache.make_key("gpt-4o-mini", "sys-different", "usr")
    k4 = LLMResponseCache.make_key("different-model", "sys", "usr")
    # Every field affects the key
    assert len({k1, k2, k3, k4}) == 4


def test_make_key_is_deterministic() -> None:
    a = LLMResponseCache.make_key("gpt-4o-mini", "sys", "usr")
    b = LLMResponseCache.make_key("gpt-4o-mini", "sys", "usr")
    assert a == b
    assert len(a) == 64  # SHA-256 hex


def test_cache_corrupted_row_treated_as_miss(cache: LLMResponseCache) -> None:
    # Simulate corruption by writing raw non-JSON directly
    cache._conn.execute(
        "INSERT OR REPLACE INTO llm_responses (key, response) VALUES (?, ?)",
        ("k-corrupt", "this-is-not-valid-json"),
    )
    cache._conn.commit()
    assert cache.get("k-corrupt") is None


# --- Integration with _llm_score ---


def _mock_client_returning(score: str) -> MagicMock:
    """Return a mock AsyncOpenAI client whose async .create returns *score*."""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = score
    resp.usage = MagicMock(prompt_tokens=50, completion_tokens=3)
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


def test_llm_score_skips_api_on_cache_hit(tmp_path: Path) -> None:
    from rag_harness.evaluation import metrics

    cache_path = tmp_path / "hit.db"
    cache = LLMResponseCache(cache_path)
    # Pre-populate cache
    key = LLMResponseCache.make_key("gpt-4o-mini", "sys prompt", "user msg")
    cache.set(key, "0.42")
    cache.close()

    mock_client = _mock_client_returning("0.99")  # would return this on API call

    with (
        patch("rag_harness.evaluation.metrics.settings.llm_cache_enabled", True),
        patch("rag_harness.evaluation.metrics.settings.llm_cache_path", str(cache_path)),
        patch("rag_harness.evaluation.metrics._client", mock_client),
        # Force _get_cache to reopen against the temp path
        patch("rag_harness.evaluation.metrics._cache", None),
    ):
        score = metrics._llm_score("sys prompt", "user msg")

    # Cache hit — got the 0.42 from the DB, NOT the API's 0.99
    assert score == 0.42
    mock_client.chat.completions.create.assert_not_called()


def test_llm_score_writes_to_cache_on_miss(tmp_path: Path) -> None:
    from rag_harness.evaluation import metrics

    cache_path = tmp_path / "miss.db"
    mock_client = _mock_client_returning("0.77")

    with (
        patch("rag_harness.evaluation.metrics.settings.llm_cache_enabled", True),
        patch("rag_harness.evaluation.metrics.settings.llm_cache_path", str(cache_path)),
        patch("rag_harness.evaluation.metrics._client", mock_client),
        patch("rag_harness.evaluation.metrics._cache", None),
    ):
        score = metrics._llm_score("sys", "usr")
        assert score == 0.77
        mock_client.chat.completions.create.assert_called_once()

    # Second call with same prompts uses the freshly-written cache entry;
    # no additional API call.
    with (
        patch("rag_harness.evaluation.metrics.settings.llm_cache_enabled", True),
        patch("rag_harness.evaluation.metrics.settings.llm_cache_path", str(cache_path)),
        patch("rag_harness.evaluation.metrics._client", mock_client),
        patch("rag_harness.evaluation.metrics._cache", None),
    ):
        score2 = metrics._llm_score("sys", "usr")
        assert score2 == 0.77
    # Total API call count across both invocations is still exactly 1
    assert mock_client.chat.completions.create.call_count == 1


def test_llm_score_bypasses_cache_when_disabled(tmp_path: Path) -> None:
    from rag_harness.evaluation import metrics

    mock_client = _mock_client_returning("0.88")

    with (
        patch("rag_harness.evaluation.metrics.settings.llm_cache_enabled", False),
        patch("rag_harness.evaluation.metrics._client", mock_client),
        patch("rag_harness.evaluation.metrics._cache", None),
    ):
        metrics._llm_score("sys", "usr")
        metrics._llm_score("sys", "usr")

    # Both calls hit the API; nothing is cached
    assert mock_client.chat.completions.create.call_count == 2


def test_llm_score_cache_hit_records_no_token_usage(tmp_path: Path) -> None:
    """Cache hits must not record TokenUsage — no tokens were consumed."""
    from rag_harness.evaluation import metrics
    from rag_harness.observability.usage import collect_usage

    cache_path = tmp_path / "hit_no_usage.db"
    cache = LLMResponseCache(cache_path)
    key = LLMResponseCache.make_key("gpt-4o-mini", "sys", "usr")
    cache.set(key, "0.55")
    cache.close()

    with (
        patch("rag_harness.evaluation.metrics.settings.llm_cache_enabled", True),
        patch("rag_harness.evaluation.metrics.settings.llm_cache_path", str(cache_path)),
        patch("rag_harness.evaluation.metrics._cache", None),
    ):
        with collect_usage() as usage_list:
            metrics._llm_score("sys", "usr")

    assert usage_list == []
