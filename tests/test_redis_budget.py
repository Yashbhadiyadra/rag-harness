"""Tests for the Redis-backed daily budget and backend factory (ADR-0024).

The headline measurement: two RedisDailyBudget instances sharing one store
enforce ONE combined cap, not one per instance - which is what makes running
more than one service instance safe.
"""

from datetime import UTC, datetime, timedelta

import fakeredis
import pytest

from rag_harness.api.budget import (
    Budget,
    DailyBudget,
    RedisDailyBudget,
    build_daily_budget,
)


def _fake() -> fakeredis.FakeStrictRedis:
    return fakeredis.FakeStrictRedis(decode_responses=True)


def test_redis_budget_enforces_cap_on_one_instance() -> None:
    budget = RedisDailyBudget(cap=2, client=_fake())
    assert budget.check_and_increment() is True
    assert budget.check_and_increment() is True
    assert budget.check_and_increment() is False
    assert budget.remaining() == 0


def test_two_instances_share_one_cap() -> None:
    """The whole point of ADR-0024: a shared store means a single global cap.

    Two budgets on the same Redis must together allow only `cap` requests, not
    `cap` each - proving the counter is shared, not per-instance.
    """
    store = _fake()
    instance_a = RedisDailyBudget(cap=3, client=store)
    instance_b = RedisDailyBudget(cap=3, client=store)

    assert instance_a.check_and_increment() is True  # 1
    assert instance_b.check_and_increment() is True  # 2
    assert instance_a.check_and_increment() is True  # 3
    # Cap of 3 reached across BOTH instances - the 4th is rejected on either.
    assert instance_b.check_and_increment() is False
    assert instance_a.check_and_increment() is False
    assert instance_b.remaining() == 0


def test_redis_budget_rolls_over_at_utc_midnight() -> None:
    store = _fake()
    clock = {"now": datetime(2026, 7, 5, 23, 59, 0, tzinfo=UTC)}
    budget = RedisDailyBudget(cap=2, client=store, now=lambda: clock["now"])

    budget.check_and_increment()
    budget.check_and_increment()
    assert budget.check_and_increment() is False, "cap reached before midnight"

    clock["now"] = clock["now"] + timedelta(minutes=2)  # cross into the next UTC day
    assert budget.check_and_increment() is True, "new day, new key, counter resets"
    assert budget.remaining() == 1


def test_redis_budget_sets_ttl_to_midnight() -> None:
    store = _fake()
    # 6 hours before midnight -> TTL should be ~21600s, never the default -1.
    clock = {"now": datetime(2026, 7, 5, 18, 0, 0, tzinfo=UTC)}
    budget = RedisDailyBudget(cap=5, client=store, now=lambda: clock["now"])
    budget.check_and_increment()
    ttl = store.ttl(budget._key())
    assert 0 < ttl <= 6 * 3600


def test_redis_budget_fails_open_on_error() -> None:
    """A Redis outage must allow the request, not reject it (ADR-0024)."""

    class BrokenRedis:
        def incr(self, *a: object, **k: object) -> int:
            raise ConnectionError("redis down")

        def get(self, *a: object, **k: object) -> object:
            raise ConnectionError("redis down")

    budget = RedisDailyBudget(cap=1, client=BrokenRedis())
    assert budget.check_and_increment() is True  # fail open
    assert budget.remaining() == 1  # reports full on read failure


def test_redis_budget_reset_clears_counter() -> None:
    store = _fake()
    budget = RedisDailyBudget(cap=1, client=store)
    assert budget.check_and_increment() is True
    assert budget.check_and_increment() is False
    budget.reset()
    assert budget.check_and_increment() is True


def test_build_daily_budget_selects_backend() -> None:
    in_memory = build_daily_budget(cap=5, redis_url="")
    assert isinstance(in_memory, DailyBudget)
    # Protocol conformance holds for the in-memory backend.
    assert isinstance(in_memory, Budget)


def test_build_daily_budget_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """A redis:// URL yields a RedisDailyBudget without a live server."""
    import redis

    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, url, **kw: _fake()))
    budget = build_daily_budget(cap=5, redis_url="redis://localhost:6379/0")
    assert isinstance(budget, RedisDailyBudget)
    assert isinstance(budget, Budget)
