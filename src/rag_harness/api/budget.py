"""Global daily request budget for the public demo.

In-memory, single-writer counter. Relies on Cloud Run ``max-instances=1``
(ADR-0010) so no external counter store is needed. Resets when the UTC
date rolls over. Thread-safe via a ``Lock`` so uvicorn worker threads
cannot double-book the counter.

The counter deliberately does not persist across container restarts. A
restart resets the count; ADR-0010 documents that worst-case restart
churn stays inside the monthly budget ceiling.
"""

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from threading import Lock
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Budget(Protocol):
    """A daily request cap. Implemented in-memory (single instance) or over Redis.

    ``DailyCapMiddleware`` depends only on this surface, so the two backends
    are interchangeable and selected at startup by :func:`build_daily_budget`.
    """

    def check_and_increment(self) -> bool: ...
    def remaining(self) -> int: ...
    def reset(self) -> None: ...


class DailyBudget:
    """A UTC-day request cap with lazy roll-over.

    Parameters
    ----------
    cap:
        Maximum requests allowed per UTC day.
    now:
        Injectable clock, primarily for tests. Defaults to
        ``datetime.now(UTC)``.
    """

    def __init__(
        self,
        cap: int,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._cap = cap
        self._now = now or (lambda: datetime.now(UTC))
        self._lock = Lock()
        self._count = 0
        self._day: date = self._now().date()

    def _maybe_roll_over(self, today: date) -> None:
        """Reset the counter if we've crossed a UTC-day boundary."""
        if today != self._day:
            self._day = today
            self._count = 0

    def check_and_increment(self) -> bool:
        """Consume one slot atomically.

        Returns True if the request is allowed (and the counter is
        incremented). Returns False if the cap is already reached - the
        counter is not incremented in that case.
        """
        with self._lock:
            today = self._now().date()
            self._maybe_roll_over(today)
            if self._count >= self._cap:
                return False
            self._count += 1
            return True

    def remaining(self) -> int:
        """Slots left in the current UTC day."""
        with self._lock:
            today = self._now().date()
            self._maybe_roll_over(today)
            return max(0, self._cap - self._count)

    def reset(self) -> None:
        """Clear the counter and re-anchor to the current UTC day.

        Used by tests via the conftest autouse fixture. Never call from
        production code - that would defeat the daily cap.
        """
        with self._lock:
            self._count = 0
            self._day = self._now().date()


class RedisDailyBudget:
    """A UTC-day request cap backed by a shared Redis counter (ADR-0024).

    Lets multiple instances enforce ONE global daily cap. The counter lives at
    a per-UTC-date key, so roll-over happens naturally when the date changes;
    a TTL to the next midnight keeps stale keys from accumulating.

    Fails OPEN: if Redis is unreachable, requests are allowed and the error is
    logged. A governance-store outage degrades rate-limiting, not the service
    (ADR-0024). The per-key/per-IP limiter, sharing the same store, is the
    backstop against abuse.

    Parameters
    ----------
    cap:
        Maximum requests allowed per UTC day, across all instances.
    client:
        A Redis client (``redis.Redis``). Injectable so tests pass a fake.
    now:
        Injectable clock, primarily for tests. Defaults to ``datetime.now(UTC)``.
    key_prefix:
        Namespace for the counter key.
    """

    def __init__(
        self,
        cap: int,
        client: Any,
        now: Callable[[], datetime] | None = None,
        key_prefix: str = "rag_harness:daily_budget",
    ) -> None:
        self._cap = cap
        self._redis = client
        self._now = now or (lambda: datetime.now(UTC))
        self._prefix = key_prefix

    def _key(self) -> str:
        return f"{self._prefix}:{self._now().date().isoformat()}"

    def _seconds_until_midnight(self) -> int:
        now = self._now()
        tomorrow = datetime.combine(now.date(), datetime.min.time(), tzinfo=UTC) + timedelta(days=1)
        return max(1, int((tomorrow - now).total_seconds()))

    def check_and_increment(self) -> bool:
        """Consume one slot atomically across instances via Redis INCR.

        The first increment of the day sets the key's TTL. A count over the cap
        returns False (rejected requests still increment, which only pushes an
        already-exceeded counter higher - harmless and resets at midnight).
        """
        try:
            key = self._key()
            count = int(self._redis.incr(key))
            if count == 1:
                self._redis.expire(key, self._seconds_until_midnight())
            return count <= self._cap
        except Exception:
            logger.warning("redis daily-cap check failed; failing open", exc_info=True)
            return True

    def remaining(self) -> int:
        """Slots left in the current UTC day; reports full on a Redis error."""
        try:
            raw = self._redis.get(self._key())
            count = int(raw) if raw is not None else 0
            return max(0, self._cap - count)
        except Exception:
            logger.warning("redis daily-cap read failed; reporting full", exc_info=True)
            return self._cap

    def reset(self) -> None:
        """Delete the current day's counter. Used by tests; never in production."""
        try:
            self._redis.delete(self._key())
        except Exception:
            logger.warning("redis daily-cap reset failed", exc_info=True)


def build_daily_budget(cap: int, redis_url: str = "") -> Budget:
    """Return the configured daily-budget backend (ADR-0024).

    An empty ``redis_url`` yields the in-memory :class:`DailyBudget` (the
    single-instance default). A ``redis://`` URL yields a shared
    :class:`RedisDailyBudget`; ``redis`` is imported lazily so the base install
    stays free of the dependency unless scale-out is configured.
    """
    if redis_url:
        import redis  # lazy: only needed when scale-out is on ([redis] extra)

        client = redis.Redis.from_url(redis_url, decode_responses=True)
        return RedisDailyBudget(cap=cap, client=client)
    return DailyBudget(cap=cap)
