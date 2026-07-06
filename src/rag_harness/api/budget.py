"""Global daily request budget for the public demo.

In-memory, single-writer counter. Relies on Cloud Run ``max-instances=1``
(ADR-0010) so no external counter store is needed. Resets when the UTC
date rolls over. Thread-safe via a ``Lock`` so uvicorn worker threads
cannot double-book the counter.

The counter deliberately does not persist across container restarts. A
restart resets the count; ADR-0010 documents that worst-case restart
churn stays inside the monthly budget ceiling.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime
from threading import Lock


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
        incremented). Returns False if the cap is already reached — the
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
        production code — that would defeat the daily cap.
        """
        with self._lock:
            self._count = 0
            self._day = self._now().date()
