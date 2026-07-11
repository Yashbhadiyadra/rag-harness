"""Unit tests for the in-memory daily budget counter (ADR-0010)."""

from datetime import UTC, datetime, timedelta

from rag_harness.api.budget import DailyBudget


def test_check_and_increment_within_cap() -> None:
    """Every call under the cap should succeed."""
    budget = DailyBudget(cap=3)
    assert budget.check_and_increment() is True
    assert budget.check_and_increment() is True
    assert budget.check_and_increment() is True


def test_check_and_increment_over_cap_returns_false() -> None:
    """The (cap+1)th call must return False and not increment further."""
    budget = DailyBudget(cap=2)
    assert budget.check_and_increment() is True
    assert budget.check_and_increment() is True
    # Cap reached - next call rejected, remaining stays at 0
    assert budget.check_and_increment() is False
    assert budget.remaining() == 0
    # Repeated rejections do not corrupt state
    assert budget.check_and_increment() is False
    assert budget.remaining() == 0


def test_remaining_reflects_consumption() -> None:
    budget = DailyBudget(cap=5)
    assert budget.remaining() == 5
    budget.check_and_increment()
    budget.check_and_increment()
    assert budget.remaining() == 3


def test_reset_clears_counter() -> None:
    budget = DailyBudget(cap=2)
    budget.check_and_increment()
    budget.check_and_increment()
    assert budget.check_and_increment() is False
    budget.reset()
    assert budget.remaining() == 2
    assert budget.check_and_increment() is True


def test_rollover_at_utc_midnight() -> None:
    """When the injected clock crosses a UTC day boundary, the counter resets."""
    # A mutable clock we can advance from within the test.
    clock = {"now": datetime(2026, 7, 5, 23, 59, 0, tzinfo=UTC)}
    budget = DailyBudget(cap=2, now=lambda: clock["now"])

    budget.check_and_increment()
    budget.check_and_increment()
    assert budget.check_and_increment() is False, "cap should be reached at day boundary"

    # Advance past midnight - the next call sees a new UTC date
    clock["now"] = clock["now"] + timedelta(minutes=2)
    assert budget.check_and_increment() is True, "counter should roll over at UTC midnight"
    assert budget.remaining() == 1
