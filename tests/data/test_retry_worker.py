"""Tests for :class:`RetryWorker`.

Drives the worker via :meth:`tick` so we don't depend on asyncio
scheduling. The lifecycle (start/stop) is exercised in a separate
test that sleeps minimally.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.data.retry_queue import BackoffPolicy, RetryQueue, RetryStatus
from src.data.retry_worker import (
    PermanentRetryFailure,
    RetryWorker,
    TransientRetryFailure,
)


def _queue(tmp_path: Path, **kwargs: object) -> RetryQueue:
    return RetryQueue(tmp_path / "retry.json", **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_tick_removes_entries_on_success(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    q.enqueue("AAPL", ("e",))
    seen: list[str] = []

    async def fetch(symbol: str) -> None:
        seen.append(symbol)

    worker = RetryWorker(q, fetch, interval_seconds=0.1)
    processed = await worker.tick()
    assert processed == 1
    assert seen == ["AAPL"]
    assert q.all_entries() == ()


@pytest.mark.asyncio
async def test_tick_reschedules_on_transient_failure(tmp_path: Path) -> None:
    q = _queue(
        tmp_path,
        backoff=BackoffPolicy(initial_seconds=60.0, max_seconds=600.0),
        max_attempts=5,
    )
    q.enqueue("AAPL", ("first",))

    async def fetch(_symbol: str) -> None:
        raise TransientRetryFailure(("fmp: rate-limited",))

    worker = RetryWorker(q, fetch, interval_seconds=0.1)
    await worker.tick()
    entry = q.all_entries()[0]
    assert entry.status is RetryStatus.PENDING
    assert entry.attempts == 1
    assert entry.last_errors == ("fmp: rate-limited",)


@pytest.mark.asyncio
async def test_tick_forwards_retry_after_hint_into_reschedule(
    tmp_path: Path,
) -> None:
    """A TransientRetryFailure carrying a 600s Retry-After hint must push
    the next-attempt past the default 60s backoff. Closes the second-attempt
    half of B-10 (the first-attempt enqueue path was already covered)."""
    from datetime import timedelta

    q = _queue(
        tmp_path,
        backoff=BackoffPolicy(initial_seconds=60.0, max_seconds=3600.0),
        max_attempts=5,
    )
    when = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    q.enqueue("AAPL", ("first",), now=when)

    async def fetch(_symbol: str) -> None:
        raise TransientRetryFailure(
            ("fmp: rate-limited",), retry_after_seconds=600.0
        )

    worker = RetryWorker(q, fetch, interval_seconds=0.1)
    await worker.tick()
    entry = q.all_entries()[0]
    # next_attempt_at - last_error_at should be ≥ 600s.
    assert entry.last_error_at is not None
    gap = entry.next_attempt_at - entry.last_error_at
    assert gap >= timedelta(seconds=600)


@pytest.mark.asyncio
async def test_tick_marks_permanent_failure_in_one_shot(tmp_path: Path) -> None:
    q = _queue(tmp_path, max_attempts=5)
    q.enqueue("AAPL", ("e",))

    async def fetch(_symbol: str) -> None:
        raise PermanentRetryFailure(("auth failed",))

    worker = RetryWorker(q, fetch, interval_seconds=0.1)
    await worker.tick()
    entry = q.all_entries()[0]
    assert entry.status is RetryStatus.PERMANENTLY_FAILED


@pytest.mark.asyncio
async def test_tick_continues_past_unexpected_exceptions(tmp_path: Path) -> None:
    """Bug in retry fn for one symbol must not strand the worker."""
    q = _queue(tmp_path)
    q.enqueue("BAD", ("e",))
    q.enqueue("GOOD", ("e",))
    handled: list[str] = []

    async def fetch(symbol: str) -> None:
        if symbol == "BAD":
            raise RuntimeError("boom")
        handled.append(symbol)

    worker = RetryWorker(q, fetch, interval_seconds=0.1)
    await worker.tick()
    # Good entry succeeded.
    assert "GOOD" in handled
    # Bad entry rescheduled (treated as transient).
    bad = next(e for e in q.all_entries() if e.symbol == "BAD")
    assert bad.attempts == 1
    assert any("unexpected" in err for err in bad.last_errors)


@pytest.mark.asyncio
async def test_tick_skips_entries_not_yet_due(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    # Enqueue, then immediately fail once to push next_attempt out
    # past now. The worker tick should not pick it up again.
    q.enqueue("AAPL", ("e",))
    q.record_failure("AAPL", ("e",))
    call_count = 0

    async def fetch(_symbol: str) -> None:
        nonlocal call_count
        call_count += 1

    worker = RetryWorker(q, fetch, interval_seconds=0.1)
    processed = await worker.tick()
    assert processed == 0
    assert call_count == 0


# -----------------------------------------------------------------------------
# Start / stop lifecycle
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_then_stop_drains_one_pass(tmp_path: Path) -> None:
    """Smoke test: start the loop, see it run a tick, stop cleanly."""
    q = _queue(tmp_path)
    q.enqueue("AAPL", ("e",))
    drained = asyncio.Event()

    async def fetch(_symbol: str) -> None:
        drained.set()

    worker = RetryWorker(q, fetch, interval_seconds=0.05)
    await worker.start()
    # Give the loop a moment to fire the first tick.
    await asyncio.wait_for(drained.wait(), timeout=2.0)
    await worker.stop()
    assert q.all_entries() == ()


@pytest.mark.asyncio
async def test_double_start_is_no_op(tmp_path: Path) -> None:
    q = _queue(tmp_path)

    async def fetch(_symbol: str) -> None:
        return None

    worker = RetryWorker(q, fetch, interval_seconds=0.1)
    await worker.start()
    first_task = worker._task
    await worker.start()  # second call is a no-op
    assert worker._task is first_task
    await worker.stop()


@pytest.mark.asyncio
async def test_stop_when_not_running_is_no_op(tmp_path: Path) -> None:
    q = _queue(tmp_path)

    async def fetch(_symbol: str) -> None:
        return None

    worker = RetryWorker(q, fetch, interval_seconds=0.1)
    # No prior start — should not raise.
    await worker.stop()
    assert not worker.is_running


def test_invalid_interval_rejected(tmp_path: Path) -> None:
    q = _queue(tmp_path)

    async def fetch(_symbol: str) -> None:
        return None

    with pytest.raises(ValueError, match="positive"):
        RetryWorker(q, fetch, interval_seconds=0)
    with pytest.raises(ValueError, match="positive"):
        RetryWorker(q, fetch, interval_seconds=-1.0)


# Pulled in so the noqa for SLF001 lands cleanly even on linters that
# refuse to honor inline ignores without a marker.
_ = datetime(2026, 1, 1, tzinfo=UTC)
