"""Unit tests for :class:`RetryQueue`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.data.retry_queue import (
    BackoffPolicy,
    RetryQueue,
    RetryStatus,
)


def _queue(tmp_path: Path, **kwargs: object) -> RetryQueue:
    return RetryQueue(tmp_path / "retry.json", **kwargs)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


def test_init_does_not_touch_disk(tmp_path: Path) -> None:
    """Same lazy contract as HealthStore — constructing is free."""
    path = tmp_path / "nested" / "retry.json"
    RetryQueue(path)
    assert not path.exists()
    assert not path.parent.exists()


def test_max_attempts_must_be_at_least_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        RetryQueue(tmp_path / "r.json", max_attempts=0)


# -----------------------------------------------------------------------------
# Enqueue
# -----------------------------------------------------------------------------


def test_enqueue_persists_entry_and_uppercases(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    q.enqueue("aapl", ("fmp: rate-limited",))
    entries = q.all_entries()
    assert len(entries) == 1
    assert entries[0].symbol == "AAPL"
    assert entries[0].attempts == 0
    assert entries[0].status is RetryStatus.PENDING
    assert entries[0].last_errors == ("fmp: rate-limited",)


def test_enqueue_replaces_existing_entry_for_same_symbol(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    q.enqueue("AAPL", ("first",))
    q.enqueue("AAPL", ("second",))
    entries = q.all_entries()
    assert len(entries) == 1
    assert entries[0].last_errors == ("second",)


def test_enqueue_sets_next_attempt_to_now(tmp_path: Path) -> None:
    """Re-enqueue with fresh-now next_attempt — worker picks up promptly."""
    q = _queue(tmp_path)
    when = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    entry = q.enqueue("AAPL", ("e",), now=when)
    assert entry.next_attempt_at == when
    assert entry.enqueued_at == when


# -----------------------------------------------------------------------------
# Due / scheduling
# -----------------------------------------------------------------------------


def test_due_entries_returns_only_pending_past_next_attempt(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    q.enqueue("AAPL", ("e",), now=now - timedelta(minutes=5))
    q.enqueue("ZZZ", ("e",), now=now + timedelta(minutes=10))
    due = q.due_entries(now=now)
    assert [d.symbol for d in due] == ["AAPL"]


def test_due_entries_sorted_oldest_first(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    base = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    q.enqueue("LATE", ("e",), now=base - timedelta(minutes=1))
    q.enqueue("EARLY", ("e",), now=base - timedelta(minutes=10))
    due = q.due_entries(now=base)
    assert [d.symbol for d in due] == ["EARLY", "LATE"]


def test_due_entries_excludes_permanently_failed(tmp_path: Path) -> None:
    q = _queue(tmp_path, max_attempts=1)
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    q.enqueue("AAPL", ("e",), now=now)
    # Single failure exceeds max_attempts → permanently_failed.
    q.record_failure("AAPL", ("e2",), now=now + timedelta(seconds=1))
    assert q.due_entries(now=now + timedelta(hours=1)) == ()


# -----------------------------------------------------------------------------
# Success / failure transitions
# -----------------------------------------------------------------------------


def test_record_success_removes_entry(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    q.enqueue("AAPL", ("e",))
    assert q.record_success("AAPL") is True
    assert q.all_entries() == ()


def test_record_success_idempotent_when_no_entry(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    assert q.record_success("MISSING") is False


def test_record_failure_reschedules_with_backoff(tmp_path: Path) -> None:
    q = _queue(
        tmp_path,
        backoff=BackoffPolicy(initial_seconds=60.0, max_seconds=3600.0),
        max_attempts=5,
    )
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    q.enqueue("AAPL", ("e",), now=now)
    # First failure: attempts=1, delay = 60 * 2**1 = 120s
    updated = q.record_failure("AAPL", ("e2",), now=now)
    assert updated is not None
    assert updated.attempts == 1
    assert updated.status is RetryStatus.PENDING
    assert updated.next_attempt_at == now + timedelta(seconds=120)
    assert updated.last_errors == ("e2",)


def test_record_failure_caps_backoff_at_max(tmp_path: Path) -> None:
    """Long-running streak should top out at max_seconds, not infinity."""
    q = _queue(
        tmp_path,
        backoff=BackoffPolicy(initial_seconds=60.0, max_seconds=300.0),
        max_attempts=10,
    )
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    q.enqueue("AAPL", ("e",), now=now)
    for _ in range(5):
        q.record_failure("AAPL", ("e",), now=now)
    entry = q.all_entries()[0]
    # 60 * 2**5 = 1920s, capped at 300s.
    assert entry.next_attempt_at == now + timedelta(seconds=300)


def test_record_failure_at_max_attempts_marks_permanently_failed(
    tmp_path: Path,
) -> None:
    q = _queue(tmp_path, max_attempts=3)
    q.enqueue("AAPL", ("e",))
    for _ in range(3):
        q.record_failure("AAPL", ("e",))
    entry = q.all_entries()[0]
    assert entry.status is RetryStatus.PERMANENTLY_FAILED
    assert entry.attempts == 3


def test_record_failure_returns_none_when_no_entry(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    assert q.record_failure("MISSING", ("e",)) is None


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def test_entries_round_trip_across_instances(tmp_path: Path) -> None:
    q1 = _queue(tmp_path)
    q1.enqueue("AAPL", ("a", "b"))
    q1.enqueue("MSFT", ("c",))
    q2 = _queue(tmp_path)
    symbols = {e.symbol for e in q2.all_entries()}
    assert symbols == {"AAPL", "MSFT"}


def test_corrupt_file_starts_fresh_does_not_raise(tmp_path: Path) -> None:
    """A truncated or malformed JSON must not bring down the queue."""
    path = tmp_path / "retry.json"
    path.write_text("{this is not valid json", encoding="utf-8")
    q = RetryQueue(path)
    # Reading must return empty rather than crash.
    assert q.all_entries() == ()
    # Subsequent writes must overwrite the corrupt file cleanly.
    q.enqueue("AAPL", ("e",))
    assert {e.symbol for e in q.all_entries()} == {"AAPL"}


def test_schema_version_mismatch_starts_fresh(tmp_path: Path) -> None:
    """Older / unknown schema versions are dropped, not migrated."""
    path = tmp_path / "retry.json"
    path.write_text(
        '{"version": 99, "entries": [{"symbol": "AAPL"}]}', encoding="utf-8"
    )
    q = RetryQueue(path)
    assert q.all_entries() == ()


def test_atomic_write_leaves_no_dot_tmp_residue(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    q.enqueue("AAPL", ("e",))
    leftover = list(tmp_path.glob(".*.tmp"))
    assert leftover == []


# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------


def test_cleanup_removes_old_permanent_failures_only(tmp_path: Path) -> None:
    q = _queue(tmp_path, max_attempts=1)
    now = datetime.now(UTC)
    q.enqueue("OLD", ("e",), now=now - timedelta(days=10))
    q.record_failure("OLD", ("e",), now=now - timedelta(days=10))
    q.enqueue("FRESH", ("e",), now=now)
    removed = q.cleanup(older_than=timedelta(days=1))
    assert removed == 1
    symbols = {e.symbol for e in q.all_entries()}
    assert symbols == {"FRESH"}


def test_cleanup_never_removes_pending_entries(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    q.enqueue("AAPL", ("e",), now=datetime.now(UTC) - timedelta(days=30))
    removed = q.cleanup(older_than=timedelta(days=1))
    assert removed == 0


# -----------------------------------------------------------------------------
# Backoff policy
# -----------------------------------------------------------------------------


def test_backoff_negative_attempts_raises() -> None:
    policy = BackoffPolicy(initial_seconds=10.0, max_seconds=100.0)
    with pytest.raises(ValueError, match="non-negative"):
        policy.next_backoff(-1)


def test_backoff_doubles_per_attempt() -> None:
    policy = BackoffPolicy(initial_seconds=10.0, max_seconds=10000.0)
    assert policy.next_backoff(0) == timedelta(seconds=10)
    assert policy.next_backoff(1) == timedelta(seconds=20)
    assert policy.next_backoff(2) == timedelta(seconds=40)
