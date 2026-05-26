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
    return RetryQueue(tmp_path / "retry.db", **kwargs)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


def test_init_does_not_touch_disk(tmp_path: Path) -> None:
    """Same lazy contract as HealthStore — constructing is free."""
    path = tmp_path / "nested" / "retry.db"
    RetryQueue(path)
    assert not path.exists()
    assert not path.parent.exists()


def test_max_attempts_must_be_at_least_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        RetryQueue(tmp_path / "r.db", max_attempts=0)


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


def test_record_failure_retry_after_overrides_backoff_when_longer(
    tmp_path: Path,
) -> None:
    """A 600s server cooldown beats the worker's 120s backoff — RFC 6585.

    Honors the larger of backoff vs Retry-After so a chatty provider that
    asked us to wait 10 minutes isn't hit again 2 minutes later."""
    q = _queue(
        tmp_path,
        backoff=BackoffPolicy(initial_seconds=60.0, max_seconds=3600.0),
        max_attempts=5,
    )
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    q.enqueue("AAPL", ("e",), now=now)
    updated = q.record_failure(
        "AAPL", ("e2",), now=now, retry_after_seconds=600.0
    )
    assert updated is not None
    # Server hint (600s) > backoff (120s) → use the hint.
    assert updated.next_attempt_at == now + timedelta(seconds=600)


def test_record_failure_backoff_wins_when_longer_than_retry_after(
    tmp_path: Path,
) -> None:
    """A 5-second server hint can't undercut a sensible backoff."""
    q = _queue(
        tmp_path,
        backoff=BackoffPolicy(initial_seconds=60.0, max_seconds=3600.0),
        max_attempts=5,
    )
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    q.enqueue("AAPL", ("e",), now=now)
    # Drive attempts up so backoff > hint.
    q.record_failure("AAPL", ("e",), now=now)  # 120s
    updated = q.record_failure(
        "AAPL", ("e2",), now=now, retry_after_seconds=5.0
    )
    assert updated is not None
    # Backoff at attempt 2 = 60 * 2**2 = 240s; hint = 5s → backoff wins.
    assert updated.next_attempt_at == now + timedelta(seconds=240)


def test_record_failure_none_retry_after_uses_pure_backoff(
    tmp_path: Path,
) -> None:
    """The None default keeps the historical contract unchanged."""
    q = _queue(
        tmp_path,
        backoff=BackoffPolicy(initial_seconds=60.0, max_seconds=3600.0),
        max_attempts=5,
    )
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    q.enqueue("AAPL", ("e",), now=now)
    updated = q.record_failure("AAPL", ("e2",), now=now)
    assert updated is not None
    assert updated.next_attempt_at == now + timedelta(seconds=120)


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


def test_non_sqlite_file_at_path_fails_fast(tmp_path: Path) -> None:
    """A path pointing at a legacy JSON file (or any non-SQLite blob)
    must surface a clear error on first use rather than silently
    starting fresh and orphaning the operator's data. ``sqlite3``
    raises ``DatabaseError("file is not a database")`` which is what
    the operator sees in the logs at startup — they then rename or
    delete the old file and restart. See BUGS.md B-23 for the
    JSON → SQLite migration note."""
    import sqlite3

    path = tmp_path / "retry.db"
    path.write_text("{legacy json blob from before B-23}", encoding="utf-8")
    q = RetryQueue(path)
    with pytest.raises(sqlite3.DatabaseError):
        q.enqueue("AAPL", ("e",))


def test_schema_migration_records_version_row(tmp_path: Path) -> None:
    """First connect runs the v1 migration which creates retry_entries
    AND records (namespace='retry_queue', version=1) in the shared
    _schema_migrations table. Second connect is a no-op."""
    import sqlite3

    q = _queue(tmp_path)
    q.enqueue("AAPL", ("e",))  # triggers _connect + migration
    # Inspect the migrations table directly via a sibling connection.
    inspect = sqlite3.connect(q.path)
    try:
        rows = inspect.execute(
            "SELECT namespace, version FROM _schema_migrations "
            "WHERE namespace = 'retry_queue' ORDER BY version"
        ).fetchall()
    finally:
        inspect.close()
    assert rows == [("retry_queue", 1)]


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


# -----------------------------------------------------------------------------
# B-23: due_entries is index-driven, not table-size dependent
# -----------------------------------------------------------------------------


def test_due_entries_returns_quickly_under_large_backlog(
    tmp_path: Path,
) -> None:
    """The old JSON queue did a full read/filter/write per call;
    ``due_entries()`` cost grew linearly with the total table size,
    including thousands of permanently-failed rows that never expire
    automatically. The SQLite rewrite (B-23) drives the hot path off
    the (status, next_attempt_at) index so cost is proportional to
    the *due* subset.

    Concrete shape: 5000 permanently-failed entries + 5 pending and
    due. ``due_entries()`` must complete in well under 500ms — a
    generous bound that still catches a regression to full-scan
    semantics (which would scale linearly to seconds on a busy
    machine)."""
    import time

    q = _queue(
        tmp_path,
        backoff=BackoffPolicy(initial_seconds=60.0, max_seconds=3600.0),
        max_attempts=2,
    )
    far_past = datetime(2024, 1, 1, tzinfo=UTC)
    # Land 5000 permanently-failed rows. Each requires enqueue + two
    # record_failure calls (max_attempts=2 → second failure promotes).
    # Batch via direct executemany would be faster but the public API
    # is the contract we're testing.
    for i in range(5000):
        sym = f"DEAD{i:05d}"
        q.enqueue(sym, ("e",), now=far_past)
        q.record_failure(sym, ("e",), now=far_past)
        q.record_failure(sym, ("e",), now=far_past)
    # Five pending entries due now.
    now = datetime.now(UTC)
    for i in range(5):
        q.enqueue(f"DUE{i}", ("e",), now=now)

    start = time.perf_counter()
    due = q.due_entries(now=now)
    elapsed = time.perf_counter() - start
    assert {e.symbol for e in due} == {f"DUE{i}" for i in range(5)}
    # Bound is intentionally loose — even a hot CI runner with the
    # index should land in single-digit ms. Anything pushing 500ms
    # signals the query stopped using the index.
    assert elapsed < 0.5, (
        f"due_entries() took {elapsed:.3f}s with 5005 rows present — "
        f"the (status, next_attempt_at) index is not being used. "
        f"Regression to full-scan semantics (BUGS.md B-23)."
    )
