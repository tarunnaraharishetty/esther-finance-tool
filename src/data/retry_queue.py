"""Durable retry queue for transient fundamentals fetch failures.

When :class:`~src.intelligence.fundamentals.service.FundamentalsService`
exhausts the provider chain with errors that are *all* transient
(rate-limit, 5xx, network blip), the symbol lands here. A worker drains
the queue on an independent cadence, retrying with exponential backoff.
On success the entry is dropped; on repeated failure it's eventually
marked ``permanently_failed`` and retained for operator visibility.

Storage
-------
SQLite (WAL, lazy-init) — same backbone as :class:`HealthStore`,
:class:`UserStore`, and :class:`AccuracyStore`. The original prototype
used a JSON-on-disk file with full read/filter/write per operation; B-23
called out the O(n) hot path under thousands of permanently-failed
rows. The SQLite rewrite makes ``due_entries`` an indexed scan and
every mutation a targeted INSERT / UPDATE / DELETE.

Single instance per process. The per-store ``threading.Lock`` serializes
writes across FastAPI's threadpool; WAL handles concurrent reads. Atomic
durability is the SQLite WAL / synchronous=NORMAL combination — a crash
mid-transaction leaves a recoverable journal.

Failure-mode policy
-------------------
Enqueueing is gated to *all-transient* chain errors. If any error in
the failed chain was permanent (missing API key, hard 4xx auth), we
don't enqueue — that wouldn't fix itself on retry. The gate keeps the
queue from clogging up with unfixable entries.

Backward compatibility
----------------------
The default settings path moved from ``data/retry_queue.json`` →
``data/retry_queue.db``. Deployments that pinned the old path via the
``RETRY_QUEUE_PATH`` env var will find sqlite3 refusing to open the
old JSON file ("file is not a database"); the operator updates the
env var and any in-flight queue state from the prior JSON file is
abandoned. That's acceptable — the queue is operational state meant
to drain in minutes, not durable user data.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from src.data.migrations import Migration, run_migrations
from src.utils.logging import get_logger

log = get_logger(__name__)


class RetryStatus(StrEnum):
    """State machine for a retry entry.

    * ``pending`` — eligible for the worker to pick up at or after
      ``next_attempt_at``.
    * ``succeeded`` — the worker fetched cleanly. Entries normally
      get removed on success; this state is kept as a transient
      label inside :meth:`RetryQueue.record_success` for callers
      that want to log it before the entry leaves.
    * ``permanently_failed`` — exceeded ``max_attempts``. Stays in
      the table so operators can see "we tried 5 times and gave up";
      :meth:`cleanup` removes old ones on demand.
    """

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    PERMANENTLY_FAILED = "permanently_failed"


@dataclass(frozen=True)
class RetryEntry:
    """One queued retry."""

    symbol: str
    enqueued_at: datetime
    next_attempt_at: datetime
    attempts: int
    last_errors: tuple[str, ...]
    status: RetryStatus
    last_error_at: datetime | None = None


@dataclass(frozen=True)
class BackoffPolicy:
    """Exponential backoff schedule.

    ``next_backoff(attempts)`` returns the delay before the
    ``attempts+1``-th try, capped at ``max_seconds``.
    """

    initial_seconds: float
    max_seconds: float
    multiplier: float = 2.0

    def next_backoff(self, attempts: int) -> timedelta:
        if attempts < 0:
            raise ValueError(f"attempts must be non-negative, got {attempts!r}")
        delay = self.initial_seconds * (self.multiplier ** attempts)
        capped = min(delay, self.max_seconds)
        return timedelta(seconds=capped)


# v1 — initial schema. One row per queued symbol. The (status,
# next_attempt_at) composite index turns ``due_entries`` into a
# range scan whose cost is independent of how many permanently-failed
# rows are sitting in the table. The (status, last_error_at) index
# accelerates :meth:`cleanup` for the same reason.
_V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS retry_entries (
    symbol TEXT PRIMARY KEY,
    enqueued_at TEXT NOT NULL,
    next_attempt_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_errors TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    last_error_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_retry_status_next
    ON retry_entries(status, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_retry_status_last_error
    ON retry_entries(status, last_error_at);
"""


def _apply_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(_V1_SCHEMA)


_MIGRATIONS = [
    Migration(version=1, name="retry_entries_init", apply=_apply_v1),
]
_MIGRATION_NAMESPACE = "retry_queue"


class RetryQueue:
    """SQLite-backed retry queue.

    Single instance per process. Reads and writes are serialized
    through a threading lock; WAL handles cross-instance read concurrency
    for any out-of-process inspector (operator CLI, future worker).

    Args:
        path: Filesystem path for the SQLite store. The parent is
            created on first write. Constructing the queue performs
            no I/O (matches the lazy-init contract every other store
            in the project follows).
        backoff: Backoff schedule for reschedules.
        max_attempts: Cap on retry attempts before marking an entry
            ``permanently_failed``.
        check_same_thread: Forwarded to ``sqlite3.connect``. Defaults to
            ``False`` so FastAPI's threadpool can call methods across
            threads.
    """

    def __init__(
        self,
        path: Path,
        *,
        backoff: BackoffPolicy | None = None,
        max_attempts: int = 5,
        check_same_thread: bool = False,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}")
        self._path = path
        self._backoff = backoff or BackoffPolicy(
            initial_seconds=30.0, max_seconds=1800.0
        )
        self._max_attempts = max_attempts
        self._check_same_thread = check_same_thread
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        """Close the underlying connection. Intended for tests."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---- internal ----

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self._path,
            check_same_thread=self._check_same_thread,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        run_migrations(conn, _MIGRATIONS, namespace=_MIGRATION_NAMESPACE)
        self._conn = conn
        return conn

    # ---- public API ----

    def enqueue(
        self,
        symbol: str,
        errors: tuple[str, ...],
        *,
        now: datetime | None = None,
        retry_after_seconds: float | None = None,
    ) -> RetryEntry:
        """Add (or refresh) an entry for ``symbol``.

        If the symbol is already in the queue, we replace its
        previous entry — re-enqueueing is the most recent operator
        signal of "still failing". The next_attempt_at is set
        immediately to ``now`` so the worker picks it up on the next
        tick rather than waiting out the prior backoff.

        ``retry_after_seconds`` is a hint from an upstream provider
        (e.g. ``Retry-After`` header on a 429). When supplied and
        positive, the first attempt is delayed by that many seconds
        — RFC 6585 says "don't retry before this", and ignoring it
        risks getting our keys banned.
        """
        sym = symbol.upper()
        moment = now or datetime.now(UTC)
        if retry_after_seconds and retry_after_seconds > 0:
            next_attempt = moment + timedelta(seconds=retry_after_seconds)
        else:
            next_attempt = moment
        entry = RetryEntry(
            symbol=sym,
            enqueued_at=moment,
            next_attempt_at=next_attempt,
            attempts=0,
            last_errors=tuple(errors),
            status=RetryStatus.PENDING,
            last_error_at=moment if errors else None,
        )
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO retry_entries
                    (symbol, enqueued_at, next_attempt_at, attempts,
                     last_errors, status, last_error_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    enqueued_at = excluded.enqueued_at,
                    next_attempt_at = excluded.next_attempt_at,
                    attempts = excluded.attempts,
                    last_errors = excluded.last_errors,
                    status = excluded.status,
                    last_error_at = excluded.last_error_at
                """,
                _entry_to_row(entry),
            )
        log.info(
            "retry_queue.enqueue",
            symbol=sym,
            error_count=len(errors),
            retry_after_seconds=retry_after_seconds,
        )
        return entry

    def due_entries(
        self, *, now: datetime | None = None
    ) -> tuple[RetryEntry, ...]:
        """Pending entries whose ``next_attempt_at`` has arrived.

        Returns entries in next_attempt_at ascending order so callers
        process the longest-overdue first. Index-driven — the cost
        is proportional to the number of due entries, not the total
        table size.
        """
        moment = now or datetime.now(UTC)
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT symbol, enqueued_at, next_attempt_at, attempts,
                   last_errors, status, last_error_at
            FROM retry_entries
            WHERE status = ? AND next_attempt_at <= ?
            ORDER BY next_attempt_at ASC
            """,
            (RetryStatus.PENDING.value, moment.isoformat()),
        )
        return tuple(_row_to_entry(row) for row in cur.fetchall())

    def record_success(self, symbol: str) -> bool:
        """Mark the entry for ``symbol`` as succeeded and remove it.

        Returns True when an entry was present and removed; False
        when no entry existed (idempotent — a worker that races with
        a manual remove won't crash).
        """
        sym = symbol.upper()
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "DELETE FROM retry_entries WHERE symbol = ?", (sym,)
            )
            removed = cur.rowcount > 0
        if removed:
            log.info("retry_queue.success", symbol=sym)
        return removed

    def record_failure(
        self,
        symbol: str,
        errors: tuple[str, ...],
        *,
        now: datetime | None = None,
        retry_after_seconds: float | None = None,
    ) -> RetryEntry | None:
        """Increment attempts on the entry and reschedule (or fail it).

        Returns the updated entry, or None when no entry existed
        (race with remove). When ``attempts`` exceeds ``max_attempts``
        the entry is marked ``permanently_failed`` and kept in the
        table for operator visibility.

        ``retry_after_seconds`` is a server-supplied cooldown hint
        (the ``Retry-After`` header on the most recent rate-limited
        attempt). When provided and longer than the worker's default
        exponential backoff, the longer of the two is used as the
        reschedule delay — RFC 6585 §4 says "MUST NOT retry before
        this", so a 5-minute server hint never gets undercut by our
        30-second default. Capped indirectly via http.py's 24h ceiling
        on the parsed value.
        """
        sym = symbol.upper()
        moment = now or datetime.now(UTC)
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                """
                SELECT symbol, enqueued_at, next_attempt_at, attempts,
                       last_errors, status, last_error_at
                FROM retry_entries
                WHERE symbol = ?
                """,
                (sym,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            existing = _row_to_entry(row)
            new_attempts = existing.attempts + 1
            if new_attempts >= self._max_attempts:
                updated = RetryEntry(
                    symbol=existing.symbol,
                    enqueued_at=existing.enqueued_at,
                    next_attempt_at=existing.next_attempt_at,
                    attempts=new_attempts,
                    last_errors=tuple(errors),
                    status=RetryStatus.PERMANENTLY_FAILED,
                    last_error_at=moment,
                )
            else:
                backoff_delay = self._backoff.next_backoff(new_attempts)
                if retry_after_seconds and retry_after_seconds > 0:
                    hint = timedelta(seconds=retry_after_seconds)
                    delay = max(backoff_delay, hint)
                else:
                    delay = backoff_delay
                updated = RetryEntry(
                    symbol=existing.symbol,
                    enqueued_at=existing.enqueued_at,
                    next_attempt_at=moment + delay,
                    attempts=new_attempts,
                    last_errors=tuple(errors),
                    status=RetryStatus.PENDING,
                    last_error_at=moment,
                )
            conn.execute(
                """
                UPDATE retry_entries SET
                    next_attempt_at = ?,
                    attempts = ?,
                    last_errors = ?,
                    status = ?,
                    last_error_at = ?
                WHERE symbol = ?
                """,
                (
                    updated.next_attempt_at.isoformat(),
                    updated.attempts,
                    json.dumps(list(updated.last_errors)),
                    updated.status.value,
                    updated.last_error_at.isoformat()
                    if updated.last_error_at is not None
                    else None,
                    sym,
                ),
            )
        log.info(
            "retry_queue.failure",
            symbol=sym,
            attempts=updated.attempts,
            status=updated.status.value,
            retry_after_seconds=retry_after_seconds,
        )
        return updated

    def all_entries(self) -> tuple[RetryEntry, ...]:
        """Snapshot of every entry currently in the queue."""
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT symbol, enqueued_at, next_attempt_at, attempts,
                   last_errors, status, last_error_at
            FROM retry_entries
            ORDER BY enqueued_at ASC
            """
        )
        return tuple(_row_to_entry(row) for row in cur.fetchall())

    def remove(self, symbol: str) -> bool:
        """Drop ``symbol`` from the queue regardless of status."""
        sym = symbol.upper()
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "DELETE FROM retry_entries WHERE symbol = ?", (sym,)
            )
            return cur.rowcount > 0

    def cleanup(self, *, older_than: timedelta) -> int:
        """Remove permanently-failed entries older than ``older_than``.

        Operator-callable. Returns the number of entries removed.
        Pending entries are never removed by cleanup — only the
        scheduling state machine can change their status. Indexed
        on (status, last_error_at) so cost is proportional to the
        permanently-failed-and-old subset, not the total table.
        """
        cutoff = (datetime.now(UTC) - older_than).isoformat()
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                """
                DELETE FROM retry_entries
                WHERE status = ?
                  AND COALESCE(last_error_at, enqueued_at) < ?
                """,
                (RetryStatus.PERMANENTLY_FAILED.value, cutoff),
            )
            return int(cur.rowcount or 0)


# ---- serialization helpers ----


def _entry_to_row(
    entry: RetryEntry,
) -> tuple[str, str, str, int, str, str, str | None]:
    return (
        entry.symbol,
        entry.enqueued_at.isoformat(),
        entry.next_attempt_at.isoformat(),
        entry.attempts,
        json.dumps(list(entry.last_errors)),
        entry.status.value,
        entry.last_error_at.isoformat() if entry.last_error_at is not None else None,
    )


def _row_to_entry(
    row: tuple[str, str, str, int, str, str, str | None],
) -> RetryEntry:
    symbol, enqueued_at, next_attempt_at, attempts, last_errors_raw, status_raw, last_error_at = row
    try:
        decoded = json.loads(last_errors_raw or "[]")
    except (TypeError, ValueError):
        decoded = []
    last_errors = tuple(str(e) for e in decoded if isinstance(decoded, list))
    return RetryEntry(
        symbol=str(symbol).upper(),
        enqueued_at=_parse_iso(enqueued_at),
        next_attempt_at=_parse_iso(next_attempt_at),
        attempts=int(attempts),
        last_errors=last_errors,
        status=_parse_status(status_raw),
        last_error_at=(
            _parse_iso(last_error_at) if last_error_at is not None else None
        ),
    )


def _parse_iso(value: str) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, UTC)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromtimestamp(0, UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_status(value: str) -> RetryStatus:
    try:
        return RetryStatus(value)
    except ValueError:
        return RetryStatus.PENDING


__all__ = [
    "BackoffPolicy",
    "RetryEntry",
    "RetryQueue",
    "RetryStatus",
]
