"""Durable retry queue for transient fundamentals fetch failures.

When :class:`~src.intelligence.fundamentals.service.FundamentalsService`
exhausts the provider chain with errors that are *all* transient
(rate-limit, 5xx, network blip), the symbol lands here. A worker drains
the queue on an independent cadence, retrying with exponential backoff.
On success the entry is dropped; on repeated failure it's eventually
marked ``permanently_failed`` and retained for operator visibility.

Why not Celery / RQ
-------------------
Esther runs single-node today and the queue depth is small (one entry
per stuck symbol). A JSON-on-disk queue is observable with ``cat``,
survives restarts, has zero deployment overhead, and never adds a
dependency on Redis or RabbitMQ. When/if we go multi-node, this
module is a small, well-tested swap target.

Failure-mode policy
-------------------
Enqueueing is gated to *all-transient* chain errors. If any error in
the failed chain was permanent (missing API key, hard 4xx auth), we
don't enqueue — that wouldn't fix itself on retry. The gate keeps the
queue from clogging up with unfixable entries.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger(__name__)


# Storage schema version. Bump on any breaking change to the on-disk
# layout. Older versions are dropped (start fresh) rather than
# attempting a migration — the queue is operational state, not
# durable user data.
_SCHEMA_VERSION = 1


class RetryStatus(StrEnum):
    """State machine for a retry entry.

    * ``pending`` — eligible for the worker to pick up at or after
      ``next_attempt_at``.
    * ``succeeded`` — the worker fetched cleanly. Entries normally
      get removed on success; this state is kept as a transient
      label inside :meth:`RetryQueue.record_success` for callers
      that want to log it before the entry leaves.
    * ``permanently_failed`` — exceeded ``max_attempts``. Stays in
      the file so operators can see "we tried 5 times and gave up";
      a cleanup helper removes old ones on demand.
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


class RetryQueue:
    """File-backed retry queue.

    Single instance per process. Reads and writes are serialized
    through a threading lock; the file path itself is the durability
    boundary. Atomic writes (tempfile + ``os.replace``) ensure a
    crash mid-write cannot leave a corrupt file.

    Args:
        path: Filesystem path for the JSON store. The parent is
            created on first write. Constructing the queue performs
            no I/O.
        backoff: Backoff schedule for reschedules.
        max_attempts: Cap on retry attempts before marking an entry
            permanently_failed.
    """

    def __init__(
        self,
        path: Path,
        *,
        backoff: BackoffPolicy | None = None,
        max_attempts: int = 5,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}")
        self._path = path
        self._backoff = backoff or BackoffPolicy(
            initial_seconds=30.0, max_seconds=1800.0
        )
        self._max_attempts = max_attempts
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    # ---- public API ----

    def enqueue(
        self,
        symbol: str,
        errors: tuple[str, ...],
        *,
        now: datetime | None = None,
    ) -> RetryEntry:
        """Add (or refresh) an entry for ``symbol``.

        If the symbol is already in the queue, we replace its
        previous entry — re-enqueueing is the most recent operator
        signal of "still failing". The next_attempt_at is set
        immediately to ``now`` so the worker picks it up on the next
        tick rather than waiting out the prior backoff.
        """
        sym = symbol.upper()
        moment = now or datetime.now(UTC)
        entry = RetryEntry(
            symbol=sym,
            enqueued_at=moment,
            next_attempt_at=moment,
            attempts=0,
            last_errors=tuple(errors),
            status=RetryStatus.PENDING,
            last_error_at=moment if errors else None,
        )
        with self._lock:
            entries = self._read_locked()
            entries = [e for e in entries if e.symbol != sym]
            entries.append(entry)
            self._write_locked(entries)
        log.info("retry_queue.enqueue", symbol=sym, error_count=len(errors))
        return entry

    def due_entries(
        self, *, now: datetime | None = None
    ) -> tuple[RetryEntry, ...]:
        """Pending entries whose ``next_attempt_at`` has arrived.

        Returns entries in next_attempt_at ascending order so callers
        process the longest-overdue first.
        """
        moment = now or datetime.now(UTC)
        with self._lock:
            entries = self._read_locked()
        due = [
            e
            for e in entries
            if e.status is RetryStatus.PENDING and e.next_attempt_at <= moment
        ]
        due.sort(key=lambda e: e.next_attempt_at)
        return tuple(due)

    def record_success(self, symbol: str) -> bool:
        """Mark the entry for ``symbol`` as succeeded and remove it.

        Returns True when an entry was present and removed; False
        when no entry existed (idempotent — a worker that races with
        a manual remove won't crash).
        """
        sym = symbol.upper()
        with self._lock:
            entries = self._read_locked()
            kept = [e for e in entries if e.symbol != sym]
            if len(kept) == len(entries):
                return False
            self._write_locked(kept)
        log.info("retry_queue.success", symbol=sym)
        return True

    def record_failure(
        self,
        symbol: str,
        errors: tuple[str, ...],
        *,
        now: datetime | None = None,
    ) -> RetryEntry | None:
        """Increment attempts on the entry and reschedule (or fail it).

        Returns the updated entry, or None when no entry existed
        (race with remove). When ``attempts`` exceeds ``max_attempts``
        the entry is marked ``permanently_failed`` and kept in the
        file for operator visibility.
        """
        sym = symbol.upper()
        moment = now or datetime.now(UTC)
        with self._lock:
            entries = self._read_locked()
            updated: RetryEntry | None = None
            new_entries: list[RetryEntry] = []
            for e in entries:
                if e.symbol != sym:
                    new_entries.append(e)
                    continue
                new_attempts = e.attempts + 1
                if new_attempts >= self._max_attempts:
                    updated = RetryEntry(
                        symbol=e.symbol,
                        enqueued_at=e.enqueued_at,
                        next_attempt_at=e.next_attempt_at,
                        attempts=new_attempts,
                        last_errors=tuple(errors),
                        status=RetryStatus.PERMANENTLY_FAILED,
                        last_error_at=moment,
                    )
                else:
                    delay = self._backoff.next_backoff(new_attempts)
                    updated = RetryEntry(
                        symbol=e.symbol,
                        enqueued_at=e.enqueued_at,
                        next_attempt_at=moment + delay,
                        attempts=new_attempts,
                        last_errors=tuple(errors),
                        status=RetryStatus.PENDING,
                        last_error_at=moment,
                    )
                new_entries.append(updated)
            if updated is None:
                return None
            self._write_locked(new_entries)
        log.info(
            "retry_queue.failure",
            symbol=sym,
            attempts=updated.attempts,
            status=updated.status.value,
        )
        return updated

    def all_entries(self) -> tuple[RetryEntry, ...]:
        """Snapshot of every entry currently in the queue."""
        with self._lock:
            return tuple(self._read_locked())

    def remove(self, symbol: str) -> bool:
        """Drop ``symbol`` from the queue regardless of status."""
        sym = symbol.upper()
        with self._lock:
            entries = self._read_locked()
            kept = [e for e in entries if e.symbol != sym]
            if len(kept) == len(entries):
                return False
            self._write_locked(kept)
        return True

    def cleanup(self, *, older_than: timedelta) -> int:
        """Remove permanently-failed entries older than ``older_than``.

        Operator-callable. Returns the number of entries removed.
        Pending entries are never removed by cleanup — only the
        scheduling state machine can change their status.
        """
        cutoff = datetime.now(UTC) - older_than
        with self._lock:
            entries = self._read_locked()
            kept = [
                e
                for e in entries
                if not (
                    e.status is RetryStatus.PERMANENTLY_FAILED
                    and (e.last_error_at or e.enqueued_at) < cutoff
                )
            ]
            removed = len(entries) - len(kept)
            if removed:
                self._write_locked(kept)
        return removed

    # ---- IO ----

    def _read_locked(self) -> list[RetryEntry]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Corrupt or truncated file. Operational state is
            # disposable — log loudly, start fresh rather than
            # propagating a crash up the worker loop.
            log.warning(
                "retry_queue.corrupt_file",
                path=str(self._path),
                error=str(exc),
            )
            return []
        if not isinstance(raw, dict) or raw.get("version") != _SCHEMA_VERSION:
            log.warning(
                "retry_queue.schema_mismatch",
                path=str(self._path),
                got=raw.get("version") if isinstance(raw, dict) else type(raw).__name__,
            )
            return []
        items = raw.get("entries", [])
        if not isinstance(items, list):
            return []
        return [_entry_from_dict(item) for item in items if isinstance(item, dict)]

    def _write_locked(self, entries: list[RetryEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _SCHEMA_VERSION,
            "entries": [_entry_to_dict(e) for e in entries],
        }
        # Atomic write: serialize to a sibling tempfile, then rename.
        # ``os.replace`` is atomic on POSIX and Windows for files on
        # the same filesystem. A crash before the rename leaves the
        # old file intact; after the rename, the new file is fully
        # written.
        import os

        tmp_name = f".{self._path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
        tmp_path = self._path.with_name(tmp_name)
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        # Atomic rename: replaces the destination atomically on POSIX
        # and Windows when both paths live on the same filesystem.
        tmp_path.replace(self._path)


# ---- serialization helpers ----


def _entry_to_dict(entry: RetryEntry) -> dict[str, object]:
    d = asdict(entry)
    d["enqueued_at"] = entry.enqueued_at.isoformat()
    d["next_attempt_at"] = entry.next_attempt_at.isoformat()
    d["status"] = entry.status.value
    d["last_errors"] = list(entry.last_errors)
    d["last_error_at"] = (
        entry.last_error_at.isoformat() if entry.last_error_at is not None else None
    )
    return d


def _entry_from_dict(item: dict[str, object]) -> RetryEntry:
    """Hand-rolled rather than pydantic — keeps the module lightweight.

    Unknown statuses fall back to PENDING so a forward-compat schema
    change doesn't lose the entry. Unparseable timestamps default to
    epoch (entry will be due immediately on the next worker tick,
    which is the right action: try the symbol again).
    """
    raw_attempts = item.get("attempts", 0)
    attempts = int(raw_attempts) if isinstance(raw_attempts, (int, float, str)) else 0
    raw_errors = item.get("last_errors", [])
    last_errors: tuple[str, ...] = (
        tuple(str(e) for e in raw_errors) if isinstance(raw_errors, list) else ()
    )
    return RetryEntry(
        symbol=str(item["symbol"]).upper(),
        enqueued_at=_parse_iso(str(item.get("enqueued_at", ""))),
        next_attempt_at=_parse_iso(str(item.get("next_attempt_at", ""))),
        attempts=attempts,
        last_errors=last_errors,
        status=_parse_status(str(item.get("status", RetryStatus.PENDING.value))),
        last_error_at=(
            _parse_iso(str(item["last_error_at"]))
            if item.get("last_error_at")
            else None
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

# Avoid unused-import lint when only ``field`` is referenced via
# ``dataclass`` defaults indirectly.
_ = field
