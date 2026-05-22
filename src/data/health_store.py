"""Persistent provider-health store.

Producers of :class:`~src.intelligence.fundamentals.models.ProviderHealth`
rows (currently the fundamentals service; later the bars / news paths)
sink them here so the platform can answer institutional-grade
observability questions:

* "What fraction of FMP calls succeeded in the last hour / day?"
* "Is Yahoo silently rotting — when did we last get a non-error response?"
* "What's the p95 latency on Finnhub fundamentals fetches today?"

Backed by SQLite with WAL journaling so a slow read can't block a
write and vice versa. The connection is lazy: ``HealthStore(path)``
performs no I/O — the DB file is created and the schema initialized
on the first :meth:`record` / :meth:`summarize` / :meth:`list_recent_failures`
call. That keeps construction cheap and lets tests that never exercise
the store sit safely on a path that doesn't exist.

Failure model
-------------
:meth:`record` is the *only* method called from a hot fetch path. It
returns the number of rows written and never raises on transient DB
issues — the caller (FundamentalsService) wraps with a broad ``except``
and logs. We do NOT want a stuck DB lock to break a fundamentals
fetch; the observability layer must degrade silently rather than
take down the request path.

Schema versioning
-----------------
v1 only. The schema is additive — adding columns later requires the
init script to ``ALTER TABLE ADD COLUMN IF NOT EXISTS``. Do not rename
or drop columns; consumers may be reading the same file via long-lived
WAL replicas.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.intelligence.fundamentals.models import ProviderHealth, ProviderName
from src.utils.logging import get_logger

log = get_logger(__name__)


# Statuses that count as a "success" for the rolling success-rate
# numerator. ``empty`` is *not* a success — it means the provider
# answered but had no data for this symbol, which is a different
# operator signal ("we asked, they shrugged"). It also is *not* a
# failure — it gets its own bucket. The denominator excludes
# ``skipped`` because skipped providers weren't actually consulted.
_SUCCESS_STATUS = "ok"
_NEUTRAL_STATUSES = frozenset({"skipped"})
_FAILURE_STATUSES = frozenset(
    {"rate_limited", "unavailable", "transient"}
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    checked_at TEXT NOT NULL,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_ph_provider_checked
    ON provider_health(provider, checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_ph_checked
    ON provider_health(checked_at DESC);
"""


@dataclass(frozen=True)
class ProviderSummary:
    """Rolling-window aggregate for one provider.

    Attributes:
        provider: Provider name (e.g. ``"fmp"``).
        window: The duration over which this summary was computed.
        total: Count of rows in the window (all statuses).
        ok: Rows with ``status == "ok"``.
        rate_limited / unavailable / transient / empty / skipped:
            Per-status counts. Sum to ``total``.
        success_rate: ``ok / (total - skipped)``; ``0.0`` when the
            denominator is zero. ``skipped`` excluded because those
            rows represent "we didn't try this provider this call",
            not real attempts.
        p50_latency_ms / p95_latency_ms: Latency percentiles over the
            *non-skipped* rows; ``None`` when no such rows exist.
        last_seen_at: Timestamp of the most recent row (any status).
        last_error_at: Timestamp of the most recent failure row
            (``rate_limited``, ``unavailable``, or ``transient``);
            ``None`` if no failures in the window.
    """

    provider: str
    window: timedelta
    total: int
    ok: int = 0
    rate_limited: int = 0
    unavailable: int = 0
    transient: int = 0
    empty: int = 0
    skipped: int = 0
    success_rate: float = 0.0
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    last_seen_at: datetime | None = None
    last_error_at: datetime | None = None
    breakdown: dict[str, int] = field(default_factory=dict)


class HealthStore:
    """SQLite-backed sink + query layer for :class:`ProviderHealth`.

    Args:
        db_path: Filesystem path to the SQLite database. Created on
            first read/write — constructing the store is free.
        check_same_thread: Forwarded to :func:`sqlite3.connect`. The
            default is ``False`` so the same connection can be used
            from FastAPI's threadpool and the async fundamentals
            service path. Concurrency is mediated by a per-store
            :class:`threading.Lock` rather than per-thread connections.
    """

    def __init__(
        self, db_path: Path, *, check_same_thread: bool = False
    ) -> None:
        self._db_path = db_path
        self._check_same_thread = check_same_thread
        self._conn: sqlite3.Connection | None = None
        # Serializes writes across threadpool workers. Reads don't
        # block writes thanks to WAL; this lock is only contention
        # under heavy writer concurrency.
        self._lock = threading.Lock()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        """Lazy connection. First call creates the parent dir + schema."""
        if self._conn is not None:
            return self._conn
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=self._check_same_thread,
            isolation_level=None,  # autocommit; explicit transactions in record()
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        self._conn = conn
        return conn

    def close(self) -> None:
        """Close the underlying connection. Safe to call multiple times."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def record(self, rows: Iterable[ProviderHealth]) -> int:
        """Append ``rows`` to the store. Returns the count written.

        Best-effort — any sqlite error propagates so the caller can
        log and continue. The fundamentals service wraps this call in
        a broad ``except``; that's the *only* designated tolerance
        point. Other callers should also catch.
        """
        materialized = list(rows)
        if not materialized:
            return 0
        with self._lock:
            conn = self._connect()
            conn.execute("BEGIN")
            try:
                conn.executemany(
                    """
                    INSERT INTO provider_health
                        (provider, symbol, status, latency_ms, checked_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            r.provider.value,
                            r.symbol,
                            r.status,
                            float(r.latency_ms),
                            r.checked_at.isoformat(),
                            r.error_message,
                        )
                        for r in materialized
                    ],
                )
                conn.execute("COMMIT")
            except sqlite3.Error:
                conn.execute("ROLLBACK")
                raise
        return len(materialized)

    def summarize(
        self,
        window: timedelta,
        *,
        now: datetime | None = None,
    ) -> tuple[ProviderSummary, ...]:
        """Aggregate health rows within ``window`` per provider.

        Returns one :class:`ProviderSummary` per provider that has at
        least one row in the window. Providers with zero rows are
        omitted — callers that need a "0 calls" entry for a known
        provider can union with the configured chain on their side.

        ``now`` defaults to ``datetime.now(UTC)``; tests pass an
        explicit value for determinism.
        """
        if window <= timedelta(0):
            raise ValueError(f"window must be positive, got {window!r}")
        moment = now or datetime.now(UTC)
        cutoff = moment - window

        conn = self._connect()
        cur = conn.execute(
            """
            SELECT provider, status, latency_ms, checked_at, error_message
            FROM provider_health
            WHERE checked_at >= ?
            ORDER BY checked_at ASC
            """,
            (cutoff.isoformat(),),
        )
        rows_by_provider: dict[str, list[_RowView]] = {}
        for provider, status, latency_ms, checked_at, error_message in cur.fetchall():
            rows_by_provider.setdefault(provider, []).append(
                _RowView(
                    status=status,
                    latency_ms=float(latency_ms),
                    checked_at=_parse_iso(checked_at),
                    error_message=error_message,
                )
            )

        summaries: list[ProviderSummary] = []
        for provider, rows in rows_by_provider.items():
            summaries.append(_summarize_rows(provider, window, rows))
        # Stable ordering: providers with the most calls first; ties
        # broken alphabetically so the response is deterministic.
        summaries.sort(key=lambda s: (-s.total, s.provider))
        return tuple(summaries)

    def list_recent_failures(
        self, *, limit: int = 20
    ) -> tuple[ProviderHealth, ...]:
        """The N most recent rows whose status is a failure.

        ``empty`` does NOT count as a failure — a provider answering
        "no data for that symbol" is not a degradation signal. Only
        ``rate_limited`` / ``unavailable`` / ``transient`` qualify.
        """
        if limit <= 0:
            return ()
        conn = self._connect()
        placeholders = ",".join("?" for _ in _FAILURE_STATUSES)
        cur = conn.execute(
            f"""
            SELECT provider, symbol, status, latency_ms, checked_at, error_message
            FROM provider_health
            WHERE status IN ({placeholders})
            ORDER BY checked_at DESC
            LIMIT ?
            """,
            (*_FAILURE_STATUSES, limit),
        )
        out: list[ProviderHealth] = []
        for provider, symbol, status, latency_ms, checked_at, error_message in cur.fetchall():
            try:
                pname = ProviderName(provider)
            except ValueError:
                # Skip rows whose provider name no longer maps to a
                # known ProviderName — a deleted provider would
                # otherwise crash the wire serializer.
                continue
            out.append(
                ProviderHealth(
                    provider=pname,
                    symbol=symbol,
                    status=status,
                    latency_ms=float(latency_ms),
                    checked_at=_parse_iso(checked_at),
                    error_message=error_message,
                )
            )
        return tuple(out)


# ---- helpers ----


@dataclass(frozen=True)
class _RowView:
    """Lightweight view of a SELECT row used inside :meth:`summarize`.

    Kept private because the public API hands callers
    :class:`ProviderSummary` aggregates, not individual rows.
    """

    status: str
    latency_ms: float
    checked_at: datetime
    error_message: str | None


def _summarize_rows(
    provider: str, window: timedelta, rows: list[_RowView]
) -> ProviderSummary:
    breakdown: dict[str, int] = {}
    for r in rows:
        breakdown[r.status] = breakdown.get(r.status, 0) + 1

    ok = breakdown.get(_SUCCESS_STATUS, 0)
    rate_limited = breakdown.get("rate_limited", 0)
    unavailable = breakdown.get("unavailable", 0)
    transient = breakdown.get("transient", 0)
    empty = breakdown.get("empty", 0)
    skipped = breakdown.get("skipped", 0)
    total = len(rows)

    denom = total - skipped
    success_rate = (ok / denom) if denom > 0 else 0.0

    non_skipped_latencies = sorted(
        r.latency_ms for r in rows if r.status not in _NEUTRAL_STATUSES
    )
    p50 = _percentile(non_skipped_latencies, 0.50)
    p95 = _percentile(non_skipped_latencies, 0.95)

    last_seen_at = max((r.checked_at for r in rows), default=None)
    last_error_at = max(
        (r.checked_at for r in rows if r.status in _FAILURE_STATUSES),
        default=None,
    )

    return ProviderSummary(
        provider=provider,
        window=window,
        total=total,
        ok=ok,
        rate_limited=rate_limited,
        unavailable=unavailable,
        transient=transient,
        empty=empty,
        skipped=skipped,
        success_rate=success_rate,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        last_seen_at=last_seen_at,
        last_error_at=last_error_at,
        breakdown=breakdown,
    )


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile on a pre-sorted list.

    Returns ``None`` for an empty list. ``q`` is in ``[0, 1]``.
    Matches numpy's ``percentile(method='linear')`` behavior so test
    expectations don't need to know the implementation detail.
    """
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q!r}")
    idx = q * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp. Always returns tz-aware UTC."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = ["HealthStore", "ProviderSummary"]
