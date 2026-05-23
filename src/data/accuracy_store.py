"""Persistent provider-accuracy ledger.

The ``provider_health`` table answers "is this provider *up*?". This
ledger answers the harder question: "is this provider *correct*?".

Every time the fundamentals reconciliation step compares a primary
provider's reading of a high-trust field (revenue / net_income /
eps_diluted / total_debt) against a secondary provider's reading of
the same field on the same fiscal period, we emit one
:class:`~src.intelligence.fundamentals.models.AccuracyEvent` recording
the comparison — both agreements and disagreements. Aggregated over
a rolling window these become the empirical accuracy signal that
multiplies the static position-based ``provider_confidence``.

Backed by the same SQLite database as :class:`HealthStore` so
operators have a single backup target and future joins ("show me
accuracy AND uptime in one row") stay cheap. The schema is additive,
the connection is lazy, and writes are best-effort: a stuck DB lock
must never break a fundamentals fetch.

Why we record agreements too
----------------------------
A naive design would only persist divergences ("FMP disagreed with
Finnhub on revenue by 12%"). That makes the denominator unknowable:
without total comparisons, "agreement rate" can't be computed. Storing
every comparison costs ~4 rows × symbols/day; SQLite handles that for
years. The denominator is the institutional contract.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.intelligence.fundamentals.models import AccuracyEvent, ProviderName
from src.utils.logging import get_logger

log = get_logger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_accuracy_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    reference_provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    field TEXT NOT NULL,
    observed_value REAL,
    reference_value REAL,
    rel_error REAL NOT NULL,
    agreed INTEGER NOT NULL,
    fiscal_date TEXT,
    observed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_acc_provider_observed
    ON provider_accuracy_event(provider, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_acc_provider_field
    ON provider_accuracy_event(provider, field);
CREATE INDEX IF NOT EXISTS idx_acc_reference_observed
    ON provider_accuracy_event(reference_provider, observed_at DESC);
"""


@dataclass(frozen=True)
class FieldAccuracy:
    """Rolling agreement rate for one provider × field."""

    provider: str
    field: str
    total: int
    agreed: int
    accuracy: float  # agreed / total; 0.0 when total == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "field": self.field,
            "total": self.total,
            "agreed": self.agreed,
            "accuracy": round(self.accuracy, 4),
        }


@dataclass(frozen=True)
class ProviderAccuracySummary:
    """Aggregate accuracy view for one provider over a rolling window.

    Attributes:
        provider: The provider being judged.
        window: The duration over which the rolling counts were taken.
        total_events: Total comparisons in window (across all fields).
        total_agreed: Comparisons that agreed within tolerance and
            without a sign flip.
        overall_accuracy: ``total_agreed / total_events``; ``0.0`` when
            ``total_events == 0``.
        by_field: Per-field breakdown. Includes only fields that had
            at least one comparison in the window.
    """

    provider: str
    window: timedelta
    total_events: int
    total_agreed: int
    overall_accuracy: float
    by_field: tuple[FieldAccuracy, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "window_seconds": int(self.window.total_seconds()),
            "total_events": self.total_events,
            "total_agreed": self.total_agreed,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "by_field": [f.to_dict() for f in self.by_field],
        }


class AccuracyStore:
    """SQLite-backed sink + query layer for :class:`AccuracyEvent`.

    Constructed cheap (no I/O until the first read or write). Shares
    a database file with :class:`HealthStore` by convention so the
    operator has a single backup target for the observability plane.

    Concurrency: a per-store :class:`threading.Lock` serializes writes
    across FastAPI's threadpool. Reads don't block writes thanks to
    WAL mode. This matches the established :class:`HealthStore`
    pattern — keep them consistent so operators don't have to learn
    two contention models.
    """

    def __init__(
        self, db_path: Path, *, check_same_thread: bool = False
    ) -> None:
        self._db_path = db_path
        self._check_same_thread = check_same_thread
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=self._check_same_thread,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def record(self, events: Iterable[AccuracyEvent]) -> int:
        """Append ``events`` to the ledger. Returns the count written.

        Best-effort — sqlite errors propagate so callers can catch and
        log. The fundamentals service wraps this call in a broad
        ``except``; do likewise from other callers.
        """
        materialized = list(events)
        if not materialized:
            return 0
        with self._lock:
            conn = self._connect()
            conn.execute("BEGIN")
            try:
                conn.executemany(
                    """
                    INSERT INTO provider_accuracy_event
                        (provider, reference_provider, symbol, field,
                         observed_value, reference_value, rel_error,
                         agreed, fiscal_date, observed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            e.provider.value,
                            e.reference_provider.value,
                            e.symbol,
                            e.field,
                            e.observed_value,
                            e.reference_value,
                            float(e.rel_error),
                            1 if e.agreed else 0,
                            e.fiscal_date.isoformat() if e.fiscal_date else None,
                            e.observed_at.isoformat(),
                        )
                        for e in materialized
                    ],
                )
                conn.execute("COMMIT")
            except sqlite3.Error:
                conn.execute("ROLLBACK")
                raise
        return len(materialized)

    def summarize(
        self,
        provider: ProviderName | str,
        window: timedelta,
        *,
        now: datetime | None = None,
    ) -> ProviderAccuracySummary:
        """Aggregate accuracy for ``provider`` over the last ``window``.

        Returns a summary even when no events exist (totals are zero,
        ``overall_accuracy = 0.0``, ``by_field = ()``). Callers must
        handle the empty case — a missing summary is meaningfully
        different from a zero-accuracy one (the trust-weight module
        bootstraps to 1.0 on absence).

        ``now`` defaults to ``datetime.now(UTC)``; tests pass an
        explicit value for determinism.
        """
        if window <= timedelta(0):
            raise ValueError(f"window must be positive, got {window!r}")
        provider_str = provider.value if isinstance(provider, ProviderName) else provider
        moment = now or datetime.now(UTC)
        cutoff = moment - window

        conn = self._connect()
        cur = conn.execute(
            """
            SELECT field,
                   COUNT(*) AS total,
                   SUM(agreed) AS agreed_count
            FROM provider_accuracy_event
            WHERE provider = ? AND observed_at >= ?
            GROUP BY field
            ORDER BY field ASC
            """,
            (provider_str, cutoff.isoformat()),
        )
        by_field: list[FieldAccuracy] = []
        total_events = 0
        total_agreed = 0
        for field_name, total, agreed_count in cur.fetchall():
            agreed_int = int(agreed_count or 0)
            total_int = int(total)
            total_events += total_int
            total_agreed += agreed_int
            accuracy = (agreed_int / total_int) if total_int > 0 else 0.0
            by_field.append(
                FieldAccuracy(
                    provider=provider_str,
                    field=field_name,
                    total=total_int,
                    agreed=agreed_int,
                    accuracy=accuracy,
                )
            )

        overall = (total_agreed / total_events) if total_events > 0 else 0.0
        return ProviderAccuracySummary(
            provider=provider_str,
            window=window,
            total_events=total_events,
            total_agreed=total_agreed,
            overall_accuracy=overall,
            by_field=tuple(by_field),
        )

    def known_providers(self) -> tuple[str, ...]:
        """Return distinct provider names that have any rows in the ledger.

        Used by the providers health endpoint to enumerate the cohort
        for trust-breakdown surfacing. Includes both the ``provider``
        and ``reference_provider`` sides so a provider that's only
        ever served as a reference still appears (its accuracy is
        observable through the symmetric comparisons).
        """
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT provider AS name FROM provider_accuracy_event
            UNION
            SELECT reference_provider AS name FROM provider_accuracy_event
            """
        )
        names = sorted({str(row[0]) for row in cur.fetchall()})
        return tuple(names)

    def has_events(self, provider: ProviderName | str) -> bool:
        """Cheap "any history?" check used by the trust-weight cold start.

        Returns ``True`` if at least one row exists for ``provider``
        as either the observed or the reference side. Used to decide
        whether to apply a computed weight or default to 1.0.
        """
        provider_str = provider.value if isinstance(provider, ProviderName) else provider
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT 1 FROM provider_accuracy_event
            WHERE provider = ? OR reference_provider = ?
            LIMIT 1
            """,
            (provider_str, provider_str),
        )
        return cur.fetchone() is not None


__all__ = [
    "AccuracyStore",
    "FieldAccuracy",
    "ProviderAccuracySummary",
]
