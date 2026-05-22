"""Signal-history calibration.

Pools score observations across symbols and converts them into
historically grounded probabilities — turning "pullback_risk = 73"
into "in 412 prior observations of pullback_risk ∈ [70, 80), the
next-5-day return was negative 64% of the time (Wilson 95% CI:
59–69%)".

Different from :mod:`~src.intelligence.analyzer.scenarios`
--------------------------------------------------------
Scenarios answer "what's today's dispersion?" via a forward-looking
lognormal. Calibration answers "what has this setup actually done in
the past?" via descriptive statistics on accumulated history. Both
are useful; they're not interchangeable. The UI surfaces both.

Why pool across symbols
-----------------------
A single symbol's history rarely has enough observations to reach a
defensible sample size in any given score bucket. Pooling washes out
per-symbol behavior — AAPL's "pullback_risk = 73" is operationally
different from a small-cap biotech's — but it's the only way to get
to N ≥ 30 per bucket inside a reasonable horizon. Per-sector or
per-vol-regime stratification is a future refinement; the
pooled-cross-symbol output is the right MVP.

Why Wilson over normal-approximation
------------------------------------
The Wilson 95% interval is well-defined at small n and at the p=0 /
p=1 corners. The normal-approximation interval (``p ± 1.96·√(p(1-p)/n)``)
breaks at the boundaries and is too narrow at small n — exactly when
the trader most needs accurate uncertainty bounds.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger(__name__)


# Schema version. Additive-only migrations going forward; an
# incompatible change bumps this and drops the calibration_buckets
# table (the observations table is the durable record).
#
# v1 → v2: added ``starting_price`` to observations so the
# maturation worker can grade returns without re-fetching historical
# bars. Existing v1 rows have NULL starting_price and are skipped at
# settlement time (the worker can't grade them).
_SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calibration_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    score_name TEXT NOT NULL,
    score_value REAL NOT NULL,
    observed_at TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    outcome_name TEXT NOT NULL,
    outcome_value INTEGER,
    starting_price REAL
);
CREATE INDEX IF NOT EXISTS idx_obs_score
    ON observations(score_name, outcome_name, horizon_days);
CREATE INDEX IF NOT EXISTS idx_obs_observed_at
    ON observations(observed_at);

CREATE TABLE IF NOT EXISTS calibration_buckets (
    score_name TEXT NOT NULL,
    bucket_lo REAL NOT NULL,
    bucket_hi REAL NOT NULL,
    horizon_days INTEGER NOT NULL,
    outcome_name TEXT NOT NULL,
    n_observations INTEGER NOT NULL,
    n_hits INTEGER NOT NULL,
    hit_rate REAL NOT NULL,
    confidence_low REAL NOT NULL,
    confidence_high REAL NOT NULL,
    last_updated TEXT NOT NULL,
    PRIMARY KEY (score_name, bucket_lo, horizon_days, outcome_name)
);
"""

# Wilson 95% interval z-score.
_WILSON_Z = 1.959963984540054  # scipy.stats.norm.ppf(0.975)


@dataclass(frozen=True)
class Observation:
    """One scored observation, optionally with a realized outcome.

    ``outcome_value`` is ``None`` until the horizon matures and the
    realized binary outcome is recorded. ``True`` means the outcome
    fired (e.g., "next-5-day return was negative" when
    ``outcome_name`` is ``"return_negative"``); ``False`` means it
    didn't.

    ``starting_price`` is the symbol's price at observation time;
    the maturation worker compares it against the symbol's price at
    :attr:`matures_at` to derive ``outcome_value``. ``None`` only on
    rows written before schema v2 — those can't be settled and the
    worker skips them.
    """

    id: int
    symbol: str
    score_name: str
    score_value: float
    observed_at: datetime
    horizon_days: int
    outcome_name: str
    outcome_value: bool | None
    starting_price: float | None = None

    @property
    def matures_at(self) -> datetime:
        """When the horizon completes and the outcome can be recorded."""
        return self.observed_at + timedelta(days=self.horizon_days)


@dataclass(frozen=True)
class CalibrationBucket:
    """Materialized hit-rate row for one score bucket.

    A bucket spans ``[bucket_lo, bucket_hi)`` except for the top
    bucket, where the upper bound is inclusive (so a score of
    exactly 100.0 falls into the [90, 100] bin).
    """

    score_name: str
    bucket_lo: float
    bucket_hi: float
    horizon_days: int
    outcome_name: str
    n_observations: int
    n_hits: int
    hit_rate: float
    confidence_low: float
    confidence_high: float
    last_updated: datetime

    def contains(self, score_value: float) -> bool:
        """Whether ``score_value`` falls into this bucket.

        Inclusive on the lower bound; exclusive on the upper bound
        *unless* this is the top bucket (upper bound 100), in which
        case the upper bound is inclusive — so a perfect 100 score
        is never a no-bucket fallthrough.
        """
        if score_value < self.bucket_lo:
            return False
        if math.isclose(self.bucket_hi, 100.0):
            return score_value <= self.bucket_hi
        return score_value < self.bucket_hi


@dataclass(frozen=True)
class CalibrationTable:
    """An ordered snapshot of every populated bucket."""

    buckets: tuple[CalibrationBucket, ...]
    built_at: datetime

    @property
    def total_observations(self) -> int:
        return sum(b.n_observations for b in self.buckets)

    def lookup(
        self,
        score_name: str,
        score_value: float,
        *,
        horizon_days: int,
        outcome_name: str,
        min_observations: int = 30,
    ) -> CalibrationBucket | None:
        """Return the bucket for ``score_value`` or ``None`` when under-sampled.

        The ``min_observations`` gate exists to prevent the UI from
        publishing a probability without a defensible sample size
        behind it. Default 30 is conservative; tune via settings.
        """
        for bucket in self.buckets:
            if bucket.score_name != score_name:
                continue
            if bucket.horizon_days != horizon_days:
                continue
            if bucket.outcome_name != outcome_name:
                continue
            if not bucket.contains(score_value):
                continue
            if bucket.n_observations < min_observations:
                return None
            return bucket
        return None


class CalibrationStore:
    """SQLite-backed observation log + materialized bucket cache.

    Lazy: constructing the store performs no I/O. The connection
    opens on the first read/write — matches the contract on
    :class:`~src.data.health_store.HealthStore`.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        check_same_thread: bool = False,
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
        _migrate_observations_v1_to_v2(conn)
        conn.execute(
            "INSERT OR REPLACE INTO calibration_meta(key, value) VALUES (?, ?)",
            ("schema_version", str(_SCHEMA_VERSION)),
        )
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---- ingest ----

    def record_observation(
        self,
        *,
        symbol: str,
        score_name: str,
        score_value: float,
        observed_at: datetime,
        horizon_days: int,
        outcome_name: str,
        starting_price: float | None = None,
    ) -> int:
        """Insert a new observation with no outcome yet.

        Returns the row id so the caller can later supply the realized
        outcome via :meth:`record_outcome`. The pair (observation_id,
        outcome) is the atomic unit the maturation worker manipulates.

        ``starting_price`` is the symbol's price at observation time.
        The maturation worker uses it to grade returns without a
        second history lookup. ``None`` is accepted for backward
        compatibility with the schema-v1 ingestion path but the
        observation will never settle automatically.
        """
        if horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {horizon_days!r}")
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                """
                INSERT INTO observations
                    (symbol, score_name, score_value, observed_at,
                     horizon_days, outcome_name, outcome_value, starting_price)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    symbol.upper(),
                    score_name,
                    float(score_value),
                    observed_at.isoformat(),
                    int(horizon_days),
                    outcome_name,
                    None if starting_price is None else float(starting_price),
                ),
            )
            row_id = cur.lastrowid
        if row_id is None:
            raise RuntimeError("INSERT did not return lastrowid")
        return int(row_id)

    def record_outcome(self, observation_id: int, outcome_value: bool) -> bool:
        """Set the realized outcome for one observation.

        Returns True when the row existed and was updated; False when
        the id is unknown (the maturation worker can no-op on a race
        with a manual delete).
        """
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                """
                UPDATE observations
                SET outcome_value = ?
                WHERE id = ?
                """,
                (1 if outcome_value else 0, observation_id),
            )
        return cur.rowcount > 0

    def matured_unsettled(
        self, *, now: datetime | None = None
    ) -> tuple[Observation, ...]:
        """Observations whose horizon has elapsed but outcome is still null.

        Drives the :class:`CalibrationMaturationWorker`: the worker
        takes each, looks up the realized price at
        ``observation.matures_at``, and calls :meth:`record_outcome`.
        """
        moment = now or datetime.now(UTC)
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                """
                SELECT id, symbol, score_name, score_value, observed_at,
                       horizon_days, outcome_name, outcome_value, starting_price
                FROM observations
                WHERE outcome_value IS NULL
                ORDER BY observed_at ASC
                """
            )
            rows = cur.fetchall()
        out: list[Observation] = []
        for r in rows:
            obs = _observation_from_row(r)
            if obs.matures_at <= moment:
                out.append(obs)
        return tuple(out)

    def cleanup(self, *, older_than: timedelta) -> int:
        """Drop unsettled observations older than ``older_than``.

        Operational-need helper: when a symbol rotates off the
        watchlist mid-horizon, its observations are left unsettleable
        — the maturation worker can't grade them because no fresh
        bar will arrive. After a sufficient quiet period (typically
        weeks past the configured horizon), it's safe to drop them.

        **Settled observations are never removed.** They're the
        calibration data; their value compounds over time. Cleanup
        is strictly for the "horizon expired without ever grading"
        case.

        Returns the number of rows removed.
        """
        if older_than.total_seconds() < 0:
            raise ValueError(f"older_than must be non-negative, got {older_than!r}")
        cutoff = datetime.now(UTC) - older_than
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                """
                DELETE FROM observations
                WHERE outcome_value IS NULL
                  AND observed_at < ?
                """,
                (cutoff.isoformat(),),
            )
            removed = cur.rowcount
        log.info("calibration.cleanup", removed=removed, older_than=str(older_than))
        return int(removed)

    def all_observations(
        self,
        *,
        score_name: str | None = None,
        outcome_name: str | None = None,
        horizon_days: int | None = None,
        settled_only: bool = False,
    ) -> tuple[Observation, ...]:
        """Read observations with optional filters. Used by the build pass."""
        where: list[str] = []
        params: list[object] = []
        if score_name is not None:
            where.append("score_name = ?")
            params.append(score_name)
        if outcome_name is not None:
            where.append("outcome_name = ?")
            params.append(outcome_name)
        if horizon_days is not None:
            where.append("horizon_days = ?")
            params.append(horizon_days)
        if settled_only:
            where.append("outcome_value IS NOT NULL")
        sql = (
            "SELECT id, symbol, score_name, score_value, observed_at, "
            "horizon_days, outcome_name, outcome_value, starting_price FROM observations"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY observed_at ASC"
        with self._lock:
            conn = self._connect()
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
        return tuple(_observation_from_row(r) for r in rows)

    # ---- bucket build ----

    def build_table(
        self,
        *,
        score_names: Iterable[str],
        horizon_days: int,
        outcome_name: str,
        bucket_width: float = 10.0,
        now: datetime | None = None,
    ) -> CalibrationTable:
        """Materialize a :class:`CalibrationTable` and persist it.

        Walks every settled observation matching the parameters,
        buckets by ``score_value``, computes hit_rate + Wilson 95%
        CI per bucket, and writes both the in-memory result and the
        ``calibration_buckets`` cache table.

        The write is atomic: a single transaction deletes the prior
        rows for the given (score_name, horizon_days, outcome_name)
        triple, then inserts the new ones. A reader running
        concurrently never sees a half-built table.
        """
        if bucket_width <= 0 or bucket_width > 100:
            raise ValueError(
                f"bucket_width must be in (0, 100], got {bucket_width!r}"
            )
        moment = now or datetime.now(UTC)
        result_buckets: list[CalibrationBucket] = []
        for score_name in score_names:
            observations = self.all_observations(
                score_name=score_name,
                outcome_name=outcome_name,
                horizon_days=horizon_days,
                settled_only=True,
            )
            bins = _bin_observations(observations, bucket_width=bucket_width)
            for (lo, hi), (n, hits) in sorted(bins.items()):
                if n == 0:
                    continue
                hit_rate = hits / n
                ci_lo, ci_hi = wilson_interval(hits, n)
                result_buckets.append(
                    CalibrationBucket(
                        score_name=score_name,
                        bucket_lo=lo,
                        bucket_hi=hi,
                        horizon_days=horizon_days,
                        outcome_name=outcome_name,
                        n_observations=n,
                        n_hits=hits,
                        hit_rate=hit_rate,
                        confidence_low=ci_lo,
                        confidence_high=ci_hi,
                        last_updated=moment,
                    )
                )

        with self._lock:
            conn = self._connect()
            conn.execute("BEGIN")
            try:
                for score_name in score_names:
                    conn.execute(
                        """
                        DELETE FROM calibration_buckets
                        WHERE score_name = ? AND horizon_days = ? AND outcome_name = ?
                        """,
                        (score_name, horizon_days, outcome_name),
                    )
                conn.executemany(
                    """
                    INSERT INTO calibration_buckets
                        (score_name, bucket_lo, bucket_hi, horizon_days,
                         outcome_name, n_observations, n_hits, hit_rate,
                         confidence_low, confidence_high, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            b.score_name,
                            b.bucket_lo,
                            b.bucket_hi,
                            b.horizon_days,
                            b.outcome_name,
                            b.n_observations,
                            b.n_hits,
                            b.hit_rate,
                            b.confidence_low,
                            b.confidence_high,
                            b.last_updated.isoformat(),
                        )
                        for b in result_buckets
                    ],
                )
                conn.execute("COMMIT")
            except sqlite3.Error:
                conn.execute("ROLLBACK")
                raise

        log.info(
            "calibration.build.done",
            score_names=list(score_names),
            horizon_days=horizon_days,
            outcome_name=outcome_name,
            bucket_count=len(result_buckets),
            total_observations=sum(b.n_observations for b in result_buckets),
        )
        return CalibrationTable(
            buckets=tuple(result_buckets), built_at=moment
        )

    def load_table(self) -> CalibrationTable:
        """Read the materialized table from disk into memory.

        Cheap; the bucket table is bounded by
        ``buckets_per_score × score_count × horizon_count × outcome_count``,
        which is small enough to keep entirely in process memory and
        avoid a SQL round-trip per lookup.
        """
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                """
                SELECT score_name, bucket_lo, bucket_hi, horizon_days,
                       outcome_name, n_observations, n_hits, hit_rate,
                       confidence_low, confidence_high, last_updated
                FROM calibration_buckets
                ORDER BY score_name, horizon_days, outcome_name, bucket_lo
                """
            )
            rows = cur.fetchall()
        buckets = tuple(
            CalibrationBucket(
                score_name=r[0],
                bucket_lo=float(r[1]),
                bucket_hi=float(r[2]),
                horizon_days=int(r[3]),
                outcome_name=r[4],
                n_observations=int(r[5]),
                n_hits=int(r[6]),
                hit_rate=float(r[7]),
                confidence_low=float(r[8]),
                confidence_high=float(r[9]),
                last_updated=_parse_iso(r[10]),
            )
            for r in rows
        )
        return CalibrationTable(buckets=buckets, built_at=datetime.now(UTC))


# ---- helpers ----


def _bin_observations(
    observations: Iterable[Observation], *, bucket_width: float
) -> dict[tuple[float, float], tuple[int, int]]:
    """Group settled observations by score-value bucket.

    Returns ``{(lo, hi): (n_observations, n_hits)}``. The top bucket
    has an inclusive upper bound (the 100 score lands in [90, 100]).
    Observations with ``outcome_value is None`` are skipped — only
    settled outcomes inform the calibration table.
    """
    bins: dict[tuple[float, float], tuple[int, int]] = {}
    for obs in observations:
        if obs.outcome_value is None:
            continue
        value = max(0.0, min(100.0, obs.score_value))
        index = min(int(value // bucket_width), int(100 // bucket_width) - 1)
        lo = index * bucket_width
        hi = lo + bucket_width
        n, hits = bins.get((lo, hi), (0, 0))
        bins[(lo, hi)] = (
            n + 1,
            hits + (1 if obs.outcome_value else 0),
        )
    return bins


def wilson_interval(hits: int, n: int) -> tuple[float, float]:
    """Return the Wilson 95% confidence interval for a binomial proportion.

    Handles ``n == 0`` (returns (0, 1) — no information), ``hits == 0``
    and ``hits == n`` without divide-by-zero. The Wilson interval is
    asymmetric near the boundaries, which is the *correct* behavior at
    small n; the normal-approximation interval is symmetric and
    over-confident.
    """
    if n <= 0:
        return 0.0, 1.0
    p_hat = hits / n
    z = _WILSON_Z
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    half = (
        z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n))
    ) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def _observation_from_row(
    row: tuple[object, ...],
) -> Observation:
    outcome_raw = row[7]
    outcome_value: bool | None = (
        None if outcome_raw is None else bool(int(outcome_raw))  # type: ignore[call-overload]
    )
    starting_price_raw = row[8] if len(row) > 8 else None
    starting_price: float | None = (
        None if starting_price_raw is None else float(starting_price_raw)  # type: ignore[arg-type]
    )
    return Observation(
        id=int(row[0]),  # type: ignore[call-overload]
        symbol=str(row[1]),
        score_name=str(row[2]),
        score_value=float(row[3]),  # type: ignore[arg-type]
        observed_at=_parse_iso(str(row[4])),
        horizon_days=int(row[5]),  # type: ignore[call-overload]
        outcome_name=str(row[6]),
        outcome_value=outcome_value,
        starting_price=starting_price,
    )


def _migrate_observations_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Add the ``starting_price`` column to a pre-v2 observations table.

    SQLite has no ``ADD COLUMN IF NOT EXISTS`` — we inspect via
    ``PRAGMA table_info`` and add only when missing. Idempotent;
    runs on every connect because the cost is one quick query.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(observations)").fetchall()}
    if "starting_price" not in cols:
        conn.execute("ALTER TABLE observations ADD COLUMN starting_price REAL")
        log.info("calibration.schema.migrated", from_version=1, to_version=2)


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = [
    "CalibrationBucket",
    "CalibrationStore",
    "CalibrationTable",
    "Observation",
    "wilson_interval",
]
