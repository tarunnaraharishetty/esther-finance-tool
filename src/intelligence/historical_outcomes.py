"""Per-symbol historical signal outcomes — drill-down on the calibration data.

The analyzer endpoint already surfaces *pooled* hit rates from the
calibration table (P2.3). That's the right primary signal — pooled
buckets reach defensible sample sizes faster — but it averages out
per-symbol behavior. This module zooms in:

    "For AAPL specifically, when pullback_risk was 70+, 8 of 12
     observations closed lower over 5 days (Wilson CI 39–87%)."

Why a separate aggregation
--------------------------
The :class:`CalibrationTable` is materialized cross-symbol. Re-using
it would let the per-symbol view drift behind a stale build pass.
This module reads observations *live* from the store and bins them
in memory — bounded per symbol (typically tens to hundreds of
settled rows), trivial to compute on demand.

Why a lower ``min_observations`` than pooled
--------------------------------------------
Per-symbol observation counts are an order of magnitude smaller
than pooled. The pooled view uses 30 as the publish floor; here
the default is 5 — the Wilson CI widens to compensate, so the
trader sees the uncertainty without losing access to early reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from src.intelligence.calibration import (
    CalibrationStore,
    Observation,
    wilson_interval,
)


@dataclass(frozen=True)
class SymbolBucket:
    """One per-symbol bucket of observed outcomes.

    Same shape as :class:`~src.intelligence.calibration.CalibrationBucket`
    plus a ``bucket_published`` flag — the per-symbol view publishes
    or hides buckets on its own threshold (typically lower than
    pooled), so the flag is computed here rather than inferred at
    render time.
    """

    score_name: str
    outcome_name: str
    horizon_days: int
    bucket_lo: float
    bucket_hi: float
    n_observations: int
    n_hits: int
    hit_rate: float
    confidence_low: float
    confidence_high: float
    bucket_published: bool


@dataclass(frozen=True)
class HistoricalOutcomes:
    """Aggregate per-symbol outcome view for one symbol.

    ``total_observations`` includes pending (unsettled) rows so the
    UI can say "12 settled · 3 still maturing"; ``settled_observations``
    is the count actually used for the buckets.
    """

    symbol: str
    horizon_days: int
    total_observations: int
    settled_observations: int
    buckets: tuple[SymbolBucket, ...]
    first_observed_at: datetime | None
    last_settled_at: datetime | None


def symbol_outcomes(
    store: CalibrationStore,
    symbol: str,
    *,
    horizon_days: int,
    bucket_width: float = 10.0,
    min_observations: int = 5,
) -> HistoricalOutcomes:
    """Read the symbol's observations from ``store`` and bucket them.

    Filters in-memory rather than at the SQL layer because the
    existing ``all_observations()`` filters by score_name + outcome
    but not by symbol — adding a symbol filter to the store API would
    couple it to this per-symbol use case for no real performance
    win (the score+outcome+horizon index already prunes the row count
    aggressively).
    """
    if bucket_width <= 0 or bucket_width > 100:
        raise ValueError(
            f"bucket_width must be in (0, 100], got {bucket_width!r}"
        )
    if min_observations < 1:
        raise ValueError(
            f"min_observations must be >= 1, got {min_observations!r}"
        )
    sym = symbol.upper()

    all_rows = store.all_observations(horizon_days=horizon_days)
    symbol_rows = [r for r in all_rows if r.symbol == sym]
    settled = [r for r in symbol_rows if r.outcome_value is not None]

    first_observed_at = (
        min(r.observed_at for r in symbol_rows) if symbol_rows else None
    )
    last_settled_at = (
        max(r.observed_at for r in settled) if settled else None
    )

    bins: dict[tuple[str, str, float, float], list[Observation]] = {}
    bucket_count = int(100 // bucket_width)
    for obs in settled:
        value = max(0.0, min(100.0, obs.score_value))
        index = min(int(value // bucket_width), bucket_count - 1)
        lo = index * bucket_width
        hi = lo + bucket_width
        key = (obs.score_name, obs.outcome_name, lo, hi)
        bins.setdefault(key, []).append(obs)

    buckets: list[SymbolBucket] = []
    for (score_name, outcome_name, lo, hi), rows in sorted(bins.items()):
        n = len(rows)
        hits = sum(1 for r in rows if r.outcome_value is True)
        hit_rate = hits / n if n else 0.0
        ci_lo, ci_hi = wilson_interval(hits, n)
        buckets.append(
            SymbolBucket(
                score_name=score_name,
                outcome_name=outcome_name,
                horizon_days=horizon_days,
                bucket_lo=lo,
                bucket_hi=hi,
                n_observations=n,
                n_hits=hits,
                hit_rate=hit_rate,
                confidence_low=ci_lo,
                confidence_high=ci_hi,
                bucket_published=n >= min_observations,
            )
        )

    return HistoricalOutcomes(
        symbol=sym,
        horizon_days=horizon_days,
        total_observations=len(symbol_rows),
        settled_observations=len(settled),
        buckets=tuple(buckets),
        first_observed_at=first_observed_at,
        last_settled_at=last_settled_at,
    )


def outcomes_to_wire(outcomes: HistoricalOutcomes) -> dict[str, object]:
    """JSON wire shape for the ``/api/history/{symbol}`` endpoint.

    Bucket fields are flattened (no nested ``bucket`` object) so the
    frontend reads them uniformly. Datetimes are ISO 8601 with tz.
    """
    return {
        "symbol": outcomes.symbol,
        "horizon_days": outcomes.horizon_days,
        "total_observations": outcomes.total_observations,
        "settled_observations": outcomes.settled_observations,
        "first_observed_at": (
            outcomes.first_observed_at.isoformat()
            if outcomes.first_observed_at is not None
            else None
        ),
        "last_settled_at": (
            outcomes.last_settled_at.isoformat()
            if outcomes.last_settled_at is not None
            else None
        ),
        "buckets": [
            {
                "score_name": b.score_name,
                "outcome_name": b.outcome_name,
                "horizon_days": b.horizon_days,
                "bucket_lo": b.bucket_lo,
                "bucket_hi": b.bucket_hi,
                "n_observations": b.n_observations,
                "n_hits": b.n_hits,
                "hit_rate": round(b.hit_rate, 4),
                "confidence_low": round(b.confidence_low, 4),
                "confidence_high": round(b.confidence_high, 4),
                "bucket_published": b.bucket_published,
            }
            for b in outcomes.buckets
        ],
    }


# ``math`` import kept for forward compatibility — future per-bucket
# computations (e.g., median return, drawdown) will land here without
# adding the import. ruff RUF059 is silenced via this explicit use.
_ = math


__all__ = [
    "HistoricalOutcomes",
    "SymbolBucket",
    "outcomes_to_wire",
    "symbol_outcomes",
]
