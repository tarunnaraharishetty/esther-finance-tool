"""Provider trust weighting from observed accuracy + uptime + latency.

Translates the rolling-window aggregates in :class:`AccuracyStore` and
:class:`HealthStore` into a single multiplicative weight applied to
the static position-based ``provider_confidence`` in
:class:`FundamentalsService`. The weight lives in ``[0.5, 1.2]``:

* No history → ``1.0`` (neutral; never punish a new provider).
* Perfect (100% accuracy, 100% uptime, fast) → ``1.2`` (small bonus).
* Half-wrong → ``≈0.85``.
* Catastrophic (0% accuracy or fully down) → ``0.5`` (floor).

The weight feeds directly into the platform's :class:`TrustScore`
via ``provider_confidence``; raising or lowering accuracy here moves
the trust grade on every report rendered from that provider.

Why a multiplier, not a gate
----------------------------
A down-weighted provider keeps getting called. We still need its
data to keep the reconciliation ledger fresh — pulling it from the
chain would freeze its accuracy at the historical moment of removal.
Multiplying its confidence preserves the feedback loop.

Cold start
----------
A provider with fewer than ``min_events`` reconciliation observations
in the window returns ``1.0`` regardless of accuracy. This protects
against thin samples dictating a heavy penalty before the ledger is
warm. ``min_events`` defaults to 20 — roughly five symbols worth of
quarterly reconciliations.

Caching
-------
The weight changes slowly (rolling 90-day window). A per-process
TTL cache (default 5 minutes) keeps the lookup cost near-free on the
hot fetch path.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from src.intelligence.fundamentals.models import ProviderName

if TYPE_CHECKING:
    from src.data.accuracy_store import AccuracyStore
    from src.data.health_store import HealthStore


# Composition coefficients (sum to 1.0). Tunable but the relative
# ordering reflects the platform's bet: accuracy dominates because
# it's what traders actually notice; uptime is the next-most-felt
# signal; latency is a tiebreaker.
_ACCURACY_WEIGHT = 0.65
_UPTIME_WEIGHT = 0.25
_LATENCY_WEIGHT = 0.10

# Multiplier range. Floor at 0.5 keeps a known-bad provider's data
# from disappearing entirely (the trader still sees it with a warning);
# ceiling at 1.2 gives the chain leader a modest bonus without
# dwarfing the static position curve.
_WEIGHT_FLOOR = 0.5
_WEIGHT_CEILING = 1.2

# A p95 latency at or below this number scores 1.0; above 2000ms
# scores 0.0; linearly interpolated in between.
_LATENCY_GREEN_MS = 200.0
_LATENCY_RED_MS = 2000.0


@dataclass(frozen=True)
class TrustBreakdown:
    """Components that produced a provider's trust weight.

    Useful for the operator dashboard ("why is FMP down-weighted to
    0.78?"). Carried alongside the numeric weight so the surfacing
    code never has to re-derive the inputs.
    """

    provider: str
    weight: float
    accuracy: float | None  # None when too few events for a signal
    uptime: float | None
    latency_p95_ms: float | None
    latency_score: float | None
    events: int
    health_calls: int
    cold_start: bool  # True when we returned 1.0 due to thin data


class ProviderTrust:
    """Compute per-provider trust weights from accuracy + health stores.

    Constructed once and shared across the fetch path. Internally
    caches per-provider weights for ``ttl_seconds`` to keep the hot
    path off the SQLite layer; the window is large enough (90 days
    default) that a 5-minute stale weight is operationally fine.
    """

    def __init__(
        self,
        *,
        accuracy_store: AccuracyStore | None,
        health_store: HealthStore | None,
        window: timedelta = timedelta(days=90),
        ttl_seconds: float = 300.0,
        min_events: int = 20,
    ) -> None:
        self._accuracy = accuracy_store
        self._health = health_store
        self._window = window
        self._ttl = ttl_seconds
        self._min_events = min_events
        self._cache: dict[str, tuple[float, TrustBreakdown]] = {}
        self._lock = threading.Lock()

    def weight_for(self, provider: ProviderName) -> float:
        """Look up the trust multiplier for ``provider`` in ``[0.5, 1.2]``.

        Falls through to ``1.0`` on any store-side error so a stuck
        SQLite lock can never break a fundamentals fetch — the
        provider-trust system is a confidence modifier, not a
        correctness gate.
        """
        return self.breakdown_for(provider).weight

    def breakdown_for(self, provider: ProviderName) -> TrustBreakdown:
        """Return the full :class:`TrustBreakdown` (weight + components).

        The same cache as :meth:`weight_for` so back-to-back lookups
        of "weight" and "breakdown" cost one query each only on the
        first call within the TTL.
        """
        key = provider.value
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                cached_at, breakdown = cached
                if now - cached_at < self._ttl:
                    return breakdown

        breakdown = self._compute(provider)
        with self._lock:
            self._cache[key] = (now, breakdown)
        return breakdown

    def invalidate(self) -> None:
        """Drop all cached weights. Called after a sink writes new events."""
        with self._lock:
            self._cache.clear()

    def _compute(self, provider: ProviderName) -> TrustBreakdown:
        accuracy_signal, events = self._accuracy_signal(provider)
        uptime_signal, health_calls, latency_p95_ms = self._health_signal(provider)
        latency_score = self._latency_score(latency_p95_ms)

        if accuracy_signal is None and uptime_signal is None:
            # No observations at all → neutral. The new-provider case
            # plus the case where stores aren't wired both land here.
            return TrustBreakdown(
                provider=provider.value,
                weight=1.0,
                accuracy=None,
                uptime=None,
                latency_p95_ms=latency_p95_ms,
                latency_score=latency_score,
                events=events,
                health_calls=health_calls,
                cold_start=True,
            )

        # Re-weight the present signals to keep the [0, 1] interpretation
        # honest. A provider with health data but no accuracy events
        # gets scored on uptime + latency only — the weight then reflects
        # what we actually know about it.
        contributions: list[tuple[float, float]] = []
        if accuracy_signal is not None:
            contributions.append((_ACCURACY_WEIGHT, accuracy_signal))
        if uptime_signal is not None:
            contributions.append((_UPTIME_WEIGHT, uptime_signal))
        if latency_score is not None:
            contributions.append((_LATENCY_WEIGHT, latency_score))
        total_weight = sum(w for w, _ in contributions)
        trust = (
            sum(w * v for w, v in contributions) / total_weight
            if total_weight > 0
            else 1.0
        )
        weight = max(
            _WEIGHT_FLOOR,
            min(_WEIGHT_CEILING, _WEIGHT_FLOOR + (_WEIGHT_CEILING - _WEIGHT_FLOOR) * trust),
        )
        return TrustBreakdown(
            provider=provider.value,
            weight=weight,
            accuracy=accuracy_signal,
            uptime=uptime_signal,
            latency_p95_ms=latency_p95_ms,
            latency_score=latency_score,
            events=events,
            health_calls=health_calls,
            cold_start=False,
        )

    def _accuracy_signal(
        self, provider: ProviderName
    ) -> tuple[float | None, int]:
        """Pull rolling overall accuracy. ``None`` when below ``min_events``."""
        if self._accuracy is None:
            return None, 0
        try:
            summary = self._accuracy.summarize(provider, self._window)
        except Exception as exc:
            log_msg = "provider_trust.accuracy_lookup.failed"
            _quiet_log(log_msg, provider=provider.value, error=str(exc))
            return None, 0
        if summary.total_events < self._min_events:
            return None, summary.total_events
        return summary.overall_accuracy, summary.total_events

    def _health_signal(
        self, provider: ProviderName
    ) -> tuple[float | None, int, float | None]:
        """Return ``(uptime, total_non_skipped, p95_latency_ms)``.

        Uses :meth:`HealthStore.summarize` for one window. Returns
        ``(None, 0, None)`` when the store isn't wired, the window is
        empty, or only ``skipped`` rows are present (a provider that
        was never tried has no uptime signal).
        """
        if self._health is None:
            return None, 0, None
        try:
            summaries = self._health.summarize(self._window)
        except Exception as exc:
            _quiet_log(
                "provider_trust.health_lookup.failed",
                provider=provider.value,
                error=str(exc),
            )
            return None, 0, None
        match = next(
            (s for s in summaries if s.provider == provider.value), None
        )
        if match is None:
            return None, 0, None
        non_skipped = match.total - match.skipped
        if non_skipped <= 0:
            return None, 0, match.p95_latency_ms
        # ``HealthStore.summarize`` already gates skipped rows out of
        # its success_rate calculation, so we can use it directly.
        return match.success_rate, non_skipped, match.p95_latency_ms

    @staticmethod
    def _latency_score(p95_ms: float | None) -> float | None:
        """Map p95 latency to a [0, 1] score.

        Below ``_LATENCY_GREEN_MS`` (200ms) → 1.0. Above
        ``_LATENCY_RED_MS`` (2000ms) → 0.0. Linear between. ``None``
        in, ``None`` out.
        """
        if p95_ms is None:
            return None
        if p95_ms <= _LATENCY_GREEN_MS:
            return 1.0
        if p95_ms >= _LATENCY_RED_MS:
            return 0.0
        span = _LATENCY_RED_MS - _LATENCY_GREEN_MS
        return 1.0 - (p95_ms - _LATENCY_GREEN_MS) / span


def _quiet_log(event: str, **kwargs: object) -> None:
    """Local late-bind import for the structlog helper.

    Avoids a top-level import cycle and keeps ``provider_trust`` clean
    when tests construct it without the logging machinery wired.
    """
    from src.utils.logging import get_logger

    get_logger(__name__).warning(event, **kwargs)


__all__ = [
    "ProviderTrust",
    "TrustBreakdown",
]
