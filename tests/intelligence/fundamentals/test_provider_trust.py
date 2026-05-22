"""Tests for the :class:`ProviderTrust` weight composer.

Covers cold-start neutrality, the per-component computation, the
``[0.5, 1.2]`` floor/ceiling, the TTL cache, and the
defense-in-depth: a store-side error must collapse to ``1.0``
rather than break a fundamentals fetch.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.data.accuracy_store import AccuracyStore
from src.data.health_store import HealthStore
from src.intelligence.fundamentals.models import (
    AccuracyEvent,
    ProviderHealth,
    ProviderName,
)
from src.intelligence.fundamentals.provider_trust import ProviderTrust

# ---------------------------------------------------------------------------
# Cold-start neutrality
# ---------------------------------------------------------------------------


def test_no_stores_returns_neutral_weight() -> None:
    """Both stores ``None`` → 1.0 with cold_start flag set."""
    trust = ProviderTrust(accuracy_store=None, health_store=None)
    bd = trust.breakdown_for(ProviderName.FMP)
    assert bd.weight == 1.0
    assert bd.cold_start is True
    assert bd.accuracy is None
    assert bd.uptime is None


def test_empty_stores_return_neutral_weight(tmp_path: Path) -> None:
    """Stores wired but never written → 1.0, not 0.0."""
    acc = AccuracyStore(tmp_path / "acc.db")
    health = HealthStore(tmp_path / "health.db")
    trust = ProviderTrust(accuracy_store=acc, health_store=health)
    assert trust.weight_for(ProviderName.FMP) == 1.0
    acc.close()
    health.close()


def test_below_min_events_returns_neutral_even_if_inaccurate(
    tmp_path: Path,
) -> None:
    """Don't punish a provider until we have statistically meaningful data.

    A handful of disagreements is noise, not signal — the weight stays
    at 1.0 until the ledger accumulates at least ``min_events`` rows.
    """
    acc = AccuracyStore(tmp_path / "acc.db")
    health = HealthStore(tmp_path / "health.db")
    now = datetime.now(UTC)
    # 5 disagreements; min_events default = 20 → too thin to act on.
    acc.record(
        [
            _event(agreed=False, observed_at=now)
            for _ in range(5)
        ]
    )
    trust = ProviderTrust(accuracy_store=acc, health_store=health)
    bd = trust.breakdown_for(ProviderName.FMP)
    assert bd.weight == 1.0
    assert bd.accuracy is None  # signal not used → reported as None
    assert bd.events == 5
    acc.close()
    health.close()


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


def test_perfect_provider_earns_ceiling_weight(tmp_path: Path) -> None:
    """100% accuracy + 100% uptime + fast latency → ~1.2 ceiling."""
    acc = AccuracyStore(tmp_path / "acc.db")
    health = HealthStore(tmp_path / "health.db")
    now = datetime.now(UTC)
    acc.record([_event(agreed=True, observed_at=now) for _ in range(50)])
    health.record(
        [_health(status="ok", latency_ms=50.0, checked_at=now) for _ in range(50)]
    )
    trust = ProviderTrust(accuracy_store=acc, health_store=health)
    bd = trust.breakdown_for(ProviderName.FMP)
    assert bd.weight == pytest.approx(1.2)
    assert bd.cold_start is False
    assert bd.accuracy == 1.0
    assert bd.uptime == 1.0
    assert bd.latency_score == 1.0
    acc.close()
    health.close()


def test_catastrophic_provider_floors_at_half(tmp_path: Path) -> None:
    """0% accuracy + 0% uptime + slow latency → 0.5 floor."""
    acc = AccuracyStore(tmp_path / "acc.db")
    health = HealthStore(tmp_path / "health.db")
    now = datetime.now(UTC)
    acc.record([_event(agreed=False, observed_at=now) for _ in range(50)])
    health.record(
        [
            _health(status="unavailable", latency_ms=3000.0, checked_at=now)
            for _ in range(50)
        ]
    )
    trust = ProviderTrust(accuracy_store=acc, health_store=health)
    bd = trust.breakdown_for(ProviderName.FMP)
    assert bd.weight == pytest.approx(0.5)
    assert bd.accuracy == 0.0
    assert bd.uptime == 0.0
    assert bd.latency_score == 0.0
    acc.close()
    health.close()


def test_mid_range_provider_lands_near_neutral(tmp_path: Path) -> None:
    """80% accuracy + 100% uptime + fast latency → weight near 1.0."""
    acc = AccuracyStore(tmp_path / "acc.db")
    health = HealthStore(tmp_path / "health.db")
    now = datetime.now(UTC)
    # 40 agreements, 10 disagreements → 80% accuracy
    acc.record(
        [_event(agreed=True, observed_at=now) for _ in range(40)]
        + [_event(agreed=False, observed_at=now) for _ in range(10)]
    )
    health.record(
        [_health(status="ok", latency_ms=100.0, checked_at=now) for _ in range(30)]
    )
    trust = ProviderTrust(accuracy_store=acc, health_store=health)
    bd = trust.breakdown_for(ProviderName.FMP)
    # 0.65*0.8 + 0.25*1.0 + 0.10*1.0 = 0.87 trust → weight = 0.5 + 0.7*0.87 ≈ 1.109
    assert 1.0 < bd.weight < 1.2
    assert bd.accuracy == pytest.approx(0.8)
    assert bd.uptime == pytest.approx(1.0)
    acc.close()
    health.close()


def test_partial_signals_renormalize_present_components(tmp_path: Path) -> None:
    """Accuracy alone (no health data) drives the weight by itself.

    Tests the renormalization in ``_compute`` — when uptime/latency
    aren't present, the accuracy component takes the full weight
    rather than being diluted by missing signals.
    """
    acc = AccuracyStore(tmp_path / "acc.db")
    now = datetime.now(UTC)
    acc.record([_event(agreed=True, observed_at=now) for _ in range(50)])
    trust = ProviderTrust(accuracy_store=acc, health_store=None)
    bd = trust.breakdown_for(ProviderName.FMP)
    # Only accuracy = 1.0 contributes; weight is the ceiling.
    assert bd.weight == pytest.approx(1.2)
    assert bd.uptime is None
    assert bd.latency_score is None
    acc.close()


# ---------------------------------------------------------------------------
# Latency scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p95_ms", "expected"),
    [
        (50.0, 1.0),
        (200.0, 1.0),
        (1100.0, 0.5),
        (2000.0, 0.0),
        (5000.0, 0.0),
    ],
)
def test_latency_score_is_piecewise_linear(p95_ms: float, expected: float) -> None:
    assert ProviderTrust._latency_score(p95_ms) == pytest.approx(expected)


def test_latency_score_none_in_none_out() -> None:
    assert ProviderTrust._latency_score(None) is None


# ---------------------------------------------------------------------------
# Cache + invalidate
# ---------------------------------------------------------------------------


def test_cache_serves_repeat_calls_without_store_access(tmp_path: Path) -> None:
    """Within TTL, repeat calls don't touch the store.

    Important on the hot fetch path — every analyzer call asks for
    the weight; without a cache that's an extra SQLite hit per request.
    """
    acc = AccuracyStore(tmp_path / "acc.db")
    trust = ProviderTrust(
        accuracy_store=acc,
        health_store=None,
        ttl_seconds=300.0,
    )
    first = trust.weight_for(ProviderName.FMP)
    # Close the store under the cache — if cache works, second call
    # still returns the same value rather than crashing on a closed DB.
    acc.close()
    second = trust.weight_for(ProviderName.FMP)
    assert first == second


def test_invalidate_drops_cache(tmp_path: Path) -> None:
    """Sink writes call invalidate() so new events surface immediately."""
    acc = AccuracyStore(tmp_path / "acc.db")
    trust = ProviderTrust(accuracy_store=acc, health_store=None, ttl_seconds=300.0)
    # Prime the cache with a cold-start 1.0.
    assert trust.weight_for(ProviderName.FMP) == 1.0
    # Now add evidence and invalidate; next call sees the new weight.
    now = datetime.now(UTC)
    acc.record([_event(agreed=True, observed_at=now) for _ in range(50)])
    trust.invalidate()
    assert trust.weight_for(ProviderName.FMP) == pytest.approx(1.2)
    acc.close()


def test_ttl_expiry_re_queries(tmp_path: Path) -> None:
    """After the TTL elapses, the next call re-queries the store."""
    acc = AccuracyStore(tmp_path / "acc.db")
    trust = ProviderTrust(accuracy_store=acc, health_store=None, ttl_seconds=0.05)
    first = trust.weight_for(ProviderName.FMP)
    now = datetime.now(UTC)
    acc.record([_event(agreed=True, observed_at=now) for _ in range(50)])
    time.sleep(0.1)
    second = trust.weight_for(ProviderName.FMP)
    assert first == 1.0
    assert second > 1.0
    acc.close()


# ---------------------------------------------------------------------------
# Defense in depth
# ---------------------------------------------------------------------------


def test_store_error_degrades_to_neutral_weight(tmp_path: Path) -> None:
    """A broken store must never break a fundamentals fetch.

    Simulated by closing the connection under the trust module's feet
    — query attempts raise ProgrammingError ("Cannot operate on a closed
    database"), which the module catches and treats as "no signal".
    """
    acc = AccuracyStore(tmp_path / "acc.db")
    # Force a connection so close() is meaningful.
    acc.record([_event()])
    acc.close()
    trust = ProviderTrust(accuracy_store=acc, health_store=None)
    assert trust.weight_for(ProviderName.FMP) == 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(
    *,
    provider: ProviderName = ProviderName.FMP,
    reference: ProviderName = ProviderName.SEC_EDGAR,
    field: str = "revenue",
    agreed: bool = True,
    observed_at: datetime | None = None,
) -> AccuracyEvent:
    return AccuracyEvent(
        provider=provider,
        reference_provider=reference,
        symbol="AAPL",
        field=field,
        observed_value=100.0,
        reference_value=100.0,
        rel_error=0.0 if agreed else 0.2,
        agreed=agreed,
        fiscal_date=datetime(2024, 12, 31, tzinfo=UTC),
        observed_at=observed_at or datetime.now(UTC),
    )


def _health(
    *,
    provider: ProviderName = ProviderName.FMP,
    status: str = "ok",
    latency_ms: float = 100.0,
    checked_at: datetime | None = None,
) -> ProviderHealth:
    return ProviderHealth(
        provider=provider,
        symbol="AAPL",
        status=status,
        latency_ms=latency_ms,
        checked_at=checked_at or datetime.now(UTC),
        error_message=None,
    )
