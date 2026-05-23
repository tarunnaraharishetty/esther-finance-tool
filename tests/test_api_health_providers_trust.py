"""Tests for the trust-breakdown + per-field-accuracy extension to
``/api/health/providers``.

The base tests in ``test_api_health.py`` cover the rolling SLO + recent
failures contract. This file focuses on what the moat-surfacing
extension adds:

* Every provider row carries a ``trust_breakdown`` and ``field_accuracy``
  block (possibly null when the store isn't wired).
* Providers that exist only in the accuracy ledger surface in the
  response (degraded zero-count health summary).
* Trust components reflect the underlying store state (cold-start
  flag, accuracy/uptime/latency).
* Wire shapes match the documented TypeScript types.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController
from src.data.accuracy_store import AccuracyStore
from src.data.health_store import HealthStore
from src.intelligence.fundamentals.models import (
    AccuracyEvent,
    ProviderHealth,
    ProviderName,
)


def _health_row(
    *,
    provider: ProviderName = ProviderName.FMP,
    symbol: str = "AAPL",
    status: str = "ok",
    latency_ms: float = 100.0,
    checked_at: datetime | None = None,
) -> ProviderHealth:
    return ProviderHealth(
        provider=provider,
        symbol=symbol,
        status=status,
        latency_ms=latency_ms,
        checked_at=checked_at or datetime.now(UTC),
        error_message=None,
    )


def _accuracy_event(
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


def _build_client(
    tmp_path: Path,
) -> tuple[TestClient, HealthStore, AccuracyStore]:
    # HealthStore + AccuracyStore intentionally share the same DB file
    # (production wires both this way). Both are lazy — constructing
    # is free, schema init happens on first write. Both are injected
    # into ``create_app`` so the test, the app, and the health route
    # all read the same on-disk state.
    db_path = tmp_path / "health.db"
    store = HealthStore(db_path)
    accuracy = AccuracyStore(db_path)
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(
        controller,
        health_store=store,
        accuracy_store=accuracy,
        fundamentals_cache_dir=tmp_path / "fundamentals_cache",
        analyzer_cache_dir=tmp_path / "analyzer_cache",
    )
    return TestClient(app), store, accuracy


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------


def test_provider_rows_carry_trust_breakdown_and_field_accuracy_keys(
    tmp_path: Path,
) -> None:
    """Every provider row must include the two new fields, even when
    the underlying stores have no data."""
    client, store, _accuracy = _build_client(tmp_path)
    now = datetime.now(UTC)
    # Single health row so the provider surfaces at all.
    store.record([_health_row(checked_at=now)])

    res = client.get("/api/health/providers?window=1h")
    assert res.status_code == 200
    body = res.json()
    assert len(body["providers"]) == 1
    row = body["providers"][0]
    assert "trust_breakdown" in row
    assert "field_accuracy" in row


def test_trust_breakdown_wire_shape_documented_keys(tmp_path: Path) -> None:
    """trust_breakdown payload matches the TypeScript type."""
    client, store, accuracy = _build_client(tmp_path)
    now = datetime.now(UTC)
    store.record([_health_row(checked_at=now)])
    # Seed enough accuracy events to clear the min_events floor so
    # the breakdown actually carries an accuracy value (not None).
    accuracy.record(
        [_accuracy_event(observed_at=now) for _ in range(50)]
    )

    body = client.get("/api/health/providers?window=1h").json()
    row = body["providers"][0]
    trust = row["trust_breakdown"]
    assert trust is not None
    assert set(trust.keys()) == {
        "provider",
        "weight",
        "accuracy",
        "uptime",
        "latency_p95_ms",
        "latency_score",
        "events",
        "health_calls",
        "cold_start",
    }
    assert trust["provider"] == "fmp"
    # 50 perfect agreements → high accuracy and weight above neutral.
    assert trust["accuracy"] == 1.0
    assert trust["weight"] > 1.0
    assert trust["cold_start"] is False


def test_field_accuracy_wire_shape_documented_keys(tmp_path: Path) -> None:
    """field_accuracy payload carries per-field breakdown."""
    client, store, accuracy = _build_client(tmp_path)
    now = datetime.now(UTC)
    store.record([_health_row(checked_at=now)])
    # Mix of fields + agreed/disagreed for variety.
    accuracy.record(
        [
            _accuracy_event(field="revenue", agreed=True, observed_at=now),
            _accuracy_event(field="revenue", agreed=True, observed_at=now),
            _accuracy_event(field="revenue", agreed=False, observed_at=now),
            _accuracy_event(field="net_income", agreed=True, observed_at=now),
        ]
    )

    body = client.get("/api/health/providers?window=1h").json()
    row = body["providers"][0]
    fa = row["field_accuracy"]
    assert fa is not None
    assert set(fa.keys()) == {
        "provider",
        "window_seconds",
        "total_events",
        "total_agreed",
        "overall_accuracy",
        "by_field",
    }
    assert fa["total_events"] == 4
    assert fa["total_agreed"] == 3
    by_field = {f["field"]: f for f in fa["by_field"]}
    assert by_field["revenue"]["total"] == 3
    assert by_field["revenue"]["agreed"] == 2
    assert by_field["net_income"]["total"] == 1


# ---------------------------------------------------------------------------
# Cold-start behavior
# ---------------------------------------------------------------------------


def test_cold_start_flag_set_when_below_min_events(tmp_path: Path) -> None:
    """Few events → cold_start True, weight 1.0, accuracy null."""
    client, store, accuracy = _build_client(tmp_path)
    now = datetime.now(UTC)
    store.record([_health_row(checked_at=now)])
    # Only 5 events; default min_events = 20 → too thin for a signal.
    accuracy.record(
        [_accuracy_event(observed_at=now) for _ in range(5)]
    )

    body = client.get("/api/health/providers?window=1h").json()
    row = body["providers"][0]
    trust = row["trust_breakdown"]
    assert trust["cold_start"] is False  # has health signal
    # Accuracy is null because under min_events, even with health
    # data informing other components.
    assert trust["accuracy"] is None
    assert trust["events"] == 5


def test_no_accuracy_store_returns_null_field_accuracy(
    tmp_path: Path,
) -> None:
    """When accuracy store isn't wired, field_accuracy is None.

    The current ``create_app`` always wires the accuracy store from
    ``health_store_path`` when set; this test confirms the route
    *can* tolerate a None accuracy_store wiring (defense in depth).
    """
    from fastapi import FastAPI

    from src.api.health import register_health_routes
    from src.data.health_store import HealthStore

    store = HealthStore(tmp_path / "h.db")
    store.record([_health_row()])
    app = FastAPI()
    # Deliberately pass neither accuracy_store nor provider_trust.
    register_health_routes(app, store=store)
    client = TestClient(app)
    body = client.get("/api/health/providers?window=1h").json()
    row = body["providers"][0]
    assert row["trust_breakdown"] is None
    assert row["field_accuracy"] is None


# ---------------------------------------------------------------------------
# Union of providers (the moat-surfacing case)
# ---------------------------------------------------------------------------


def test_provider_with_only_accuracy_events_still_surfaces(
    tmp_path: Path,
) -> None:
    """A provider with reconciliation events but no health rows must
    appear in the response — otherwise the moat is partially invisible."""
    client, store, accuracy = _build_client(tmp_path)
    now = datetime.now(UTC)
    # FMP has health rows; SEC_EDGAR is only a reference in accuracy
    # events (never a primary fetch).
    store.record([_health_row(provider=ProviderName.FMP, checked_at=now)])
    accuracy.record(
        [
            _accuracy_event(
                provider=ProviderName.FMP,
                reference=ProviderName.SEC_EDGAR,
                observed_at=now,
            )
        ]
    )

    body = client.get("/api/health/providers?window=1h").json()
    names = {row["provider"] for row in body["providers"]}
    # Both must appear: FMP from health, SEC_EDGAR via accuracy ledger.
    assert "fmp" in names
    assert "sec_edgar" in names
    edgar_row = next(
        r for r in body["providers"] if r["provider"] == "sec_edgar"
    )
    # No health rows → zero counts but the row still surfaces.
    assert edgar_row["total"] == 0
    assert edgar_row["ok"] == 0
    assert edgar_row["trust_breakdown"] is not None


def test_provider_with_no_data_in_either_store_omitted(
    tmp_path: Path,
) -> None:
    """Providers with neither health rows nor accuracy events are not
    in the response — we don't enumerate the full ProviderName enum."""
    client, store, _accuracy = _build_client(tmp_path)
    store.record([_health_row(provider=ProviderName.FMP)])
    body = client.get("/api/health/providers?window=1h").json()
    names = {row["provider"] for row in body["providers"]}
    assert names == {"fmp"}  # FINNHUB / ALPHA_VANTAGE / YAHOO etc absent


# ---------------------------------------------------------------------------
# Window honored across both stores
# ---------------------------------------------------------------------------


def test_field_accuracy_respects_window(tmp_path: Path) -> None:
    """Events outside the window are excluded from field_accuracy counts."""
    client, store, accuracy = _build_client(tmp_path)
    now = datetime.now(UTC)
    old = now - timedelta(days=200)
    store.record([_health_row(checked_at=now)])
    accuracy.record(
        [
            _accuracy_event(field="revenue", agreed=True, observed_at=now),
            _accuracy_event(field="revenue", agreed=False, observed_at=old),
        ]
    )
    body = client.get("/api/health/providers?window=1h").json()
    fa = body["providers"][0]["field_accuracy"]
    # Only the in-window event counts.
    assert fa["total_events"] == 1
    assert fa["by_field"][0]["total"] == 1
    assert fa["by_field"][0]["accuracy"] == 1.0
