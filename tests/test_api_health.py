"""Tests for the /api/health/providers route."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController
from src.data.health_store import HealthStore
from src.data.retry_queue import RetryQueue
from src.intelligence.fundamentals.models import ProviderHealth, ProviderName


def _row(
    *,
    provider: ProviderName = ProviderName.FMP,
    symbol: str = "AAPL",
    status: str = "ok",
    latency_ms: float = 100.0,
    checked_at: datetime | None = None,
    error_message: str | None = None,
) -> ProviderHealth:
    return ProviderHealth(
        provider=provider,
        symbol=symbol,
        status=status,
        latency_ms=latency_ms,
        checked_at=checked_at or datetime.now(UTC),
        error_message=error_message,
    )


def _build_client(
    tmp_path: Path, *, with_queue: bool = False
) -> tuple[TestClient, HealthStore, RetryQueue | None]:
    store = HealthStore(tmp_path / "health.db")
    queue: RetryQueue | None = (
        RetryQueue(tmp_path / "retry.json") if with_queue else None
    )
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(
        controller,
        health_store=store,
        retry_queue=queue,
        fundamentals_cache_dir=tmp_path / "fundamentals_cache",
        analyzer_cache_dir=tmp_path / "analyzer_cache",
    )
    return TestClient(app), store, queue


def test_providers_endpoint_empty_on_fresh_store(tmp_path: Path) -> None:
    """No data → empty arrays + 200. Endpoint must never 500 on cold start."""
    client, _, _ = _build_client(tmp_path)
    res = client.get("/api/health/providers?window=1h")
    assert res.status_code == 200
    body = res.json()
    assert body["providers"] == []
    assert body["recent_failures"] == []
    assert body["window_seconds"] == 3600


def test_providers_endpoint_aggregates_recorded_rows(tmp_path: Path) -> None:
    client, store, _ = _build_client(tmp_path)
    now = datetime.now(UTC)
    store.record(
        [
            _row(provider=ProviderName.FMP, status="ok", latency_ms=100.0, checked_at=now),
            _row(provider=ProviderName.FMP, status="ok", latency_ms=200.0, checked_at=now),
            _row(
                provider=ProviderName.FMP,
                status="rate_limited",
                latency_ms=80.0,
                checked_at=now,
                error_message="429",
            ),
            _row(provider=ProviderName.FINNHUB, status="ok", latency_ms=150.0, checked_at=now),
        ]
    )

    body = client.get("/api/health/providers?window=24h").json()
    by_provider = {p["provider"]: p for p in body["providers"]}

    fmp = by_provider["fmp"]
    assert fmp["total"] == 3
    assert fmp["ok"] == 2
    assert fmp["rate_limited"] == 1
    # 2 ok / 3 non-skipped = 0.6667
    assert fmp["success_rate"] == pytest.approx(0.6667, abs=1e-4)

    finnhub = by_provider["finnhub"]
    assert finnhub["total"] == 1
    assert finnhub["success_rate"] == pytest.approx(1.0)


def test_providers_endpoint_surfaces_recent_failures(tmp_path: Path) -> None:
    client, store, _ = _build_client(tmp_path)
    now = datetime.now(UTC)
    store.record(
        [
            _row(status="ok", checked_at=now),
            _row(
                provider=ProviderName.FMP,
                symbol="XYZ",
                status="rate_limited",
                checked_at=now,
                error_message="429",
            ),
        ]
    )
    body = client.get("/api/health/providers?window=1h").json()
    assert len(body["recent_failures"]) == 1
    failure = body["recent_failures"][0]
    assert failure["provider"] == "fmp"
    assert failure["symbol"] == "XYZ"
    assert failure["status"] == "rate_limited"
    assert failure["error_message"] == "429"


def test_window_parsing_accepts_each_unit(tmp_path: Path) -> None:
    client, _, _ = _build_client(tmp_path)
    cases = {"60s": 60, "15m": 900, "1h": 3600, "7d": 7 * 86400}
    for window, seconds in cases.items():
        body = client.get(f"/api/health/providers?window={window}").json()
        assert body["window_seconds"] == seconds, window


def test_window_parsing_rejects_garbage(tmp_path: Path) -> None:
    client, _, _ = _build_client(tmp_path)
    for garbage in ("forever", "1y", "-1h", "0h", "1", "1hh", "abc"):
        res = client.get(f"/api/health/providers?window={garbage}")
        assert res.status_code == 400, garbage


def test_window_caps_at_30_days(tmp_path: Path) -> None:
    client, _, _ = _build_client(tmp_path)
    res = client.get("/api/health/providers?window=31d")
    assert res.status_code == 400
    assert "maximum" in res.json()["detail"]


def test_route_not_registered_when_health_store_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settings.health_store_path=None → no route registered.

    Lets an operator opt-out of persistence when they only want
    stateless behavior. The endpoint should 404 cleanly rather than
    silently degrade to "empty data".
    """
    monkeypatch.setenv("HEALTH_STORE_PATH", "")
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    # Need to also recompute the cache for this app build.
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(
        controller,
        fundamentals_cache_dir=tmp_path / "fc",
        analyzer_cache_dir=tmp_path / "ac",
    )
    client = TestClient(app)
    res = client.get("/api/health/providers?window=1h")
    assert res.status_code == 404


def test_route_appears_in_openapi(tmp_path: Path) -> None:
    client, _, _ = _build_client(tmp_path)
    schema = client.get("/openapi.json").json()
    assert "/api/health/providers" in schema["paths"]


def test_recent_limit_query_param(tmp_path: Path) -> None:
    client, store, _ = _build_client(tmp_path)
    base = datetime.now(UTC) - timedelta(minutes=5)
    store.record(
        [
            _row(
                status="rate_limited",
                symbol=f"S{i:02d}",
                checked_at=base + timedelta(seconds=i),
            )
            for i in range(10)
        ]
    )
    body = client.get(
        "/api/health/providers?window=1h&recent_limit=3"
    ).json()
    assert len(body["recent_failures"]) == 3


def test_recent_limit_rejects_negative(tmp_path: Path) -> None:
    client, _, _ = _build_client(tmp_path)
    res = client.get("/api/health/providers?window=1h&recent_limit=-1")
    assert res.status_code == 422


# -----------------------------------------------------------------------------
# Retry queue endpoint (P1.5)
# -----------------------------------------------------------------------------


def test_queue_endpoint_404s_when_queue_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a retry queue, the queue endpoint should not be registered.

    Disabling = ``RETRY_QUEUE_PATH=`` in env, which makes the settings
    validator coerce the path to None. ``with_queue=False`` on
    ``_build_client`` only suppresses the kwarg injection; settings
    would otherwise still construct a default queue at the project
    data dir, defeating the test.
    """
    monkeypatch.setenv("RETRY_QUEUE_PATH", "")
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()

    client, _, _ = _build_client(tmp_path, with_queue=False)
    res = client.get("/api/health/queue")
    assert res.status_code == 404


def test_queue_endpoint_empty_on_fresh_queue(tmp_path: Path) -> None:
    client, _, _ = _build_client(tmp_path, with_queue=True)
    res = client.get("/api/health/queue")
    assert res.status_code == 200
    body = res.json()
    assert body["pending"] == 0
    assert body["permanently_failed"] == 0
    assert body["total"] == 0
    assert body["entries"] == []


def test_queue_endpoint_shows_pending_and_failed_counts(tmp_path: Path) -> None:
    client, _, queue = _build_client(tmp_path, with_queue=True)
    assert queue is not None
    queue.enqueue("AAPL", ("fmp: rate-limited",))
    queue.enqueue("MSFT", ("finnhub: transient",))
    for _ in range(10):
        queue.record_failure("MSFT", ("transient again",))

    body = client.get("/api/health/queue").json()
    by_symbol = {e["symbol"]: e for e in body["entries"]}
    assert by_symbol["AAPL"]["status"] == "pending"
    assert by_symbol["MSFT"]["status"] == "permanently_failed"
    assert body["pending"] == 1
    assert body["permanently_failed"] == 1
    assert body["total"] == 2


def test_queue_endpoint_wire_shape_complete(tmp_path: Path) -> None:
    """Per-entry shape must include every field the UI needs."""
    client, _, queue = _build_client(tmp_path, with_queue=True)
    assert queue is not None
    queue.enqueue("AAPL", ("fmp: rate-limited",))
    body = client.get("/api/health/queue").json()
    entry = body["entries"][0]
    expected_keys = {
        "symbol",
        "enqueued_at",
        "next_attempt_at",
        "attempts",
        "last_errors",
        "status",
        "last_error_at",
    }
    assert expected_keys.issubset(entry.keys())
