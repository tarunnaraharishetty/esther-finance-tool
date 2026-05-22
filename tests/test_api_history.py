"""Integration test for the per-symbol /api/history/{symbol} endpoint."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController
from src.intelligence.calibration import CalibrationStore


def _build_client(
    tmp_path: Path, *, with_store: bool = True
) -> tuple[TestClient, CalibrationStore | None]:
    store: CalibrationStore | None = None
    if with_store:
        store = CalibrationStore(tmp_path / "calibration.db")
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(
        controller,
        calibration_store=store,
        fundamentals_cache_dir=tmp_path / "fc",
        analyzer_cache_dir=tmp_path / "ac",
        research_cache_dir=tmp_path / "rc",
    )
    return TestClient(app), store


def _seed_observations(
    store: CalibrationStore,
    *,
    symbol: str,
    n: int,
    hits: int,
) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(n):
        oid = store.record_observation(
            symbol=symbol,
            score_name="pullback_risk",
            score_value=75.0,
            observed_at=base + timedelta(hours=i),
            horizon_days=5,
            outcome_name="return_negative",
            starting_price=100.0,
        )
        store.record_outcome(oid, i < hits)


def test_history_endpoint_empty_store_returns_zero_buckets(
    tmp_path: Path,
) -> None:
    client, _ = _build_client(tmp_path)
    res = client.get("/api/history/AAPL")
    assert res.status_code == 200
    body = res.json()
    assert body["symbol"] == "AAPL"
    assert body["total_observations"] == 0
    assert body["settled_observations"] == 0
    assert body["buckets"] == []


def test_history_endpoint_reports_per_symbol_buckets(tmp_path: Path) -> None:
    client, store = _build_client(tmp_path)
    assert store is not None
    _seed_observations(store, symbol="AAPL", n=10, hits=7)

    body = client.get("/api/history/AAPL").json()
    assert body["total_observations"] == 10
    assert body["settled_observations"] == 10
    assert len(body["buckets"]) == 1
    bucket = body["buckets"][0]
    assert bucket["bucket_lo"] == 70.0
    assert bucket["n_observations"] == 10
    assert bucket["n_hits"] == 7
    assert bucket["hit_rate"] == pytest.approx(0.7)
    assert bucket["bucket_published"] is True
    assert bucket["confidence_low"] < bucket["hit_rate"] < bucket["confidence_high"]


def test_history_endpoint_404s_when_calibration_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a calibration store, the route should not be registered."""
    monkeypatch.setenv("CALIBRATION_STORE_PATH", "")
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    client, _ = _build_client(tmp_path, with_store=False)
    res = client.get("/api/history/AAPL")
    assert res.status_code == 404


def test_history_endpoint_honors_min_observations_query(
    tmp_path: Path,
) -> None:
    """``?min_observations=20`` raises the bar so a 10-obs bucket goes unpublished."""
    client, store = _build_client(tmp_path)
    assert store is not None
    _seed_observations(store, symbol="AAPL", n=10, hits=7)

    body = client.get("/api/history/AAPL?min_observations=20").json()
    assert body["buckets"][0]["bucket_published"] is False
    # Hit rate still surfaces — the trader can apply their own threshold.
    assert body["buckets"][0]["hit_rate"] == pytest.approx(0.7)


def test_history_endpoint_isolates_symbols(tmp_path: Path) -> None:
    client, store = _build_client(tmp_path)
    assert store is not None
    _seed_observations(store, symbol="AAPL", n=10, hits=8)
    _seed_observations(store, symbol="MSFT", n=10, hits=3)

    aapl = client.get("/api/history/AAPL").json()
    msft = client.get("/api/history/MSFT").json()
    assert aapl["buckets"][0]["n_hits"] == 8
    assert msft["buckets"][0]["n_hits"] == 3
