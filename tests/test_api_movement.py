"""End-to-end tests for /api/movement/{symbol}/drivers.

Confirms the endpoint composes analyzer + sector + snapshot row,
runs the driver ranker, and returns a well-shaped response. Uses
the same stub-fundamentals pattern the other API tests use.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController
from src.data.envelope import DataEnvelope
from src.intelligence.fundamentals import (
    CompanyProfile,
    FundamentalsResult,
    IncomeStatement,
    KeyRatios,
    NormalizedFundamentals,
    ProviderHealth,
    ProviderName,
    ProviderRawResponse,
    ReportPeriod,
)


class _StubFundamentalsService:
    def __init__(self) -> None:
        self.responses: dict[str, list[object]] = {}

    def queue(self, symbol: str, *outcomes: object) -> None:
        self.responses.setdefault(symbol.upper(), []).extend(outcomes)

    async def fetch(self, symbol: str) -> FundamentalsResult:
        queue = self.responses.get(symbol.upper(), [])
        if not queue:
            raise AssertionError(f"no stub queued for {symbol!r}")
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, FundamentalsResult):
            return outcome
        raise AssertionError(f"unknown outcome: {outcome!r}")


def _result_for(
    symbol: str,
    *,
    sector: str | None = "Information Technology",
) -> FundamentalsResult:
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    incomes = tuple(
        IncomeStatement(
            period=ReportPeriod.ANNUAL,
            fiscal_date=datetime(year, 12, 31, tzinfo=UTC),
            revenue=100_000_000 * (1.15 ** i),
            net_income=20_000_000 * (1.15 ** i),
            ebitda=30_000_000 * (1.15 ** i),
            eps_diluted=2.0 * (1.15 ** i),
            shares_diluted=10_000_000,
        )
        for i, year in enumerate(range(2020, 2025))
    )
    fundamentals = NormalizedFundamentals(
        symbol=symbol,
        fetched_at=now,
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(
            symbol=symbol,
            name=f"{symbol} Co",
            sector=sector,
            market_cap=1.5e9,
            shares_outstanding=10_000_000,
        ),
        income_statements=incomes,
        key_ratios=KeyRatios(pe_ratio=32.0, peg_ratio=2.0, debt_to_equity=0.4),
    )
    envelope: DataEnvelope[NormalizedFundamentals] = DataEnvelope(
        data=fundamentals,
        as_of=datetime(2024, 12, 31, tzinfo=UTC),
        fetched_at=now,
        source_chain=("fmp",),
        freshness="fresh",
        provider_confidence=1.0,
    )
    raw = ProviderRawResponse(
        provider=ProviderName.FMP,
        symbol=symbol,
        endpoint="stub",
        fetched_at=now,
        payload={},
    )
    health = (
        ProviderHealth(
            provider=ProviderName.FMP,
            symbol=symbol,
            status="ok",
            latency_ms=10.0,
            checked_at=now,
            error_message=None,
        ),
    )
    return FundamentalsResult(envelope=envelope, raw=raw, health=health)


def _build_client(tmp_path: Path) -> tuple[TestClient, _StubFundamentalsService]:
    stub = _StubFundamentalsService()
    controller = MockDashboardController(watchlist=["AAPL", "MSFT"], seed=7)
    app = create_app(
        controller,
        fundamentals_service=stub,
        fundamentals_cache_dir=tmp_path / "fundamentals_cache",
        analyzer_cache_dir=tmp_path / "analyzer_cache",
        sector_cache_dir=tmp_path / "sector_cache",
        movement_cache_dir=tmp_path / "movement_cache",
    )
    return TestClient(app), stub


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_movement_endpoint_returns_well_shaped_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    # The route assembles AAPL once for analyzer, once for sector primary,
    # then peers for sector. Queue generously.
    for _ in range(4):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))

    res = client.get("/api/movement/AAPL/drivers")
    assert res.status_code == 200, res.text
    body = res.json()

    expected_keys = {
        "symbol",
        "drivers",
        "considered",
        "suppressed",
        "sources",
        "cache",
    }
    assert expected_keys.issubset(body.keys())
    assert body["symbol"] == "AAPL"
    assert isinstance(body["drivers"], list)
    assert body["cache"] == "miss"

    # Sources block describes which inputs assembled successfully.
    sources = body["sources"]
    assert set(sources.keys()) == {"analyzer", "sector", "snapshot_row"}
    assert sources["analyzer"] is True
    # Snapshot row should be present for the watchlist symbol.
    assert sources["snapshot_row"] is True

    # Every driver has the documented per-entry schema.
    for d in body["drivers"]:
        assert set(d.keys()) == {
            "kind",
            "label",
            "detail",
            "tone",
            "salience",
            "citations",
        }
        assert d["tone"] in {"bull", "bear", "warn", "neutral"}
        assert 0.0 <= d["salience"] <= 1.0


def test_movement_endpoint_is_case_insensitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    for _ in range(4):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))
    body = client.get("/api/movement/aapl/drivers").json()
    assert body["symbol"] == "AAPL"


def test_movement_endpoint_drivers_sorted_by_salience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever drivers fire, they must be ordered descending by salience."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    for _ in range(4):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))
    body = client.get("/api/movement/AAPL/drivers").json()
    saliences = [d["salience"] for d in body["drivers"]]
    assert saliences == sorted(saliences, reverse=True)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_movement_endpoint_cache_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    for _ in range(4):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))
    first = client.get("/api/movement/AAPL/drivers").json()
    assert first["cache"] == "miss"
    second = client.get("/api/movement/AAPL/drivers").json()
    assert second["cache"] == "hit"


def test_movement_endpoint_fresh_bypasses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    for _ in range(8):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))
    client.get("/api/movement/AAPL/drivers")
    fresh = client.get("/api/movement/AAPL/drivers?fresh=true").json()
    assert fresh["cache"] == "miss"


# ---------------------------------------------------------------------------
# Degraded behavior
# ---------------------------------------------------------------------------


def test_movement_endpoint_unknown_sector_still_returns_drivers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sector driver simply doesn't fire when the sector is unknown."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    for _ in range(4):
        stub.queue("AAPL", _result_for("AAPL", sector=None))
        stub.queue("MSFT", _result_for("MSFT", sector=None))
    res = client.get("/api/movement/AAPL/drivers")
    assert res.status_code == 200
    body = res.json()
    # Sector source flag reflects the missing-sector state.
    assert body["sources"]["sector"] is False
    # Non-sector drivers may still fire (technicals etc.)
    kinds = [d["kind"] for d in body["drivers"]]
    assert "sector" not in kinds
