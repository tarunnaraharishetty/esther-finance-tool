"""End-to-end tests for /api/spotlight.

Drives the cross-watchlist driver assembly via the real analyzer
pipeline (stub fundamentals) and confirms the wire shape, the cache
behavior, and the empty-watchlist degradation.
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


def _result_for(symbol: str) -> FundamentalsResult:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
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
            sector="Information Technology",
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


def _build_client(
    tmp_path: Path,
    *,
    watchlist: list[str] | None = None,
) -> tuple[TestClient, _StubFundamentalsService]:
    stub = _StubFundamentalsService()
    controller = MockDashboardController(
        watchlist=watchlist or ["AAPL", "MSFT"], seed=7
    )
    app = create_app(
        controller,
        fundamentals_service=stub,
        fundamentals_cache_dir=tmp_path / "fundamentals_cache",
        analyzer_cache_dir=tmp_path / "analyzer_cache",
        sector_cache_dir=tmp_path / "sector_cache",
        movement_cache_dir=tmp_path / "movement_cache",
        spotlight_cache_dir=tmp_path / "spotlight_cache",
    )
    return TestClient(app), stub


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_spotlight_endpoint_returns_well_shaped_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both watchlist symbols assembled → ranked spotlight returned."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    # The spotlight needs analyzer reports for both symbols, plus
    # sector-rank inputs (analyzer for primary + peer per symbol).
    for _ in range(8):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))

    res = client.get("/api/spotlight")
    assert res.status_code == 200, res.text
    body = res.json()

    expected_keys = {
        "entries",
        "considered_symbols",
        "surfaced_symbols",
        "watchlist_size",
        "assembled_symbols",
        "cache",
    }
    assert expected_keys.issubset(body.keys())
    assert body["cache"] == "miss"
    assert body["watchlist_size"] == 2
    # Both symbols assembled successfully.
    assert body["assembled_symbols"] == 2
    assert body["considered_symbols"] == 2

    # Every entry has the documented per-row schema.
    for entry in body["entries"]:
        assert set(entry.keys()) == {"rank", "symbol", "driver"}
        driver = entry["driver"]
        assert set(driver.keys()) == {
            "kind",
            "label",
            "detail",
            "tone",
            "salience",
            "citations",
        }
        assert driver["tone"] in {"bull", "bear", "warn", "neutral"}
        assert 0.0 <= driver["salience"] <= 1.0


def test_spotlight_entries_sorted_by_salience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever drivers fire, entries are ordered descending by salience."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    for _ in range(8):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))
    body = client.get("/api/spotlight").json()
    saliences = [e["driver"]["salience"] for e in body["entries"]]
    assert saliences == sorted(saliences, reverse=True)


def test_spotlight_default_caps_at_one_driver_per_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No symbol can dominate — at most one entry per symbol by default."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    for _ in range(8):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))
    body = client.get("/api/spotlight").json()
    symbol_counts: dict[str, int] = {}
    for entry in body["entries"]:
        symbol_counts[entry["symbol"]] = symbol_counts.get(entry["symbol"], 0) + 1
    for sym, count in symbol_counts.items():
        assert count <= 1, f"{sym} got {count} entries; cap is 1"


# ---------------------------------------------------------------------------
# Empty / degraded states
# ---------------------------------------------------------------------------


def test_spotlight_empty_watchlist_returns_empty_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty watchlist short-circuits without calling the assembler.

    ``MockDashboardController`` rejects an empty constructor list (falls
    back to a default 5-symbol watchlist), so we mutate the live
    attribute after construction to simulate the degenerate state.
    """
    monkeypatch.chdir(tmp_path)
    stub = _StubFundamentalsService()
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    controller.watchlist = []  # simulate the empty-watchlist state
    app = create_app(
        controller,
        fundamentals_service=stub,
        fundamentals_cache_dir=tmp_path / "fundamentals_cache",
        analyzer_cache_dir=tmp_path / "analyzer_cache",
        sector_cache_dir=tmp_path / "sector_cache",
        movement_cache_dir=tmp_path / "movement_cache",
        spotlight_cache_dir=tmp_path / "spotlight_cache",
    )
    client = TestClient(app)
    res = client.get("/api/spotlight")
    assert res.status_code == 200
    body = res.json()
    assert body["entries"] == []
    assert body["watchlist_size"] == 0
    assert body["reason"] == "watchlist is empty"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_spotlight_cache_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    for _ in range(8):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))
    first = client.get("/api/spotlight").json()
    assert first["cache"] == "miss"
    second = client.get("/api/spotlight").json()
    assert second["cache"] == "hit"


def test_spotlight_fresh_bypasses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    for _ in range(16):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))
    client.get("/api/spotlight")
    fresh = client.get("/api/spotlight?fresh=true").json()
    assert fresh["cache"] == "miss"
