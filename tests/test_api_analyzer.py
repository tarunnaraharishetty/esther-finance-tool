"""Tests for the Financial Analyzer endpoint (Phase 6).

We inject a stub :class:`FundamentalsService` so no network is touched
and the chain behavior is deterministic. The mock controller already
fabricates synthetic OHLCV in :meth:`fetch_bars`, which gives us a
real technical-scoring path to exercise.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController
from src.intelligence.fundamentals import (
    CompanyProfile,
    FundamentalsResult,
    IncomeStatement,
    KeyRatios,
    NormalizedFundamentals,
    ProviderChainExhausted,
    ProviderHealth,
    ProviderName,
    ProviderRawResponse,
    ReportPeriod,
)


class _StubFundamentalsService:
    def __init__(self) -> None:
        self.responses: dict[str, list[object]] = {}
        self.calls = 0

    def queue(self, symbol: str, *outcomes: object) -> None:
        self.responses.setdefault(symbol.upper(), []).extend(outcomes)

    async def fetch(self, symbol: str) -> FundamentalsResult:
        self.calls += 1
        queue = self.responses.get(symbol.upper(), [])
        if not queue:
            raise AssertionError(f"no stub response queued for {symbol!r}")
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, FundamentalsResult):
            return outcome
        raise AssertionError(f"unknown stub outcome: {outcome!r}")


def _result_for(symbol: str) -> FundamentalsResult:
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
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
    return FundamentalsResult(fundamentals=fundamentals, raw=raw, health=health)


def _build_client(tmp_path: Path) -> tuple[TestClient, _StubFundamentalsService]:
    """Bind a mock controller + stub service onto temp cache dirs.

    Both the fundamentals cache and the analyzer cache are redirected
    onto ``tmp_path`` so per-test state can't leak via the on-disk
    blobs.
    """
    stub = _StubFundamentalsService()
    controller = MockDashboardController(watchlist=["AAPL", "MSFT"], seed=7)
    app = create_app(
        controller,
        fundamentals_service=stub,
        fundamentals_cache_dir=tmp_path / "fundamentals_cache",
        analyzer_cache_dir=tmp_path / "analyzer_cache",
    )
    return TestClient(app), stub


def test_analyzer_endpoint_returns_assembled_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))

    res = client.get("/api/analyzer/AAPL")
    assert res.status_code == 200
    body = res.json()

    # Shape contract — keep this list in sync with the TypeScript
    # AnalyzerReport interface.
    expected_keys = {
        "symbol",
        "generated_at",
        "last_price",
        "overall_analyzer_score",
        "technical_score",
        "fundamental_score",
        "valuation_score",
        "technicals",
        "valuation",
        "fundamentals_profile",
        "explanation",
        "warnings",
        "cache",
        "cache_age_seconds",
    }
    assert expected_keys.issubset(body.keys())

    assert body["symbol"] == "AAPL"
    assert body["technicals"] is not None
    assert body["valuation"] is not None
    assert body["fundamentals_profile"]["sector"] == "Information Technology"
    assert isinstance(body["explanation"]["claims"], list)
    # Header rings expect numeric sub-scores when data is present.
    assert isinstance(body["overall_analyzer_score"], (int, float))
    assert body["cache"] == "miss"


def test_analyzer_endpoint_degrades_without_fundamentals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chain exhaustion → 200 with the technical-only report + a warning."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    health = (
        ProviderHealth(
            provider=ProviderName.FMP,
            symbol="AAPL",
            status="unavailable",
            latency_ms=0.0,
            checked_at=datetime.now(UTC),
            error_message="no key",
        ),
    )
    stub.queue(
        "AAPL",
        ProviderChainExhausted("AAPL", health, ("fmp: unavailable",)),
    )

    res = client.get("/api/analyzer/AAPL")
    assert res.status_code == 200
    body = res.json()
    assert body["technicals"] is not None  # mock controller still has bars
    assert body["valuation"] is None
    assert body["fundamentals_profile"] is None
    assert any("Fundamentals chain exhausted" in w for w in body["warnings"])
    # The grounded explanation still passes citation validation —
    # ungrounded streams are simply absent.
    claims = body["explanation"]["claims"]
    for claim in claims:
        # Every claim ends with at least one [tag].
        assert "[" in claim and "]" in claim


def test_analyzer_endpoint_caches_between_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))

    first = client.get("/api/analyzer/AAPL").json()
    second = client.get("/api/analyzer/AAPL").json()

    assert stub.calls == 1  # cache hit on second call
    assert first["cache"] == "miss"
    assert second["cache"] == "hit"
    assert second["cache_age_seconds"] >= 0


def test_fresh_query_param_bypasses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"), _result_for("AAPL"))

    client.get("/api/analyzer/AAPL")
    res = client.get("/api/analyzer/AAPL?fresh=true")
    assert stub.calls == 2
    assert res.json()["cache"] == "miss"


def test_symbol_outside_watchlist_still_renders_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock controller can't synthesize bars for unknown symbols → graceful degrade."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("ZZZ", _result_for("ZZZ"))

    res = client.get("/api/analyzer/ZZZ")
    assert res.status_code == 200
    body = res.json()
    # No bars for an unknown mock symbol → technical scoring drops.
    assert body["technicals"] is None
    # Warning surfaces both unknown-watchlist + no-bars.
    assert any("not on the active watchlist" in w for w in body["warnings"])
    # Valuation still fires because fundamentals were provided.
    assert body["valuation"] is not None


def test_cache_delete_endpoint_clears_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"), _result_for("AAPL"))

    client.get("/api/analyzer/AAPL")
    delete = client.delete("/api/analyzer/AAPL/cache").json()
    assert delete == {"symbol": "AAPL", "removed": 1}
    # Subsequent GET re-runs the assembly.
    client.get("/api/analyzer/AAPL")
    assert stub.calls == 2


def test_analyzer_endpoint_appears_in_openapi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path)
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"].keys())
    assert "/api/analyzer/{symbol}" in paths
    assert "/api/analyzer/{symbol}/cache" in paths
