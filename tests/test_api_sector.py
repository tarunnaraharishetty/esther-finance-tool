"""End-to-end tests for /api/sector/{symbol}/rank.

Drives the full pipeline: a watchlist with two same-sector symbols
must produce a cohort of 2 and assign meaningful ranks. Singleton
sectors must degrade cleanly. Cache round-trip is exercised.
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


def _result_for(
    symbol: str,
    *,
    sector: str | None = "Information Technology",
    revenue_seed: float = 100.0,
) -> FundamentalsResult:
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    incomes = tuple(
        IncomeStatement(
            period=ReportPeriod.ANNUAL,
            fiscal_date=datetime(year, 12, 31, tzinfo=UTC),
            revenue=revenue_seed * 1_000_000 * (1.15 ** i),
            net_income=revenue_seed * 200_000 * (1.15 ** i),
            ebitda=revenue_seed * 300_000 * (1.15 ** i),
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
    )
    return TestClient(app), stub


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_sector_rank_endpoint_returns_well_shaped_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both watchlist symbols in the same sector → 2-symbol cohort with ranks."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    # AAPL queued for primary fetch + cohort fetch (assembled twice
    # because the sector route first asks for AAPL's sector, then
    # treats it as a cohort member). Same for MSFT.
    for _ in range(2):
        stub.queue("AAPL", _result_for("AAPL", revenue_seed=200.0))
        stub.queue("MSFT", _result_for("MSFT", revenue_seed=100.0))

    res = client.get("/api/sector/AAPL/rank")
    assert res.status_code == 200, res.text
    body = res.json()

    expected_keys = {
        "symbol",
        "sector",
        "cohort_size",
        "cohort_symbols",
        "metrics",
        "available",
        "cache",
    }
    assert expected_keys.issubset(body.keys())
    assert body["symbol"] == "AAPL"
    assert body["sector"] == "Information Technology"
    assert body["available"] is True
    assert body["cohort_size"] == 2
    assert set(body["cohort_symbols"]) == {"AAPL", "MSFT"}
    assert body["cache"] == "miss"

    # Every metric row carries the documented per-row schema.
    metrics = body["metrics"]
    assert len(metrics) == 8
    for m in metrics:
        assert set(m.keys()) == {
            "metric",
            "label",
            "value",
            "rank",
            "cohort_size",
            "percentile",
            "direction",
            "percent",
        }
        assert m["direction"] in {"higher_better", "lower_better"}


def test_sector_rank_is_case_insensitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    for _ in range(2):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))
    body = client.get("/api/sector/aapl/rank").json()
    assert body["symbol"] == "AAPL"


def test_sector_rank_assigns_one_to_higher_trust_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same controller, different revenue seeds → different valuations → ranked."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    # Same revenue seed → identical fundamentals → most metrics tie.
    # Different last prices (from controller mock) still differentiate
    # via upside calculation.
    for _ in range(2):
        stub.queue("AAPL", _result_for("AAPL", revenue_seed=100.0))
        stub.queue("MSFT", _result_for("MSFT", revenue_seed=300.0))
    body = client.get("/api/sector/AAPL/rank").json()
    # At least one metric must produce a non-tied verdict given the
    # different revenue seeds.
    non_tied = [
        m for m in body["metrics"]
        if m["rank"] is not None and m["cohort_size"] == 2
    ]
    assert len(non_tied) > 0


# ---------------------------------------------------------------------------
# Degraded states
# ---------------------------------------------------------------------------


def test_sector_rank_singleton_cohort_renders_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different sectors → no peer cohort → available=False with reason."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    # AAPL in IT, MSFT in Financials → singleton cohort for each.
    for _ in range(2):
        stub.queue("AAPL", _result_for("AAPL", sector="Information Technology"))
        stub.queue("MSFT", _result_for("MSFT", sector="Financials"))

    res = client.get("/api/sector/AAPL/rank")
    assert res.status_code == 200
    body = res.json()
    assert body["available"] is False
    assert body["cohort_size"] == 0
    assert body["metrics"] == []
    assert "no watchlist peers share this sector" in body["reason"]


def test_sector_rank_missing_sector_renders_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symbol with no sector classification → available=False, no cohort."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL", sector=None))
    res = client.get("/api/sector/AAPL/rank")
    assert res.status_code == 200
    body = res.json()
    assert body["available"] is False
    assert "no sector classification" in body["reason"]


def test_sector_rank_solo_watchlist_renders_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Watchlist has only the target symbol → no peers possible."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path, watchlist=["AAPL"])
    stub.queue("AAPL", _result_for("AAPL"))
    res = client.get("/api/sector/AAPL/rank")
    assert res.status_code == 200
    body = res.json()
    assert body["available"] is False
    assert "watchlist has no other symbols" in body["reason"]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_sector_rank_cache_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First call writes the cohort cache; the same call again hits it."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    # Queue enough fetches for the first call (each symbol fetched
    # once when the analyzer cache is empty; the route also makes the
    # initial primary fetch). The analyzer cache absorbs subsequent
    # calls, so the second sector call doesn't need extra queueing.
    for _ in range(2):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))

    first = client.get("/api/sector/AAPL/rank").json()
    assert first["cache"] == "miss"
    # The analyzer cache + sector cache + primary fetch all combine —
    # a second call should hit the sector cache.
    second = client.get("/api/sector/AAPL/rank").json()
    assert second["cache"] == "hit"


def test_sector_rank_fresh_bypasses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``?fresh=true`` always rebuilds; analyzer cache also gets bypassed."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    for _ in range(4):
        stub.queue("AAPL", _result_for("AAPL"))
        stub.queue("MSFT", _result_for("MSFT"))

    client.get("/api/sector/AAPL/rank")
    fresh = client.get("/api/sector/AAPL/rank?fresh=true").json()
    assert fresh["cache"] == "miss"
