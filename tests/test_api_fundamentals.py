"""Tests for the fundamentals API surface (Phase 2).

We inject a stub :class:`FundamentalsService` into :func:`create_app`
so no network or API keys are needed and the chain behavior is fully
deterministic per test.
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
    ProviderChainExhausted,
    ProviderHealth,
    ProviderName,
    ProviderRawResponse,
    ReportPeriod,
)


class _StubFundamentalsService:
    """Pretends to be a FundamentalsService for the route module.

    Each test pre-seeds ``responses`` keyed by symbol; ``fetch`` pops
    the next entry off the list and either returns a
    :class:`FundamentalsResult` or raises a
    :class:`ProviderChainExhausted`. ``calls`` is incremented on every
    invocation so tests can assert cache hits skipped the chain.
    """

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


def _make_result(
    symbol: str,
    *,
    primary: ProviderName = ProviderName.FMP,
    fetched_at: datetime | None = None,
    freshness: str = "fresh",
    provider_confidence: float = 1.0,
) -> FundamentalsResult:
    now = fetched_at or datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)
    fundamentals = NormalizedFundamentals(
        symbol=symbol,
        fetched_at=now,
        primary_provider=primary,
        contributing_providers=(primary,),
        profile=CompanyProfile(
            symbol=symbol,
            name=f"{symbol} Co",
            sector="Technology",
            market_cap=1.0e12,
        ),
        income_statements=(
            IncomeStatement(
                period=ReportPeriod.ANNUAL,
                fiscal_date=datetime(2024, 12, 31, tzinfo=UTC),
                revenue=100.0,
                net_income=20.0,
            ),
        ),
        key_ratios=KeyRatios(pe_ratio=25.0, peg_ratio=1.5),
    )
    raw = ProviderRawResponse(
        provider=primary, symbol=symbol, endpoint="stub", fetched_at=now, payload={}
    )
    health = (
        ProviderHealth(
            provider=primary,
            symbol=symbol,
            status="ok",
            latency_ms=12.3,
            checked_at=now,
            error_message=None,
        ),
    )
    envelope: DataEnvelope[NormalizedFundamentals] = DataEnvelope(
        data=fundamentals,
        as_of=datetime(2024, 12, 31, tzinfo=UTC),
        fetched_at=now,
        source_chain=(primary.value,),
        freshness=freshness,  # type: ignore[arg-type]
        provider_confidence=provider_confidence,
    )
    return FundamentalsResult(envelope=envelope, raw=raw, health=health)


def _build_client(
    tmp_path: Path,
) -> tuple[TestClient, _StubFundamentalsService]:
    stub = _StubFundamentalsService()
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(
        controller,
        fundamentals_service=stub,
        fundamentals_cache_dir=tmp_path / "fundamentals_cache",
    )
    return TestClient(app), stub


def test_fundamentals_endpoint_returns_normalized_shape(tmp_path: Path) -> None:
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _make_result("AAPL"))

    res = client.get("/api/fundamentals/AAPL")
    assert res.status_code == 200
    body = res.json()

    assert body["symbol"] == "AAPL"
    assert body["primary_provider"] == "fmp"
    assert body["contributing_providers"] == ["fmp"]
    assert body["profile"]["name"] == "AAPL Co"
    assert body["profile"]["market_cap"] == 1.0e12
    assert body["key_ratios"]["pe_ratio"] == 25.0
    assert body["income_statements"][0]["revenue"] == 100.0
    assert body["cache"] == "miss"
    assert body["cache_age_seconds"] == 0
    # Health log surfaces the chain trace from this fetch.
    assert isinstance(body["health"], list) and body["health"][0]["status"] == "ok"


def test_fundamentals_endpoint_surfaces_freshness_envelope(tmp_path: Path) -> None:
    """The wire payload exposes every envelope field consumers need."""
    client, stub = _build_client(tmp_path)
    stub.queue(
        "AAPL",
        _make_result("AAPL", freshness="aging", provider_confidence=0.85),
    )

    body = client.get("/api/fundamentals/AAPL").json()

    assert body["freshness"] == "aging"
    assert body["provider_confidence"] == pytest.approx(0.85)
    assert body["source_chain"] == ["fmp"]
    # as_of / fetched_at are ISO-8601 strings; data_age_days is a float.
    assert body["as_of"].startswith("2024-12-31")
    assert body["fetched_at"].startswith("2026-05-21")
    assert isinstance(body["data_age_days"], (int, float))
    assert body["data_age_days"] > 0


def test_fundamentals_endpoint_caches_between_calls(tmp_path: Path) -> None:
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _make_result("AAPL"))

    first = client.get("/api/fundamentals/AAPL").json()
    second = client.get("/api/fundamentals/AAPL").json()

    # Second call hits the file cache; the service is consulted exactly once.
    assert stub.calls == 1
    assert first["cache"] == "miss"
    assert second["cache"] == "hit"
    assert second["cache_age_seconds"] >= 0
    # Payload identity is preserved.
    assert second["profile"]["name"] == first["profile"]["name"]


def test_fresh_query_param_bypasses_cache(tmp_path: Path) -> None:
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _make_result("AAPL"), _make_result("AAPL", primary=ProviderName.FINNHUB))

    first = client.get("/api/fundamentals/AAPL").json()
    second = client.get("/api/fundamentals/AAPL?fresh=true").json()

    assert stub.calls == 2
    assert first["primary_provider"] == "fmp"
    assert second["primary_provider"] == "finnhub"
    assert second["cache"] == "miss"


def test_symbol_is_uppercased_before_chain_walk(tmp_path: Path) -> None:
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _make_result("AAPL"))

    res = client.get("/api/fundamentals/aapl")
    assert res.status_code == 200
    assert res.json()["symbol"] == "AAPL"


def test_chain_exhausted_returns_503_with_health_log(tmp_path: Path) -> None:
    client, stub = _build_client(tmp_path)
    health = (
        ProviderHealth(
            provider=ProviderName.FMP,
            symbol="XYZ",
            status="rate_limited",
            latency_ms=80.0,
            checked_at=datetime.now(UTC),
            error_message="429",
        ),
        ProviderHealth(
            provider=ProviderName.FINNHUB,
            symbol="XYZ",
            status="unavailable",
            latency_ms=0.0,
            checked_at=datetime.now(UTC),
            error_message="no key",
        ),
    )
    stub.queue(
        "XYZ",
        ProviderChainExhausted(
            "XYZ", health, ("fmp: rate-limited", "finnhub: unavailable")
        ),
    )

    res = client.get("/api/fundamentals/XYZ")
    assert res.status_code == 503
    detail = res.json()["detail"]
    assert detail["symbol"] == "XYZ"
    assert "fmp: rate-limited" in detail["errors"]
    assert len(detail["health"]) == 2
    # Statuses come through unmodified for the UI to render badge colors.
    statuses = {row["status"] for row in detail["health"]}
    assert statuses == {"rate_limited", "unavailable"}


def test_chain_exhausted_response_is_not_cached(tmp_path: Path) -> None:
    client, stub = _build_client(tmp_path)
    health = (
        ProviderHealth(
            provider=ProviderName.FMP,
            symbol="XYZ",
            status="unavailable",
            latency_ms=0.0,
            checked_at=datetime.now(UTC),
            error_message="no key",
        ),
    )
    # First call fails; second call succeeds. If we were caching the
    # 503 by accident, the second call would still 503.
    stub.queue(
        "XYZ",
        ProviderChainExhausted("XYZ", health, ("fmp: unavailable",)),
        _make_result("XYZ"),
    )

    first = client.get("/api/fundamentals/XYZ")
    second = client.get("/api/fundamentals/XYZ")
    assert first.status_code == 503
    assert second.status_code == 200
    assert second.json()["primary_provider"] == "fmp"


def test_health_endpoint_walks_chain_each_call(tmp_path: Path) -> None:
    client, stub = _build_client(tmp_path)
    stub.queue(
        "AAPL",
        _make_result("AAPL", primary=ProviderName.FINNHUB),
        _make_result("AAPL", primary=ProviderName.FMP),
    )

    first = client.get("/api/fundamentals/AAPL/health").json()
    second = client.get("/api/fundamentals/AAPL/health").json()

    assert stub.calls == 2  # not cached
    assert first["primary_provider"] == "finnhub"
    assert second["primary_provider"] == "fmp"
    for body in (first, second):
        assert body["chain_exhausted"] is False
        assert body["errors"] == []
        assert len(body["health"]) == 1


def test_health_endpoint_returns_200_on_chain_exhausted(tmp_path: Path) -> None:
    """Monitoring panel should poll without alarm — 200 + flag, not 5xx."""
    client, stub = _build_client(tmp_path)
    health = (
        ProviderHealth(
            provider=ProviderName.FMP,
            symbol="XYZ",
            status="unavailable",
            latency_ms=0.0,
            checked_at=datetime.now(UTC),
            error_message="no key",
        ),
    )
    stub.queue(
        "XYZ",
        ProviderChainExhausted("XYZ", health, ("fmp: unavailable",)),
    )
    res = client.get("/api/fundamentals/XYZ/health")
    assert res.status_code == 200
    body = res.json()
    assert body["chain_exhausted"] is True
    assert body["primary_provider"] is None
    assert body["errors"] == ["fmp: unavailable"]
    assert body["health"][0]["status"] == "unavailable"


def test_cache_delete_removes_persisted_blob(tmp_path: Path) -> None:
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _make_result("AAPL"), _make_result("AAPL"))

    client.get("/api/fundamentals/AAPL")
    # The persistence file exists on disk after the first call.
    cache_file = tmp_path / "fundamentals_cache" / "AAPL.json"
    assert cache_file.exists()

    delete = client.delete("/api/fundamentals/AAPL/cache").json()
    assert delete == {"symbol": "AAPL", "removed": 1}
    assert not cache_file.exists()

    # Subsequent GET re-runs the chain (no cache file to read).
    client.get("/api/fundamentals/AAPL")
    assert stub.calls == 2


def test_cache_delete_is_idempotent_for_unknown_symbols(tmp_path: Path) -> None:
    client, _ = _build_client(tmp_path)
    res = client.delete("/api/fundamentals/UNSEEN/cache").json()
    assert res == {"symbol": "UNSEEN", "removed": 0}


def test_status_endpoint_reports_provider_configuration(tmp_path: Path) -> None:
    client, _ = _build_client(tmp_path)
    res = client.get("/api/fundamentals/_/status").json()

    # Defaults in conftest (no FMP/Finnhub/AlphaVantage keys, empty
    # SEC UA) → only Yahoo is "configured".
    assert res["chain_order"] == ["fmp", "finnhub", "alpha_vantage", "sec_edgar", "yahoo"]
    assert res["providers_configured"] == {
        "fmp": False,
        "finnhub": False,
        "alpha_vantage": False,
        "sec_edgar": False,
        "yahoo": True,
    }
    assert res["cache_ttl_hours"] == pytest.approx(12.0)
    assert res["cache_dir"].endswith("fundamentals_cache")


def test_status_endpoint_reflects_configured_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FMP_API_KEY", "fake-fmp-key")
    monkeypatch.setenv("FINNHUB_API_KEY", "fake-finnhub-key")
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Esther esther@example.com")
    # Bust the lru cache so the next get_settings() picks up the env.
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()

    client, _ = _build_client(tmp_path)
    res = client.get("/api/fundamentals/_/status").json()

    configured = res["providers_configured"]
    assert configured["fmp"] is True
    assert configured["finnhub"] is True
    assert configured["alpha_vantage"] is False
    assert configured["sec_edgar"] is True


def test_endpoint_is_listed_in_openapi_schema(tmp_path: Path) -> None:
    client, _ = _build_client(tmp_path)
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"].keys())
    assert "/api/fundamentals/{symbol}" in paths
    assert "/api/fundamentals/{symbol}/health" in paths
    assert "/api/fundamentals/{symbol}/cache" in paths
    assert "/api/fundamentals/_/status" in paths
