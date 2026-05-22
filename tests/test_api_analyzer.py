"""Tests for the Financial Analyzer endpoint (Phase 6).

We inject a stub :class:`FundamentalsService` so no network is touched
and the chain behavior is deterministic. The mock controller already
fabricates synthetic OHLCV in :meth:`fetch_bars`, which gives us a
real technical-scoring path to exercise.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController
from src.data.envelope import DataEnvelope
from src.intelligence.calibration import CalibrationStore
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
    envelope: DataEnvelope[NormalizedFundamentals] = DataEnvelope(
        data=fundamentals,
        as_of=datetime(2024, 12, 31, tzinfo=UTC),
        fetched_at=now,
        source_chain=("fmp",),
        freshness="fresh",
        provider_confidence=1.0,
    )
    return FundamentalsResult(envelope=envelope, raw=raw, health=health)


def _build_client(
    tmp_path: Path,
    *,
    calibration_store: object | None = None,
) -> tuple[TestClient, _StubFundamentalsService]:
    """Bind a mock controller + stub service onto temp cache dirs.

    Both the fundamentals cache and the analyzer cache are redirected
    onto ``tmp_path`` so per-test state can't leak via the on-disk
    blobs. ``calibration_store`` is opt-in — tests that don't exercise
    P2.3 leave it None and the analyzer report's ``calibrations``
    array is empty.
    """
    stub = _StubFundamentalsService()
    controller = MockDashboardController(watchlist=["AAPL", "MSFT"], seed=7)
    app = create_app(
        controller,
        fundamentals_service=stub,
        fundamentals_cache_dir=tmp_path / "fundamentals_cache",
        analyzer_cache_dir=tmp_path / "analyzer_cache",
        calibration_store=calibration_store,  # type: ignore[arg-type]
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
        "fundamentals_freshness",
        "scenarios",
        "calibrations",
        "explanation",
        "trust_score",
        "warnings",
        "cache",
        "cache_age_seconds",
    }
    assert expected_keys.issubset(body.keys())

    # Trust score is composed from the same envelope fields the test
    # asserts below; a fully-populated analyzer report should grade
    # at least B and carry the documented wire shape.
    trust = body["trust_score"]
    assert set(trust.keys()) == {"score", "grade", "components"}
    assert trust["grade"] in {"A+", "A", "B"}
    assert 0.0 <= trust["score"] <= 100.0
    assert len(trust["components"]) == 6  # six named ingredients, always.

    # Freshness envelope surfaces alongside the profile.
    fresh_block = body["fundamentals_freshness"]
    assert fresh_block is not None
    assert fresh_block["freshness"] == "fresh"
    assert fresh_block["provider_confidence"] == pytest.approx(1.0)
    assert fresh_block["source_chain"] == ["fmp"]

    # Scenarios block: present when bars + last_price exist. The mock
    # controller produces enough bars to cross the 30-day lookback.
    scenarios = body["scenarios"]
    assert scenarios is not None
    assert scenarios["horizon_days"] == 30
    assert scenarios["vol_method"] == "realized_log"
    assert scenarios["quantile_20"] < scenarios["quantile_50"] < scenarios["quantile_80"]
    # Notes carry the model-card caveats verbatim.
    assert any("zero drift" in n for n in scenarios["notes"])

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
    # Chain exhausted → no freshness block; UI must handle absence.
    assert body["fundamentals_freshness"] is None
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


# -----------------------------------------------------------------------------
# Calibration wiring (P2.3)
# -----------------------------------------------------------------------------


def test_calibrations_empty_when_store_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No CalibrationStore → empty calibrations array (never absent)."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))
    body = client.get("/api/analyzer/AAPL").json()
    assert body["calibrations"] == []


def _seed_calibration_store(path: Path) -> CalibrationStore:
    """A store pre-populated so pullback_risk lookups publish.

    50 observations at every 10-point step from 5 to 95 → every
    pullback_risk bucket has n=50, well above the default 30
    floor. Hit rate hardcoded at 70% so the exact value is
    assertable. rebound_potential is deliberately seeded thinly
    (10 obs at a single score value) so the analyzer's reading for
    that pairing exercises the under-sampled path *only when the
    technical score falls in the matching bucket*.
    """
    store = CalibrationStore(path / "calibration.db")
    base = datetime(2026, 1, 1, tzinfo=UTC)
    counter = 0
    for score_value in range(5, 100, 10):
        for i in range(50):
            counter += 1
            oid = store.record_observation(
                symbol=f"S{i % 5:02d}",
                score_name="pullback_risk",
                score_value=float(score_value),
                observed_at=base + timedelta(hours=counter),
                horizon_days=5,
                outcome_name="return_negative",
            )
            store.record_outcome(oid, i < 35)
    for i in range(10):
        counter += 1
        oid = store.record_observation(
            symbol="UND",
            score_name="rebound_potential",
            score_value=15.0,
            observed_at=base + timedelta(hours=counter),
            horizon_days=5,
            outcome_name="return_positive",
        )
        store.record_outcome(oid, i < 6)
    store.build_table(
        score_names=["pullback_risk", "rebound_potential"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    store.build_table(
        score_names=["pullback_risk", "rebound_potential"],
        horizon_days=5,
        outcome_name="return_positive",
    )
    return store


def test_calibrations_published_when_bucket_well_sampled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A populated bucket should surface a published reading with CI."""
    monkeypatch.chdir(tmp_path)
    store = _seed_calibration_store(tmp_path)
    stub = _StubFundamentalsService()
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(
        controller,
        fundamentals_service=stub,
        fundamentals_cache_dir=tmp_path / "fc",
        analyzer_cache_dir=tmp_path / "ac",
        calibration_store=store,
    )
    client = TestClient(app)
    stub.queue("AAPL", _result_for("AAPL"))

    body = client.get("/api/analyzer/AAPL").json()
    by_name = {r["score_name"]: r for r in body["calibrations"]}
    # Pairings whose technical score is computable show up as
    # readings. We don't pin the exact mock-generated score, so we
    # assert structural invariants on whichever readings landed.
    assert "pullback_risk" in by_name
    pullback = by_name["pullback_risk"]
    assert pullback["outcome_name"] == "return_negative"
    assert pullback["horizon_days"] == 5
    # All pullback_risk buckets are seeded at n=50 → bucket_published
    # must be True for any score the mock controller produces.
    assert pullback["bucket_published"] is True
    assert pullback["n_observations"] == 50
    assert pullback["hit_rate"] == pytest.approx(0.70)
    assert pullback["confidence_low"] is not None
    assert pullback["confidence_high"] is not None
    # Score value is the actual technical reading, not the bucket midpoint.
    assert isinstance(pullback["score_value"], (int, float))


def test_calibrations_under_sampled_bucket_emits_unpublished_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under-sampled buckets surface with bucket_published=false."""
    monkeypatch.chdir(tmp_path)
    store = _seed_calibration_store(tmp_path)
    stub = _StubFundamentalsService()
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(
        controller,
        fundamentals_service=stub,
        fundamentals_cache_dir=tmp_path / "fc",
        analyzer_cache_dir=tmp_path / "ac",
        calibration_store=store,
    )
    client = TestClient(app)
    stub.queue("AAPL", _result_for("AAPL"))

    body = client.get("/api/analyzer/AAPL").json()
    by_name = {r["score_name"]: r for r in body["calibrations"]}
    # rebound_potential is seeded at score 15 only (10 obs). For any
    # other score the analyzer computes, the bucket has zero
    # observations → bucket lookup returns None → bucket_published
    # is False. Either way (no matching bucket OR under-sampled
    # matching bucket), the reading must be unpublished.
    rebound = by_name["rebound_potential"]
    assert rebound["bucket_published"] is False
    assert rebound["hit_rate"] is None
    assert rebound["confidence_low"] is None
    assert rebound["n_observations"] is None
    # Score value preserved so the UI can render "calibration pending"
    # next to the actual read.
    assert isinstance(rebound["score_value"], (int, float))
