"""Tests for the side-by-side comparison endpoint.

Drives ``GET /api/compare/{left}/{right}`` end-to-end through the
shared analyzer assembler, using the same stub fundamentals service
the analyzer endpoint tests use. Confirms the wire shape, the
parallel-fetch semantics, the same-symbol guard, and the degraded
behavior when one side fails.
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
    """Per-symbol queue of pre-built results — same shape the analyzer
    tests use, replicated here so the compare test file is self-contained."""

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


def _result_for(symbol: str, *, revenue_seed: float = 100.0) -> FundamentalsResult:
    """Minimal but realistic FundamentalsResult fixture."""
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
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


def _build_client(tmp_path: Path) -> tuple[TestClient, _StubFundamentalsService]:
    stub = _StubFundamentalsService()
    controller = MockDashboardController(watchlist=["AAPL", "MSFT"], seed=7)
    app = create_app(
        controller,
        fundamentals_service=stub,
        fundamentals_cache_dir=tmp_path / "fundamentals_cache",
        analyzer_cache_dir=tmp_path / "analyzer_cache",
        compare_narrative_cache_dir=tmp_path / "compare_narrative_cache",
    )
    return TestClient(app), stub


def _disable_llm_narrator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the lazy LLM narrator factory to behave as 'not configured'.

    The settings cache + a possibly-real ``.env`` value mean
    monkeypatching the env var alone isn't enough. We replace the
    factory with a stub that records the reason and returns None,
    which is the contract the endpoint expects when LLM mode is
    unavailable.
    """
    import src.api.compare as compare_module

    compare_module._llm_narrator_cache = None
    compare_module._llm_init_attempted = True
    compare_module._llm_init_error = "ANTHROPIC_API_KEY not configured (test)"
    monkeypatch.setattr(
        compare_module,
        "_get_llm_narrator",
        lambda: None,
    )


# ---------------------------------------------------------------------------
# Happy path + wire contract
# ---------------------------------------------------------------------------


def test_compare_endpoint_returns_well_shaped_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two healthy symbols → 200 with the documented shape.

    Frontend keys off ``sections[*].metrics[*].winner`` for the
    chip; an absent field would surface as a render crash, so we
    pin the contract here.
    """
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))
    stub.queue("MSFT", _result_for("MSFT", revenue_seed=200.0))

    res = client.get("/api/compare/AAPL/MSFT")
    assert res.status_code == 200, res.text
    body = res.json()

    expected_keys = {
        "left_symbol",
        "right_symbol",
        "headline",
        "sections",
        "overall_winner",
        "warnings",
        "left_report",
        "right_report",
    }
    assert expected_keys.issubset(body.keys())
    assert body["left_symbol"] == "AAPL"
    assert body["right_symbol"] == "MSFT"
    assert body["overall_winner"] in {"left", "right", "tie", "n/a"}
    assert isinstance(body["sections"], list)
    # Four sections is the comparison contract.
    titles = [s["title"] for s in body["sections"]]
    assert titles == ["Trust", "Valuation", "Technical", "Risk"]
    # Every metric row carries the documented per-row schema.
    for section in body["sections"]:
        for metric in section["metrics"]:
            assert set(metric.keys()) == {
                "metric",
                "label",
                "left_value",
                "right_value",
                "winner",
                "direction",
                "detail",
                "percent",
            }


def test_compare_endpoint_lower_cases_via_case_insensitive_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Endpoint normalizes symbol case → AAPL and aapl yield the same view."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))
    stub.queue("MSFT", _result_for("MSFT"))

    res = client.get("/api/compare/aapl/msft")
    assert res.status_code == 200
    body = res.json()
    assert body["left_symbol"] == "AAPL"
    assert body["right_symbol"] == "MSFT"


def test_compare_endpoint_carries_both_raw_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The view embeds both analyzer reports so the hero panel can
    render trust score badges without an extra round-trip."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))
    stub.queue("MSFT", _result_for("MSFT"))

    body = client.get("/api/compare/AAPL/MSFT").json()
    assert body["left_report"] is not None
    assert body["right_report"] is not None
    assert body["left_report"]["symbol"] == "AAPL"
    assert body["right_report"]["symbol"] == "MSFT"
    assert "trust_score" in body["left_report"]


# ---------------------------------------------------------------------------
# Same-symbol guard
# ---------------------------------------------------------------------------


def test_same_symbol_on_both_sides_400s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Comparing AAPL to AAPL is meaningless — we reject early."""
    monkeypatch.chdir(tmp_path)
    client, _stub = _build_client(tmp_path)
    res = client.get("/api/compare/AAPL/AAPL")
    assert res.status_code == 400
    assert "distinct" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Degraded behavior
# ---------------------------------------------------------------------------


def test_one_side_chain_exhausted_returns_200_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed fundamentals chain on one side degrades, doesn't 500.

    The endpoint should still render a comparison; the failed side
    gets the technical-only view and the merged warnings call out
    the degraded data.
    """
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))
    health = (
        ProviderHealth(
            provider=ProviderName.FMP,
            symbol="MSFT",
            status="unavailable",
            latency_ms=0.0,
            checked_at=datetime.now(UTC),
            error_message="no key",
        ),
    )
    stub.queue(
        "MSFT", ProviderChainExhausted("MSFT", health, ("fmp: unavailable",))
    )

    res = client.get("/api/compare/AAPL/MSFT")
    assert res.status_code == 200
    body = res.json()
    # Both reports present (the analyzer endpoint degrades to
    # technical-only rather than 500).
    assert body["left_report"] is not None
    assert body["right_report"] is not None
    # MSFT side has no fundamentals → fundamentals_freshness is null.
    assert body["right_report"]["fundamentals_freshness"] is None
    # Merged warnings carry the MSFT chain-exhausted message.
    assert any("MSFT" in w and "exhausted" in w.lower() for w in body["warnings"])


def test_overall_winner_documented_as_one_of_the_four_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``overall_winner`` is always one of left|right|tie|n/a."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))
    stub.queue("MSFT", _result_for("MSFT", revenue_seed=200.0))
    res = client.get("/api/compare/AAPL/MSFT")
    assert res.status_code == 200
    assert res.json()["overall_winner"] in {"left", "right", "tie", "n/a"}


# ---------------------------------------------------------------------------
# Narrative endpoint
# ---------------------------------------------------------------------------


def test_narrative_endpoint_template_mode_returns_grounded_narrative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Template mode never touches Anthropic and validator drops zero."""
    monkeypatch.chdir(tmp_path)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))
    stub.queue("MSFT", _result_for("MSFT", revenue_seed=200.0))

    res = client.get("/api/compare/AAPL/MSFT/narrative?mode=template")
    assert res.status_code == 200, res.text
    body = res.json()

    expected_keys = {
        "left_symbol",
        "right_symbol",
        "tagline",
        "headline",
        "momentum",
        "valuation",
        "risk",
        "quality",
        "bottom_line",
        "model",
        "generated_at",
        "warnings",
        "validation",
        "trust_score",
        "mode",
        "assembly_warnings",
        "cache",
    }
    assert expected_keys.issubset(body.keys())
    assert body["left_symbol"] == "AAPL"
    assert body["right_symbol"] == "MSFT"
    assert body["mode"] == "template"
    # Template is grounded by construction.
    assert body["validation"]["drop_count"] == 0
    # Trust score wired in via the research-style scorer.
    assert body["trust_score"]["grade"] in {"A+", "A", "B", "C", "D", "F"}
    # Every section has the documented shape.
    for key in (
        "headline",
        "momentum",
        "valuation",
        "risk",
        "quality",
        "bottom_line",
    ):
        section = body[key]
        assert set(section.keys()) == {"title", "body", "bullets", "provenance"}
        assert isinstance(section["bullets"], list)


def test_narrative_endpoint_same_symbol_400s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _stub = _build_client(tmp_path)
    res = client.get("/api/compare/AAPL/AAPL/narrative?mode=template")
    assert res.status_code == 400


def test_narrative_endpoint_llm_mode_503s_when_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mode=llm`` without a configured narrator surfaces a 503 — opt-in only."""
    monkeypatch.chdir(tmp_path)
    _disable_llm_narrator(monkeypatch)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))
    stub.queue("MSFT", _result_for("MSFT"))
    res = client.get("/api/compare/AAPL/MSFT/narrative?mode=llm")
    assert res.status_code == 503
    assert "ANTHROPIC_API_KEY" in res.json()["detail"]


def test_narrative_endpoint_auto_mode_falls_back_to_template_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No API key + ``mode=auto`` → template output + a warning string."""
    monkeypatch.chdir(tmp_path)
    _disable_llm_narrator(monkeypatch)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))
    stub.queue("MSFT", _result_for("MSFT"))
    res = client.get("/api/compare/AAPL/MSFT/narrative?mode=auto")
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "template"
    assert "warning" in body
    assert "LLM narrative unavailable" in body["warning"]


def test_narrative_endpoint_cache_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second call within TTL returns ``cache=hit``."""
    monkeypatch.chdir(tmp_path)
    _disable_llm_narrator(monkeypatch)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))
    stub.queue("MSFT", _result_for("MSFT"))
    # First call → miss (the narrative cache is empty per-test).
    first = client.get("/api/compare/AAPL/MSFT/narrative?mode=template").json()
    assert first["cache"] == "miss"
    # Second call: analyzer cache + narrative cache both hit.
    stub.queue("AAPL", _result_for("AAPL"))
    stub.queue("MSFT", _result_for("MSFT"))
    second = client.get("/api/compare/AAPL/MSFT/narrative?mode=template").json()
    assert second["cache"] == "hit"


def test_narrative_endpoint_fresh_bypasses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``?fresh=true`` always regenerates regardless of cache state."""
    monkeypatch.chdir(tmp_path)
    _disable_llm_narrator(monkeypatch)
    client, stub = _build_client(tmp_path)
    stub.queue("AAPL", _result_for("AAPL"))
    stub.queue("MSFT", _result_for("MSFT"))
    client.get("/api/compare/AAPL/MSFT/narrative?mode=template")
    stub.queue("AAPL", _result_for("AAPL"))
    stub.queue("MSFT", _result_for("MSFT"))
    fresh = client.get(
        "/api/compare/AAPL/MSFT/narrative?mode=template&fresh=true"
    ).json()
    assert fresh["cache"] == "miss"
