"""FMP provider mapper tests.

We don't hit the network. We invoke the private ``_map_*`` functions
directly against realistic JSON fragments so the contract between
FMP's payload shape and our normalized dataclasses is locked down.
The HTTP path is exercised in ``test_service.py`` via fake providers.
"""

from __future__ import annotations

import pytest

from src.intelligence.fundamentals.fmp import (
    _map_balance,
    _map_cashflow,
    _map_income,
    _map_profile,
    _map_ratios,
    _map_targets,
)
from src.intelligence.fundamentals.models import ReportPeriod


def test_map_profile_pulls_canonical_fields() -> None:
    raw = [
        {
            "companyName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "exchangeShortName": "NASDAQ",
            "country": "US",
            "currency": "USD",
            "mktCap": 3_000_000_000_000,
            "sharesOutstanding": 15_000_000_000,
            "description": "Designs phones.",
            "cik": "0000320193",
        }
    ]
    profile = _map_profile("AAPL", raw)
    assert profile.symbol == "AAPL"
    assert profile.name == "Apple Inc."
    assert profile.sector == "Technology"
    assert profile.exchange == "NASDAQ"
    assert profile.market_cap == pytest.approx(3e12)
    assert profile.shares_outstanding == pytest.approx(1.5e10)
    assert profile.cik == "0000320193"


def test_map_profile_handles_empty_response() -> None:
    profile = _map_profile("UNKNOWN", [])
    assert profile.symbol == "UNKNOWN"
    assert profile.name is None
    assert profile.market_cap is None


def test_map_income_marks_annual_vs_quarterly() -> None:
    raw = [
        {
            "period": "FY",
            "date": "2024-09-30",
            "revenue": 100,
            "operatingIncome": 30,
            "netIncome": 25,
            "eps": 1.5,
            "epsdiluted": 1.45,
            "reportedCurrency": "USD",
        },
        {
            "period": "Q3",
            "date": "2024-06-30",
            "revenue": 25,
            "netIncome": 6,
        },
    ]
    statements = _map_income(raw)
    assert len(statements) == 2
    annual = statements[0]
    quarterly = statements[1]
    assert annual.period is ReportPeriod.ANNUAL
    assert quarterly.period is ReportPeriod.QUARTERLY
    assert annual.revenue == 100
    assert annual.eps_diluted == pytest.approx(1.45)


def test_map_balance_and_cashflow_handle_missing_fields() -> None:
    bs_raw = [{"date": "2024-09-30", "period": "FY", "totalAssets": 50}]
    cf_raw = [
        {
            "date": "2024-09-30",
            "period": "FY",
            "operatingCashFlow": 10,
            "capitalExpenditure": -3,
            "freeCashFlow": 7,
        }
    ]
    bs = _map_balance(bs_raw)
    cf = _map_cashflow(cf_raw)
    assert len(bs) == 1
    assert bs[0].total_assets == 50
    assert bs[0].total_equity is None
    assert cf[0].free_cash_flow == 7
    assert cf[0].dividends_paid is None


def test_map_ratios_handles_partial_payload() -> None:
    raw = [
        {
            "peRatioTTM": 30.1,
            "pegRatioTTM": 1.8,
            "priceToSalesRatioTTM": 7.0,
            "netProfitMarginTTM": 0.25,
        }
    ]
    ratios = _map_ratios(raw)
    assert ratios.pe_ratio == pytest.approx(30.1)
    assert ratios.peg_ratio == pytest.approx(1.8)
    assert ratios.price_to_sales == pytest.approx(7.0)
    assert ratios.net_margin == pytest.approx(0.25)
    assert ratios.beta is None  # FMP TTM endpoint doesn't carry beta


def test_map_targets_returns_none_when_all_missing() -> None:
    assert _map_targets([]) is None
    assert _map_targets([{"targetHigh": None, "targetLow": None}]) is None


def test_map_targets_populates_when_present() -> None:
    targets = _map_targets(
        [
            {
                "targetHigh": 250,
                "targetLow": 180,
                "targetConsensus": 215,
                "targetMedian": 220,
                "numberOfAnalystsOpinions": 30,
            }
        ]
    )
    assert targets is not None
    assert targets.target_high == 250
    assert targets.target_mean == 215
    assert targets.number_of_analysts == 30


def test_map_income_tolerates_non_numeric_values() -> None:
    raw = [
        {
            "period": "FY",
            "date": "2024-09-30",
            "revenue": "garbage",
            "netIncome": None,
        }
    ]
    statements = _map_income(raw)
    assert len(statements) == 1
    assert statements[0].revenue is None
    assert statements[0].net_income is None
