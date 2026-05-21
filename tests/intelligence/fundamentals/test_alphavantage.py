"""Alpha Vantage provider mapper + quota-detection tests."""

from __future__ import annotations

import pytest

from src.intelligence.fundamentals.alphavantage import (
    _map_balance,
    _map_cashflow,
    _map_income,
    _map_profile,
    _map_ratios,
    _raise_for_quota,
)
from src.intelligence.fundamentals.base import ProviderRateLimited
from src.intelligence.fundamentals.models import ProviderName, ReportPeriod


def test_quota_message_raises_rate_limited() -> None:
    quota_payload = {
        "Information": (
            "Thank you for using Alpha Vantage! Our standard API rate limit "
            "is 25 requests per day. Please subscribe to a premium plan ..."
        )
    }
    with pytest.raises(ProviderRateLimited):
        _raise_for_quota(ProviderName.ALPHA_VANTAGE, quota_payload)


def test_note_field_with_rate_limit_text_raises() -> None:
    with pytest.raises(ProviderRateLimited):
        _raise_for_quota(
            ProviderName.ALPHA_VANTAGE,
            {"Note": "Our standard API rate limit is ..."},
        )


def test_quota_check_passes_for_normal_payload() -> None:
    _raise_for_quota(ProviderName.ALPHA_VANTAGE, {"Symbol": "AAPL", "PERatio": "30"})


def test_map_profile_pulls_canonical_fields() -> None:
    raw = {
        "Name": "Apple Inc",
        "Sector": "Technology",
        "Industry": "Consumer Electronics",
        "Exchange": "NASDAQ",
        "Country": "USA",
        "Currency": "USD",
        "MarketCapitalization": "3000000000000",
        "SharesOutstanding": "15000000000",
        "Description": "iPhones.",
        "CIK": "320193",
    }
    profile = _map_profile("AAPL", raw)
    assert profile.name == "Apple Inc"
    assert profile.sector == "Technology"
    assert profile.market_cap == 3e12
    assert profile.cik == "320193"


def test_map_ratios_handles_none_and_dash_sentinels() -> None:
    raw = {
        "PERatio": "30.5",
        "ForwardPE": "None",
        "PEGRatio": "-",
        "Beta": "1.23",
    }
    ratios = _map_ratios(raw)
    assert ratios.pe_ratio == pytest.approx(30.5)
    # AV emits "None" and "-" as sentinels; those should normalize to None.
    assert ratios.forward_pe is None
    assert ratios.peg_ratio is None
    assert ratios.beta == pytest.approx(1.23)


def test_map_income_balance_cashflow_handle_annual_arrays() -> None:
    income_raw = {
        "annualReports": [
            {
                "fiscalDateEnding": "2024-09-30",
                "totalRevenue": "390000000000",
                "netIncome": "100000000000",
                "reportedCurrency": "USD",
            }
        ]
    }
    balance_raw = {
        "annualReports": [
            {
                "fiscalDateEnding": "2024-09-30",
                "totalAssets": "350000000000",
                "totalShareholderEquity": "60000000000",
            }
        ]
    }
    cashflow_raw = {
        "annualReports": [
            {
                "fiscalDateEnding": "2024-09-30",
                "operatingCashflow": "120000000000",
                "capitalExpenditures": "10000000000",
            }
        ]
    }
    income = _map_income(income_raw)
    balance = _map_balance(balance_raw)
    cashflow = _map_cashflow(cashflow_raw)
    assert income[0].revenue == 3.9e11
    assert income[0].period is ReportPeriod.ANNUAL
    assert balance[0].total_equity == 6e10
    assert cashflow[0].capital_expenditure == 1e10
