"""Unit tests for the normalized fundamentals dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.intelligence.fundamentals.models import (
    BalanceSheet,
    CashFlow,
    CompanyProfile,
    IncomeStatement,
    KeyRatios,
    NormalizedFundamentals,
    ProviderName,
    ReportPeriod,
)


def _income(year: int, period: ReportPeriod = ReportPeriod.ANNUAL) -> IncomeStatement:
    return IncomeStatement(
        period=period,
        fiscal_date=datetime(year, 12, 31, tzinfo=UTC),
        revenue=1.0,
    )


def test_normalized_fundamentals_latest_annual_picks_most_recent() -> None:
    statements = (_income(2023), _income(2025), _income(2024))
    record = NormalizedFundamentals(
        symbol="AAPL",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="AAPL"),
        income_statements=statements,
    )
    latest = record.latest_annual_income
    assert latest is not None
    assert latest.fiscal_date.year == 2025


def test_normalized_fundamentals_latest_skips_quarterlies() -> None:
    statements = (_income(2025, ReportPeriod.QUARTERLY), _income(2023))
    record = NormalizedFundamentals(
        symbol="AAPL",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="AAPL"),
        income_statements=statements,
    )
    latest = record.latest_annual_income
    assert latest is not None
    assert latest.period is ReportPeriod.ANNUAL
    assert latest.fiscal_date.year == 2023


def test_latest_helpers_return_none_when_no_annuals() -> None:
    record = NormalizedFundamentals(
        symbol="AAPL",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="AAPL"),
        income_statements=(),
        balance_sheets=(),
        cash_flows=(),
    )
    assert record.latest_annual_income is None
    assert record.latest_annual_balance is None
    assert record.latest_annual_cashflow is None


def test_normalized_fundamentals_is_frozen() -> None:
    record = NormalizedFundamentals(
        symbol="AAPL",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="AAPL"),
    )
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        record.symbol = "MSFT"  # type: ignore[misc]


def test_key_ratios_default_is_all_none() -> None:
    ratios = KeyRatios()
    # Spot check — every field defaults to None so a partial-coverage
    # provider doesn't silently zero out missing values.
    assert ratios.pe_ratio is None
    assert ratios.peg_ratio is None
    assert ratios.beta is None


def test_balance_sheet_and_cashflow_defaults() -> None:
    bs = BalanceSheet(
        period=ReportPeriod.ANNUAL, fiscal_date=datetime(2024, 12, 31, tzinfo=UTC)
    )
    cf = CashFlow(
        period=ReportPeriod.ANNUAL, fiscal_date=datetime(2024, 12, 31, tzinfo=UTC)
    )
    assert bs.total_assets is None
    assert cf.free_cash_flow is None
    assert bs.reported_currency == "USD"
    assert cf.reported_currency == "USD"
