"""Tests for ``src.intelligence.fundamentals.sanity``.

The sanitizer is a single chokepoint: a freshly normalized record from
any provider walks through it before the analyzer sees the data. These
tests pin the rejection rules so an absurd provider value can never
silently propagate into a recommendation.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.intelligence.fundamentals.models import (
    AnalystTargets,
    BalanceSheet,
    CompanyProfile,
    IncomeStatement,
    KeyRatios,
    NormalizedFundamentals,
    ProviderName,
    ReportPeriod,
)
from src.intelligence.fundamentals.sanity import sanitize_normalized


def _make_record(
    *,
    profile: CompanyProfile | None = None,
    income: tuple[IncomeStatement, ...] = (),
    balance: tuple[BalanceSheet, ...] = (),
    ratios: KeyRatios | None = None,
    targets: AnalystTargets | None = None,
) -> NormalizedFundamentals:
    return NormalizedFundamentals(
        symbol="AAPL",
        fetched_at=datetime(2026, 5, 24, tzinfo=UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=profile or CompanyProfile(symbol="AAPL"),
        income_statements=income,
        balance_sheets=balance,
        key_ratios=ratios or KeyRatios(),
        analyst_targets=targets,
    )


def test_clean_record_passes_through_with_no_rejections() -> None:
    record = _make_record(
        profile=CompanyProfile(symbol="AAPL", market_cap=3_000_000_000.0),
        ratios=KeyRatios(pe_ratio=22.0, payout_ratio=0.3),
    )
    sanitized, rejections = sanitize_normalized(record)
    assert rejections == []
    assert sanitized.profile.market_cap == 3_000_000_000.0
    assert sanitized.key_ratios.pe_ratio == 22.0
    # Warnings are unchanged on a clean record.
    assert sanitized.warnings == ()


def test_negative_market_cap_dropped() -> None:
    record = _make_record(
        profile=CompanyProfile(symbol="AAPL", market_cap=-1_000.0),
    )
    sanitized, rejections = sanitize_normalized(record)
    assert sanitized.profile.market_cap is None
    assert ("profile.market_cap", -1_000.0) in rejections
    assert any("market_cap" in w for w in sanitized.warnings)


def test_huge_eps_dropped() -> None:
    bad = IncomeStatement(
        period=ReportPeriod.ANNUAL,
        fiscal_date=datetime(2025, 12, 31, tzinfo=UTC),
        eps_diluted=1_000_000.0,
    )
    sanitized, rejections = sanitize_normalized(_make_record(income=(bad,)))
    assert sanitized.income_statements[0].eps_diluted is None
    assert any("eps_diluted" in field for field, _ in rejections)


def test_negative_pe_dropped() -> None:
    sanitized, rejections = sanitize_normalized(
        _make_record(ratios=KeyRatios(pe_ratio=-5.0))
    )
    assert sanitized.key_ratios.pe_ratio is None
    assert any("pe_ratio" in field for field, _ in rejections)


def test_negative_total_debt_dropped() -> None:
    bad = BalanceSheet(
        period=ReportPeriod.ANNUAL,
        fiscal_date=datetime(2025, 12, 31, tzinfo=UTC),
        total_debt=-500.0,
    )
    sanitized, rejections = sanitize_normalized(_make_record(balance=(bad,)))
    assert sanitized.balance_sheets[0].total_debt is None
    assert any("total_debt" in field for field, _ in rejections)


def test_inverted_analyst_targets_both_dropped() -> None:
    targets = AnalystTargets(target_high=100.0, target_low=200.0)
    sanitized, rejections = sanitize_normalized(_make_record(targets=targets))
    assert sanitized.analyst_targets is not None
    assert sanitized.analyst_targets.target_high is None
    assert sanitized.analyst_targets.target_low is None
    assert any("target_high<low" in field for field, _ in rejections)


def test_payout_ratio_over_cap_dropped() -> None:
    sanitized, rejections = sanitize_normalized(
        _make_record(ratios=KeyRatios(payout_ratio=10.0))
    )
    assert sanitized.key_ratios.payout_ratio is None
    assert any("payout_ratio" in field for field, _ in rejections)


def test_warnings_appended_when_record_has_existing_warnings() -> None:
    record = _make_record(
        profile=CompanyProfile(symbol="AAPL", market_cap=-1.0),
    )
    record_with_prior = record.__class__(
        **{**record.__dict__, "warnings": ("provider:fmp:missing_ebitda",)}
    )
    sanitized, _ = sanitize_normalized(record_with_prior)
    assert sanitized.warnings[0] == "provider:fmp:missing_ebitda"
    assert any("market_cap" in w for w in sanitized.warnings)


# ---------------------------------------------------------------------------
# B-11 expanded rules — physical impossibilities on additional scalars
# ---------------------------------------------------------------------------


def test_negative_cost_of_revenue_dropped() -> None:
    """Cost of producing goods is non-negative by accounting definition."""
    bad = IncomeStatement(
        period=ReportPeriod.ANNUAL,
        fiscal_date=datetime(2025, 12, 31, tzinfo=UTC),
        cost_of_revenue=-500.0,
    )
    sanitized, rejections = sanitize_normalized(_make_record(income=(bad,)))
    assert sanitized.income_statements[0].cost_of_revenue is None
    assert any("cost_of_revenue" in field for field, _ in rejections)


def test_negative_total_assets_dropped() -> None:
    """Assets cannot be negative — only equity can flip sign."""
    bad = BalanceSheet(
        period=ReportPeriod.ANNUAL,
        fiscal_date=datetime(2025, 12, 31, tzinfo=UTC),
        total_assets=-1_000_000.0,
    )
    sanitized, rejections = sanitize_normalized(_make_record(balance=(bad,)))
    assert sanitized.balance_sheets[0].total_assets is None
    assert any("total_assets" in field for field, _ in rejections)


def test_negative_cash_dropped() -> None:
    bad = BalanceSheet(
        period=ReportPeriod.ANNUAL,
        fiscal_date=datetime(2025, 12, 31, tzinfo=UTC),
        cash_and_equivalents=-1.0,
    )
    sanitized, rejections = sanitize_normalized(_make_record(balance=(bad,)))
    assert sanitized.balance_sheets[0].cash_and_equivalents is None
    assert any("cash_and_equivalents" in field for field, _ in rejections)


def test_negative_price_multiples_dropped() -> None:
    """price_to_sales / price_to_book / ev_to_revenue all need price ≥ 0."""
    sanitized, rejections = sanitize_normalized(
        _make_record(
            ratios=KeyRatios(
                price_to_sales=-1.0,
                price_to_book=-2.0,
                ev_to_revenue=-3.0,
            )
        )
    )
    assert sanitized.key_ratios.price_to_sales is None
    assert sanitized.key_ratios.price_to_book is None
    assert sanitized.key_ratios.ev_to_revenue is None
    rejected_fields = {field for field, _ in rejections}
    assert "key_ratios.price_to_sales" in rejected_fields
    assert "key_ratios.price_to_book" in rejected_fields
    assert "key_ratios.ev_to_revenue" in rejected_fields


def test_huge_ev_ebitda_dropped() -> None:
    sanitized, rejections = sanitize_normalized(
        _make_record(ratios=KeyRatios(ev_to_ebitda=99_999.0))
    )
    assert sanitized.key_ratios.ev_to_ebitda is None
    assert any("ev_to_ebitda" in field for field, _ in rejections)


def test_negative_ev_ebitda_kept_when_in_bounds() -> None:
    """Negative EV/EBITDA is real when EBITDA < 0 — keep, don't drop."""
    sanitized, rejections = sanitize_normalized(
        _make_record(ratios=KeyRatios(ev_to_ebitda=-12.0))
    )
    assert sanitized.key_ratios.ev_to_ebitda == -12.0
    assert all("ev_to_ebitda" not in field for field, _ in rejections)


def test_negative_current_ratio_dropped() -> None:
    sanitized, rejections = sanitize_normalized(
        _make_record(ratios=KeyRatios(current_ratio=-0.5, quick_ratio=-0.3))
    )
    assert sanitized.key_ratios.current_ratio is None
    assert sanitized.key_ratios.quick_ratio is None
    fields = {field for field, _ in rejections}
    assert "key_ratios.current_ratio" in fields
    assert "key_ratios.quick_ratio" in fields


def test_huge_margins_dropped_keeping_sign_logic_intact() -> None:
    """Margins past ±10x indicate a percent/fraction parsing error."""
    sanitized, rejections = sanitize_normalized(
        _make_record(
            ratios=KeyRatios(
                gross_margin=42.0,
                operating_margin=-50.0,
                net_margin=0.18,
            )
        )
    )
    assert sanitized.key_ratios.gross_margin is None
    assert sanitized.key_ratios.operating_margin is None
    # In-bounds negative margin survives — real losses produce negatives.
    assert sanitized.key_ratios.net_margin == 0.18
    fields = {field for field, _ in rejections}
    assert "key_ratios.gross_margin" in fields
    assert "key_ratios.operating_margin" in fields


def test_dividend_yield_above_one_dropped() -> None:
    """A yield > 100% is a percent-not-fraction parsing error."""
    sanitized, rejections = sanitize_normalized(
        _make_record(ratios=KeyRatios(dividend_yield=4.5))
    )
    assert sanitized.key_ratios.dividend_yield is None
    assert any("dividend_yield" in field for field, _ in rejections)


def test_dividend_yield_below_zero_dropped() -> None:
    sanitized, rejections = sanitize_normalized(
        _make_record(ratios=KeyRatios(dividend_yield=-0.01))
    )
    assert sanitized.key_ratios.dividend_yield is None
    assert any("dividend_yield" in field for field, _ in rejections)


def test_recommendation_mean_out_of_range_dropped() -> None:
    """Five-point scale: outside [1, 5] is an enum error."""
    sanitized, rejections = sanitize_normalized(
        _make_record(targets=AnalystTargets(recommendation_mean=6.2))
    )
    assert sanitized.analyst_targets is not None
    assert sanitized.analyst_targets.recommendation_mean is None
    assert any(
        "recommendation_mean" in field for field, _ in rejections
    )


def test_number_of_analysts_negative_dropped() -> None:
    sanitized, rejections = sanitize_normalized(
        _make_record(targets=AnalystTargets(number_of_analysts=-3))
    )
    assert sanitized.analyst_targets is not None
    assert sanitized.analyst_targets.number_of_analysts is None
    assert any(
        "number_of_analysts" in field for field, _ in rejections
    )


def test_extreme_beta_dropped() -> None:
    sanitized, rejections = sanitize_normalized(
        _make_record(ratios=KeyRatios(beta=42.0))
    )
    assert sanitized.key_ratios.beta is None
    assert any("beta" in field for field, _ in rejections)
