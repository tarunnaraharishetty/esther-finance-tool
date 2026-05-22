"""Pure-function tests for :func:`reconcile`.

The orchestration test (does the service call reconcile at the right
times?) lives in ``test_service.py``. This module focuses on the
comparison rules: which fields fire, at which threshold, with which
fiscal-date semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.intelligence.fundamentals.models import (
    BalanceSheet,
    CompanyProfile,
    IncomeStatement,
    NormalizedFundamentals,
    ProviderName,
    ReportPeriod,
)
from src.intelligence.fundamentals.reconciliation import reconcile

_FISCAL = datetime(2024, 12, 31, tzinfo=UTC)


def _income(
    *,
    fiscal_date: datetime = _FISCAL,
    revenue: float | None = 100.0,
    net_income: float | None = 20.0,
    eps_diluted: float | None = 2.0,
    eps_basic: float | None = None,
) -> IncomeStatement:
    return IncomeStatement(
        period=ReportPeriod.ANNUAL,
        fiscal_date=fiscal_date,
        revenue=revenue,
        net_income=net_income,
        eps_diluted=eps_diluted,
        eps_basic=eps_basic,
    )


def _balance(
    *,
    fiscal_date: datetime = _FISCAL,
    total_debt: float | None = 50.0,
) -> BalanceSheet:
    return BalanceSheet(
        period=ReportPeriod.ANNUAL,
        fiscal_date=fiscal_date,
        total_debt=total_debt,
    )


def _fundamentals(
    *,
    provider: ProviderName = ProviderName.FMP,
    income: IncomeStatement | None = None,
    balance: BalanceSheet | None = None,
) -> NormalizedFundamentals:
    return NormalizedFundamentals(
        symbol="AAPL",
        fetched_at=datetime.now(UTC),
        primary_provider=provider,
        contributing_providers=(provider,),
        profile=CompanyProfile(symbol="AAPL"),
        income_statements=(income or _income(),),
        balance_sheets=(balance or _balance(),),
    )


# -----------------------------------------------------------------------------
# Agreement / no divergence
# -----------------------------------------------------------------------------


def test_identical_fundamentals_produce_no_divergences() -> None:
    primary = _fundamentals(provider=ProviderName.FINNHUB)
    secondary = _fundamentals(provider=ProviderName.ALPHA_VANTAGE)
    result = reconcile(primary, secondary, threshold=0.05)
    assert result.divergences == ()
    assert result.warnings == ()


def test_within_threshold_does_not_fire() -> None:
    """4% difference at a 5% threshold → agreement."""
    primary = _fundamentals(income=_income(revenue=100.0))
    secondary = _fundamentals(
        provider=ProviderName.FINNHUB, income=_income(revenue=104.0)
    )
    result = reconcile(primary, secondary, threshold=0.05)
    assert result.divergences == ()


# -----------------------------------------------------------------------------
# Magnitude divergence
# -----------------------------------------------------------------------------


def test_above_threshold_fires_with_correct_metadata() -> None:
    primary = _fundamentals(
        provider=ProviderName.FMP, income=_income(revenue=100.0)
    )
    secondary = _fundamentals(
        provider=ProviderName.FINNHUB, income=_income(revenue=120.0)
    )
    result = reconcile(primary, secondary, threshold=0.05)
    revenue_divs = [d for d in result.divergences if d.field == "revenue"]
    assert len(revenue_divs) == 1
    d = revenue_divs[0]
    assert d.primary_value == 100.0
    assert d.secondary_value == 120.0
    # |100-120| / max(|100|,|120|) = 20/120 ≈ 0.1667
    assert d.relative_divergence == pytest.approx(0.16667, abs=1e-4)
    assert d.primary_provider == "fmp"
    assert d.secondary_provider == "finnhub"
    assert d.fiscal_date == _FISCAL


def test_each_high_trust_field_can_diverge() -> None:
    primary = _fundamentals(
        income=_income(revenue=100, net_income=10, eps_diluted=1.0),
        balance=_balance(total_debt=50),
    )
    secondary = _fundamentals(
        provider=ProviderName.FINNHUB,
        income=_income(revenue=200, net_income=20, eps_diluted=2.0),
        balance=_balance(total_debt=200),
    )
    result = reconcile(primary, secondary, threshold=0.05)
    fields = {d.field for d in result.divergences}
    assert fields == {"revenue", "net_income", "eps_diluted", "total_debt"}


def test_eps_falls_back_to_basic_when_diluted_missing() -> None:
    """A provider that emits only basic EPS still participates."""
    primary = _fundamentals(
        income=_income(eps_diluted=2.0)
    )
    secondary = _fundamentals(
        provider=ProviderName.FINNHUB,
        # Only basic, no diluted → reconciliation should still compare
        income=_income(eps_diluted=None, eps_basic=2.5),
    )
    result = reconcile(primary, secondary, threshold=0.05)
    eps_divs = [d for d in result.divergences if d.field == "eps_diluted"]
    assert len(eps_divs) == 1
    assert eps_divs[0].secondary_value == 2.5


# -----------------------------------------------------------------------------
# Sign-flip rule (always fires regardless of magnitude)
# -----------------------------------------------------------------------------


def test_sign_flip_on_net_income_fires_at_any_magnitude() -> None:
    """Net income +1 vs −1 must surface even though magnitudes are small."""
    primary = _fundamentals(income=_income(net_income=1.0))
    secondary = _fundamentals(
        provider=ProviderName.FINNHUB, income=_income(net_income=-1.0)
    )
    # Threshold 100% would normally never fire on close magnitudes —
    # but a sign flip always does.
    result = reconcile(primary, secondary, threshold=1.0)
    flips = [d for d in result.divergences if d.field == "net_income"]
    assert len(flips) == 1


def test_both_zero_is_agreement_not_divergence() -> None:
    """Edge case: both providers report 0 → no signal, no warning."""
    primary = _fundamentals(income=_income(revenue=0.0))
    secondary = _fundamentals(
        provider=ProviderName.FINNHUB, income=_income(revenue=0.0)
    )
    result = reconcile(primary, secondary, threshold=0.05)
    revenue_divs = [d for d in result.divergences if d.field == "revenue"]
    assert revenue_divs == []


# -----------------------------------------------------------------------------
# Missing / sparse data
# -----------------------------------------------------------------------------


def test_missing_field_on_either_side_does_not_fire() -> None:
    primary = _fundamentals(income=_income(revenue=100.0))
    secondary = _fundamentals(
        provider=ProviderName.FINNHUB,
        income=_income(revenue=None),
    )
    result = reconcile(primary, secondary, threshold=0.05)
    revenue_divs = [d for d in result.divergences if d.field == "revenue"]
    assert revenue_divs == []


def test_missing_income_statement_emits_warnings() -> None:
    """No annual income statement on one side → warnings, no divergences."""
    primary_with_income = _fundamentals(income=_income())
    secondary_no_income = NormalizedFundamentals(
        symbol="AAPL",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FINNHUB,
        contributing_providers=(ProviderName.FINNHUB,),
        profile=CompanyProfile(symbol="AAPL"),
        # No income statements at all.
        balance_sheets=(_balance(),),
    )
    result = reconcile(primary_with_income, secondary_no_income, threshold=0.05)
    income_fields = {"revenue", "net_income", "eps_diluted"}
    warning_fields = {w.field for w in result.warnings}
    # Each income field should have a warning explaining the absence.
    assert income_fields.issubset(warning_fields)
    # No income-field divergences.
    income_divs = [d for d in result.divergences if d.field in income_fields]
    assert income_divs == []


# -----------------------------------------------------------------------------
# Fiscal-date mismatch
# -----------------------------------------------------------------------------


def test_fiscal_date_mismatch_emits_warnings_not_divergences() -> None:
    """Different reporting periods are release-timing, not disagreement."""
    primary = _fundamentals(
        income=_income(fiscal_date=datetime(2024, 12, 31, tzinfo=UTC), revenue=100.0)
    )
    secondary = _fundamentals(
        provider=ProviderName.FINNHUB,
        income=_income(fiscal_date=datetime(2023, 12, 31, tzinfo=UTC), revenue=200.0),
    )
    result = reconcile(primary, secondary, threshold=0.05)
    income_divs = [
        d
        for d in result.divergences
        if d.field in {"revenue", "net_income", "eps_diluted"}
    ]
    assert income_divs == []
    # Per-field warnings should explain the fiscal-date mismatch.
    for w in result.warnings:
        if w.field in {"revenue", "net_income", "eps_diluted"}:
            assert "fiscal_date mismatch" in w.reason


def test_fiscal_date_matches_on_one_family_only() -> None:
    """Income family matches but balance family doesn't → split outcome."""
    primary = _fundamentals(
        income=_income(revenue=100.0),
        balance=_balance(
            fiscal_date=datetime(2024, 12, 31, tzinfo=UTC), total_debt=50.0
        ),
    )
    secondary = _fundamentals(
        provider=ProviderName.FINNHUB,
        income=_income(revenue=150.0),
        balance=_balance(
            fiscal_date=datetime(2023, 12, 31, tzinfo=UTC), total_debt=200.0
        ),
    )
    result = reconcile(primary, secondary, threshold=0.05)
    revenue_divs = [d for d in result.divergences if d.field == "revenue"]
    debt_divs = [d for d in result.divergences if d.field == "total_debt"]
    debt_warnings = [w for w in result.warnings if w.field == "total_debt"]
    # Income matched → divergence fires.
    assert len(revenue_divs) == 1
    # Balance fiscal date mismatch → warning, no divergence.
    assert debt_divs == []
    assert len(debt_warnings) == 1
    assert "fiscal_date mismatch" in debt_warnings[0].reason


# -----------------------------------------------------------------------------
# Threshold validation
# -----------------------------------------------------------------------------


def test_negative_threshold_rejected() -> None:
    primary = _fundamentals()
    secondary = _fundamentals(provider=ProviderName.FINNHUB)
    with pytest.raises(ValueError, match="non-negative"):
        reconcile(primary, secondary, threshold=-0.01)


def test_zero_threshold_fires_on_any_difference() -> None:
    primary = _fundamentals(income=_income(revenue=100.0))
    secondary = _fundamentals(
        provider=ProviderName.FINNHUB, income=_income(revenue=100.01)
    )
    result = reconcile(primary, secondary, threshold=0.0)
    revenue_divs = [d for d in result.divergences if d.field == "revenue"]
    assert len(revenue_divs) == 1
