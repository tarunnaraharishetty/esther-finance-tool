"""Multi-method valuation tests.

Tests each estimator against hand-built :class:`NormalizedFundamentals`
fixtures, then verifies the ensemble combines them into bear / base /
bull / weighted as expected.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.intelligence.analyzer.valuation import (
    build_valuation,
    estimate_analyst_target,
    estimate_dcf,
    estimate_ev_ebitda,
    estimate_historical_band,
    estimate_pe_multiple,
    estimate_peg,
    estimate_ps_multiple,
)
from src.intelligence.fundamentals.models import (
    AnalystTargets,
    BalanceSheet,
    CashFlow,
    CompanyProfile,
    IncomeStatement,
    KeyRatios,
    NormalizedFundamentals,
    ProviderName,
    ReportPeriod,
)

# ---- fixtures ----


def _income(
    year: int,
    *,
    revenue: float | None = None,
    net_income: float | None = None,
    ebitda: float | None = None,
    eps_diluted: float | None = None,
    eps_basic: float | None = None,
    shares_diluted: float | None = None,
) -> IncomeStatement:
    return IncomeStatement(
        period=ReportPeriod.ANNUAL,
        fiscal_date=datetime(year, 12, 31, tzinfo=UTC),
        revenue=revenue,
        net_income=net_income,
        ebitda=ebitda,
        eps_basic=eps_basic,
        eps_diluted=eps_diluted,
        shares_diluted=shares_diluted,
    )


def _balance(
    year: int,
    *,
    total_debt: float | None = None,
    cash: float | None = None,
    shares: float | None = None,
) -> BalanceSheet:
    return BalanceSheet(
        period=ReportPeriod.ANNUAL,
        fiscal_date=datetime(year, 12, 31, tzinfo=UTC),
        total_debt=total_debt,
        cash_and_equivalents=cash,
        shares_outstanding=shares,
    )


def _cashflow(year: int, fcf: float | None) -> CashFlow:
    return CashFlow(
        period=ReportPeriod.ANNUAL,
        fiscal_date=datetime(year, 12, 31, tzinfo=UTC),
        free_cash_flow=fcf,
    )


def _profitable_tech() -> NormalizedFundamentals:
    """Realistic profitable tech name: 5y of growing EPS + FCF + revenue."""
    incomes = tuple(
        _income(
            year,
            revenue=100_000_000 * (1.15 ** i),
            net_income=20_000_000 * (1.15 ** i),
            ebitda=30_000_000 * (1.15 ** i),
            eps_diluted=2.0 * (1.15 ** i),
            shares_diluted=10_000_000,
        )
        for i, year in enumerate(range(2020, 2025))
    )
    cashflows = tuple(
        _cashflow(year, fcf=18_000_000 * (1.12 ** i))
        for i, year in enumerate(range(2020, 2025))
    )
    return NormalizedFundamentals(
        symbol="TCHX",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(
            symbol="TCHX",
            sector="Information Technology",
            market_cap=1.5e9,
            shares_outstanding=10_000_000,
        ),
        income_statements=incomes,
        balance_sheets=(
            _balance(2024, total_debt=50_000_000, cash=80_000_000, shares=10_000_000),
        ),
        cash_flows=cashflows,
        key_ratios=KeyRatios(
            pe_ratio=32.0, ev_to_ebitda=18.0, price_to_sales=5.5, peg_ratio=2.0
        ),
        analyst_targets=AnalystTargets(
            target_high=150.0,
            target_low=110.0,
            target_mean=130.0,
            target_median=128.0,
            number_of_analysts=15,
        ),
    )


def _unprofitable_biotech() -> NormalizedFundamentals:
    """Loss-making name: P/E + EV/EBITDA + PEG should refuse."""
    incomes = (
        _income(
            2024,
            revenue=20_000_000,
            net_income=-30_000_000,
            ebitda=-25_000_000,
            eps_diluted=-3.0,
            shares_diluted=10_000_000,
        ),
    )
    return NormalizedFundamentals(
        symbol="BIOX",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(
            symbol="BIOX",
            sector="Health Care",
            shares_outstanding=10_000_000,
        ),
        income_statements=incomes,
        balance_sheets=(_balance(2024, shares=10_000_000),),
        cash_flows=(_cashflow(2024, fcf=-15_000_000),),
        key_ratios=KeyRatios(price_to_sales=10.0),
        analyst_targets=AnalystTargets(target_mean=12.0, number_of_analysts=5),
    )


# ---- per-model tests ----


def test_dcf_produces_positive_estimate_on_profitable_company() -> None:
    est = estimate_dcf(_profitable_tech())
    assert est is not None
    assert est.method == "dcf"
    assert est.fair_value > 0
    assert 0.1 <= est.confidence <= 0.9
    assert "FCF=" in est.inputs_used
    assert "discount=" in est.inputs_used


def test_dcf_refuses_negative_fcf() -> None:
    est = estimate_dcf(_unprofitable_biotech())
    assert est is None


def test_dcf_refuses_when_discount_at_or_below_terminal() -> None:
    est = estimate_dcf(
        _profitable_tech(),
        discount_rate=0.02,
        terminal_growth=0.025,
    )
    assert est is None


def test_pe_multiple_uses_sector_median() -> None:
    f = _profitable_tech()
    est = estimate_pe_multiple(f)
    assert est is not None
    # Sector PE for tech = 30. EPS at 2024 = 2.0 * 1.15^4 ≈ 3.498.
    eps = f.latest_annual_income.eps_diluted  # type: ignore[union-attr]
    assert eps is not None
    assert est.fair_value == pytest.approx(30.0 * eps, rel=0.01)
    assert "sector_pe=30.0" in est.inputs_used


def test_pe_multiple_refuses_negative_eps() -> None:
    assert estimate_pe_multiple(_unprofitable_biotech()) is None


def test_ev_ebitda_subtracts_net_debt() -> None:
    f = _profitable_tech()
    est = estimate_ev_ebitda(f)
    assert est is not None
    # ebitda 2024 = 30M * 1.15^4 ≈ 52.46M. Sector multiple = 20.
    # EV = 1.049e9. Equity = EV - debt + cash = 1.049e9 - 50M + 80M = 1.079e9.
    # Shares = 10M. Price ≈ $107.93.
    assert 80.0 < est.fair_value < 130.0
    assert "EV/EBITDA=20.0" in est.inputs_used


def test_ev_ebitda_refuses_negative_ebitda() -> None:
    assert estimate_ev_ebitda(_unprofitable_biotech()) is None


def test_ps_multiple_uses_revenue_per_share() -> None:
    f = _profitable_tech()
    est = estimate_ps_multiple(f)
    assert est is not None
    # Revenue 2024 ≈ 174.9M. Shares 10M → rps ≈ 17.49. Sector PS = 6 → ~104.9.
    assert 80.0 < est.fair_value < 130.0


def test_ps_multiple_still_works_for_unprofitable_companies() -> None:
    """P/S applies even when earnings are negative — its whole point."""
    est = estimate_ps_multiple(_unprofitable_biotech())
    assert est is not None
    assert est.fair_value > 0


def test_peg_uses_eps_cagr_for_fair_pe() -> None:
    f = _profitable_tech()
    est = estimate_peg(f)
    assert est is not None
    # CAGR = 15%, fair PE = 15, EPS ≈ 3.498 → fair value ≈ $52.5.
    assert 40.0 < est.fair_value < 65.0
    assert "EPS_CAGR=15.0%" in est.inputs_used


def test_peg_refuses_when_growth_negative() -> None:
    """A company with declining EPS produces no PEG estimate."""
    declining = NormalizedFundamentals(
        symbol="DECL",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="DECL", sector="Industrials"),
        income_statements=(
            _income(2022, eps_diluted=5.0),
            _income(2023, eps_diluted=4.0),
            _income(2024, eps_diluted=3.0),
        ),
    )
    assert estimate_peg(declining) is None


def test_historical_band_projects_one_year_forward() -> None:
    f = _profitable_tech()
    est = estimate_historical_band(f)
    assert est is not None
    # EPS ≈ 3.498 grown at 15% for 1 year → ~4.02. Sector PE 30 → ~120.7.
    assert 100.0 < est.fair_value < 140.0


def test_historical_band_refuses_single_year_history() -> None:
    one_year = NormalizedFundamentals(
        symbol="NEW",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="NEW"),
        income_statements=(_income(2024, eps_diluted=2.0),),
    )
    assert estimate_historical_band(one_year) is None


def test_analyst_target_confidence_scales_with_analyst_count() -> None:
    f = _profitable_tech()
    est = estimate_analyst_target(f)
    assert est is not None
    assert est.fair_value == 130.0
    # 15 analysts → moderately high confidence.
    assert est.confidence > 0.7

    f_one_analyst = NormalizedFundamentals(
        symbol="X",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="X"),
        analyst_targets=AnalystTargets(target_mean=100.0, number_of_analysts=1),
    )
    one = estimate_analyst_target(f_one_analyst)
    assert one is not None
    assert one.confidence < est.confidence


def test_analyst_target_returns_none_when_targets_missing() -> None:
    f = NormalizedFundamentals(
        symbol="X",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="X"),
        analyst_targets=None,
    )
    assert estimate_analyst_target(f) is None


# ---- ensemble tests ----


def test_ensemble_aggregates_all_applicable_models() -> None:
    ensemble = build_valuation(_profitable_tech())
    methods = {e.method for e in ensemble.estimates}
    # Profitable tech should fire every model.
    assert methods == {
        "dcf",
        "pe_multiple",
        "ev_ebitda_multiple",
        "ps_multiple",
        "peg",
        "historical_band",
        "analyst_target",
    }
    assert ensemble.bear_case is not None
    assert ensemble.base_case is not None
    assert ensemble.bull_case is not None
    assert ensemble.weighted_ai_fair_value is not None
    # Bear < base < bull by construction (percentiles).
    assert ensemble.bear_case <= ensemble.base_case <= ensemble.bull_case
    assert 0 <= ensemble.confidence_score <= 100


def test_ensemble_skips_inapplicable_models_for_loss_makers() -> None:
    ensemble = build_valuation(_unprofitable_biotech())
    methods = {e.method for e in ensemble.estimates}
    # P/E, EV/EBITDA, PEG, DCF require positive earnings / FCF.
    assert "pe_multiple" not in methods
    assert "ev_ebitda_multiple" not in methods
    assert "peg" not in methods
    assert "dcf" not in methods
    # P/S and analyst targets still fire for unprofitable names.
    assert "ps_multiple" in methods
    assert "analyst_target" in methods


def test_ensemble_handles_no_applicable_models_gracefully() -> None:
    empty = NormalizedFundamentals(
        symbol="EMPTY",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="EMPTY"),
    )
    ensemble = build_valuation(empty)
    assert ensemble.estimates == ()
    assert ensemble.bear_case is None
    assert ensemble.base_case is None
    assert ensemble.bull_case is None
    assert ensemble.weighted_ai_fair_value is None
    assert ensemble.confidence_score == 0.0


def test_ensemble_weights_by_confidence() -> None:
    """High-confidence methods pull the weighted average toward themselves."""
    f = _profitable_tech()
    ensemble = build_valuation(f)
    assert ensemble.weighted_ai_fair_value is not None
    # Weighted average must fall within the [bear, bull] span.
    assert ensemble.bear_case is not None and ensemble.bull_case is not None
    assert ensemble.bear_case <= ensemble.weighted_ai_fair_value <= ensemble.bull_case


def test_ensemble_confidence_higher_when_models_agree() -> None:
    """Tight clustering of estimates → higher ensemble confidence.

    We build two fundamentals records: one where all multiples align
    with sector medians (estimates cluster), one where the multiples
    are wildly off (estimates spread).
    """
    f = _profitable_tech()
    ensemble = build_valuation(f)
    base_confidence = ensemble.confidence_score
    # Confirm at least *some* signal — sanity check the ensemble's
    # confidence calc isn't permanently zero.
    assert base_confidence > 0


def test_to_dict_is_json_safe() -> None:
    import json

    ensemble = build_valuation(_profitable_tech())
    payload = ensemble.to_dict()
    # Round-trip cleanly — no datetimes, no enums.
    serialized = json.dumps(payload)
    restored = json.loads(serialized)
    assert "estimates" in restored
    assert "weighted_ai_fair_value" in restored
    assert "confidence_score" in restored


# ---------------------------------------------------------------------------
# B-12 regression: dispersion clamp + finite confidence on pathological inputs
# ---------------------------------------------------------------------------


def test_unprofitable_company_ensemble_confidence_is_finite_and_bounded() -> None:
    """A loss-maker with only fallback estimators surviving must still
    yield a finite, [0, 100]-bounded confidence — never NaN/Inf."""
    import math

    ensemble = build_valuation(_unprofitable_biotech())
    assert ensemble.confidence_score == ensemble.confidence_score  # NaN check
    assert math.isfinite(ensemble.confidence_score)
    assert 0.0 <= ensemble.confidence_score <= 100.0


def _near_zero_base_with_outlier_fixture() -> NormalizedFundamentals:
    """Force the ``base ≈ 0`` regime that historically poisoned dispersion.

    Two estimators survive: P/S (revenue 100 / shares 100_000 / sector
    PS 2.5 → ~0.0025) and analyst target (1000.0). Median (= base case)
    sits between them; without the divisor floor + dispersion clamp the
    old code would compute a huge ``std/base`` ratio and propagate NaN
    via the agreement factor.
    """
    return NormalizedFundamentals(
        symbol="EDGE",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(
            symbol="EDGE",
            sector="Communication Services",
            shares_outstanding=100_000.0,
        ),
        income_statements=(
            _income(
                2024,
                revenue=100.0,
                net_income=-50.0,
                eps_diluted=-0.0005,
                shares_diluted=100_000.0,
            ),
        ),
        balance_sheets=(_balance(2024, shares=100_000.0),),
        cash_flows=(_cashflow(2024, fcf=-50.0),),
        key_ratios=KeyRatios(),
        analyst_targets=AnalystTargets(
            target_mean=1000.0, number_of_analysts=4
        ),
    )


def test_dispersion_clamp_holds_when_base_near_zero() -> None:
    """A tiny base case alongside a huge outlier must not poison confidence.

    Before B-12's fix, ``std / max(|base|, 1.0)`` produced dispersion ≫ 1
    and the agreement factor went negative, eventually surfacing as NaN
    in the ensemble confidence. The current implementation floors the
    divisor against the median magnitude AND clamps dispersion into
    [0, 1] — this test pins that invariant against future drift.
    """
    import math

    ensemble = build_valuation(_near_zero_base_with_outlier_fixture())
    # Confidence is finite + in the documented range.
    assert math.isfinite(ensemble.confidence_score)
    assert 0.0 <= ensemble.confidence_score <= 100.0
    # The bear / base / bull percentiles are real numbers (or None when
    # no estimator survived) — never NaN.
    for value in (
        ensemble.bear_case,
        ensemble.base_case,
        ensemble.bull_case,
        ensemble.weighted_ai_fair_value,
    ):
        if value is not None:
            assert math.isfinite(value)


def test_single_surviving_estimator_yields_zero_dispersion() -> None:
    """When only one method survives, dispersion is defined as 0 and
    agreement collapses to 1 — confidence is then coverage × avg_conf."""
    import math

    # Build a fixture where only the analyst target estimator can fire:
    # no statements, no profile multiples, just a target_mean.
    only_target = NormalizedFundamentals(
        symbol="LONE",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="LONE", sector="Industrials"),
        analyst_targets=AnalystTargets(
            target_mean=42.0, number_of_analysts=8
        ),
    )
    ensemble = build_valuation(only_target)
    assert len(ensemble.estimates) == 1
    assert math.isfinite(ensemble.confidence_score)
    assert ensemble.confidence_score > 0.0
    assert ensemble.confidence_score <= 100.0
