"""Sanity bounds on normalized provider data.

Providers occasionally ship absurd numbers: negative market cap from a
data-loading hiccup, EPS larger than the stock price from a bad shares
count, P/E ratios from missing earnings denominators. None of these are
meaningful inputs to the valuation ensemble — at best they propagate as
nonsense, at worst they flip the final recommendation.

This module is a single chokepoint that scrubs a freshly normalized
:class:`NormalizedFundamentals` before it reaches the analyzer. The
contract:

* Bad scalars are replaced with ``None`` (the existing "missing" signal
  every downstream model already handles).
* A human-readable warning is appended to the record so operators can
  see what was rejected via ``/api/fundamentals/{symbol}``.
* The rejected (field, value) pairs are returned alongside the cleaned
  record so the orchestrator can sink them into the accuracy ledger as
  evidence of provider quality drift.

Only obvious impossibilities are rejected — we don't try to enforce
business-correctness (e.g. flagging negative net income, which is real).
The bar is *physically impossible* or *catastrophically corrupted*.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from src.intelligence.fundamentals.models import (
    AnalystTargets,
    BalanceSheet,
    CompanyProfile,
    IncomeStatement,
    KeyRatios,
    NormalizedFundamentals,
)

# Caps tuned for US equities. They're loose on purpose — the goal is
# "this number cannot exist," not "we don't believe this number."
_MAX_REASONABLE_EPS: Final[float] = 5_000.0  # BRK.A class A is ~ $7K EPS; cap above.
_MAX_REASONABLE_PE: Final[float] = 10_000.0  # Beyond this, the denominator is broken.
_MAX_REASONABLE_PAYOUT: Final[float] = 5.0  # 500% payout = data error, not policy.
# Magnitude caps for ensemble multiples. Both signs are legal (a
# negative EV/EBITDA describes a money-losing balance sheet) but
# numbers this large indicate a broken denominator rather than a real
# read.
_MAX_REASONABLE_EV_EBITDA: Final[float] = 10_000.0
_MAX_REASONABLE_PEG: Final[float] = 100.0
# Margins and returns are fractions in [-N, +N]. Real margins live in
# [-2, +1] roughly; anything past 10x in either direction is a parsing
# error (percent reported as fraction, or a missing denominator).
_MAX_REASONABLE_MARGIN: Final[float] = 10.0
_MAX_REASONABLE_RETURN: Final[float] = 10.0
# Beta lives in roughly [-3, +3]; values past ±20 indicate a broken
# regression denominator.
_MAX_REASONABLE_BETA: Final[float] = 20.0
# Recommendation mean is on a five-point scale (1=Strong Buy ...
# 5=Strong Sell). Anything outside is an enum error.
_RECOMMENDATION_MEAN_MIN: Final[float] = 1.0
_RECOMMENDATION_MEAN_MAX: Final[float] = 5.0
# Dividend yield is a non-negative fraction; > 100% is a parsing error
# (e.g. raw percent emitted without dividing by 100).
_MAX_REASONABLE_DIVIDEND_YIELD: Final[float] = 1.0


Rejection = tuple[str, float]
"""``(field_path, original_value)`` for a single rejection."""


def sanitize_normalized(
    record: NormalizedFundamentals,
) -> tuple[NormalizedFundamentals, list[Rejection]]:
    """Return a cleaned copy of ``record`` plus the list of rejections.

    The shape is `(sanitized, rejections)` so callers can log / sink the
    rejections to the accuracy ledger without re-walking the record. An
    empty rejection list means every scalar passed the bounds.

    Pure function: ``record`` is never mutated; the cleaned copy is
    built via :func:`dataclasses.replace`.
    """
    rejections: list[Rejection] = []

    new_profile = _sanitize_profile(record.profile, rejections)
    new_income = tuple(
        _sanitize_income(stmt, idx, rejections)
        for idx, stmt in enumerate(record.income_statements)
    )
    new_balance = tuple(
        _sanitize_balance(stmt, idx, rejections)
        for idx, stmt in enumerate(record.balance_sheets)
    )
    new_ratios = _sanitize_ratios(record.key_ratios, rejections)
    new_targets = (
        _sanitize_targets(record.analyst_targets, rejections)
        if record.analyst_targets is not None
        else None
    )

    warnings = record.warnings
    if rejections:
        warning_lines = tuple(
            f"sanity: dropped {field} (value {value!r})"
            for field, value in rejections
        )
        warnings = warnings + warning_lines

    sanitized = replace(
        record,
        profile=new_profile,
        income_statements=new_income,
        balance_sheets=new_balance,
        key_ratios=new_ratios,
        analyst_targets=new_targets,
        warnings=warnings,
    )
    return sanitized, rejections


# ---------------------------------------------------------------------------
# Per-record sanitizers
# ---------------------------------------------------------------------------


def _sanitize_profile(
    profile: CompanyProfile, rejections: list[Rejection]
) -> CompanyProfile:
    market_cap = _drop_if_negative(profile.market_cap, "profile.market_cap", rejections)
    ev = _drop_if_negative(
        profile.enterprise_value, "profile.enterprise_value", rejections
    )
    shares = _drop_if_negative(
        profile.shares_outstanding, "profile.shares_outstanding", rejections
    )
    return replace(
        profile,
        market_cap=market_cap,
        enterprise_value=ev,
        shares_outstanding=shares,
    )


def _sanitize_income(
    stmt: IncomeStatement, idx: int, rejections: list[Rejection]
) -> IncomeStatement:
    eps_basic = _drop_if_unreasonable(
        stmt.eps_basic,
        _MAX_REASONABLE_EPS,
        f"income_statements[{idx}].eps_basic",
        rejections,
    )
    eps_diluted = _drop_if_unreasonable(
        stmt.eps_diluted,
        _MAX_REASONABLE_EPS,
        f"income_statements[{idx}].eps_diluted",
        rejections,
    )
    shares = _drop_if_negative(
        stmt.shares_diluted,
        f"income_statements[{idx}].shares_diluted",
        rejections,
    )
    # Cost of revenue is the amount of money spent producing the goods —
    # always non-negative by accounting definition. A negative value is
    # a sign flip on the provider side.
    cost = _drop_if_negative(
        stmt.cost_of_revenue,
        f"income_statements[{idx}].cost_of_revenue",
        rejections,
    )
    return replace(
        stmt,
        eps_basic=eps_basic,
        eps_diluted=eps_diluted,
        shares_diluted=shares,
        cost_of_revenue=cost,
    )


def _sanitize_balance(
    stmt: BalanceSheet, idx: int, rejections: list[Rejection]
) -> BalanceSheet:
    total_debt = _drop_if_negative(
        stmt.total_debt, f"balance_sheets[{idx}].total_debt", rejections
    )
    long_term_debt = _drop_if_negative(
        stmt.long_term_debt,
        f"balance_sheets[{idx}].long_term_debt",
        rejections,
    )
    shares = _drop_if_negative(
        stmt.shares_outstanding,
        f"balance_sheets[{idx}].shares_outstanding",
        rejections,
    )
    # Assets / cash / investments are physical quantities — negative is
    # nonsensical and indicates a parsing error (typically a sign-flipped
    # liability extracted into an asset column). Equity is deliberately
    # NOT bounded here — negative equity is a real financial state
    # (liabilities > assets).
    total_assets = _drop_if_negative(
        stmt.total_assets, f"balance_sheets[{idx}].total_assets", rejections
    )
    cash = _drop_if_negative(
        stmt.cash_and_equivalents,
        f"balance_sheets[{idx}].cash_and_equivalents",
        rejections,
    )
    short_term = _drop_if_negative(
        stmt.short_term_investments,
        f"balance_sheets[{idx}].short_term_investments",
        rejections,
    )
    return replace(
        stmt,
        total_debt=total_debt,
        long_term_debt=long_term_debt,
        shares_outstanding=shares,
        total_assets=total_assets,
        cash_and_equivalents=cash,
        short_term_investments=short_term,
    )


def _sanitize_ratios(
    ratios: KeyRatios, rejections: list[Rejection]
) -> KeyRatios:
    # P/E below zero or huge → drop. Forward P/E follows the same shape.
    pe = _drop_if_unreasonable_pe(
        ratios.pe_ratio, "key_ratios.pe_ratio", rejections
    )
    forward_pe = _drop_if_unreasonable_pe(
        ratios.forward_pe, "key_ratios.forward_pe", rejections
    )
    # Payout ratios outside [0, _MAX_REASONABLE_PAYOUT] are data errors.
    payout = _drop_if_unreasonable_payout(
        ratios.payout_ratio, "key_ratios.payout_ratio", rejections
    )
    # Price-based multiples must be non-negative: price ≥ 0 always, and
    # the denominators (sales, book value, revenue) are non-negative too.
    # A negative read is a parsing error.
    p_to_s = _drop_if_negative(
        ratios.price_to_sales, "key_ratios.price_to_sales", rejections
    )
    p_to_b = _drop_if_negative(
        ratios.price_to_book, "key_ratios.price_to_book", rejections
    )
    ev_to_rev = _drop_if_negative(
        ratios.ev_to_revenue, "key_ratios.ev_to_revenue", rejections
    )
    # EV/EBITDA: negative is real (EBITDA < 0); cap absurd magnitudes.
    ev_to_ebitda = _drop_if_unreasonable(
        ratios.ev_to_ebitda,
        _MAX_REASONABLE_EV_EBITDA,
        "key_ratios.ev_to_ebitda",
        rejections,
    )
    # PEG can be negative (negative growth). Cap magnitudes.
    peg = _drop_if_unreasonable(
        ratios.peg_ratio, _MAX_REASONABLE_PEG, "key_ratios.peg_ratio", rejections
    )
    # Liquidity ratios are amount-over-amount on non-negative quantities.
    current = _drop_if_negative(
        ratios.current_ratio, "key_ratios.current_ratio", rejections
    )
    quick = _drop_if_negative(
        ratios.quick_ratio, "key_ratios.quick_ratio", rejections
    )
    # Margins and returns are fractions; magnitudes past ±10 indicate a
    # percent-vs-fraction parsing error or a broken denominator. Keep
    # the sign (real losses produce negative margins).
    gross_m = _drop_if_unreasonable(
        ratios.gross_margin,
        _MAX_REASONABLE_MARGIN,
        "key_ratios.gross_margin",
        rejections,
    )
    operating_m = _drop_if_unreasonable(
        ratios.operating_margin,
        _MAX_REASONABLE_MARGIN,
        "key_ratios.operating_margin",
        rejections,
    )
    net_m = _drop_if_unreasonable(
        ratios.net_margin,
        _MAX_REASONABLE_MARGIN,
        "key_ratios.net_margin",
        rejections,
    )
    roe = _drop_if_unreasonable(
        ratios.return_on_equity,
        _MAX_REASONABLE_RETURN,
        "key_ratios.return_on_equity",
        rejections,
    )
    roa = _drop_if_unreasonable(
        ratios.return_on_assets,
        _MAX_REASONABLE_RETURN,
        "key_ratios.return_on_assets",
        rejections,
    )
    beta = _drop_if_unreasonable(
        ratios.beta, _MAX_REASONABLE_BETA, "key_ratios.beta", rejections
    )
    # Dividend yield is a non-negative fraction; > 100% means the
    # provider emitted a raw percent without dividing by 100.
    div_yield = _drop_if_outside(
        ratios.dividend_yield,
        lo=0.0,
        hi=_MAX_REASONABLE_DIVIDEND_YIELD,
        field_path="key_ratios.dividend_yield",
        rejections=rejections,
    )
    return replace(
        ratios,
        pe_ratio=pe,
        forward_pe=forward_pe,
        payout_ratio=payout,
        price_to_sales=p_to_s,
        price_to_book=p_to_b,
        ev_to_revenue=ev_to_rev,
        ev_to_ebitda=ev_to_ebitda,
        peg_ratio=peg,
        current_ratio=current,
        quick_ratio=quick,
        gross_margin=gross_m,
        operating_margin=operating_m,
        net_margin=net_m,
        return_on_equity=roe,
        return_on_assets=roa,
        beta=beta,
        dividend_yield=div_yield,
    )


def _sanitize_targets(
    targets: AnalystTargets, rejections: list[Rejection]
) -> AnalystTargets:
    high = _drop_if_negative(targets.target_high, "analyst_targets.target_high", rejections)
    low = _drop_if_negative(targets.target_low, "analyst_targets.target_low", rejections)
    mean = _drop_if_negative(targets.target_mean, "analyst_targets.target_mean", rejections)
    median = _drop_if_negative(
        targets.target_median, "analyst_targets.target_median", rejections
    )
    # Inverted high/low pair: drop both rather than guess which to trust.
    if high is not None and low is not None and high < low:
        rejections.append(("analyst_targets.target_high<low", high))
        rejections.append(("analyst_targets.target_low>high", low))
        high = None
        low = None
    # Recommendation mean is a five-point enum (1=Strong Buy ... 5=Strong
    # Sell). A value outside [1, 5] is a provider parsing error.
    rec_mean = _drop_if_outside(
        targets.recommendation_mean,
        lo=_RECOMMENDATION_MEAN_MIN,
        hi=_RECOMMENDATION_MEAN_MAX,
        field_path="analyst_targets.recommendation_mean",
        rejections=rejections,
    )
    # ``number_of_analysts`` is an int field; sanitize via the float
    # primitives by widening — the cast back preserves the original int
    # type when the value survives.
    number_raw = targets.number_of_analysts
    if number_raw is not None and number_raw < 0:
        rejections.append(("analyst_targets.number_of_analysts", float(number_raw)))
        number = None
    else:
        number = number_raw
    return replace(
        targets,
        target_high=high,
        target_low=low,
        target_mean=mean,
        target_median=median,
        recommendation_mean=rec_mean,
        number_of_analysts=number,
    )


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _drop_if_negative(
    value: float | None, field_path: str, rejections: list[Rejection]
) -> float | None:
    """Return ``None`` (and record the rejection) if value is negative."""
    if value is None:
        return None
    if value < 0:
        rejections.append((field_path, value))
        return None
    return value


def _drop_if_unreasonable(
    value: float | None,
    cap: float,
    field_path: str,
    rejections: list[Rejection],
) -> float | None:
    """Drop if absolute value exceeds ``cap``. Negative-but-reasonable kept."""
    if value is None:
        return None
    if abs(value) > cap:
        rejections.append((field_path, value))
        return None
    return value


def _drop_if_unreasonable_pe(
    value: float | None, field_path: str, rejections: list[Rejection]
) -> float | None:
    """Drop P/E values that are negative (no earnings) or above the cap.

    Negative P/E technically describes "loss-making" companies but the
    valuation ensemble cannot use a negative multiple — treat as missing.
    """
    if value is None:
        return None
    if value < 0 or value > _MAX_REASONABLE_PE:
        rejections.append((field_path, value))
        return None
    return value


def _drop_if_unreasonable_payout(
    value: float | None, field_path: str, rejections: list[Rejection]
) -> float | None:
    if value is None:
        return None
    if value < 0 or value > _MAX_REASONABLE_PAYOUT:
        rejections.append((field_path, value))
        return None
    return value


def _drop_if_outside(
    value: float | None,
    *,
    lo: float,
    hi: float,
    field_path: str,
    rejections: list[Rejection],
) -> float | None:
    """Drop if ``value`` is outside the inclusive ``[lo, hi]`` window."""
    if value is None:
        return None
    if value < lo or value > hi:
        rejections.append((field_path, value))
        return None
    return value


__all__ = [
    "Rejection",
    "sanitize_normalized",
]
