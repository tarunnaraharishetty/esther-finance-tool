"""Multi-method valuation ensemble.

Replaces the legacy "DCF + sentiment fudge" with a portfolio of
independent valuation methods, each producing a per-share fair-value
estimate and a confidence in ``[0, 1]``. The ensemble layer combines
them into bear / base / bull cases and a weighted "AI fair value".

Why multiple methods
--------------------
No single valuation method is right for every company:

* DCF is theoretically pure but pathologically sensitive to growth
  + discount-rate assumptions, and useless for unprofitable names.
* P/E works on profitable, steady businesses; meaningless when EPS
  is negative or volatile.
* EV/EBITDA bypasses capital-structure noise but requires positive
  EBITDA.
* P/S applies everywhere but says nothing about profitability.
* PEG bakes in growth expectations.
* Historical bands capture mean-reversion behavior.
* Analyst targets summarize what the street thinks (a Bayesian prior,
  not ground truth).

Combining them with explicit confidences forces the ensemble to
down-weight methods that lack the inputs they need (a net-income-
negative biotech contributes nothing via P/E, but DCF, P/S, and
analyst targets still fire).

Output shape
------------
:class:`ValuationEnsemble` carries:

* ``estimates`` — one :class:`ValuationEstimate` per *applicable*
  method, with the per-method confidence + a string of "inputs used"
  so the AI explanation layer can cite numerics.
* ``bear_case`` / ``base_case`` / ``bull_case`` — 25th / 50th / 75th
  percentile of populated estimates. Percentile (not min / max)
  prevents one wildly outlying model from defining the range.
* ``weighted_ai_fair_value`` — confidence-weighted mean of populated
  estimates.
* ``confidence_score`` — overall confidence, scaled to ``[0, 100]``
  to compose with the technical-scoring layer.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from src.config import get_settings
from src.intelligence.analyzer import sector_medians
from src.intelligence.fundamentals import NormalizedFundamentals


@dataclass(frozen=True)
class ValuationEstimate:
    """One per-share fair-value estimate from one method.

    The ``inputs_used`` string is a human-readable summary of the
    numeric inputs that produced this estimate — used verbatim by the
    AI explanation layer's citation block.
    """

    method: str
    fair_value: float
    confidence: float  # [0, 1]
    inputs_used: str


@dataclass(frozen=True)
class ValuationEnsemble:
    """Aggregated multi-method valuation output."""

    estimates: tuple[ValuationEstimate, ...]
    bear_case: float | None
    base_case: float | None
    bull_case: float | None
    weighted_ai_fair_value: float | None
    confidence_score: float  # [0, 100]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---- Per-method estimators ----
#
# Each estimator returns ``ValuationEstimate | None``. None means the
# method's required inputs weren't present — the ensemble skips it
# silently rather than zero-weighting and biasing the average.


def estimate_dcf(
    fundamentals: NormalizedFundamentals,
    *,
    discount_rate: float | None = None,
    terminal_growth: float | None = None,
    projection_years: int | None = None,
) -> ValuationEstimate | None:
    """Simple two-stage DCF.

    Stage 1 grows the latest free cash flow at the trailing FCF CAGR
    (clamped to a sensible range) for ``projection_years``. Stage 2 is
    a Gordon-growth terminal at ``terminal_growth``. Discounted at
    ``discount_rate``. Equity value = enterprise value (we don't
    subtract net debt here — the multiple-based EV/EBITDA model
    handles that explicitly). Divided by shares for per-share fair
    value.

    Confidence is bounded by how reliable the inputs look — negative
    or zero FCF, missing shares, or unrealistic CAGR pull confidence
    toward zero rather than forcing the caller to special-case.
    """
    settings = get_settings()
    discount_rate = discount_rate or settings.dcf_default_discount_rate
    terminal_growth = (
        terminal_growth if terminal_growth is not None else settings.dcf_default_terminal_growth
    )
    projection_years = projection_years or settings.dcf_projection_years
    if discount_rate <= terminal_growth:
        # Gordon growth blows up; refuse to produce a fantasy number.
        return None

    cashflows = fundamentals.cash_flows
    if not cashflows:
        return None
    latest = max(cashflows, key=lambda c: c.fiscal_date)
    fcf = latest.free_cash_flow
    if fcf is None or fcf <= 0:
        return None

    shares = _shares_outstanding(fundamentals)
    if shares is None or shares <= 0:
        return None

    cagr = _fcf_cagr(fundamentals)
    if cagr is None:
        # Sensible default growth when we can't compute trailing CAGR:
        # fall back to terminal growth.
        cagr = terminal_growth
    # Clamp to ±25% to keep DCF from going parabolic on noisy data.
    growth = max(-0.25, min(0.25, cagr))

    projected = 0.0
    fcf_t = fcf
    for year in range(1, projection_years + 1):
        fcf_t = fcf_t * (1.0 + growth)
        projected += fcf_t / ((1.0 + discount_rate) ** year)

    terminal_fcf = fcf_t * (1.0 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    terminal_discounted = terminal_value / ((1.0 + discount_rate) ** projection_years)

    enterprise_value = projected + terminal_discounted
    fair_value = enterprise_value / shares

    # Confidence: high FCF + multi-year history → high; single data
    # point and clamped growth → lower.
    cf_count = len(cashflows)
    coverage = min(1.0, cf_count / 5.0)
    growth_stability = 1.0 - abs(growth - cagr) / 0.25 if abs(cagr) > 0.001 else 1.0
    confidence = max(0.1, min(0.9, 0.5 * coverage + 0.5 * growth_stability))

    return ValuationEstimate(
        method="dcf",
        fair_value=float(fair_value),
        confidence=confidence,
        inputs_used=(
            f"FCF={fcf:.0f}, growth={growth:.1%}, "
            f"discount={discount_rate:.1%}, terminal={terminal_growth:.1%}, "
            f"horizon={projection_years}y, shares={shares:.0f}"
        ),
    )


def estimate_pe_multiple(
    fundamentals: NormalizedFundamentals,
) -> ValuationEstimate | None:
    """Sector-median P/E × diluted EPS.

    Requires a positive diluted EPS — applies to profitable companies
    only. Loss-making names contribute nothing via this method
    (intentional: P/E is meaningless when E is negative).
    """
    eps = _diluted_eps(fundamentals)
    if eps is None or eps <= 0:
        return None
    medians = sector_medians.lookup(fundamentals.profile.sector)
    fair_value = medians.pe * eps
    # Confidence drops when company multiple deviates wildly from
    # sector — extreme outliers usually mean the market sees something
    # the median doesn't.
    company_pe = fundamentals.key_ratios.pe_ratio
    confidence = _multiple_confidence(company_pe, medians.pe)
    return ValuationEstimate(
        method="pe_multiple",
        fair_value=float(fair_value),
        confidence=confidence,
        inputs_used=(
            f"EPS={eps:.2f}, sector_pe={medians.pe:.1f}, "
            f"company_pe={_fmt_multiple(company_pe)}, "
            f"sector={fundamentals.profile.sector or 'unknown'}"
        ),
    )


def estimate_ev_ebitda(
    fundamentals: NormalizedFundamentals,
) -> ValuationEstimate | None:
    """Sector-median EV/EBITDA mapped back to per-share equity value.

    EV = EBITDA × sector_median.
    Equity = EV − net_debt = EV − (total_debt − cash).
    Price = Equity / shares.

    Returns ``None`` when EBITDA is non-positive (the multiple is
    nonsensical) or when balance-sheet data is missing.
    """
    income = fundamentals.latest_annual_income
    balance = fundamentals.latest_annual_balance
    if income is None or balance is None:
        return None
    ebitda = income.ebitda
    if ebitda is None or ebitda <= 0:
        return None
    shares = _shares_outstanding(fundamentals)
    if shares is None or shares <= 0:
        return None

    medians = sector_medians.lookup(fundamentals.profile.sector)
    enterprise_value = ebitda * medians.ev_ebitda
    cash = balance.cash_and_equivalents or 0.0
    debt = balance.total_debt or balance.long_term_debt or 0.0
    equity_value = enterprise_value - debt + cash
    fair_value = equity_value / shares

    company_multiple = fundamentals.key_ratios.ev_to_ebitda
    confidence = _multiple_confidence(company_multiple, medians.ev_ebitda)
    return ValuationEstimate(
        method="ev_ebitda_multiple",
        fair_value=float(fair_value),
        confidence=confidence,
        inputs_used=(
            f"EBITDA={ebitda:.0f}, EV/EBITDA={medians.ev_ebitda:.1f}, "
            f"debt={debt:.0f}, cash={cash:.0f}, shares={shares:.0f}"
        ),
    )


def estimate_ps_multiple(
    fundamentals: NormalizedFundamentals,
) -> ValuationEstimate | None:
    """Sector-median P/S × revenue per share."""
    income = fundamentals.latest_annual_income
    if income is None:
        return None
    revenue = income.revenue
    if revenue is None or revenue <= 0:
        return None
    shares = _shares_outstanding(fundamentals)
    if shares is None or shares <= 0:
        return None

    medians = sector_medians.lookup(fundamentals.profile.sector)
    revenue_per_share = revenue / shares
    fair_value = medians.ps * revenue_per_share

    company_multiple = fundamentals.key_ratios.price_to_sales
    confidence = _multiple_confidence(company_multiple, medians.ps)
    return ValuationEstimate(
        method="ps_multiple",
        fair_value=float(fair_value),
        confidence=confidence,
        inputs_used=(
            f"revenue={revenue:.0f}, sector_ps={medians.ps:.1f}, "
            f"revenue_per_share={revenue_per_share:.2f}, "
            f"company_ps={_fmt_multiple(company_multiple)}"
        ),
    )


def estimate_peg(
    fundamentals: NormalizedFundamentals,
) -> ValuationEstimate | None:
    """Fair P/E = growth_pct (PEG = 1) × EPS.

    Uses trailing EPS CAGR. The naive Lynch rule: a stock growing
    earnings at G% per year is "fairly valued" at P/E = G. We compute
    G from the historical income statements rather than relying on
    forward estimates we don't have.
    """
    eps = _diluted_eps(fundamentals)
    if eps is None or eps <= 0:
        return None
    growth = _eps_cagr(fundamentals)
    if growth is None or growth <= 0:
        return None
    # Clamp G to [5%, 40%] — under 5% PEG is unstable, over 40% the
    # rule of thumb breaks down.
    growth_pct = max(5.0, min(40.0, growth * 100.0))
    fair_pe = growth_pct
    fair_value = fair_pe * eps
    # PEG already encodes growth assumptions — confidence reflects
    # the EPS-history coverage.
    cf_count = len(fundamentals.income_statements)
    confidence = max(0.2, min(0.7, cf_count / 5.0))
    return ValuationEstimate(
        method="peg",
        fair_value=float(fair_value),
        confidence=confidence,
        inputs_used=(
            f"EPS={eps:.2f}, EPS_CAGR={growth_pct:.1f}%, fair_pe={fair_pe:.1f}"
        ),
    )


def estimate_historical_band(
    fundamentals: NormalizedFundamentals,
) -> ValuationEstimate | None:
    """Trend-extrapolated EPS × sector P/E.

    Computes EPS CAGR from the historical income statements, projects
    the latest EPS forward one year, applies the sector-median P/E.
    Returns ``None`` when we have fewer than two annual income
    statements (need ≥2 to compute a CAGR) or when EPS history is
    non-positive.
    """
    if len(fundamentals.income_statements) < 2:
        return None
    eps = _diluted_eps(fundamentals)
    if eps is None or eps <= 0:
        return None
    growth = _eps_cagr(fundamentals)
    if growth is None:
        return None
    # Clamp projected growth so a single anomalous year doesn't blow
    # the projection into the stratosphere.
    growth = max(-0.10, min(0.30, growth))
    projected_eps = eps * (1.0 + growth)
    medians = sector_medians.lookup(fundamentals.profile.sector)
    fair_value = medians.pe * projected_eps
    # Confidence climbs with the number of statements we used.
    n = len(fundamentals.income_statements)
    confidence = max(0.2, min(0.7, n / 5.0))
    return ValuationEstimate(
        method="historical_band",
        fair_value=float(fair_value),
        confidence=confidence,
        inputs_used=(
            f"EPS={eps:.2f}, EPS_CAGR={growth:.1%}, "
            f"projected_EPS={projected_eps:.2f}, sector_pe={medians.pe:.1f}, "
            f"history={n} statements"
        ),
    )


def estimate_analyst_target(
    fundamentals: NormalizedFundamentals,
) -> ValuationEstimate | None:
    """Direct passthrough of analyst consensus target.

    Confidence scales with the number of analysts contributing —
    a 1-analyst consensus is much weaker than a 30-analyst one.
    """
    targets = fundamentals.analyst_targets
    if targets is None:
        return None
    consensus = targets.target_mean or targets.target_median
    if consensus is None or consensus <= 0:
        return None
    n_analysts = targets.number_of_analysts or 0
    # Confidence: 1 analyst → 0.3, 10 analysts → 0.8, saturate at 20.
    confidence = max(0.2, min(0.9, 0.3 + (n_analysts / 20.0) * 0.6))
    spread_note = ""
    if targets.target_high and targets.target_low:
        spread_note = f", range=[{targets.target_low:.2f}, {targets.target_high:.2f}]"
    return ValuationEstimate(
        method="analyst_target",
        fair_value=float(consensus),
        confidence=confidence,
        inputs_used=(
            f"consensus={consensus:.2f}, analysts={n_analysts}{spread_note}"
        ),
    )


# ---- Ensemble ----


def build_valuation(
    fundamentals: NormalizedFundamentals,
) -> ValuationEnsemble:
    """Run every model + combine into bear/base/bull/weighted output."""
    estimators = (
        estimate_dcf,
        estimate_pe_multiple,
        estimate_ev_ebitda,
        estimate_ps_multiple,
        estimate_peg,
        estimate_historical_band,
        estimate_analyst_target,
    )
    estimates: list[ValuationEstimate] = []
    for fn in estimators:
        est = fn(fundamentals)
        if est is not None and math.isfinite(est.fair_value):
            estimates.append(est)

    if not estimates:
        return ValuationEnsemble(
            estimates=(),
            bear_case=None,
            base_case=None,
            bull_case=None,
            weighted_ai_fair_value=None,
            confidence_score=0.0,
        )

    values = np.array([e.fair_value for e in estimates], dtype=float)
    weights = np.array([e.confidence for e in estimates], dtype=float)
    bear = float(np.percentile(values, 25))
    base = float(np.percentile(values, 50))
    bull = float(np.percentile(values, 75))
    if weights.sum() > 0:
        weighted = float(np.average(values, weights=weights))
    else:
        weighted = base

    # Overall confidence: coverage (how many models fired / 7) × average
    # per-model confidence × an agreement factor (1 - normalized stdev).
    coverage = len(estimates) / 7.0
    avg_confidence = float(weights.mean())
    if len(values) > 1 and base != 0:
        dispersion = float(np.std(values) / max(abs(base), 1.0))
    else:
        dispersion = 0.0
    agreement = max(0.0, 1.0 - dispersion)
    confidence = max(0.0, min(100.0, coverage * avg_confidence * agreement * 100.0))

    return ValuationEnsemble(
        estimates=tuple(estimates),
        bear_case=bear,
        base_case=base,
        bull_case=bull,
        weighted_ai_fair_value=weighted,
        confidence_score=confidence,
    )


# ---- helpers ----


def _diluted_eps(fundamentals: NormalizedFundamentals) -> float | None:
    income = fundamentals.latest_annual_income
    if income is None:
        return None
    if income.eps_diluted is not None:
        return float(income.eps_diluted)
    if income.eps_basic is not None:
        return float(income.eps_basic)
    # Synthesize from net income / diluted shares when both are present.
    if income.net_income is not None and income.shares_diluted:
        try:
            return float(income.net_income) / float(income.shares_diluted)
        except ZeroDivisionError:
            return None
    return None


def _shares_outstanding(fundamentals: NormalizedFundamentals) -> float | None:
    """Pick the best available shares-outstanding count.

    Order: latest annual balance sheet → company profile → latest
    diluted-shares figure from the income statement.
    """
    balance = fundamentals.latest_annual_balance
    if balance and balance.shares_outstanding:
        return float(balance.shares_outstanding)
    profile_shares = fundamentals.profile.shares_outstanding
    if profile_shares:
        return float(profile_shares)
    income = fundamentals.latest_annual_income
    if income and income.shares_diluted:
        return float(income.shares_diluted)
    return None


def _fcf_cagr(fundamentals: NormalizedFundamentals) -> float | None:
    cashflows = sorted(
        [c for c in fundamentals.cash_flows if c.free_cash_flow and c.free_cash_flow > 0],
        key=lambda c: c.fiscal_date,
    )
    return _cagr([c.free_cash_flow for c in cashflows])  # type: ignore[arg-type]


def _eps_cagr(fundamentals: NormalizedFundamentals) -> float | None:
    statements = sorted(
        [
            s
            for s in fundamentals.income_statements
            if (s.eps_diluted or s.eps_basic) is not None
        ],
        key=lambda s: s.fiscal_date,
    )
    eps_series: list[float] = []
    for s in statements:
        value = s.eps_diluted if s.eps_diluted is not None else s.eps_basic
        if value is None:
            continue
        if value <= 0:
            # Negative EPS makes CAGR undefined. Stop accumulating once
            # we hit one to avoid a sign-flipped CAGR.
            return None
        eps_series.append(float(value))
    return _cagr(eps_series)


def _cagr(series: list[float]) -> float | None:
    """Compound growth rate from the first to the last positive value.

    Returns ``None`` when the series has fewer than two positive entries
    (we need two anchor points to compute growth).
    """
    if len(series) < 2:
        return None
    first, last = series[0], series[-1]
    if first <= 0 or last <= 0:
        return None
    years = len(series) - 1
    return (last / first) ** (1.0 / years) - 1.0


def _multiple_confidence(
    company_multiple: float | None, sector_median: float
) -> float:
    """Confidence shrinks as the company multiple diverges from sector.

    Returns 0.7 when they line up (the multiple is broadly applicable
    here), down to 0.2 when the company multiple is 3x or 1/3 the
    sector. Pure passthrough when we have no company multiple — the
    sector median is still a defensible benchmark.
    """
    if company_multiple is None or company_multiple <= 0 or sector_median <= 0:
        return 0.5
    ratio = company_multiple / sector_median
    deviation = abs(math.log(ratio))  # ~0 when ratio=1, 1.1 when ratio=3
    confidence = 0.7 * math.exp(-deviation)
    return max(0.2, min(0.7, confidence))


def _fmt_multiple(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


__all__ = [
    "ValuationEnsemble",
    "ValuationEstimate",
    "build_valuation",
    "estimate_analyst_target",
    "estimate_dcf",
    "estimate_ev_ebitda",
    "estimate_historical_band",
    "estimate_pe_multiple",
    "estimate_peg",
    "estimate_ps_multiple",
]
