"""Closed-form lognormal scenario model.

Layered on top of :class:`~src.intelligence.analyzer.valuation.ValuationEnsemble`
to give the trader a *probabilistic* read on near-term price paths.
The ensemble answers "what do my valuation methods think the right
price is?". This module answers a different question: "given today's
realized volatility and no directional drift, what's the realistic
30-day price range, and what's the probability of breaking through
the bear / bull boundaries?"

Why zero drift
--------------
Estimating expected return requires a forecast model. We do not have
one — and a single bad drift assumption would invalidate every
probability in this module. Setting μ=0 makes the output strictly a
function of *current price + current vol*. The framing is "this is
the dispersion the market is currently pricing in", not "this is
where I think the price is going".

Why closed form
---------------
Monte Carlo is unnecessary for the lognormal case; the cumulative
distribution and quantiles are analytical. Closed form is faster,
deterministic, and the test fixtures can pin known textbook values
to four decimal places. A future jump-diffusion or stochastic-vol
extension can fall back to Monte Carlo, but the lognormal baseline
is the right starting place: it has known limitations and a clear
model card.

Limitations (documented in :attr:`ScenarioModel.notes`)
------------------------------------------------------
* μ=0 means our distribution is symmetric in log-space. Real stocks
  have positive drift on average; we under-shoot upside probabilities
  and over-shoot downside probabilities by a small amount.
* Lognormal under-estimates fat tails. Realized return distributions
  have heavier tails than the model predicts; large moves are more
  likely than the model says.
* Realized vol is backward-looking. A regime shift on day T+1 makes
  the model wrong; the UI must show the vol lookback window so the
  trader can judge.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.intelligence.analyzer.valuation import ValuationEnsemble


# Annualization constant: trading days per year. Matches the analyzer
# layer's existing convention (technical indicators use the same when
# they annualize). Calendar-day annualization is a different model and
# should be its own function if we ever need it.
_TRADING_DAYS_PER_YEAR = 252.0

# Below this realized-vol threshold the lognormal collapses to a
# degenerate spike. Skip rather than emit pathologically tight bands.
_VOL_FLOOR = 1e-4

# Stock-data vol guard: yearly returns past this are almost certainly
# a data artifact (split-adjustment glitch, missing dividend, etc.).
# Past this we still return a model — flagged in notes — so the
# operator can see the issue rather than the data being silently dropped.
_VOL_SANITY_CEILING = 5.0  # 500% annualized

_STANDARD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


@dataclass(frozen=True)
class ScenarioModel:
    """Probabilistic price-range model for one symbol over one horizon.

    Attributes:
        horizon_days: How far ahead the model projects, in trading days.
        current_price: The starting S₀ used in the lognormal math.
        annualized_vol: Realized log-return σ × √252 over the lookback.
        vol_lookback_days: How many daily closes fed the σ estimate.
        vol_method: How we computed σ. Currently always
            ``"realized_log"`` (close-to-close log returns). Future
            extensions (Yang-Zhang, Parkinson, EWMA) would land their
            own method tag.
        prob_below_bear: ``P(S_T < bear_case)`` per the lognormal CDF.
            ``None`` when the valuation ensemble didn't produce a
            bear case.
        prob_above_bull: ``P(S_T > bull_case)``. ``None`` when no bull.
        quantile_20 / quantile_50 / quantile_80: The 20th / 50th /
            80th percentile of S_T. Bounds a 60%-confidence range.
            ``quantile_50`` is the lognormal median (≈ S₀ at μ=0).
        notes: Human-readable caveats the UI renders verbatim. Always
            non-empty — the model has limitations and we surface them.
    """

    horizon_days: int
    current_price: float
    annualized_vol: float
    vol_lookback_days: int
    vol_method: str
    prob_below_bear: float | None
    prob_above_bull: float | None
    quantile_20: float
    quantile_50: float
    quantile_80: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-safe dict for the API layer."""
        return asdict(self)


def build_scenarios(
    current_price: float,
    ohlcv: pd.DataFrame,
    *,
    valuation: ValuationEnsemble | None = None,
    horizon_days: int = 30,
    vol_lookback_days: int = 30,
) -> ScenarioModel | None:
    """Build a :class:`ScenarioModel` from price + OHLCV + optional valuation.

    Returns ``None`` (not an exception) when the inputs can't support
    the model:

    * ``current_price`` non-positive or non-finite.
    * ``ohlcv`` has fewer than ``vol_lookback_days + 1`` rows (we need
      that many to compute ``vol_lookback_days`` log returns).
    * Realized vol below :data:`_VOL_FLOOR` — the distribution would
      collapse to a spike.
    * ``horizon_days`` non-positive.

    Callers should render "scenarios unavailable" on ``None`` rather
    than treat it as an error. We deliberately don't raise here so the
    analyzer report degrades gracefully when bars are sparse.
    """
    if not _is_positive_finite(current_price):
        return None
    if horizon_days <= 0 or vol_lookback_days < 2:
        return None
    if ohlcv.empty or "close" not in ohlcv.columns:
        return None
    if len(ohlcv) < vol_lookback_days + 1:
        return None

    closes = ohlcv["close"].astype(float).dropna()
    if len(closes) < vol_lookback_days + 1:
        return None

    # Use the most recent ``vol_lookback_days + 1`` closes so the
    # window of log returns is exactly ``vol_lookback_days``.
    window = closes.iloc[-(vol_lookback_days + 1) :]
    annualized_vol = _annualized_log_vol(window)
    if annualized_vol is None or annualized_vol < _VOL_FLOOR:
        return None

    notes = list(_BASE_CAVEATS)
    if annualized_vol > _VOL_SANITY_CEILING:
        notes.append(
            f"Annualized volatility of {annualized_vol:.0%} is unusually high — "
            "verify the price history is split-adjusted."
        )

    horizon_years = horizon_days / _TRADING_DAYS_PER_YEAR
    sigma_t = annualized_vol * math.sqrt(horizon_years)

    quantile_20 = _quantile(current_price, sigma_t, 0.20)
    quantile_50 = _quantile(current_price, sigma_t, 0.50)
    quantile_80 = _quantile(current_price, sigma_t, 0.80)

    bear = valuation.bear_case if valuation is not None else None
    bull = valuation.bull_case if valuation is not None else None
    prob_below_bear = _prob_below(current_price, sigma_t, bear)
    prob_above_bull = _prob_above(current_price, sigma_t, bull)

    return ScenarioModel(
        horizon_days=horizon_days,
        current_price=float(current_price),
        annualized_vol=annualized_vol,
        vol_lookback_days=vol_lookback_days,
        vol_method="realized_log",
        prob_below_bear=prob_below_bear,
        prob_above_bull=prob_above_bull,
        quantile_20=quantile_20,
        quantile_50=quantile_50,
        quantile_80=quantile_80,
        notes=tuple(notes),
    )


# ---- helpers ----


_BASE_CAVEATS: tuple[str, ...] = (
    "Probabilities assume zero drift — they describe today's dispersion, "
    "not a forecast.",
    "Lognormal model under-estimates the likelihood of large moves; treat "
    "tail probabilities as conservative lower bounds.",
    "Realized volatility is backward-looking. A regime shift after the "
    "lookback window invalidates the model.",
)


def _annualized_log_vol(closes: pd.Series) -> float | None:
    """Annualize the std of close-to-close log returns.

    ``closes`` is a series of daily prices. Returns ``None`` when
    fewer than two returns can be computed (degenerate window).
    """
    if len(closes) < 2:
        return None
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if len(log_returns) < 2:
        return None
    daily_std = float(log_returns.std(ddof=1))
    if not math.isfinite(daily_std) or daily_std <= 0:
        return None
    return daily_std * math.sqrt(_TRADING_DAYS_PER_YEAR)


def _quantile(s0: float, sigma_t: float, q: float) -> float:
    """The q-quantile of S_T under lognormal(μ=0, σ²T).

    ``S_T(q) = S₀ · exp(σ·√T · Φ⁻¹(q))``. q in (0, 1).
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q!r}")
    z = _STANDARD_NORMAL.inv_cdf(q)
    return float(s0 * math.exp(sigma_t * z))


def _prob_below(s0: float, sigma_t: float, threshold: float | None) -> float | None:
    """``P(S_T < threshold)`` under lognormal(μ=0, σ²T).

    Returns ``None`` when ``threshold`` is missing or non-positive.
    """
    if threshold is None or not _is_positive_finite(threshold):
        return None
    z = math.log(threshold / s0) / sigma_t
    return float(_STANDARD_NORMAL.cdf(z))


def _prob_above(s0: float, sigma_t: float, threshold: float | None) -> float | None:
    """``P(S_T > threshold)`` — complement of :func:`_prob_below`."""
    below = _prob_below(s0, sigma_t, threshold)
    if below is None:
        return None
    return 1.0 - below


def _is_positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0


__all__ = ["ScenarioModel", "build_scenarios"]
