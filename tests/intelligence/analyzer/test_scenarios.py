"""Unit tests for :func:`build_scenarios`.

The math is deterministic, so the assertions pin specific textbook
values. Where the spec says "lognormal with μ=0, σ=20%, T=30/252", the
test expects the precise probability that pops out of a clean
implementation — fixed-point invariants we can defend in code review.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src.intelligence.analyzer.scenarios import (
    ScenarioModel,
    build_scenarios,
)
from src.intelligence.analyzer.valuation import (
    ValuationEnsemble,
    ValuationEstimate,
)

_TRADING_DAYS_PER_YEAR = 252.0


def _ohlcv_with_constant_log_vol(
    *,
    n: int,
    daily_log_std: float,
    start_price: float = 100.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Synthetic OHLCV whose close-to-close log returns have a known std.

    Generates ``n`` returns with the exact std we want by drawing from
    a fixed-seed normal and then z-scoring + rescaling. The realized
    vol of the resulting series is *exactly* ``daily_log_std`` to
    numerical precision — so tests can pin the implied annualized vol.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, size=n)
    raw = (raw - raw.mean()) / raw.std(ddof=1)
    log_returns = raw * daily_log_std
    closes = start_price * np.exp(np.cumsum(log_returns))
    closes = np.concatenate([[start_price], closes])
    idx = pd.date_range(end=datetime(2026, 5, 21, tzinfo=UTC), periods=n + 1, freq="D")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": np.full(n + 1, 1_000_000, dtype=int),
        },
        index=idx,
    )


def _valuation(bear: float | None = 90.0, bull: float | None = 115.0) -> ValuationEnsemble:
    return ValuationEnsemble(
        estimates=(
            ValuationEstimate(
                method="dcf", fair_value=100.0, confidence=0.5, inputs_used="fixture"
            ),
        ),
        bear_case=bear,
        base_case=100.0,
        bull_case=bull,
        weighted_ai_fair_value=100.0,
        confidence_score=50.0,
    )


# -----------------------------------------------------------------------------
# Core math
# -----------------------------------------------------------------------------


def test_quantiles_match_lognormal_closed_form() -> None:
    """Pin q20/q50/q80 against the analytical closed form for σ=20% / T=30d."""
    sigma = 0.20  # annualized
    horizon_days = 30
    sigma_t = sigma * math.sqrt(horizon_days / _TRADING_DAYS_PER_YEAR)

    # Build OHLCV with that exact realized vol. n=30 returns →
    # 31 closes, which is exactly vol_lookback_days+1 — so the
    # scenarios slice covers the whole window and the synthetic
    # std lands on the target precisely.
    daily_log_std = sigma / math.sqrt(_TRADING_DAYS_PER_YEAR)
    df = _ohlcv_with_constant_log_vol(n=30, daily_log_std=daily_log_std)

    model = build_scenarios(
        current_price=100.0,
        ohlcv=df,
        horizon_days=horizon_days,
        vol_lookback_days=30,
    )
    assert model is not None
    nd = NormalDist(mu=0.0, sigma=1.0)
    expected_q20 = 100.0 * math.exp(sigma_t * nd.inv_cdf(0.20))
    expected_q50 = 100.0 * math.exp(sigma_t * nd.inv_cdf(0.50))
    expected_q80 = 100.0 * math.exp(sigma_t * nd.inv_cdf(0.80))
    assert model.quantile_20 == pytest.approx(expected_q20, abs=1e-6)
    assert model.quantile_50 == pytest.approx(expected_q50, abs=1e-6)
    assert model.quantile_80 == pytest.approx(expected_q80, abs=1e-6)
    # μ=0 → median equals current price.
    assert model.quantile_50 == pytest.approx(100.0, abs=1e-6)
    assert model.quantile_20 < model.quantile_50 < model.quantile_80


def test_prob_below_bear_matches_lognormal_cdf() -> None:
    """P(S_T < bear) = Φ(ln(bear/S₀) / (σ√T))."""
    sigma = 0.20
    horizon_days = 30
    sigma_t = sigma * math.sqrt(horizon_days / _TRADING_DAYS_PER_YEAR)
    daily_log_std = sigma / math.sqrt(_TRADING_DAYS_PER_YEAR)
    df = _ohlcv_with_constant_log_vol(n=30, daily_log_std=daily_log_std)

    model = build_scenarios(
        current_price=100.0,
        ohlcv=df,
        valuation=_valuation(bear=90.0, bull=115.0),
        horizon_days=horizon_days,
        vol_lookback_days=30,
    )
    assert model is not None
    nd = NormalDist(mu=0.0, sigma=1.0)
    expected_below_bear = nd.cdf(math.log(90.0 / 100.0) / sigma_t)
    expected_above_bull = 1.0 - nd.cdf(math.log(115.0 / 100.0) / sigma_t)
    assert model.prob_below_bear == pytest.approx(expected_below_bear, abs=1e-6)
    assert model.prob_above_bull == pytest.approx(expected_above_bull, abs=1e-6)


def test_prob_below_current_price_is_one_half_under_zero_drift() -> None:
    """μ=0 invariant: median equals S₀, so P(S_T < S₀) = 0.5 by definition.

    We don't surface this directly on ScenarioModel, but the math
    flows from quantile_50: it should match current_price within
    float tolerance.
    """
    daily_log_std = 0.25 / math.sqrt(_TRADING_DAYS_PER_YEAR)
    df = _ohlcv_with_constant_log_vol(n=31, daily_log_std=daily_log_std)
    model = build_scenarios(current_price=100.0, ohlcv=df)
    assert model is not None
    assert model.quantile_50 == pytest.approx(model.current_price, abs=1e-6)


# -----------------------------------------------------------------------------
# Degradation / None paths
# -----------------------------------------------------------------------------


def test_empty_ohlcv_returns_none() -> None:
    assert build_scenarios(current_price=100.0, ohlcv=pd.DataFrame()) is None


def test_short_ohlcv_returns_none() -> None:
    """Need at least vol_lookback_days + 1 rows."""
    df = _ohlcv_with_constant_log_vol(n=10, daily_log_std=0.01)
    # Default vol_lookback_days=30 → needs 31 rows; we have 11.
    assert build_scenarios(current_price=100.0, ohlcv=df) is None


def test_non_positive_price_returns_none() -> None:
    df = _ohlcv_with_constant_log_vol(n=40, daily_log_std=0.01)
    assert build_scenarios(current_price=0.0, ohlcv=df) is None
    assert build_scenarios(current_price=-10.0, ohlcv=df) is None
    assert build_scenarios(current_price=float("nan"), ohlcv=df) is None


def test_zero_vol_returns_none() -> None:
    """Flat prices → degenerate distribution; skip rather than emit."""
    closes = np.full(40, 100.0)
    idx = pd.date_range(end=datetime(2026, 5, 21, tzinfo=UTC), periods=40, freq="D")
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.full(40, 1_000_000, dtype=int),
        },
        index=idx,
    )
    assert build_scenarios(current_price=100.0, ohlcv=df) is None


def test_non_positive_horizon_returns_none() -> None:
    df = _ohlcv_with_constant_log_vol(n=40, daily_log_std=0.01)
    assert build_scenarios(current_price=100.0, ohlcv=df, horizon_days=0) is None
    assert build_scenarios(current_price=100.0, ohlcv=df, horizon_days=-5) is None


def test_lookback_below_two_returns_none() -> None:
    df = _ohlcv_with_constant_log_vol(n=40, daily_log_std=0.01)
    assert build_scenarios(current_price=100.0, ohlcv=df, vol_lookback_days=1) is None


def test_missing_valuation_means_tail_probabilities_are_none() -> None:
    """Quantiles still computed; only the tail probs go None."""
    df = _ohlcv_with_constant_log_vol(n=40, daily_log_std=0.015)
    model = build_scenarios(current_price=100.0, ohlcv=df, valuation=None)
    assert model is not None
    assert model.prob_below_bear is None
    assert model.prob_above_bull is None
    assert model.quantile_50 == pytest.approx(100.0)


def test_valuation_with_missing_bear_or_bull_handled() -> None:
    """A one-sided valuation (only bull, no bear) still produces the model."""
    df = _ohlcv_with_constant_log_vol(n=40, daily_log_std=0.015)
    val = _valuation(bear=None, bull=115.0)
    model = build_scenarios(current_price=100.0, ohlcv=df, valuation=val)
    assert model is not None
    assert model.prob_below_bear is None
    assert model.prob_above_bull is not None


# -----------------------------------------------------------------------------
# Annualization / convention
# -----------------------------------------------------------------------------


def test_annualization_uses_252_trading_days() -> None:
    """A daily log-return std of σ_d should annualize to σ_d * sqrt(252)."""
    daily_log_std = 0.01
    df = _ohlcv_with_constant_log_vol(n=30, daily_log_std=daily_log_std)
    model = build_scenarios(current_price=100.0, ohlcv=df, vol_lookback_days=30)
    assert model is not None
    expected = daily_log_std * math.sqrt(_TRADING_DAYS_PER_YEAR)
    assert model.annualized_vol == pytest.approx(expected, abs=1e-9)


def test_vol_lookback_window_uses_last_n_returns() -> None:
    """Drop-in: a noisy first half + a quiet recent half should yield
    a vol close to the recent half's vol when lookback covers only it."""
    n_old = 30
    n_recent = 30
    high = _ohlcv_with_constant_log_vol(n=n_old, daily_log_std=0.05, start_price=100.0)
    last_close = float(high["close"].iloc[-1])
    low = _ohlcv_with_constant_log_vol(
        n=n_recent, daily_log_std=0.005, start_price=last_close, seed=11
    )
    # Stitch them; drop the duplicated overlap close on the seam.
    df = pd.concat([high, low.iloc[1:]], axis=0)
    df.index = pd.date_range(end=datetime(2026, 5, 21, tzinfo=UTC), periods=len(df), freq="D")
    model = build_scenarios(current_price=100.0, ohlcv=df, vol_lookback_days=30)
    assert model is not None
    # The model should reflect the *recent* low-vol regime (~0.5% daily),
    # not the early high-vol period (~5% daily).
    expected = 0.005 * math.sqrt(_TRADING_DAYS_PER_YEAR)
    assert model.annualized_vol == pytest.approx(expected, rel=0.05)


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------


def test_notes_always_contain_the_caveat_block() -> None:
    """Model-card statements must be present on every output."""
    df = _ohlcv_with_constant_log_vol(n=40, daily_log_std=0.015)
    model = build_scenarios(current_price=100.0, ohlcv=df)
    assert model is not None
    assert any("zero drift" in n for n in model.notes)
    assert any("Lognormal" in n for n in model.notes)
    assert any("backward-looking" in n for n in model.notes)


def test_unusually_high_vol_emits_warning_note() -> None:
    """A vol past the sanity ceiling should flag a data-quality concern."""
    daily_log_std = 0.40  # ~635% annualized; clearly a data artifact
    df = _ohlcv_with_constant_log_vol(n=40, daily_log_std=daily_log_std)
    model = build_scenarios(current_price=100.0, ohlcv=df)
    assert model is not None
    assert any("split-adjusted" in n for n in model.notes)


def test_to_dict_round_trips_json_safe_fields() -> None:
    df = _ohlcv_with_constant_log_vol(n=40, daily_log_std=0.015)
    model = build_scenarios(
        current_price=100.0,
        ohlcv=df,
        valuation=_valuation(),
    )
    assert model is not None
    d = model.to_dict()
    # Every key the API serializer will surface.
    expected_keys = {
        "horizon_days",
        "current_price",
        "annualized_vol",
        "vol_lookback_days",
        "vol_method",
        "prob_below_bear",
        "prob_above_bull",
        "quantile_20",
        "quantile_50",
        "quantile_80",
        "notes",
    }
    assert set(d.keys()) == expected_keys
    assert isinstance(d["notes"], tuple)


def test_frozen_dataclass_is_immutable() -> None:
    df = _ohlcv_with_constant_log_vol(n=40, daily_log_std=0.015)
    model = build_scenarios(current_price=100.0, ohlcv=df)
    assert model is not None
    with pytest.raises(Exception):
        # FrozenInstanceError on a frozen dataclass; pinned broadly so
        # the test pins the *immutability invariant*, not the exact
        # exception type.
        model.current_price = 999.0  # type: ignore[misc]


def test_model_is_scenariomodel_type() -> None:
    df = _ohlcv_with_constant_log_vol(n=40, daily_log_std=0.015)
    model = build_scenarios(current_price=100.0, ohlcv=df)
    assert isinstance(model, ScenarioModel)
