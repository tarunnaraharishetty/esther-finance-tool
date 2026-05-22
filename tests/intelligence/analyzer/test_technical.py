"""Normalized technical scoring tests.

Two layers:

1. Direct tests of each private helper against a hand-picked numeric
   input — locks down the [0, 100] mapping behavior independent of
   the indicators that feed them.
2. End-to-end ``score_technicals`` tests against deterministic OHLCV
   fixtures shaped to simulate overbought / oversold / neutral
   regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.intelligence.analyzer.technical import (
    TechnicalSubscores,
    _atr_volatility_score,
    _bollinger_extension_score,
    _confidence,
    _ma_extension_score,
    _momentum_exhaustion_score,
    _rsi_overbought_score,
    _rsi_oversold_score,
    _volume_spike_score,
    score_technicals,
)

# ---------- sub-score unit tests ----------


def test_rsi_overbought_score_endpoints() -> None:
    assert _rsi_overbought_score(70) == 0.0
    assert _rsi_overbought_score(85) == pytest.approx(50.0)
    assert _rsi_overbought_score(100) == 100.0
    assert _rsi_overbought_score(50) == 0.0
    assert _rsi_overbought_score(None) is None


def test_rsi_oversold_score_endpoints() -> None:
    assert _rsi_oversold_score(30) == 0.0
    assert _rsi_oversold_score(15) == pytest.approx(50.0)
    assert _rsi_oversold_score(0) == 100.0
    assert _rsi_oversold_score(60) == 0.0
    assert _rsi_oversold_score(None) is None


def test_bollinger_extension_score_inside_bands_is_zero() -> None:
    assert _bollinger_extension_score(0.5) == 0.0
    assert _bollinger_extension_score(-1.0) == 0.0
    assert _bollinger_extension_score(None) is None


def test_bollinger_extension_score_saturates_at_three_sigma() -> None:
    # Just past ±1 sigma → small score.
    assert _bollinger_extension_score(1.5) == pytest.approx(25.0)
    # At ±3 sigma → saturates at 100.
    assert _bollinger_extension_score(3.0) == 100.0
    assert _bollinger_extension_score(-3.0) == 100.0
    # Beyond saturation still clips, not blows up.
    assert _bollinger_extension_score(5.0) == 100.0


def test_ma_extension_score_uses_absolute_deviation() -> None:
    # 0% → 0; 30% → 100; 60% saturates.
    assert _ma_extension_score(0.0) == 0.0
    assert _ma_extension_score(15.0) == pytest.approx(50.0)
    assert _ma_extension_score(-15.0) == pytest.approx(50.0)
    assert _ma_extension_score(30.0) == 100.0
    assert _ma_extension_score(50.0) == 100.0


def test_volume_spike_score_ignores_negative_z() -> None:
    assert _volume_spike_score(-1.0) == 0.0
    assert _volume_spike_score(0.0) == 0.0
    assert _volume_spike_score(1.5) == pytest.approx(50.0)
    assert _volume_spike_score(3.0) == 100.0
    assert _volume_spike_score(None) is None


def test_atr_volatility_score_clips_at_baseline() -> None:
    assert _atr_volatility_score(0.5) == 0.0
    assert _atr_volatility_score(1.0) == 0.0
    assert _atr_volatility_score(1.5) == pytest.approx(50.0)
    assert _atr_volatility_score(2.0) == 100.0
    assert _atr_volatility_score(3.0) == 100.0


def test_momentum_exhaustion_needs_extreme_peak() -> None:
    # Recent peak is only RSI = 60 — no exhaustion signal regardless
    # of how much it's pulled back.
    series = pd.Series([60.0, 58.0, 55.0, 50.0, 45.0])
    assert _momentum_exhaustion_score(series) == 0.0

    # Peak in window was 90, current is 65 → 25-point pullback from an
    # extreme peak → saturated exhaustion.
    series = pd.Series([60, 70, 90, 88, 75, 65], dtype=float)
    score = _momentum_exhaustion_score(series)
    assert score == 100.0


def test_momentum_exhaustion_mirrors_for_oversold() -> None:
    # Recent trough at RSI = 15, RSI now back to 45 → 30 points up
    # from an extreme trough → saturated rebound exhaustion.
    series = pd.Series([60, 50, 30, 15, 25, 40, 45], dtype=float)
    assert _momentum_exhaustion_score(series) == 100.0


def test_momentum_exhaustion_returns_none_on_short_series() -> None:
    series = pd.Series([50.0, 51.0, 52.0])
    assert _momentum_exhaustion_score(series) is None


# ---------- confidence tests ----------


def test_confidence_drops_when_most_fields_are_none() -> None:
    subs = TechnicalSubscores(
        rsi_overbought_score=50.0,
        rsi_oversold_score=None,
        bollinger_extension_score=None,
        ma_extension_score=None,
        volume_spike_score=None,
        atr_volatility_score=None,
        momentum_exhaustion_score=None,
    )
    # Coverage = 1/7, dispersion = 0 → confidence ~14.3.
    assert _confidence(subs) == pytest.approx(100.0 / 7, abs=0.5)


def test_confidence_drops_with_dispersion() -> None:
    """Full coverage but wildly disagreeing sub-scores → low confidence."""
    consistent = TechnicalSubscores(
        rsi_overbought_score=70.0,
        rsi_oversold_score=70.0,
        bollinger_extension_score=70.0,
        ma_extension_score=70.0,
        volume_spike_score=70.0,
        atr_volatility_score=70.0,
        momentum_exhaustion_score=70.0,
    )
    mixed = TechnicalSubscores(
        rsi_overbought_score=100.0,
        rsi_oversold_score=0.0,
        bollinger_extension_score=100.0,
        ma_extension_score=0.0,
        volume_spike_score=100.0,
        atr_volatility_score=0.0,
        momentum_exhaustion_score=50.0,
    )
    assert _confidence(consistent) > _confidence(mixed)
    assert _confidence(consistent) == 100.0


def test_confidence_zero_when_no_subscores_populated() -> None:
    subs = TechnicalSubscores(
        rsi_overbought_score=None,
        rsi_oversold_score=None,
        bollinger_extension_score=None,
        ma_extension_score=None,
        volume_spike_score=None,
        atr_volatility_score=None,
        momentum_exhaustion_score=None,
    )
    assert _confidence(subs) == 0.0


# ---------- end-to-end fixtures + tests ----------


def _ramp_up_ohlcv(n: int = 80, start: float = 100.0, daily_pct: float = 0.015) -> pd.DataFrame:
    """Strong uptrend with deterministic daily-return noise.

    Pure-up ramps trigger an RSI edge case (zero average loss → RSI
    undefined). We build daily returns as ``daily_pct + N(0, 0.015)``
    so a handful of bars print negative returns, keeping the RSI
    definition intact while leaving the *aggregate* trend overwhelmingly
    positive.
    """
    rng = np.random.default_rng(seed=11)
    daily_returns = rng.normal(loc=daily_pct, scale=0.015, size=n)
    close = start * np.cumprod(1.0 + daily_returns)
    high = close * 1.005
    low = close * 0.995
    open_ = np.concatenate([[start], close[:-1]])
    volume = np.full(n, 1_000_000, dtype=int)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _ramp_down_ohlcv(n: int = 80, start: float = 200.0, daily_pct: float = 0.015) -> pd.DataFrame:
    rng = np.random.default_rng(seed=13)
    daily_returns = rng.normal(loc=-daily_pct, scale=0.015, size=n)
    close = start * np.cumprod(1.0 + daily_returns)
    high = close * 1.005
    low = close * 0.995
    open_ = np.concatenate([[start], close[:-1]])
    volume = np.full(n, 1_000_000, dtype=int)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _flat_ohlcv(n: int = 80, price: float = 100.0) -> pd.DataFrame:
    close = np.full(n, price, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=int),
        }
    )


def test_score_technicals_empty_frame_returns_zero_confidence() -> None:
    result = score_technicals(pd.DataFrame())
    assert result.confidence_score == 0.0
    assert result.overbought_score is None
    assert result.oversold_score is None
    assert result.raw_rsi is None


def test_score_technicals_on_ramp_up_is_overbought() -> None:
    df = _ramp_up_ohlcv()
    scores = score_technicals(df)
    subs = scores.subscores

    # Direct overbought signals all light up.
    assert subs.rsi_overbought_score is not None and subs.rsi_overbought_score >= 60
    assert subs.ma_extension_score is not None and subs.ma_extension_score >= 60
    assert subs.bollinger_extension_score is not None
    # Oversold signal stays at 0 — the move is upward.
    assert subs.rsi_oversold_score == 0.0

    assert scores.overbought_score is not None
    assert scores.oversold_score is not None
    assert scores.overbought_score > scores.oversold_score
    # Raw RSI on a 2.5%/day ramp pins near 100.
    assert scores.raw_rsi is not None and scores.raw_rsi >= 80
    assert scores.raw_ma_distance_pct is not None and scores.raw_ma_distance_pct > 0
    assert 0.0 <= scores.confidence_score <= 100.0


def test_score_technicals_on_ramp_down_is_oversold() -> None:
    df = _ramp_down_ohlcv()
    scores = score_technicals(df)
    subs = scores.subscores

    assert subs.rsi_oversold_score is not None and subs.rsi_oversold_score >= 60
    assert subs.ma_extension_score is not None and subs.ma_extension_score >= 60
    # RSI overbought sub-score is mirror-flat.
    assert subs.rsi_overbought_score == 0.0

    assert scores.oversold_score is not None and scores.overbought_score is not None
    assert scores.oversold_score > scores.overbought_score
    assert scores.raw_rsi is not None and scores.raw_rsi <= 20
    assert scores.raw_ma_distance_pct is not None and scores.raw_ma_distance_pct < 0


def test_score_technicals_on_flat_series_is_neutral() -> None:
    df = _flat_ohlcv()
    scores = score_technicals(df)
    subs = scores.subscores
    # On a strictly flat series RSI is mathematically undefined
    # (0 gain AND 0 loss). The analyzer surfaces that as ``None``
    # rather than fabricating a midpoint, which would silently bias
    # the combiners.
    assert subs.rsi_overbought_score is None
    assert subs.rsi_oversold_score is None
    # MA distance is 0 → ma_extension_score is 0.
    assert subs.ma_extension_score == 0.0
    # No volume change → volume spike = 0 (or None when std=0).
    assert (subs.volume_spike_score or 0.0) == 0.0
    # Neither directional combiner should fire on a flat tape.
    assert (scores.overbought_score or 0.0) < 10.0
    assert (scores.oversold_score or 0.0) < 10.0


def test_score_technicals_volume_spike_lights_up_on_anomaly() -> None:
    df = _flat_ohlcv(n=60)
    # Bake a real volume spike onto a slowly trending close so the
    # rolling-std denominator is non-zero.
    rng = np.random.default_rng(seed=12)
    df = df.copy()
    df["volume"] = rng.integers(900_000, 1_100_000, size=len(df))
    df.loc[df.index[-1], "volume"] = 20_000_000
    scores = score_technicals(df)
    assert scores.subscores.volume_spike_score is not None
    assert scores.subscores.volume_spike_score >= 60.0
    assert scores.raw_volume_z is not None and scores.raw_volume_z > 2.0


def test_score_technicals_short_history_has_partial_coverage() -> None:
    # Only 25 bars — the 50-bar SMA + ATR baseline can't be computed.
    df = _ramp_up_ohlcv(n=25)
    scores = score_technicals(df)
    # MA extension needs 50-bar SMA → drops to None.
    assert scores.subscores.ma_extension_score is None
    # ATR baseline (50-bar) also unavailable.
    assert scores.subscores.atr_volatility_score is None
    # RSI / Bollinger are fine on 25 bars.
    assert scores.subscores.rsi_overbought_score is not None
    # Coverage gap is reflected in the raw fields — what we can't
    # compute is honestly reported as None rather than imputed.
    assert scores.raw_ma_distance_pct is None
    assert scores.raw_atr_ratio is None


def test_pullback_risk_tracks_overbought_plus_exhaustion() -> None:
    """Pullback risk is the average of overbought + momentum exhaustion."""
    df = _ramp_up_ohlcv()
    # Force a recent pullback in close to trigger momentum exhaustion.
    df = df.copy()
    last_idx = df.index[-1]
    df.loc[last_idx, "close"] = float(df["close"].iloc[-2]) * 0.95
    df.loc[last_idx, "high"] = df.loc[last_idx, "close"] * 1.005
    df.loc[last_idx, "low"] = df.loc[last_idx, "close"] * 0.995
    scores = score_technicals(df)
    assert scores.pullback_risk is not None
    assert scores.overbought_score is not None
    # Pullback risk averages overbought + exhaustion → should sit between
    # the two when exhaustion is materially lower than overbought.
    if scores.subscores.momentum_exhaustion_score is not None:
        ob = scores.overbought_score
        exh = scores.subscores.momentum_exhaustion_score
        expected = (ob + exh) / 2.0
        assert scores.pullback_risk == pytest.approx(expected, abs=0.01)


def test_to_dict_is_json_safe() -> None:
    import json

    scores = score_technicals(_ramp_up_ohlcv())
    payload = scores.to_dict()
    # Round-trip through json — no datetimes, only floats / None / nested dicts.
    serialized = json.dumps(payload)
    restored = json.loads(serialized)
    assert "subscores" in restored
    assert "confidence_score" in restored
