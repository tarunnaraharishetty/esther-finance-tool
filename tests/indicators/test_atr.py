"""ATR indicator tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.atr import ATR


def test_atr_shape_and_non_negative(ohlcv_df: pd.DataFrame) -> None:
    atr = ATR(period=14).compute(ohlcv_df)
    assert len(atr) == len(ohlcv_df)
    valid = atr.dropna()
    assert (valid >= 0).all()


def test_atr_on_flat_series_is_zero() -> None:
    # Constant high == low == close → true range = 0 on every bar.
    n = 30
    df = pd.DataFrame(
        {
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "open": [100.0] * n,
            "volume": [1] * n,
        }
    )
    atr = ATR(period=14).compute(df).dropna()
    assert np.isclose(atr.iloc[-1], 0.0)


def test_atr_increases_with_widening_bars() -> None:
    """A bar with 2x the range of its neighbors should lift ATR."""
    n = 50
    rng = np.random.default_rng(seed=3)
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    high = close + 0.5
    low = close - 0.5
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": [1] * n}
    )
    baseline = ATR(period=14).compute(df).dropna().iloc[-1]

    # Make the final bar's range 10x baseline.
    df_shocked = df.copy()
    df_shocked.loc[df_shocked.index[-1], "high"] = float(close[-1]) + 5.0
    df_shocked.loc[df_shocked.index[-1], "low"] = float(close[-1]) - 5.0
    shocked = ATR(period=14).compute(df_shocked).dropna().iloc[-1]
    assert shocked > baseline
