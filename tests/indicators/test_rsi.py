from __future__ import annotations

import pandas as pd

from src.indicators.rsi import RSI


def test_rsi_shape_and_range(ohlcv_df: pd.DataFrame) -> None:
    rsi = RSI(period=14).compute(ohlcv_df)
    assert len(rsi) == len(ohlcv_df)
    valid = rsi.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_rsi_constant_price_is_neutral_or_nan() -> None:
    df = pd.DataFrame({"close": [100.0] * 50})
    rsi = RSI(period=14).compute(df)
    # all-zero gain and loss → undefined (NaN) is acceptable
    valid = rsi.dropna()
    if not valid.empty:
        assert (valid == 100).all() or valid.isna().all()
