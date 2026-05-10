from __future__ import annotations

import pandas as pd

from src.indicators.macd import MACD


def test_macd_full_returns_three_columns(ohlcv_df: pd.DataFrame) -> None:
    out = MACD().compute_full(ohlcv_df)
    assert set(out.columns) == {"macd", "signal", "hist"}
    assert len(out) == len(ohlcv_df)


def test_macd_hist_is_difference(ohlcv_df: pd.DataFrame) -> None:
    out = MACD().compute_full(ohlcv_df)
    assert ((out["macd"] - out["signal"]) - out["hist"]).abs().max() < 1e-9
