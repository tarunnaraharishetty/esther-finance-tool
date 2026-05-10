from __future__ import annotations

import pandas as pd

from src.indicators.bollinger import BollingerBands


def test_bollinger_band_ordering(ohlcv_df: pd.DataFrame) -> None:
    out = BollingerBands(period=20, num_std=2.0).compute_full(ohlcv_df)
    valid = out.dropna()
    assert (valid["upper"] >= valid["middle"]).all()
    assert (valid["middle"] >= valid["lower"]).all()
