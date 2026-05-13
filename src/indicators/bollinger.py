"""Bollinger Bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.indicators.base import Indicator

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BollingerBands(Indicator):
    period: int = 20
    num_std: float = 2.0
    price_col: str = "close"
    name: str = "bollinger"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return self.compute_full(df)["middle"]

    def compute_full(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        close = df[self.price_col]
        middle = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = middle + self.num_std * std
        lower = middle - self.num_std * std
        return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})
