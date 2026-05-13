"""Relative Strength Index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.indicators.base import Indicator

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class RSI(Indicator):
    """Wilder's RSI over ``period`` bars (default 14)."""

    period: int = 14
    price_col: str = "close"
    name: str = "rsi"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        delta = df[self.price_col].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        return rsi.rename(self.name)
