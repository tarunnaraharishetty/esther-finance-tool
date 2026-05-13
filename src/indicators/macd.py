"""Moving Average Convergence Divergence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.indicators.base import Indicator

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class MACD(Indicator):
    """Returns a DataFrame-like Series with MACD line.

    Use :meth:`compute_full` for (macd, signal, histogram).
    """

    fast: int = 12
    slow: int = 26
    signal: int = 9
    price_col: str = "close"
    name: str = "macd"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return self.compute_full(df)["macd"]

    def compute_full(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        close = df[self.price_col]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})
