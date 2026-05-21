"""Average True Range — Wilder's volatility indicator.

ATR measures volatility from the typical bar range, not from price
returns. Used by the analyzer to gauge whether current volatility is
elevated (overheated names tend to print large true ranges as their
move accelerates and again as it reverses).

True range for one bar is the max of:

* ``high - low``
* ``|high - previous close|``
* ``|low - previous close|``

ATR is the Wilder-smoothed (1/period EWM) running average of true
range. The first ``period - 1`` values are NaN — callers must drop or
guard before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.indicators.base import Indicator

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ATR(Indicator):
    """Wilder's ATR over ``period`` bars (default 14)."""

    period: int = 14
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    name: str = "atr"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        import pandas as pd

        high = df[self.high_col]
        low = df[self.low_col]
        prev_close = df[self.close_col].shift(1)
        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        # Wilder smoothing: EWM with alpha = 1/period, mirrors the
        # RSI implementation so behavior across indicators is consistent.
        return true_range.ewm(alpha=1 / self.period, adjust=False).mean().rename(
            self.name
        )
