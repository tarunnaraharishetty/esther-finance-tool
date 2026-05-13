"""Base class for technical indicators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class Indicator(ABC):
    """All indicators take an OHLCV DataFrame and return a pandas Series.

    Implementations must be deterministic and side-effect-free so they can
    be safely reused across strategies and backtests.
    """

    name: str = "indicator"

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series: ...

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.compute(df)
