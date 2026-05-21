"""Volume z-score / spike detection.

The analyzer's "volume spike" sub-score wants to know whether the
most recent bar's volume is anomalously high relative to the recent
norm. A simple rolling z-score on log-volume captures that cleanly
and behaves well across symbols with different volume scales (mega-
caps vs small-caps).

Why log volume: raw volume is heavy-tailed (a single earnings day
prints 5-10x the median). A z-score on raw volume gets dominated by
outliers in the trailing window itself. Working in log space keeps
the standard deviation stable and the z-score interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.indicators.base import Indicator

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class VolumeZScore(Indicator):
    """Rolling z-score of log-volume over ``period`` bars."""

    period: int = 20
    volume_col: str = "volume"
    name: str = "volume_z"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        import numpy as np

        vol = df[self.volume_col].astype(float)
        # log1p so a zero-volume bar (rare but real for OTC names)
        # doesn't blow the rolling mean to -inf.
        log_vol = np.log1p(vol)
        mean = log_vol.rolling(self.period).mean()
        std = log_vol.rolling(self.period).std()
        z = (log_vol - mean) / std.replace(0, float("nan"))
        return z.rename(self.name)
