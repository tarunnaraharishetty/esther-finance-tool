"""Performance and risk metrics: Sharpe, Sortino, max drawdown."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# Treat std below this as effectively zero — a constant series produces
# float-noise std on the order of 1e-19, which would explode the ratio.
_STD_EPS = 1e-12


def sharpe_ratio(returns: "pd.Series", risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio. ``returns`` are periodic (e.g. daily)."""
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    std = excess.std()
    if std < _STD_EPS or np.isnan(std):
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / std)


def max_drawdown(equity_curve: "pd.Series") -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (0.2 == 20%)."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(-drawdown.min())
