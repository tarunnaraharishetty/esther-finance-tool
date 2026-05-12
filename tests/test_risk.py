from __future__ import annotations

import pandas as pd

from src.risk.metrics import max_drawdown, sharpe_ratio


def test_sharpe_zero_for_zero_variance() -> None:
    assert sharpe_ratio(pd.Series([0.001] * 100)) == 0.0


def test_sharpe_positive_for_positive_drift(synthetic_returns: pd.Series) -> None:
    assert sharpe_ratio(synthetic_returns) > 0


def test_max_drawdown_non_negative(equity_curve: pd.Series) -> None:
    assert max_drawdown(equity_curve) >= 0.0
