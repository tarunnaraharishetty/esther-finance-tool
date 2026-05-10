"""Backtesting engine wrapping Backtrader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import backtrader as bt
    import pandas as pd

log = get_logger(__name__)


@dataclass
class BacktestConfig:
    starting_cash: float = 100_000.0
    commission: float = 0.0
    slippage_pct: float = 0.0


@dataclass
class BacktestResult:
    final_value: float
    total_return_pct: float
    sharpe: float | None
    max_drawdown_pct: float
    trades: int


class BacktestEngine:
    """Run a strategy over historical OHLCV via Backtrader."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        df: "pd.DataFrame",
        strategy_cls: type["bt.Strategy"],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> BacktestResult:
        raise NotImplementedError(
            "wire up backtrader Cerebro with PandasData feed + analyzers"
        )
