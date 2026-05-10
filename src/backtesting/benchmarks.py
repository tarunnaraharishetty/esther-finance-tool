"""Benchmark comparisons (default: SPY)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.data.market_data import MarketDataService
from src.data.models import TimeFrame


@dataclass
class BenchmarkComparison:
    benchmark_symbol: str
    benchmark_return_pct: float
    strategy_return_pct: float
    alpha_pct: float


async def compare_to_spy(
    strategy_return_pct: float,
    start: datetime,
    end: datetime,
    market_data: MarketDataService | None = None,
) -> BenchmarkComparison:
    md = market_data or MarketDataService()
    bars = await md.get_bars(["SPY"], TimeFrame.DAY_1, start, end)
    if not bars:
        raise RuntimeError("no SPY bars returned for benchmark window")
    first = float(bars[0].close)
    last = float(bars[-1].close)
    spy_return = (last - first) / first * 100
    return BenchmarkComparison(
        benchmark_symbol="SPY",
        benchmark_return_pct=spy_return,
        strategy_return_pct=strategy_return_pct,
        alpha_pct=strategy_return_pct - spy_return,
    )
