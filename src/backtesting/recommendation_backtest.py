"""Walk-forward backtester for the RecommendationEngine.

Vectorised-ish loop over a single symbol's daily OHLCV: at each bar, run
the engine on the trailing ``lookback_bars`` window, decide
LONG / FLAT / SHORT for the next bar, and accrue equity.

Why not Backtrader for this first pass:
* Backtrader's overhead (Cerebro, PandasData, analyzers) is heavy for a
  daily, single-instrument, rule-based backtest.
* Keeping the loop in pandas makes it trivial to plug in a fake
  sentiment analyzer and run the test suite offline.
The Backtrader engine in ``src/backtesting/engine.py`` is still on the
roadmap for multi-asset / intraday / event-driven scenarios.

The backtest is **deterministic** for a given (engine, df, config). No
randomness; no hidden state across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.backtesting.benchmarks import BenchmarkComparison
from src.risk.metrics import max_drawdown, sharpe_ratio
from src.strategy.base import SignalAction
from src.strategy.recommendation import RecommendationEngine
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from datetime import date

    from src.data.models import NewsArticle

log = get_logger(__name__)


class Position(StrEnum):
    LONG = "long"
    FLAT = "flat"
    SHORT = "short"


@dataclass
class BacktestRunConfig:
    """Knobs for one backtest run."""

    starting_cash: float = 100_000.0
    lookback_bars: int = 60
    min_confidence: float = 0.3
    commission_pct: float = 0.0005  # 5 bps per fill (round-trip ~ 10 bps)
    allow_short: bool = False
    rebalance_every: int = 1  # bars between engine evaluations
    # Sentiment off by default — FinBERT per bar over years is expensive.
    use_sentiment: bool = False


@dataclass
class RecommendationBacktestResult:
    """All artifacts of one backtest run."""

    symbol: str
    equity_curve: pd.Series
    returns: pd.Series
    actions: pd.Series  # action *taken* per bar (after min_confidence filter)
    positions: pd.Series  # position held going INTO each bar
    confidence: pd.Series
    trades: int
    final_value: float
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    buy_and_hold_return_pct: float
    alpha_pct: float
    config: BacktestRunConfig = field(repr=False)

    @property
    def benchmark(self) -> BenchmarkComparison:
        return BenchmarkComparison(
            benchmark_symbol="buy_and_hold",
            benchmark_return_pct=self.buy_and_hold_return_pct,
            strategy_return_pct=self.total_return_pct,
            alpha_pct=self.alpha_pct,
        )


class RecommendationBacktest:
    """Run :class:`RecommendationEngine` over an OHLCV DataFrame.

    Convention:
    * ``df`` must have a sorted datetime index and columns
      ``open, high, low, close, volume``.
    * At bar ``i``, the engine sees ``df.iloc[i - lookback : i + 1]``
      (close included). The resulting action is applied to the **next**
      bar's return — strictly close-to-close, no peeking.
    """

    def __init__(
        self,
        engine: RecommendationEngine,
        config: BacktestRunConfig | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or BacktestRunConfig()

    def run(
        self,
        df: pd.DataFrame,
        *,
        symbol: str = "SPY",
        news_by_date: "dict[date, list[NewsArticle]] | None" = None,
    ) -> RecommendationBacktestResult:
        self._validate(df)
        cfg = self.config

        n = len(df)
        if n <= cfg.lookback_bars + 1:
            raise ValueError(
                f"need > {cfg.lookback_bars + 1} bars, got {n}"
            )

        closes = df["close"].astype(float).to_numpy()
        index = df.index

        # State
        position: Position = Position.FLAT
        cash = cfg.starting_cash
        shares = 0.0
        equity = np.full(n, cfg.starting_cash, dtype=float)
        actions = np.array(["hold"] * n, dtype=object)
        positions = np.array([Position.FLAT.value] * n, dtype=object)
        confidence = np.zeros(n, dtype=float)
        trades = 0

        # Walk forward. We can only decide from bar lookback_bars onward
        # (engine needs the window). We hold the position into bar i, and
        # the engine's decision at the *close* of bar i takes effect for
        # the return from bar i to bar i+1.
        for i in range(n):
            close_i = closes[i]

            # mark-to-market equity at bar i (before any new trade)
            equity[i] = cash + shares * close_i
            positions[i] = position.value

            # Only re-evaluate after we have enough history and on rebalance bars
            if i < cfg.lookback_bars or (i - cfg.lookback_bars) % cfg.rebalance_every != 0:
                continue
            if i == n - 1:
                # last bar: nothing to act on (no next bar)
                continue

            window = df.iloc[i - cfg.lookback_bars : i + 1]
            news = self._news_for(news_by_date, index[i])
            rec = self.engine.recommend(symbol, window, news=news)
            confidence[i] = rec.confidence

            action = rec.action
            if rec.confidence < cfg.min_confidence:
                action = SignalAction.HOLD
            actions[i] = action.value

            new_position = self._target_position(position, action, cfg.allow_short)
            if new_position == position:
                continue

            # Realise the position change at bar i's close.
            position, cash, shares, fills = self._transition(
                position=position,
                new_position=new_position,
                cash=cash,
                shares=shares,
                price=close_i,
                commission_pct=cfg.commission_pct,
            )
            trades += fills

        # Final mark-to-market is already in equity[-1]
        equity_curve = pd.Series(equity, index=index, name="equity")
        returns = equity_curve.pct_change().fillna(0.0)
        actions_s = pd.Series(actions, index=index, name="action")
        positions_s = pd.Series(positions, index=index, name="position")
        confidence_s = pd.Series(confidence, index=index, name="confidence")

        total_return = float(equity_curve.iloc[-1] / cfg.starting_cash - 1.0)
        bh = float(closes[-1] / closes[0] - 1.0)
        result = RecommendationBacktestResult(
            symbol=symbol,
            equity_curve=equity_curve,
            returns=returns,
            actions=actions_s,
            positions=positions_s,
            confidence=confidence_s,
            trades=trades,
            final_value=float(equity_curve.iloc[-1]),
            total_return_pct=total_return * 100.0,
            sharpe=sharpe_ratio(returns),
            max_drawdown_pct=max_drawdown(equity_curve) * 100.0,
            buy_and_hold_return_pct=bh * 100.0,
            alpha_pct=(total_return - bh) * 100.0,
            config=cfg,
        )
        log.info(
            "backtest.done",
            symbol=symbol,
            trades=trades,
            total_return_pct=round(result.total_return_pct, 3),
            buy_hold_pct=round(result.buy_and_hold_return_pct, 3),
            alpha_pct=round(result.alpha_pct, 3),
            sharpe=round(result.sharpe, 3),
            max_dd_pct=round(result.max_drawdown_pct, 3),
        )
        return result

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"OHLCV df missing columns: {sorted(missing)}")
        if not df.index.is_monotonic_increasing:
            raise ValueError("OHLCV df index must be sorted ascending")

    @staticmethod
    def _news_for(
        news_by_date: "dict[date, list[NewsArticle]] | None",
        ts: object,
    ) -> "list[NewsArticle] | None":
        if not news_by_date:
            return None
        d = getattr(ts, "date", None)
        key = d() if callable(d) else ts
        return news_by_date.get(key)  # type: ignore[arg-type]

    @staticmethod
    def _target_position(
        current: Position, action: SignalAction, allow_short: bool
    ) -> Position:
        if action == SignalAction.BUY:
            return Position.LONG
        if action == SignalAction.SELL:
            return Position.SHORT if allow_short else Position.FLAT
        return current  # HOLD

    @staticmethod
    def _transition(
        *,
        position: Position,
        new_position: Position,
        cash: float,
        shares: float,
        price: float,
        commission_pct: float,
    ) -> tuple[Position, float, float, int]:
        """Apply a position change at ``price``. Returns updated state + #fills."""
        fills = 0
        # First, close any existing position into cash.
        if position == Position.LONG and shares > 0:
            gross = shares * price
            cash += gross * (1.0 - commission_pct)
            shares = 0.0
            fills += 1
        elif position == Position.SHORT and shares < 0:
            # Buying back: shares < 0, abs(shares) * price is what we owe.
            cost = (-shares) * price
            cash -= cost * (1.0 + commission_pct)
            shares = 0.0
            fills += 1

        # Now open the new position with available cash (long-only sizing:
        # 100% of cash into the new position).
        if new_position == Position.LONG and cash > 0 and price > 0:
            notional = cash
            gross_shares = notional / price
            # Pay commission from the gross.
            shares = gross_shares * (1.0 - commission_pct)
            cash = 0.0
            fills += 1
        elif new_position == Position.SHORT and price > 0:
            notional = cash
            gross_shares = notional / price
            shares = -gross_shares * (1.0 - commission_pct)
            # Short proceeds add to cash less commission.
            cash = cash + (-shares) * price * (1.0 - commission_pct) - notional
            fills += 1

        return new_position, cash, shares, fills
