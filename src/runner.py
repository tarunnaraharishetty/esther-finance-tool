"""Trading loop orchestrator.

Ties together the data layer, recommendation engine, and risk-gated order
manager. This is the heart of ``esther run``.

Pipeline per tick (paper-only):

    fetch bars + news (per symbol)
        → RecommendationEngine.recommend(symbol, df, news)
        → recommendation_to_signal(rec)
        → OrderManager.submit_signal(signal, last_price)   [risk gate runs here]
        → OrderResult (or None if HOLD / blocked / dry-run)

The loop is paper-only and *defaults to dry-run* (no orders submitted).
Order submission must be an explicit, deliberate flip via ``execute=True``
on the loop or ``--execute`` on the CLI.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from src.config import Settings, get_settings
from src.data.market_data import MarketDataService
from src.data.models import TimeFrame
from src.data.news_ingestion import NewsSource, get_news_source
from src.execution.broker import AlpacaBroker, Broker
from src.execution.order_manager import OrderManager
from src.risk.exposure import RiskGate
from src.strategy.base import Signal
from src.strategy.recommendation import RecommendationEngine, TradingRecommendation
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.execution.broker import OrderResult

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Signal conversion
# ---------------------------------------------------------------------------


def recommendation_to_signal(rec: TradingRecommendation) -> Signal:
    """Convert engine output to the Signal type the OrderManager consumes."""
    return Signal(
        symbol=rec.symbol,
        action=rec.action,
        confidence=rec.confidence,
        timestamp=rec.timestamp,
        source="recommendation_engine",
        rationale=rec.reasoning,
        metadata={
            "combined_score": rec.combined_score,
            "technical_score": rec.technical_score,
            "sentiment_score": rec.sentiment_score,
            "indicator_scores": rec.indicator_scores,
            "num_news_articles": rec.num_news_articles,
        },
    )


# ---------------------------------------------------------------------------
# Config + tick result
# ---------------------------------------------------------------------------


@dataclass
class TradingLoopConfig:
    """Knobs for the trading loop."""

    watchlist: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "SPY"])
    timeframe: TimeFrame = TimeFrame.DAY_1
    lookback_days: int = 60
    news_hours: int = 48
    interval_seconds: float = 60.0
    max_iterations: int | None = None  # None = run forever
    min_confidence: float = 0.3  # below this, treat as HOLD even if BUY/SELL
    execute: bool = False  # paper order submission. Default is dry-run.


@dataclass(frozen=True)
class TickResult:
    """Outcome of evaluating a single symbol in one tick."""

    symbol: str
    recommendation: TradingRecommendation | None
    order: "OrderResult | None"
    note: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class TradingLoop:
    """Async loop: data → recommend → risk-gated order.

    Constructor takes the fully-built dependencies so tests can inject mocks
    cheaply. Use :meth:`build` to construct the production wiring.
    """

    def __init__(
        self,
        *,
        config: TradingLoopConfig,
        engine: RecommendationEngine,
        market: MarketDataService,
        news_source: NewsSource,
        order_manager: OrderManager,
        settings: Settings | None = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.market = market
        self.news_source = news_source
        self.order_manager = order_manager
        self.settings = settings or get_settings()

        if not self.settings.is_paper_trading:
            raise RuntimeError(
                "TradingLoop refuses to start against a non-paper Alpaca URL "
                f"(got {self.settings.alpaca_base_url!r})"
            )

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        config: TradingLoopConfig,
        *,
        engine: RecommendationEngine | None = None,
        broker: Broker | None = None,
        settings: Settings | None = None,
    ) -> "TradingLoop":
        """Construct with production defaults (Alpaca-backed broker/data/news)."""
        s = settings or get_settings()
        market = MarketDataService()
        news_source = get_news_source()
        engine = engine or RecommendationEngine()
        broker = broker or AlpacaBroker()
        risk_gate = RiskGate(settings=s)
        order_manager = OrderManager(broker=broker, risk_gate=risk_gate, settings=s)
        return cls(
            config=config,
            engine=engine,
            market=market,
            news_source=news_source,
            order_manager=order_manager,
            settings=s,
        )

    # -- one tick over the watchlist --------------------------------------

    async def tick(self, *, now: datetime | None = None) -> list[TickResult]:
        ts = now or datetime.now(UTC)
        results: list[TickResult] = []
        for symbol in self.config.watchlist:
            try:
                result = await self._evaluate_symbol(symbol, ts)
            except Exception as e:  # noqa: BLE001 — keep loop going on per-symbol errors
                log.warning("loop.symbol_error", symbol=symbol, error=str(e))
                result = TickResult(
                    symbol=symbol, recommendation=None, order=None, error=str(e)
                )
            results.append(result)
        return results

    async def _evaluate_symbol(self, symbol: str, ts: datetime) -> TickResult:
        bars_start = ts - timedelta(days=self.config.lookback_days)
        news_start = ts - timedelta(hours=self.config.news_hours)

        bars = await self.market.get_bars(
            [symbol], self.config.timeframe, bars_start, ts
        )
        df = self.market.to_dataframe(bars)
        if df.empty:
            return TickResult(
                symbol=symbol, recommendation=None, order=None, note="no bars"
            )

        news = await self.news_source.fetch([symbol], news_start, ts, limit=20)
        rec = self.engine.recommend(symbol, df, news=news, now=ts)

        # Below-confidence trades are dropped, not submitted.
        if rec.confidence < self.config.min_confidence:
            return TickResult(
                symbol=symbol,
                recommendation=rec,
                order=None,
                note=f"confidence {rec.confidence:.2f} < min {self.config.min_confidence:.2f}",
            )

        signal = recommendation_to_signal(rec)
        last_price = Decimal(str(df["close"].iloc[-1]))

        if not self.config.execute:
            return TickResult(
                symbol=symbol, recommendation=rec, order=None, note="dry-run"
            )

        order = await self.order_manager.submit_signal(signal, last_price)
        note = "submitted" if order else "blocked or HOLD"
        return TickResult(symbol=symbol, recommendation=rec, order=order, note=note)

    # -- run the loop ------------------------------------------------------

    async def run(self) -> None:
        """Run until ``max_iterations`` (if set) or until cancelled."""
        log.info(
            "loop.start",
            watchlist=self.config.watchlist,
            execute=self.config.execute,
            interval=self.config.interval_seconds,
        )
        i = 0
        while True:
            i += 1
            results = await self.tick()
            for r in results:
                self._log_result(i, r)
            if self.config.max_iterations is not None and i >= self.config.max_iterations:
                log.info("loop.done", iterations=i)
                return
            await asyncio.sleep(self.config.interval_seconds)

    @staticmethod
    def _log_result(iteration: int, r: TickResult) -> None:
        if r.error:
            log.warning("loop.tick", iter=iteration, symbol=r.symbol, error=r.error)
            return
        if r.recommendation is None:
            log.info("loop.tick", iter=iteration, symbol=r.symbol, note=r.note)
            return
        log.info(
            "loop.tick",
            iter=iteration,
            symbol=r.symbol,
            action=r.recommendation.action.value,
            confidence=round(r.recommendation.confidence, 3),
            combined=round(r.recommendation.combined_score, 3),
            order_id=r.order.id if r.order else None,
            note=r.note,
        )
