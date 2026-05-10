"""WebSocket streaming for live bars / quotes / trades / news.

Wraps alpaca-py's ``StockDataStream`` and ``NewsDataStream``. Handlers receive
our domain models (:class:`Bar`, :class:`Quote`, :class:`Trade`,
:class:`NewsArticle`) rather than alpaca-py raw types.

Usage::

    stream = MarketStream()
    stream.on_bar(my_handler)
    stream.subscribe(["AAPL", "MSFT"])
    await stream.start()
    # ... run other tasks ...
    await stream.stop()

The alpaca-py ``run()`` call is blocking and creates its own asyncio loop, so
we run it in a worker thread via :func:`asyncio.to_thread`. Our public API
stays fully async.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import TYPE_CHECKING

from src.data.alpaca_client import AlpacaClient
from src.data.models import Bar, NewsArticle, Quote, TimeFrame, Trade
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from alpaca.data.live import NewsDataStream, StockDataStream

log = get_logger(__name__)

BarHandler = Callable[[Bar], Awaitable[None]]
QuoteHandler = Callable[[Quote], Awaitable[None]]
TradeHandler = Callable[[Trade], Awaitable[None]]
NewsHandler = Callable[[NewsArticle], Awaitable[None]]


class MarketStream:
    """Async wrapper around alpaca-py's market + news streams.

    Multiple handlers can be registered per event type; each is awaited per
    incoming message. Handler exceptions are logged but never propagated to
    avoid killing the stream loop.
    """

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        include_news: bool = False,
    ) -> None:
        self.client = client or AlpacaClient()
        self._include_news = include_news

        self._bar_handlers: list[BarHandler] = []
        self._quote_handlers: list[QuoteHandler] = []
        self._trade_handlers: list[TradeHandler] = []
        self._news_handlers: list[NewsHandler] = []

        self._symbols: set[str] = set()
        self._news_symbols: set[str] = set()

        self._market_task: asyncio.Task[None] | None = None
        self._news_task: asyncio.Task[None] | None = None
        self._started = False

    # ---- handler registration ------------------------------------------------

    def on_bar(self, handler: BarHandler) -> None:
        self._bar_handlers.append(handler)

    def on_quote(self, handler: QuoteHandler) -> None:
        self._quote_handlers.append(handler)

    def on_trade(self, handler: TradeHandler) -> None:
        self._trade_handlers.append(handler)

    def on_news(self, handler: NewsHandler) -> None:
        self._news_handlers.append(handler)

    # ---- subscriptions -------------------------------------------------------

    def subscribe(self, symbols: list[str]) -> None:
        """Add symbols to the market-data subscription set."""
        self._symbols.update(symbols)

    def subscribe_news(self, symbols: list[str]) -> None:
        """Add symbols to the news subscription set."""
        self._news_symbols.update(symbols)

    # ---- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Wire handlers, then run the underlying streams in worker threads."""
        if self._started:
            raise RuntimeError("MarketStream.start() called twice")
        if not self._has_any_handler():
            raise RuntimeError("no handlers registered; nothing to stream")
        if not self._symbols and not self._news_symbols:
            raise RuntimeError("no symbols subscribed")

        market_stream = self.client.stream
        if self._symbols:
            symbols_tuple = tuple(self._symbols)
            if self._bar_handlers:
                market_stream.subscribe_bars(self._dispatch_bar, *symbols_tuple)
            if self._quote_handlers:
                market_stream.subscribe_quotes(self._dispatch_quote, *symbols_tuple)
            if self._trade_handlers:
                market_stream.subscribe_trades(self._dispatch_trade, *symbols_tuple)
            log.info("stream.market_starting", symbols=sorted(self._symbols))
            self._market_task = asyncio.create_task(
                asyncio.to_thread(market_stream.run), name="esther-market-stream"
            )

        if self._include_news and self._news_handlers and self._news_symbols:
            news_stream = self.client.news_stream
            news_stream.subscribe_news(self._dispatch_news, *tuple(self._news_symbols))
            log.info("stream.news_starting", symbols=sorted(self._news_symbols))
            self._news_task = asyncio.create_task(
                asyncio.to_thread(news_stream.run), name="esther-news-stream"
            )

        self._started = True

    async def stop(self) -> None:
        """Signal the SDK to stop, then await the worker tasks."""
        if not self._started:
            return

        await asyncio.gather(
            self._stop_one(self.client.stream, self._market_task, "market"),
            self._stop_one(
                self.client.news_stream if self._include_news else None,
                self._news_task,
                "news",
            ),
            return_exceptions=True,
        )
        self._market_task = None
        self._news_task = None
        self._started = False

    async def run_forever(self) -> None:
        """Convenience: start and block until both worker tasks complete."""
        await self.start()
        tasks = [t for t in (self._market_task, self._news_task) if t is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ---- internals -----------------------------------------------------------

    def _has_any_handler(self) -> bool:
        return bool(
            self._bar_handlers
            or self._quote_handlers
            or self._trade_handlers
            or (self._include_news and self._news_handlers)
        )

    @staticmethod
    async def _stop_one(
        stream: "StockDataStream | NewsDataStream | None",
        task: asyncio.Task[None] | None,
        label: str,
    ) -> None:
        if stream is None or task is None:
            return
        try:
            await asyncio.to_thread(stream.stop)
        except Exception:
            log.exception("stream.stop_failed", which=label)
        try:
            await asyncio.wait_for(task, timeout=10.0)
        except asyncio.TimeoutError:
            log.warning("stream.stop_timeout", which=label)
            task.cancel()

    async def _dispatch_bar(self, raw: object) -> None:
        bar = _to_domain_bar(raw)
        await self._fan_out(self._bar_handlers, bar, "bar")

    async def _dispatch_quote(self, raw: object) -> None:
        quote = _to_domain_quote(raw)
        await self._fan_out(self._quote_handlers, quote, "quote")

    async def _dispatch_trade(self, raw: object) -> None:
        trade = _to_domain_trade(raw)
        await self._fan_out(self._trade_handlers, trade, "trade")

    async def _dispatch_news(self, raw: object) -> None:
        article = _to_domain_news(raw)
        await self._fan_out(self._news_handlers, article, "news")

    @staticmethod
    async def _fan_out(handlers: list, payload: object, label: str) -> None:
        for handler in handlers:
            try:
                await handler(payload)
            except Exception:
                log.exception("stream.handler_error", which=label)


# ---------------------------------------------------------------------------
# domain conversion (alpaca-py models / dicts -> our models)
# ---------------------------------------------------------------------------


def _attr(raw: object, name: str, default: object = None) -> object:
    """Get attribute from a model OR dict from raw_data mode."""
    if isinstance(raw, dict):
        return raw.get(name, default)
    return getattr(raw, name, default)


def _to_domain_bar(raw: object) -> Bar:
    return Bar(
        symbol=str(_attr(raw, "symbol", "")),
        timestamp=_attr(raw, "timestamp"),  # type: ignore[arg-type]
        open=Decimal(str(_attr(raw, "open"))),
        high=Decimal(str(_attr(raw, "high"))),
        low=Decimal(str(_attr(raw, "low"))),
        close=Decimal(str(_attr(raw, "close"))),
        volume=int(_attr(raw, "volume", 0)),  # type: ignore[arg-type]
        timeframe=TimeFrame.MIN_1,  # streaming bars are 1-minute by default
    )


def _to_domain_quote(raw: object) -> Quote:
    return Quote(
        symbol=str(_attr(raw, "symbol", "")),
        timestamp=_attr(raw, "timestamp"),  # type: ignore[arg-type]
        bid_price=Decimal(str(_attr(raw, "bid_price", 0))),
        bid_size=int(_attr(raw, "bid_size", 0)),  # type: ignore[arg-type]
        ask_price=Decimal(str(_attr(raw, "ask_price", 0))),
        ask_size=int(_attr(raw, "ask_size", 0)),  # type: ignore[arg-type]
    )


def _to_domain_trade(raw: object) -> Trade:
    return Trade(
        symbol=str(_attr(raw, "symbol", "")),
        timestamp=_attr(raw, "timestamp"),  # type: ignore[arg-type]
        price=Decimal(str(_attr(raw, "price", 0))),
        size=int(_attr(raw, "size", 0)),  # type: ignore[arg-type]
    )


def _to_domain_news(raw: object) -> NewsArticle:
    return NewsArticle(
        id=str(_attr(raw, "id", "")),
        headline=str(_attr(raw, "headline", "")),
        summary=str(_attr(raw, "summary", "") or ""),
        author=_attr(raw, "author"),  # type: ignore[arg-type]
        url=_attr(raw, "url"),  # type: ignore[arg-type]
        source=str(_attr(raw, "source", "alpaca")),
        symbols=list(_attr(raw, "symbols", []) or []),  # type: ignore[arg-type]
        published_at=_attr(raw, "created_at") or _attr(raw, "timestamp"),  # type: ignore[arg-type]
        updated_at=_attr(raw, "updated_at"),  # type: ignore[arg-type]
    )
