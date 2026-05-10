"""WebSocket streaming for live bars / quotes / trades / news."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from src.data.alpaca_client import AlpacaClient
from src.utils.logging import get_logger

log = get_logger(__name__)

Handler = Callable[[Any], Awaitable[None]]


class MarketStream:
    """Async wrapper around alpaca-py StockDataStream."""

    def __init__(self, client: AlpacaClient | None = None) -> None:
        self.client = client or AlpacaClient()
        self._bar_handlers: list[Handler] = []
        self._quote_handlers: list[Handler] = []
        self._trade_handlers: list[Handler] = []

    def on_bar(self, handler: Handler) -> None:
        self._bar_handlers.append(handler)

    def on_quote(self, handler: Handler) -> None:
        self._quote_handlers.append(handler)

    def on_trade(self, handler: Handler) -> None:
        self._trade_handlers.append(handler)

    async def subscribe(self, symbols: list[str]) -> None:
        raise NotImplementedError("wire up StockDataStream subscribe + run loop")

    async def run(self) -> None:
        raise NotImplementedError
