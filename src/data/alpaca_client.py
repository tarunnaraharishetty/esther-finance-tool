"""Thin wrapper over alpaca-py clients.

Centralises authentication, rate limiting, and retries so the rest of the
codebase never instantiates Alpaca SDK objects directly.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from src.config import Settings, get_settings
from src.utils.logging import get_logger
from src.utils.rate_limiter import TokenBucket

if TYPE_CHECKING:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.live import NewsDataStream, StockDataStream
    from alpaca.trading.client import TradingClient

log = get_logger(__name__)


class AlpacaClient:
    """Lazily-instantiated facade over Alpaca trading + data clients.

    Subclients are constructed on first access via cached_property to keep
    startup cheap and tests fast (no network at import).
    """

    # Alpaca's documented limit is 200 req/min for free tier; tune per plan.
    _DEFAULT_BUCKET = TokenBucket(capacity=200, refill_rate=200 / 60)

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.bucket = AlpacaClient._DEFAULT_BUCKET
        if not self._settings.is_paper_trading:
            # Defence in depth: Esther's `safety` contract (see CLAUDE.md)
            # bans order submission and pins data ingestion to the paper
            # endpoint. Refusing in the client constructor means future
            # code paths (background workers, REPL exploration) can't
            # silently start hitting live by setting ``ALPACA_BASE_URL``.
            # If genuinely needed for an integration test, instantiate
            # the underlying alpaca-py clients directly.
            raise RuntimeError(
                "ALPACA_BASE_URL points at the live endpoint "
                f"({self._settings.alpaca_base_url!r}). Esther refuses to "
                "instantiate AlpacaClient against a live URL — only the paper "
                "endpoint (paper-api.alpaca.markets) is supported."
            )

    @cached_property
    def trading(self) -> TradingClient:
        from alpaca.trading.client import TradingClient

        return TradingClient(
            api_key=self._settings.alpaca_api_key.get_secret_value(),
            secret_key=self._settings.alpaca_api_secret.get_secret_value(),
            paper=self._settings.is_paper_trading,
        )

    @cached_property
    def market_data(self) -> StockHistoricalDataClient:
        from alpaca.data.historical import StockHistoricalDataClient

        return StockHistoricalDataClient(
            api_key=self._settings.alpaca_api_key.get_secret_value(),
            secret_key=self._settings.alpaca_api_secret.get_secret_value(),
        )

    @cached_property
    def news(self) -> NewsClient:
        from alpaca.data.historical.news import NewsClient

        return NewsClient(
            api_key=self._settings.alpaca_api_key.get_secret_value(),
            secret_key=self._settings.alpaca_api_secret.get_secret_value(),
        )

    @cached_property
    def stream(self) -> StockDataStream:
        from alpaca.data.enums import DataFeed
        from alpaca.data.live import StockDataStream

        return StockDataStream(
            api_key=self._settings.alpaca_api_key.get_secret_value(),
            secret_key=self._settings.alpaca_api_secret.get_secret_value(),
            feed=DataFeed(self._settings.alpaca_data_feed.value),
        )

    @cached_property
    def news_stream(self) -> NewsDataStream:
        from alpaca.data.live import NewsDataStream

        return NewsDataStream(
            api_key=self._settings.alpaca_api_key.get_secret_value(),
            secret_key=self._settings.alpaca_api_secret.get_secret_value(),
        )

    async def acquire_quota(self, weight: float = 1.0) -> None:
        """Throttle before issuing an Alpaca request."""
        await self.bucket.acquire(weight)
