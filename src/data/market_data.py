"""Historical OHLCV / quote / trade fetching."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from src.data.alpaca_client import AlpacaClient
from src.data.models import Bar, TimeFrame
from src.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)


class MarketDataService:
    """Fetches historical bars/quotes/trades."""

    def __init__(self, client: AlpacaClient | None = None) -> None:
        self.client = client or AlpacaClient()

    async def get_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Fetch OHLCV bars for ``symbols`` between ``start`` and ``end``."""
        await self.client.acquire_quota()
        raise NotImplementedError("wire up alpaca-py StockBarsRequest")

    async def get_latest_bar(self, symbol: str) -> Bar:
        await self.client.acquire_quota()
        raise NotImplementedError

    def to_dataframe(self, bars: list[Bar]) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame([b.model_dump() for b in bars])
