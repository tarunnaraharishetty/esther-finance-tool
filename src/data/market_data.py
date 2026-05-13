"""Historical OHLCV / quote / trade fetching via Alpaca."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from src.config import get_settings
from src.data.alpaca_client import AlpacaClient
from src.data.models import Bar, TimeFrame
from src.utils.logging import get_logger
from src.utils.rate_limiter import with_async_retry

if TYPE_CHECKING:
    import pandas as pd
    from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame

log = get_logger(__name__)


def _to_alpaca_timeframe(tf: TimeFrame) -> "AlpacaTimeFrame":
    from alpaca.data.timeframe import TimeFrame as AlpacaTF
    from alpaca.data.timeframe import TimeFrameUnit

    mapping: dict[TimeFrame, "AlpacaTimeFrame"] = {
        TimeFrame.MIN_1: AlpacaTF(1, TimeFrameUnit.Minute),
        TimeFrame.MIN_5: AlpacaTF(5, TimeFrameUnit.Minute),
        TimeFrame.MIN_15: AlpacaTF(15, TimeFrameUnit.Minute),
        TimeFrame.HOUR_1: AlpacaTF(1, TimeFrameUnit.Hour),
        TimeFrame.DAY_1: AlpacaTF(1, TimeFrameUnit.Day),
    }
    return mapping[tf]


class MarketDataService:
    """Fetches historical bars/quotes/trades. Async wrapper over the sync SDK."""

    def __init__(self, client: AlpacaClient | None = None) -> None:
        self.client = client or AlpacaClient()

    @with_async_retry(attempts=3, min_wait=1.0, max_wait=10.0)
    async def get_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
        *,
        limit: int | None = None,
    ) -> list[Bar]:
        """Fetch OHLCV bars for ``symbols`` between ``start`` and ``end``."""
        if not symbols:
            return []
        await self.client.acquire_quota()

        from alpaca.data.requests import StockBarsRequest

        feed_value = get_settings().alpaca_data_feed.value
        # alpaca-py's StockBarsRequest is typed as `DataFeed | None` for
        # `feed`, but accepts the raw enum string at runtime.
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=_to_alpaca_timeframe(timeframe),
            start=start,
            end=end,
            limit=limit,
            feed=feed_value,  # type: ignore[arg-type]
        )
        bar_set = await asyncio.to_thread(self.client.market_data.get_stock_bars, request)
        return self._convert_bars(bar_set, timeframe)

    @with_async_retry(attempts=3, min_wait=1.0, max_wait=10.0)
    async def get_latest_bar(self, symbol: str) -> Bar:
        """Latest available bar for a single symbol."""
        await self.client.acquire_quota()

        from alpaca.data.requests import StockLatestBarRequest

        feed_value = get_settings().alpaca_data_feed.value
        # See note on `get_bars`: SDK's `feed` typing is narrower than runtime.
        request = StockLatestBarRequest(
            symbol_or_symbols=symbol, feed=feed_value  # type: ignore[arg-type]
        )
        result = await asyncio.to_thread(self.client.market_data.get_stock_latest_bar, request)

        # Response is a dict-like keyed by symbol with a single Bar object.
        # alpaca-py's return type is unioned over a couple of shapes; the
        # hasattr branch covers both at runtime.
        raw = result[symbol] if hasattr(result, "__getitem__") else result.data[symbol]  # type: ignore[union-attr]
        return _bar_from_alpaca(symbol, raw, TimeFrame.MIN_1)

    def _convert_bars(self, bar_set: object, timeframe: TimeFrame) -> list[Bar]:
        """Flatten alpaca-py BarSet (dict[symbol, list[Bar]]) into our Bar list."""
        # alpaca-py returns a BarSet where .data is dict[str, list[AlpacaBar]];
        # mypy can't follow the duck-typed fallback through getattr.
        data = getattr(bar_set, "data", bar_set)
        out: list[Bar] = []
        for symbol, raw_bars in data.items():  # type: ignore[attr-defined]
            for raw in raw_bars:
                out.append(_bar_from_alpaca(symbol, raw, timeframe))
        return out

    def to_dataframe(self, bars: list[Bar]) -> "pd.DataFrame":
        import pandas as pd

        if not bars:
            return pd.DataFrame(columns=["symbol", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame([b.model_dump() for b in bars])
        df = df.set_index("timestamp").sort_index()
        return df


def _bar_from_alpaca(symbol: str, raw: object, timeframe: TimeFrame) -> Bar:
    """Map alpaca-py Bar -> our domain Bar. Raw is a Pydantic-ish object."""
    return Bar(
        symbol=symbol,
        timestamp=getattr(raw, "timestamp"),
        open=Decimal(str(getattr(raw, "open"))),
        high=Decimal(str(getattr(raw, "high"))),
        low=Decimal(str(getattr(raw, "low"))),
        close=Decimal(str(getattr(raw, "close"))),
        volume=int(getattr(raw, "volume")),
        timeframe=timeframe,
    )
