"""Domain models shared across the data layer."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TimeFrame(StrEnum):
    MIN_1 = "1Min"
    MIN_5 = "5Min"
    MIN_15 = "15Min"
    HOUR_1 = "1Hour"
    DAY_1 = "1Day"


class Bar(BaseModel):
    """OHLCV bar."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    timeframe: TimeFrame


class Quote(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    bid_price: Decimal
    bid_size: int
    ask_price: Decimal
    ask_size: int


class Trade(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    price: Decimal
    size: int


class NewsArticle(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    headline: str
    summary: str = ""
    author: str | None = None
    url: str | None = None
    source: str
    symbols: list[str] = Field(default_factory=list)
    published_at: datetime
    updated_at: datetime | None = None
