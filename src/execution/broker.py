"""Broker abstraction. Currently Alpaca-backed; designed for swappability."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from src.data.alpaca_client import AlpacaClient
from src.utils.logging import get_logger

log = get_logger(__name__)


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    qty: Decimal
    side: OrderSide
    type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    take_profit: Decimal | None = None
    stop_loss: Decimal | None = None
    client_order_id: str | None = None


@dataclass(frozen=True)
class OrderResult:
    id: str
    status: str
    filled_qty: Decimal
    filled_avg_price: Decimal | None


class Broker(ABC):
    @abstractmethod
    async def submit(self, order: OrderRequest) -> OrderResult: ...

    @abstractmethod
    async def cancel(self, order_id: str) -> None: ...

    @abstractmethod
    async def get_account_equity(self) -> Decimal: ...


class AlpacaBroker(Broker):
    def __init__(self, client: AlpacaClient | None = None) -> None:
        self.client = client or AlpacaClient()

    async def submit(self, order: OrderRequest) -> OrderResult:
        await self.client.acquire_quota()
        raise NotImplementedError("wire up alpaca-py TradingClient.submit_order")

    async def cancel(self, order_id: str) -> None:
        await self.client.acquire_quota()
        raise NotImplementedError

    async def get_account_equity(self) -> Decimal:
        await self.client.acquire_quota()
        raise NotImplementedError
