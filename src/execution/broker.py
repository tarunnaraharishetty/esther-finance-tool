"""Broker abstraction. Currently Alpaca-backed; designed for swappability."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from src.data.alpaca_client import AlpacaClient
from src.utils.logging import get_logger
from src.utils.rate_limiter import with_async_retry

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
    """Alpaca-py backed broker. Submits bracket orders when stop_loss/take_profit set."""

    def __init__(self, client: AlpacaClient | None = None) -> None:
        self.client = client or AlpacaClient()

    @with_async_retry(attempts=3, min_wait=1.0, max_wait=10.0)
    async def submit(self, order: OrderRequest) -> OrderResult:
        await self.client.acquire_quota()
        request = self._build_request(order)
        raw = await asyncio.to_thread(self.client.trading.submit_order, request)
        return _to_result(raw)

    @with_async_retry(attempts=3, min_wait=0.5, max_wait=5.0)
    async def cancel(self, order_id: str) -> None:
        await self.client.acquire_quota()
        await asyncio.to_thread(self.client.trading.cancel_order_by_id, order_id)

    @with_async_retry(attempts=3, min_wait=0.5, max_wait=5.0)
    async def get_account_equity(self) -> Decimal:
        await self.client.acquire_quota()
        account = await asyncio.to_thread(self.client.trading.get_account)
        equity_str = getattr(account, "equity", None) or getattr(account, "portfolio_value")
        return Decimal(str(equity_str))

    def _build_request(self, order: OrderRequest) -> object:
        from alpaca.trading.enums import (
            OrderClass,
            OrderSide as AlpacaSide,
            TimeInForce as AlpacaTIF,
        )
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        side = AlpacaSide(order.side.value)
        tif = AlpacaTIF(order.time_in_force.value)

        bracket_kwargs: dict[str, object] = {}
        if order.take_profit is not None and order.stop_loss is not None:
            bracket_kwargs = {
                "order_class": OrderClass.BRACKET,
                "take_profit": TakeProfitRequest(limit_price=float(order.take_profit)),
                "stop_loss": StopLossRequest(stop_price=float(order.stop_loss)),
            }

        common = dict(
            symbol=order.symbol,
            qty=float(order.qty),
            side=side,
            time_in_force=tif,
            client_order_id=order.client_order_id,
            **bracket_kwargs,
        )

        if order.type == OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("limit_price required for LIMIT orders")
            return LimitOrderRequest(limit_price=float(order.limit_price), **common)
        if order.type == OrderType.MARKET:
            return MarketOrderRequest(**common)
        raise NotImplementedError(f"order type {order.type} not yet supported")


def _to_result(raw: object) -> OrderResult:
    filled_qty_raw = getattr(raw, "filled_qty", 0) or 0
    filled_avg_raw = getattr(raw, "filled_avg_price", None)
    return OrderResult(
        id=str(getattr(raw, "id")),
        status=str(getattr(raw, "status")),
        filled_qty=Decimal(str(filled_qty_raw)),
        filled_avg_price=Decimal(str(filled_avg_raw)) if filled_avg_raw is not None else None,
    )
