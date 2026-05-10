"""Translates Signals into bracketed orders, applying risk checks first."""

from __future__ import annotations

from decimal import Decimal

from src.config import Settings, get_settings
from src.execution.broker import (
    Broker,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    TimeInForce,
)
from src.execution.position_sizer import FixedFractionSizer
from src.risk.exposure import RiskGate
from src.strategy.base import Signal, SignalAction
from src.utils.logging import get_logger

log = get_logger(__name__)


class OrderManager:
    """Owns the path from Signal -> risk-checked bracket OrderRequest -> Broker."""

    def __init__(
        self,
        broker: Broker,
        risk_gate: RiskGate,
        sizer: FixedFractionSizer | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.broker = broker
        self.risk_gate = risk_gate
        self.settings = settings or get_settings()
        self.sizer = sizer or FixedFractionSizer(settings=self.settings)

    async def submit_signal(self, signal: Signal, last_price: Decimal) -> OrderResult | None:
        if signal.action == SignalAction.HOLD:
            return None

        equity = await self.broker.get_account_equity()
        gate_result = self.risk_gate.check(signal=signal, equity=equity, price=last_price)
        if not gate_result.allowed:
            log.warning(
                "order.blocked_by_risk",
                symbol=signal.symbol,
                reason=gate_result.reason,
            )
            return None

        qty = self.sizer.size(equity=equity, price=last_price)
        if qty <= 0:
            return None

        side = OrderSide.BUY if signal.action == SignalAction.BUY else OrderSide.SELL
        sl_pct = Decimal(str(self.settings.default_stop_loss_pct))
        tp_pct = Decimal(str(self.settings.default_take_profit_pct))
        if side == OrderSide.BUY:
            stop_loss = last_price * (Decimal(1) - sl_pct)
            take_profit = last_price * (Decimal(1) + tp_pct)
        else:
            stop_loss = last_price * (Decimal(1) + sl_pct)
            take_profit = last_price * (Decimal(1) - tp_pct)

        order = OrderRequest(
            symbol=signal.symbol,
            qty=qty,
            side=side,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            stop_loss=stop_loss.quantize(Decimal("0.01")),
            take_profit=take_profit.quantize(Decimal("0.01")),
        )
        log.info("order.submit", symbol=signal.symbol, side=side.value, qty=str(qty))
        return await self.broker.submit(order)
