"""Pre-trade risk gate. Every order must pass through this."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.config import Settings, get_settings
from src.strategy.base import Signal


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason: str = ""


@dataclass
class RiskGate:
    """Synchronous, deterministic pre-trade checks.

    Plug into :class:`OrderManager`. Extend with daily PnL / drawdown / open
    position counts as those state stores come online.
    """

    settings: Settings | None = None

    def check(self, *, signal: Signal, equity: Decimal, price: Decimal) -> GateResult:
        s = self.settings or get_settings()

        if equity <= 0:
            return GateResult(False, "non-positive equity")
        if price <= 0:
            return GateResult(False, "non-positive price")

        notional = equity * Decimal(str(s.max_position_pct))
        if notional <= 0:
            return GateResult(False, "max_position_pct yields zero notional")

        # TODO: integrate daily-drawdown check + open-position cap once
        # PortfolioState is wired in.
        return GateResult(True)
