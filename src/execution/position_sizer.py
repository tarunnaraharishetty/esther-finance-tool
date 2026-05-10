"""Position sizing rules. Always references Settings — never hardcode."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.config import Settings, get_settings


@dataclass
class FixedFractionSizer:
    """Risk a fixed fraction of equity per position.

    The fraction is read from settings (``max_position_pct``) by default but
    can be overridden per-instance for tighter strategies.
    """

    settings: Settings | None = None
    fraction: float | None = None

    def size(self, equity: Decimal, price: Decimal) -> Decimal:
        s = self.settings or get_settings()
        f = self.fraction if self.fraction is not None else s.max_position_pct
        if price <= 0:
            return Decimal(0)
        notional = equity * Decimal(str(f))
        return (notional / price).quantize(Decimal("0.0001"))
