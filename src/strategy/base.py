"""Base classes for trading strategies and signals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class SignalAction(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class Signal:
    """A normalized trading signal emitted by a Strategy.

    ``confidence`` is in [0, 1]. ``size_hint`` is an optional fractional
    position size; the execution layer may clip per risk rules.
    """

    symbol: str
    action: SignalAction
    confidence: float
    timestamp: datetime
    source: str
    size_hint: float | None = None
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Strategies consume market data + sentiment and emit Signals."""

    name: str = "strategy"

    @abstractmethod
    def generate(self, df: "pd.DataFrame", **context: Any) -> list[Signal]:
        """Return zero or more signals for the latest bar in ``df``."""
