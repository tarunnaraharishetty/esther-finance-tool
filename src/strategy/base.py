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


class RecommendationTier(StrEnum):
    """Five-tier directional label surfaced to the trader.

    Strict refinement of :class:`SignalAction`: every tier maps back to
    exactly one base action. ``STRONG_*`` tiers only appear when the
    promoter in :mod:`src.intelligence.tier` confirms multi-system
    alignment + stability + high confidence — see that module for the
    gating rules.

    Kept as a separate enum from ``SignalAction`` so existing code
    paths (alerts, pulse, opportunities, rankings, history) that switch
    on the base 3-tier action keep working unchanged. The tier is
    additive: render in the UI, never required by core logic.
    """

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

    @classmethod
    def from_action(cls, action: SignalAction) -> RecommendationTier:
        """Default tier for a base action — no promotion."""
        return {
            SignalAction.BUY: cls.BUY,
            SignalAction.SELL: cls.SELL,
            SignalAction.HOLD: cls.HOLD,
        }[action]

    @property
    def is_strong(self) -> bool:
        return self in (RecommendationTier.STRONG_BUY, RecommendationTier.STRONG_SELL)

    @property
    def direction(self) -> int:
        """+1 bullish, -1 bearish, 0 neutral."""
        if self in (RecommendationTier.STRONG_BUY, RecommendationTier.BUY):
            return 1
        if self in (RecommendationTier.STRONG_SELL, RecommendationTier.SELL):
            return -1
        return 0

    @property
    def display(self) -> str:
        """Human-friendly label ('STRONG BUY' rather than 'strong_buy')."""
        return self.value.replace("_", " ").upper()


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
