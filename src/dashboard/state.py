"""Pure-data structures the dashboard UI renders.

No Textual / Rich imports here — keeping these decoupled lets the
controller layer be unit-tested without spinning up a TUI.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from src.strategy.base import SignalAction


@dataclass(frozen=True)
class RecommendationRow:
    """One symbol's worth of dashboard data for the watchlist table."""

    symbol: str
    action: SignalAction
    confidence: float  # [0, 1]
    combined_score: float  # [-1, 1]
    technical_score: float
    sentiment_score: float
    rsi: float
    macd: float
    bollinger: float
    last_price: float
    num_news_articles: int
    reasoning: str
    timestamp: datetime
    error: str | None = None  # populated if the per-symbol fetch failed


@dataclass(frozen=True)
class AccountSnapshot:
    """Account-side panel data."""

    equity: Decimal
    paper_trading: bool
    timestamp: datetime
    note: str = ""


@dataclass(frozen=True)
class EventEntry:
    """A single line in the events / log panel."""

    timestamp: datetime
    level: str  # "info" | "warn" | "error"
    message: str


@dataclass
class DashboardSnapshot:
    """One frame of dashboard state, produced by the controller each tick."""

    tick: int
    rows: list[RecommendationRow]
    account: AccountSnapshot | None
    events: list[EventEntry] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class EventBuffer:
    """Bounded FIFO of EventEntry — used by the controller to accumulate log lines."""

    def __init__(self, capacity: int = 200) -> None:
        self._buf: deque[EventEntry] = deque(maxlen=capacity)

    def push(self, level: str, message: str) -> EventEntry:
        ev = EventEntry(timestamp=datetime.now(UTC), level=level, message=message)
        self._buf.append(ev)
        return ev

    def info(self, message: str) -> EventEntry:
        return self.push("info", message)

    def warn(self, message: str) -> EventEntry:
        return self.push("warn", message)

    def error(self, message: str) -> EventEntry:
        return self.push("error", message)

    def snapshot(self) -> list[EventEntry]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)
