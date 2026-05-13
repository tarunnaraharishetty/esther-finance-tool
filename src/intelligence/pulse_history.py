"""Rolling-window history of recent :class:`MarketPulse` readings.

A single pulse answers "what's the watchlist's mood right now?" — useful
but static. Pulse history adds the time dimension: is momentum breadth
building or fading? Is sentiment breadth firming? Did the alert count
just spike?

The dashboard's HIST line renders ASCII sparklines from this data so
the trader can see the trajectory at a glance without scrolling
through the events log.

Pure data. The controller owns one :class:`PulseHistoryTracker` per
session; each :class:`~src.dashboard.state.DashboardSnapshot` carries
an immutable :class:`PulseHistory` reflecting the rolling window.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.intelligence.pulse import MarketPulse

if TYPE_CHECKING:
    from src.persistence.session_store import PulseRecord


@dataclass(frozen=True)
class PulseHistory:
    """Immutable snapshot of the recent pulse window.

    All series are oldest-first; the rightmost element is the most
    recent pulse. ``window`` is the configured cap; ``len(series)``
    may be smaller early in a session when the buffer is still filling.

    Categorical fields (sentiment / conviction / activity) are kept as
    tuples for future use (e.g. recap prose) but the dashboard
    sparkline path consumes only the numeric ones below.
    """

    momentum_breadth: tuple[float, ...]
    sentiment_breadth: tuple[float, ...]
    bullish_count: tuple[int, ...]
    bearish_count: tuple[int, ...]
    reversal_intensity: tuple[int, ...]
    alert_intensity: tuple[int, ...]
    sentiment: tuple[str, ...]
    conviction: tuple[str, ...]
    activity: tuple[str, ...]
    window: int

    @property
    def length(self) -> int:
        """Number of pulses currently held — at most ``window``."""
        return len(self.momentum_breadth)

    @property
    def has_trend(self) -> bool:
        """True when there are at least 2 readings — i.e. a sparkline
        renders something other than a single dot."""
        return self.length >= 2


class PulseHistoryTracker:
    """Rolling-window tracker that captures one :class:`MarketPulse`
    reading per tick.

    Skips empty pulses (no healthy rows yet) so the first useful tick
    becomes the buffer's first entry — otherwise the sparkline opens
    with a meaningless zero.
    """

    def __init__(self, window: int = 20) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self._buf: deque[MarketPulse] = deque(maxlen=window)

    def record(self, pulse: MarketPulse) -> None:
        """Append one pulse reading. Empty pulses are skipped so the
        sparkline starts at the first meaningful frame."""
        if pulse.is_empty:
            return
        self._buf.append(pulse)

    def summary(self) -> PulseHistory:
        """Project the rolling buffer into a frozen :class:`PulseHistory`."""
        pulses = tuple(self._buf)
        return PulseHistory(
            momentum_breadth=tuple(p.momentum_breadth for p in pulses),
            sentiment_breadth=tuple(p.sentiment_breadth for p in pulses),
            bullish_count=tuple(p.bullish_count for p in pulses),
            bearish_count=tuple(p.bearish_count for p in pulses),
            reversal_intensity=tuple(p.reversal_intensity for p in pulses),
            alert_intensity=tuple(p.alert_intensity for p in pulses),
            sentiment=tuple(p.sentiment for p in pulses),
            conviction=tuple(p.conviction for p in pulses),
            activity=tuple(p.activity for p in pulses),
            window=self.window,
        )

    def __len__(self) -> int:
        return len(self._buf)

    # -- persistence -------------------------------------------------------

    def to_snapshot(self) -> list[PulseRecord]:
        """Return each buffered pulse as a serialization-friendly record."""
        from src.persistence.session_store import PulseRecord

        return [
            PulseRecord(
                sentiment=p.sentiment,
                conviction=p.conviction,
                activity=p.activity,
                summary=p.summary,
                bullish_count=p.bullish_count,
                bearish_count=p.bearish_count,
                healthy_count=p.healthy_count,
                momentum_breadth=p.momentum_breadth,
                sentiment_breadth=p.sentiment_breadth,
                reversal_intensity=p.reversal_intensity,
                alert_intensity=p.alert_intensity,
                strongest_symbols=[list(t) for t in p.strongest_symbols],  # type: ignore[misc]
            )
            for p in self._buf
        ]

    def apply_snapshot(self, records: list[PulseRecord]) -> None:
        """Rehydrate the rolling buffer from ``records``.

        Records beyond the current ``self.window`` are dropped
        (oldest first) — restored sessions respect the current window
        even if the prior run had a larger one. Empty pulses are kept
        as-is here (unlike :meth:`record`) because the snapshot only
        contains non-empty pulses by construction.
        """
        trimmed = records[-self.window :]
        new_buf: deque[MarketPulse] = deque(maxlen=self.window)
        for rec in trimmed:
            new_buf.append(
                MarketPulse(
                    sentiment=rec.sentiment,
                    conviction=rec.conviction,
                    activity=rec.activity,
                    summary=rec.summary,
                    bullish_count=rec.bullish_count,
                    bearish_count=rec.bearish_count,
                    healthy_count=rec.healthy_count,
                    momentum_breadth=rec.momentum_breadth,
                    sentiment_breadth=rec.sentiment_breadth,
                    reversal_intensity=rec.reversal_intensity,
                    alert_intensity=rec.alert_intensity,
                    strongest_symbols=tuple(
                        (sym, display) for sym, display in rec.strongest_symbols
                    ),
                )
            )
        self._buf = new_buf


__all__ = ["PulseHistory", "PulseHistoryTracker"]
