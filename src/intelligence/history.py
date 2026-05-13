"""Per-symbol signal history for the dashboard.

Tracks runs of consecutive same-action recommendations ("episodes") as
they unfold tick-by-tick. The dashboard uses this to answer:

- What is the current recommendation, and how long has it held?
- What was the prior recommendation before it flipped?
- Is confidence rising or falling within the current run?

In-memory only — a fresh ``SignalHistory`` is created per dashboard
session. Episodes are bounded per symbol so a long-running session
doesn't grow unboundedly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime

from src.strategy.base import SignalAction

# Confidence delta below this counts as "flat" within an episode.
_TREND_FLAT_EPS = 0.05

# Per-symbol cap on how many episodes we retain (oldest dropped first).
DEFAULT_MAX_EPISODES = 10


@dataclass(frozen=True)
class SignalEpisode:
    """A continuous run of the same recommended action for one symbol."""

    action: SignalAction
    started_at: datetime
    last_seen_at: datetime
    tick_count: int
    confidence_first: float
    confidence_last: float

    @property
    def confidence_trend(self) -> str:
        """``"rising"`` / ``"falling"`` / ``"flat"`` over the episode."""
        delta = self.confidence_last - self.confidence_first
        if abs(delta) < _TREND_FLAT_EPS:
            return "flat"
        return "rising" if delta > 0 else "falling"


@dataclass(frozen=True)
class SignalHistorySummary:
    """The frozen view of one symbol's history that the dashboard renders.

    ``recent`` excludes the current episode; the dashboard treats them
    separately so the "Now / Was" framing is unambiguous.
    """

    current: SignalEpisode
    recent: tuple[SignalEpisode, ...]  # most-recent-first, capped


class SignalHistory:
    """Append-only per-symbol log of episodes.

    ``record()`` is called once per tick per healthy row. Same-action ticks
    extend the current episode; different-action ticks close it and start a
    new one. Error rows should NOT be recorded — they don't represent a
    real signal change, and recording them would muddy the duration count.
    """

    def __init__(self, max_episodes_per_symbol: int = DEFAULT_MAX_EPISODES) -> None:
        self._by_symbol: dict[str, list[SignalEpisode]] = defaultdict(list)
        self._max = max_episodes_per_symbol

    def record(
        self,
        symbol: str,
        action: SignalAction,
        confidence: float,
        when: datetime,
    ) -> None:
        """Add one observation. Either extends the current episode or
        starts a new one."""
        episodes = self._by_symbol[symbol]
        if episodes and episodes[-1].action == action:
            # Extend the current run.
            current = episodes[-1]
            episodes[-1] = replace(
                current,
                last_seen_at=when,
                tick_count=current.tick_count + 1,
                confidence_last=confidence,
            )
        else:
            # New episode (first ever or action flipped).
            episodes.append(
                SignalEpisode(
                    action=action,
                    started_at=when,
                    last_seen_at=when,
                    tick_count=1,
                    confidence_first=confidence,
                    confidence_last=confidence,
                )
            )
            # Bound the history — drop oldest if we exceed the cap.
            if len(episodes) > self._max:
                del episodes[: len(episodes) - self._max]

    def episodes_for(self, symbol: str) -> list[SignalEpisode]:
        return list(self._by_symbol.get(symbol, ()))

    def summary_for(self, symbol: str, *, recent: int = 4) -> SignalHistorySummary | None:
        """Snapshot of the current + last few episodes for ``symbol``.

        Returns ``None`` if no observations have been recorded yet.
        """
        episodes = self._by_symbol.get(symbol)
        if not episodes:
            return None
        current = episodes[-1]
        # Most-recent-first, excluding the current episode.
        prior = episodes[:-1][-recent:][::-1]
        return SignalHistorySummary(current=current, recent=tuple(prior))


__all__ = [
    "DEFAULT_MAX_EPISODES",
    "SignalEpisode",
    "SignalHistory",
    "SignalHistorySummary",
]
