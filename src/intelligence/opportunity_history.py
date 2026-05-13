"""Per-symbol rolling-window history of top-N opportunity membership.

The dashboard renders a ranked OPP list each tick. Without history,
every line looks the same — there's no way to tell a flash-in-the-pan
setup (one tick in top-N then gone) from a durable one (sticky across
many ticks). This module gives each top-N symbol a small rolling
record of its membership over the last ``window`` ticks so the
renderer can surface a "NEW" / "Nx" badge.

Pure data. The controller owns one
:class:`OpportunityMembershipTracker` per session; each
:class:`~src.dashboard.state.DashboardSnapshot` carries an immutable
:class:`OpportunityHistory` per currently-ranked symbol.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class OpportunityHistory:
    """One symbol's top-N membership history within a fixed window.

    All counts are non-negative; ``streak`` is always at most
    ``appearances`` is at most ``window``.

    streak — consecutive ticks (ending this tick) the symbol was in
        top-N. 0 if not currently in top-N. 1 the tick it appears
        after an absence; bumps each consecutive tick of presence.

    appearances — total ticks the symbol was in top-N within the
        window. Useful for distinguishing "returned after a long
        gap" (streak=1, appearances=1) from "in and out a lot"
        (streak=1, appearances>=3).

    window — size of the rolling window the counts are computed over.
    """

    streak: int
    appearances: int
    window: int


class OpportunityMembershipTracker:
    """Rolling-window tracker of top-N opportunity membership per symbol.

    The controller calls :meth:`record` once per tick with the set of
    symbols currently in top-N. Symbols whose buffer is fully False
    (absent for the entire window) are pruned to keep the tracker's
    memory footprint bounded by the watchlist size.

    First-tick behavior: a symbol present in top-N on the very first
    call gets ``streak=1, appearances=1`` — i.e. "NEW". That's
    intentional: on a fresh dashboard launch every top-N symbol is by
    definition new to the trader's view.
    """

    def __init__(self, window: int = 10) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self._membership: dict[str, deque[bool]] = {}

    def record(self, top_symbols: Iterable[str]) -> None:
        """Append one tick's worth of presence/absence for tracked symbols.

        New symbols (in ``top_symbols`` but not yet tracked) start a
        fresh buffer with True. Existing symbols absent from
        ``top_symbols`` get a False appended; if their buffer becomes
        fully False, they're pruned.
        """
        top_set = set(top_symbols)
        # Update existing tracked symbols (presence or absence).
        for symbol in list(self._membership):
            buf = self._membership[symbol]
            buf.append(symbol in top_set)
            # Prune symbols absent for the entire window.
            if not any(buf):
                del self._membership[symbol]
        # Initialize newly-appearing symbols.
        for symbol in top_set:
            if symbol not in self._membership:
                buf = deque(maxlen=self.window)
                buf.append(True)
                self._membership[symbol] = buf

    def summary_for(self, symbol: str) -> OpportunityHistory | None:
        """Per-symbol summary, or ``None`` if the symbol isn't tracked."""
        buf = self._membership.get(symbol)
        if buf is None:
            return None
        # Streak = trailing run of True values ending at the most-recent entry.
        streak = 0
        for present in reversed(buf):
            if not present:
                break
            streak += 1
        appearances = sum(buf)
        return OpportunityHistory(
            streak=streak,
            appearances=appearances,
            window=self.window,
        )

    def tracked_symbols(self) -> tuple[str, ...]:
        """All symbols currently tracked (have non-fully-absent history)."""
        return tuple(self._membership)


__all__ = ["OpportunityHistory", "OpportunityMembershipTracker"]
