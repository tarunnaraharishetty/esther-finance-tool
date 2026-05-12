"""Watchlist-level intelligence: diffs between snapshots, top movers, etc.

Decision-support helpers that operate on a whole
:class:`~src.dashboard.state.DashboardSnapshot` (or list of rows) rather
than a single symbol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.strategy.base import SignalAction

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot, RecommendationRow


# Same threshold the explain layer uses for confidence-jump classification.
_CONFIDENCE_JUMP = 0.2


@dataclass(frozen=True)
class WatchlistChange:
    """A noteworthy change for one symbol between two snapshots."""

    symbol: str
    kind: str  # "action_changed" | "confidence_jump" | "new" | "removed"
    description: str


def diff_snapshots(
    current: DashboardSnapshot,
    previous: DashboardSnapshot | None,
) -> list[WatchlistChange]:
    """List notable changes from ``previous`` to ``current``.

    Returns an empty list when ``previous`` is None (first snapshot).
    """
    if previous is None:
        return []

    prev_by_sym = {r.symbol: r for r in previous.rows}
    curr_by_sym = {r.symbol: r for r in current.rows}
    changes: list[WatchlistChange] = []

    for sym, row in curr_by_sym.items():
        prev = prev_by_sym.get(sym)
        if prev is None:
            changes.append(
                WatchlistChange(symbol=sym, kind="new", description=f"{sym} added to watchlist")
            )
            continue
        if row.error or prev.error:
            continue
        if row.action != prev.action:
            changes.append(
                WatchlistChange(
                    symbol=sym,
                    kind="action_changed",
                    description=(
                        f"{sym}: {prev.action.value.upper()} → {row.action.value.upper()}"
                    ),
                )
            )
            continue  # don't double-fire confidence-jump for the same row
        if abs(row.confidence - prev.confidence) >= _CONFIDENCE_JUMP:
            direction = "up" if row.confidence > prev.confidence else "down"
            changes.append(
                WatchlistChange(
                    symbol=sym,
                    kind="confidence_jump",
                    description=(
                        f"{sym}: confidence {direction} "
                        f"{prev.confidence:.2f} → {row.confidence:.2f}"
                    ),
                )
            )

    for sym in prev_by_sym.keys() - curr_by_sym.keys():
        changes.append(
            WatchlistChange(symbol=sym, kind="removed", description=f"{sym} removed from watchlist")
        )

    return changes


def top_movers(
    snapshot: DashboardSnapshot,
    *,
    n: int = 3,
) -> list[RecommendationRow]:
    """Top ``n`` rows ranked by absolute combined score (strongest signal first).

    Rows with errors are excluded — they can't be ranked meaningfully.
    """
    healthy = [r for r in snapshot.rows if not r.error]
    return sorted(healthy, key=lambda r: abs(r.combined_score), reverse=True)[:n]


def action_breakdown(snapshot: DashboardSnapshot) -> dict[SignalAction, int]:
    """Count rows by recommended action. Error rows count as HOLD."""
    counts: dict[SignalAction, int] = {
        SignalAction.BUY: 0,
        SignalAction.HOLD: 0,
        SignalAction.SELL: 0,
    }
    for row in snapshot.rows:
        action = SignalAction.HOLD if row.error else row.action
        counts[action] += 1
    return counts
