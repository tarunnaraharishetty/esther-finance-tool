"""Watchlist Driver Spotlight — global rank across per-symbol drivers.

Takes the :class:`~src.intelligence.movement_drivers.KeyDrivers` dict
the spotlight endpoint assembled for every watchlist symbol, flattens
the drivers across symbols, and surfaces the top N by salience.

This is the fleet-scanning layer over the per-symbol synthesis the
Key Drivers panel ships. A trader's morning workflow becomes "open
Dashboard → see the loudest signals across my whole watchlist" rather
than clicking through every symbol.

Per-symbol cap
--------------
Without a cap, one noisy symbol (lots of high-salience drivers) would
push everything else off the list. ``max_per_symbol`` defaults to 1 —
one driver per symbol, the symbol's loudest one. Set to 2-3 if the
caller wants richer per-symbol context at the cost of less breadth.

What it isn't
-------------
Not a recommendation engine. The spotlight surfaces *what's notable*
across the watchlist; the trader decides whether notable = buy, sell,
or watch. Same framing discipline the Key Drivers panel uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.intelligence.movement_drivers import Driver, KeyDrivers

_DEFAULT_MAX_ITEMS = 8
_DEFAULT_MAX_PER_SYMBOL = 1


@dataclass(frozen=True)
class SpotlightEntry:
    """One driver promoted to the watchlist-level spotlight.

    ``rank`` is 1-indexed across the global spotlight list — useful
    for the UI to render position chips ("#1 of 8").
    """

    rank: int
    symbol: str
    driver: Driver

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "symbol": self.symbol,
            "driver": self.driver.to_dict(),
        }


@dataclass(frozen=True)
class SpotlightView:
    """Aggregate spotlight bundle for the watchlist.

    ``considered_symbols`` is the total number of symbols offered to
    the composer (i.e. watchlist size after dropping assembly failures).
    ``surfaced_symbols`` is how many actually contributed at least one
    spotlight entry (= ``len({e.symbol for e in entries})``).
    """

    entries: tuple[SpotlightEntry, ...]
    considered_symbols: int
    surfaced_symbols: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "considered_symbols": self.considered_symbols,
            "surfaced_symbols": self.surfaced_symbols,
        }


def compute_spotlight(
    symbol_drivers: dict[str, KeyDrivers],
    *,
    max_items: int = _DEFAULT_MAX_ITEMS,
    max_per_symbol: int = _DEFAULT_MAX_PER_SYMBOL,
) -> SpotlightView:
    """Rank drivers across symbols, applying per-symbol + total caps.

    Args:
        symbol_drivers: Per-symbol :class:`KeyDrivers` produced by
            the upstream assembly step. Symbols whose key drivers
            tuple is empty contribute nothing — they're counted in
            ``considered_symbols`` but not in ``surfaced_symbols``.
        max_items: Total spotlight list cap.
        max_per_symbol: Per-symbol entry cap. Prevents a single symbol
            from monopolizing the spotlight when it has multiple
            high-salience drivers.
    """
    if max_items < 0 or max_per_symbol < 0:
        raise ValueError(
            f"max_items and max_per_symbol must be non-negative; got "
            f"max_items={max_items}, max_per_symbol={max_per_symbol}"
        )

    # Flatten: every (symbol, driver) pair from every KeyDrivers
    # bundle. Preserves per-symbol driver ordering as a tie-break
    # signal (the driver ranker already sorted within each symbol).
    flat: list[tuple[str, int, Driver]] = []
    for symbol, kd in symbol_drivers.items():
        for in_symbol_index, driver in enumerate(kd.drivers):
            flat.append((symbol.upper(), in_symbol_index, driver))

    # Global sort: salience desc, then alphabetic symbol, then
    # within-symbol index. The two tie-breaks keep the order
    # deterministic (important for the file cache).
    flat.sort(key=lambda item: (-item[2].salience, item[0], item[1]))

    # Greedy fill respecting the per-symbol cap. Stop when the total
    # cap is hit.
    per_symbol_count: dict[str, int] = {}
    chosen: list[tuple[str, Driver]] = []
    for symbol, _idx, driver in flat:
        if len(chosen) >= max_items:
            break
        if per_symbol_count.get(symbol, 0) >= max_per_symbol:
            continue
        chosen.append((symbol, driver))
        per_symbol_count[symbol] = per_symbol_count.get(symbol, 0) + 1

    entries = tuple(
        SpotlightEntry(rank=i + 1, symbol=symbol, driver=driver)
        for i, (symbol, driver) in enumerate(chosen)
    )
    surfaced = len({e.symbol for e in entries})
    return SpotlightView(
        entries=entries,
        considered_symbols=len(symbol_drivers),
        surfaced_symbols=surfaced,
    )


__all__ = [
    "SpotlightEntry",
    "SpotlightView",
    "compute_spotlight",
]
