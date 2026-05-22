"""Sector-median multiples lookup.

The multiple-based valuation models (P/E, EV/EBITDA, P/S, PEG) need
a per-sector benchmark to compare against. Until we have a market-
data pipeline producing live medians, we ship a static seed table at
``config/sector_medians.toml`` and load it lazily.

Sectors come in slightly different shapes from each provider. We
normalize on lookup: lowercase + collapse whitespace, then try the
exact key, then a few common aliases (e.g. "Tech" → "Information
Technology"). Unknown sectors fall through to the ``_default`` block.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.config import get_settings


@dataclass(frozen=True)
class SectorMultiples:
    """Median multiples for one sector."""

    pe: float
    ev_ebitda: float
    ps: float
    peg: float


# Provider sectors are inconsistent — these aliases map common
# variants onto the canonical GICS Level-1 keys in our TOML file.
# Keys are lowercased. Anything not on the list falls through to the
# raw lowercased sector string.
_ALIASES: dict[str, str] = {
    "tech": "information technology",
    "technology": "information technology",
    "infotech": "information technology",
    "it": "information technology",
    "telecom": "communication services",
    "communications": "communication services",
    "consumer cyclical": "consumer discretionary",
    "consumer defensive": "consumer staples",
    "healthcare": "health care",
    "basic materials": "materials",
    "financial services": "financials",
    "financial": "financials",
    "reit": "real estate",
    "reits": "real estate",
}


def normalize_sector_key(sector: str | None) -> str | None:
    """Return the canonical lowercase sector key, or ``None`` when unknown.

    Used by callers that need to group symbols by sector (e.g. the
    sector ranker). The same alias table powers ``lookup()`` so a
    symbol that lands on multiples for "Information Technology" also
    lands on the same cohort key here.

    Returns ``None`` when the input is empty/None; that signals to
    grouping code that the symbol has no usable cohort assignment
    and should be excluded from sector cohorts.
    """
    if not sector:
        return None
    key = sector.strip().lower()
    if not key:
        return None
    return _ALIASES.get(key, key)


def lookup(sector: str | None) -> SectorMultiples:
    """Return the sector-median multiples for ``sector``.

    ``None``, empty, or unknown sectors return the ``_default`` block
    so callers never have to branch on missing data — they can always
    multiply by a number.
    """
    table = _load_table()
    if not sector:
        return table["_default"]
    key = sector.strip().lower()
    if key in table:
        return table[key]
    aliased = _ALIASES.get(key)
    if aliased and aliased in table:
        return table[aliased]
    return table["_default"]


def _load_table() -> dict[str, SectorMultiples]:
    """Lazy-cached loader. Path comes from settings so tests can swap it.

    Stored as a module-level cache keyed by the resolved path string
    so a test that points ``settings.sector_medians_path`` at a tmp
    file picks up the new table immediately.
    """
    path = get_settings().sector_medians_path
    return _load_table_cached(str(path.resolve()))


@lru_cache(maxsize=8)
def _load_table_cached(path_str: str) -> dict[str, SectorMultiples]:
    path = Path(path_str)
    if not path.exists():
        # Defensive — return a default-only table so the analyzer
        # doesn't explode if the config file disappears in production.
        return {"_default": SectorMultiples(pe=20.0, ev_ebitda=13.0, ps=2.5, peg=2.0)}
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    table: dict[str, SectorMultiples] = {}
    for key, block in raw.items():
        if not isinstance(block, dict):
            continue
        try:
            table[key.lower()] = SectorMultiples(
                pe=float(block["pe"]),
                ev_ebitda=float(block["ev_ebitda"]),
                ps=float(block["ps"]),
                peg=float(block["peg"]),
            )
        except (KeyError, TypeError, ValueError):
            # Skip malformed entries rather than crashing the analyzer;
            # the _default fallback keeps the chain usable.
            continue
    if "_default" not in table:
        table["_default"] = SectorMultiples(
            pe=20.0, ev_ebitda=13.0, ps=2.5, peg=2.0
        )
    return table


def _clear_cache() -> None:
    """Test hook — invalidate the lru cache between fixtures."""
    _load_table_cached.cache_clear()


__all__ = ["SectorMultiples", "lookup", "normalize_sector_key"]
