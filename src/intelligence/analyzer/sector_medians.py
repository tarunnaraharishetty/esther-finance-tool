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

import statistics
import threading
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from src.config import get_settings
from src.intelligence.fundamentals.models import NormalizedFundamentals

# In-memory override populated by ``register_computed_medians`` (or
# implicitly by ``refresh_computed_medians``). When non-None, ``lookup``
# consults this table FIRST and falls back to the static TOML for any
# sector not in the override. None means "no live data — use static
# only" (the v1 startup state and the test-isolation baseline).
_computed_overrides_lock = threading.Lock()
_computed_overrides: dict[str, SectorMultiples] | None = None


@dataclass(frozen=True)
class SectorMultiples:
    """Median multiples for one sector.

    ``source`` distinguishes the seed TOML table (``"static"``) from a
    cohort computed at runtime (``"computed"``). ``as_of`` is populated
    for computed entries so the analyzer can carry the freshness label
    forward into reports — static entries leave it ``None`` because the
    TOML file isn't timestamped.
    """

    pe: float
    ev_ebitda: float
    ps: float
    peg: float
    source: str = "static"
    as_of: datetime | None = None
    cohort_size: int = 0


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

    Resolution order:

    1. The in-memory computed overrides (when populated via
       :func:`register_computed_medians` or
       :func:`refresh_computed_medians`). These reflect the live
       watchlist cohort and carry ``source="computed"`` + ``as_of``
       metadata.
    2. The static TOML table (``config/sector_medians.toml``).
    3. The TOML ``_default`` block.

    Step 1 misses fall through to step 2 — a partial override (only
    a few sectors have enough live cohort) still uses the seed for
    every other sector.
    """
    static_table = _load_table()
    if not sector:
        return static_table["_default"]
    key = sector.strip().lower()
    aliased = _ALIASES.get(key, key)
    # Step 1: computed overrides take precedence when available.
    with _computed_overrides_lock:
        overrides = _computed_overrides
    if overrides is not None:
        if aliased in overrides:
            return overrides[aliased]
        if key in overrides:
            return overrides[key]
    # Step 2/3: static seed.
    if aliased in static_table:
        return static_table[aliased]
    if key in static_table:
        return static_table[key]
    return static_table["_default"]


def register_computed_medians(
    table: dict[str, SectorMultiples] | None,
) -> None:
    """Install (or clear) the computed-medians override.

    ``None`` clears the override — ``lookup`` then resolves entirely
    through the static seed. Pass a dict (typically the result of
    :func:`compute_medians_from_cohort`) to make subsequent lookups
    prefer the live cohort.

    Threadsafe — protected by a module-level lock so a worker
    refreshing the table can't be observed mid-write by an in-flight
    ``lookup``.
    """
    global _computed_overrides
    with _computed_overrides_lock:
        # Copy defensively so caller mutations don't leak into the
        # shared override (None passes through as "clear").
        _computed_overrides = None if table is None else dict(table)


def refresh_computed_medians(
    records: Iterable[NormalizedFundamentals],
) -> dict[str, SectorMultiples]:
    """Compute medians from ``records`` and install them as the override.

    Convenience wrapper: run :func:`compute_medians_from_cohort` then
    :func:`register_computed_medians` in one call. Returns the freshly
    installed table so a worker can log cohort sizes or surface the
    ``as_of`` timestamp.
    """
    table = compute_medians_from_cohort(records)
    register_computed_medians(table)
    return table


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


# Minimum cohort size before a computed sector median is considered
# trustworthy. Below this we keep the static seed value rather than
# anchoring valuation against a noisy 2-sample median.
_MIN_COHORT_SIZE = 5


def compute_medians_from_cohort(
    records: Iterable[NormalizedFundamentals],
) -> dict[str, SectorMultiples]:
    """Compute live sector medians from a cohort of normalized records.

    Groups ``records`` by canonical sector key, computes the median of
    available pe / ev_ebitda / ps / peg per sector, and returns a dict
    keyed by canonical sector name. Sectors with fewer than
    ``_MIN_COHORT_SIZE`` non-null samples on a given metric fall back
    to the static seed value for that metric only — partial coverage is
    better than dropping the whole sector.

    The output is wired into :func:`lookup` via
    :func:`register_computed_medians` (or the convenience wrapper
    :func:`refresh_computed_medians`). :class:`SectorCohort` is the
    production driver — it observes every successful fundamentals
    fetch and re-runs this computation on the rolling watchlist.
    """
    # Group records by canonical sector key, dropping anything we can't
    # classify (Yahoo doesn't always tag sector).
    by_sector: dict[str, list[NormalizedFundamentals]] = {}
    for record in records:
        key = normalize_sector_key(record.profile.sector)
        if key is None:
            continue
        by_sector.setdefault(key, []).append(record)

    now = datetime.now(UTC)
    static_table = _load_table()
    out: dict[str, SectorMultiples] = {}
    for sector_key, cohort in by_sector.items():
        pe_samples = _collect_positive(cohort, lambda r: r.key_ratios.pe_ratio)
        eveb_samples = _collect_positive(cohort, lambda r: r.key_ratios.ev_to_ebitda)
        ps_samples = _collect_positive(cohort, lambda r: r.key_ratios.price_to_sales)
        peg_samples = _collect_positive(cohort, lambda r: r.key_ratios.peg_ratio)
        seed = static_table.get(sector_key) or static_table["_default"]
        out[sector_key] = SectorMultiples(
            pe=_median_with_floor(pe_samples, seed.pe),
            ev_ebitda=_median_with_floor(eveb_samples, seed.ev_ebitda),
            ps=_median_with_floor(ps_samples, seed.ps),
            peg=_median_with_floor(peg_samples, seed.peg),
            source="computed",
            as_of=now,
            cohort_size=len(cohort),
        )
    return out


def _collect_positive(
    cohort: list[NormalizedFundamentals],
    pick: object,  # Callable[[NormalizedFundamentals], float | None]
) -> list[float]:
    """Pull a per-record float and keep only positive finite samples.

    Multiples are meaningless when zero or negative (negative P/E =
    no earnings); excluding them yields a cleaner median than mixing
    sentinels into the percentile.
    """
    out: list[float] = []
    for record in cohort:
        value = pick(record)  # type: ignore[operator]
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if f > 0 and f == f and f != float("inf"):  # finite + positive
            out.append(f)
    return out


def _median_with_floor(samples: list[float], seed: float) -> float:
    """Median of ``samples`` when the sample count meets the cohort floor.

    Below the floor we keep the static seed — a 2-sample median is a
    coin flip, not a benchmark.
    """
    if len(samples) >= _MIN_COHORT_SIZE:
        return float(statistics.median(samples))
    return seed


# ---------------------------------------------------------------------------
# SectorCohort — production driver that turns live fetches into computed medians
# ---------------------------------------------------------------------------


class SectorCohort:
    """Bounded per-symbol record cohort driving computed sector medians.

    Each call to :meth:`observe` replaces the cohort's entry for that
    symbol (latest wins) and re-runs
    :func:`refresh_computed_medians` so subsequent :func:`lookup`
    calls reflect the live watchlist instead of the static seed.

    Eviction is FIFO when the cohort grows past ``max_symbols`` —
    a Python dict preserves insertion order, so popping the first key
    drops the oldest observation. ``observe`` re-inserts the observed
    symbol so a frequently-updated ticker stays warm.

    Thread-safe via a single lock around the cohort dict. The
    downstream :func:`register_computed_medians` has its own lock; we
    release ours before calling it so a slow refresh can't block other
    observations.
    """

    def __init__(self, *, max_symbols: int = 500) -> None:
        if max_symbols < 1:
            raise ValueError(
                f"max_symbols must be >= 1, got {max_symbols!r}"
            )
        self._records: dict[str, NormalizedFundamentals] = {}
        self._max_symbols = max_symbols
        self._lock = threading.Lock()

    def observe(self, record: NormalizedFundamentals) -> None:
        """Add (or replace) ``record`` for its symbol; refresh the medians.

        Best-effort with respect to the refresh: a failing refresh is
        the caller's signal that the medians table is stale, not a
        reason to drop the observation. The dict itself is updated
        before the refresh fires.
        """
        symbol = (record.symbol or "").upper()
        if not symbol:
            return
        with self._lock:
            # Pop + re-insert so the observed symbol moves to the
            # "newest" slot — protects active symbols from FIFO
            # eviction when the cohort fills.
            self._records.pop(symbol, None)
            self._records[symbol] = record
            while len(self._records) > self._max_symbols:
                # FIFO eviction — drop the oldest insertion.
                oldest = next(iter(self._records))
                del self._records[oldest]
            snapshot = list(self._records.values())
        # Refresh outside our lock — `register_computed_medians` has
        # its own lock and holding both invites contention.
        refresh_computed_medians(snapshot)

    def snapshot(self) -> list[NormalizedFundamentals]:
        """Return the current cohort. Order is oldest → newest."""
        with self._lock:
            return list(self._records.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


# Re-export for convenience so ``field`` import lint doesn't flag unused.
_ = field

__all__ = [
    "SectorCohort",
    "SectorMultiples",
    "compute_medians_from_cohort",
    "lookup",
    "normalize_sector_key",
    "refresh_computed_medians",
    "register_computed_medians",
]
