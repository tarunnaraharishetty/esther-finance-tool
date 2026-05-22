"""Sector-relative ranking endpoint.

``GET /api/sector/{symbol}/rank`` returns the requested symbol's
rank + percentile on every metric within its sector cohort.

Cohort assembly
---------------
The cohort is every watchlist symbol that normalizes onto the same
sector key (per :mod:`src.intelligence.analyzer.sector_medians`).
We assemble each cohort member's analyzer report via the shared
``app.state.analyzer_assembler`` — same pipeline the
``/api/analyzer/{symbol}`` route uses, so cohort members' values
agree exactly with what the user sees on the analyzer tab.

Cache strategy
--------------
Per-sector file cache with a 1-hour TTL. Sectors move slowly; a 1h
window is long enough to absorb intraday cache churn and short
enough that a fresh fundamentals release surfaces the same day.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

from src.intelligence.analyzer.sector_medians import normalize_sector_key
from src.intelligence.sector_ranker import (
    compute_sector_rankings,
    empty_ranking,
)
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController

log = get_logger(__name__)


_Assembler = Callable[[str, bool], Awaitable[dict[str, Any]]]


_CACHE_TTL_SECONDS = 3600  # 1 hour


def register_sector_routes(
    app: FastAPI,
    controller: BaseController,
    *,
    cache_dir: Path | None = None,
) -> None:
    """Attach ``/api/sector/{symbol}/rank`` to ``app``.

    Requires :func:`register_analyzer_routes` to have run first — the
    sector ranker reads the analyzer assembler off ``app.state``. We
    503 with a clear message rather than silently mis-routing if the
    assembler isn't there.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/api/sector/{symbol}/rank")
    async def sector_rank(symbol: str, fresh: bool = False) -> dict[str, Any]:
        """Return the requested symbol's sector ranking + cohort summary.

        Args:
            symbol: Ticker. Case-insensitive on input.
            fresh: When ``True``, bypasses both the sector cache and
                the underlying analyzer caches (forces a fresh cohort
                assembly).
        """
        sym = symbol.upper()
        assembler = _resolve_assembler(app)

        # Step 1: get the symbol's own analyzer report (we need its
        # sector key to define the cohort).
        try:
            primary_report = await assembler(sym, fresh)
        except Exception as exc:
            log.warning(
                "sector.primary_assemble.failed", symbol=sym, error=str(exc)
            )
            raise HTTPException(
                status_code=503,
                detail=f"Could not assemble analyzer report for {sym}: {exc}",
            ) from None

        sector_raw = _extract_sector(primary_report)
        sector_key = normalize_sector_key(sector_raw)
        if sector_key is None:
            log.info("sector.rank.no_sector", symbol=sym)
            return _wire_empty_response(
                sym, sector_raw, reason="no sector classification available"
            )

        # Step 2: find cohort members from the watchlist.
        watchlist_symbols = [s.upper() for s in controller.watchlist]
        candidate_peers = [s for s in watchlist_symbols if s != sym]
        if not candidate_peers:
            return _wire_empty_response(
                sym, sector_raw, reason="watchlist has no other symbols"
            )

        # Step 3: cache lookup. Cache is keyed on the cohort fingerprint
        # so a watchlist change naturally invalidates.
        cache_file: Path | None = None
        if cache_dir is not None and not fresh:
            cache_file = cache_dir / _cache_key(sector_key, watchlist_symbols)
            cached = _read_cache(cache_file)
            if cached is not None:
                ranking = cached.get("rankings", {}).get(sym)
                if ranking is not None:
                    return _wire_response(
                        sym=sym,
                        sector_raw=sector_raw,
                        ranking_dict=ranking,
                        cache="hit",
                    )
            # Cache existed but our symbol wasn't in it (likely the
            # watchlist changed since the cache was written). Fall
            # through to a fresh assembly.

        # Step 4: parallel assemble cohort members + run the ranker.
        log.info(
            "sector.rank.assemble.start",
            symbol=sym,
            sector=sector_key,
            cohort_candidates=len(candidate_peers),
        )
        peer_reports = await _assemble_cohort(
            assembler, candidate_peers, fresh
        )
        all_reports = [primary_report, *peer_reports]

        rankings = compute_sector_rankings(all_reports)
        target = rankings.get(sym)

        if target is None or target.cohort_size < 2:
            log.info(
                "sector.rank.cohort_too_small",
                symbol=sym,
                sector=sector_key,
                size=target.cohort_size if target else 0,
            )
            response = _wire_empty_response(
                sym,
                sector_raw,
                reason=(
                    "no watchlist peers share this sector "
                    f"({sector_raw or sector_key})"
                ),
            )
        else:
            response = _wire_response(
                sym=sym,
                sector_raw=sector_raw,
                ranking_dict=target.to_dict(),
                cache="miss",
            )

        # Step 5: persist the full per-sector ranking dict so siblings
        # benefit from this cohort assembly on their next lookup.
        if cache_file is not None:
            wire_rankings = {
                k: v.to_dict() for k, v in rankings.items()
            }
            persistable = {
                "sector_key": sector_key,
                "rankings": wire_rankings,
            }
            try:
                cache_file.write_text(
                    json.dumps(jsonable_encoder(persistable)),
                    encoding="utf-8",
                )
            except OSError as exc:
                log.warning(
                    "sector.cache.write_failed",
                    error=str(exc),
                    path=str(cache_file),
                )
        log.info(
            "sector.rank.done",
            symbol=sym,
            sector=sector_key,
            cohort_size=target.cohort_size if target else 0,
        )
        return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _assemble_cohort(
    assembler: _Assembler, symbols: list[str], fresh: bool
) -> list[dict[str, Any]]:
    """Run the analyzer assembler concurrently for every cohort candidate.

    Failures degrade per-symbol — a symbol whose assembly raises is
    dropped from the cohort and logged. Returns a list of successful
    report dicts.
    """
    if not symbols:
        return []
    results = await asyncio.gather(
        *(assembler(s, fresh) for s in symbols),
        return_exceptions=True,
    )
    out: list[dict[str, Any]] = []
    for symbol, result in zip(symbols, results, strict=True):
        if isinstance(result, BaseException):
            log.warning(
                "sector.cohort.assemble.failed",
                symbol=symbol,
                error=str(result),
            )
            continue
        out.append(result)
    return out


def _extract_sector(report: dict[str, Any]) -> str | None:
    profile = report.get("fundamentals_profile")
    if not isinstance(profile, dict):
        return None
    sector = profile.get("sector")
    if isinstance(sector, str) and sector.strip():
        return sector
    return None


def _cache_key(sector_key: str, watchlist: list[str]) -> str:
    """Hash the cohort identity (sector + watchlist) for the cache file name."""
    fingerprint = {
        "sector": sector_key,
        "watchlist": sorted(set(watchlist)),
    }
    blob = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(blob, usedforsecurity=False).hexdigest()[:16]
    safe_sector = sector_key.replace(" ", "_")
    return f"{safe_sector}_{digest}.json"


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age >= _CACHE_TTL_SECONDS:
        return None
    try:
        body: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return body
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "sector.cache.read_failed",
            error=str(exc),
            path=str(path),
        )
        return None


def _wire_response(
    *,
    sym: str,
    sector_raw: str | None,
    ranking_dict: dict[str, Any],
    cache: str,
) -> dict[str, Any]:
    return {
        "symbol": sym,
        "sector": sector_raw or ranking_dict.get("sector"),
        "cohort_size": ranking_dict.get("cohort_size", 0),
        "cohort_symbols": ranking_dict.get("cohort_symbols", []),
        "metrics": ranking_dict.get("metrics", []),
        "available": True,
        "cache": cache,
    }


def _wire_empty_response(
    sym: str, sector_raw: str | None, *, reason: str
) -> dict[str, Any]:
    """Wire shape for the "no usable cohort" case.

    Frontend keys off ``available=False`` to render the "no sector
    peers" message rather than an empty table.
    """
    return empty_ranking(sym, sector_raw).to_dict() | {
        "available": False,
        "reason": reason,
        "cache": "miss",
    }


def _resolve_assembler(app: FastAPI) -> _Assembler:
    assembler = getattr(app.state, "analyzer_assembler", None)
    if assembler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Sector rankings require the analyzer pipeline; register "
                "analyzer routes on this app before /api/sector."
            ),
        )
    return assembler  # type: ignore[no-any-return]


__all__ = ["register_sector_routes"]
