"""Key Drivers endpoint — ``GET /api/movement/{symbol}/drivers``.

Composes the analyzer report, the snapshot row, and (optionally) the
sector ranking for a symbol, then runs :func:`compute_key_drivers`
to return a ranked tuple of "what's most notable right now."

The endpoint deliberately reuses the analyzer assembler + the sector
ranker so the drivers never disagree with the cards rendered
elsewhere on the same page. No new data sources — this is purely a
synthesis layer over the existing signals.

Caching
-------
Drivers shift intraday (sentiment turns, headlines arrive, RSI moves)
so the cache TTL is short — 15 minutes by default. Fingerprinted on
the analyzer cache state implicitly: a fresh analyzer rebuild
invalidates the driver cache too via the underlying analyzer's
``cache_age_seconds`` change.
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

from src.intelligence.movement_drivers import compute_key_drivers
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController

log = get_logger(__name__)


_Assembler = Callable[[str, bool], Awaitable[dict[str, Any]]]


# Drivers shift more often than analyzer cards (intraday sentiment +
# headlines), so the cache window is short.
_CACHE_TTL_SECONDS = 15 * 60


def register_movement_routes(
    app: FastAPI,
    controller: BaseController,
    *,
    cache_dir: Path | None = None,
) -> None:
    """Attach ``/api/movement/{symbol}/drivers`` to ``app``.

    Requires :func:`register_analyzer_routes` to have run first so
    ``app.state.analyzer_assembler`` is wired. We 503 with a clear
    message rather than silently mis-routing if the assembler is
    missing. The route is also tolerant of the sector endpoint not
    being registered — sector drivers simply don't fire in that case.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/api/movement/{symbol}/drivers")
    async def key_drivers(symbol: str, fresh: bool = False) -> dict[str, Any]:
        """Return the ranked Key Drivers bundle for ``symbol``.

        Args:
            symbol: Ticker. Case-insensitive on input.
            fresh: When ``True``, bypasses the driver cache and forces
                a fresh analyzer + sector + row assembly.
        """
        sym = symbol.upper()
        assembler = _resolve_assembler(app)

        cache_file: Path | None = None
        if cache_dir is not None and not fresh:
            cache_file = cache_dir / _cache_key(sym, controller.watchlist)
            cached = _read_cache(cache_file)
            if cached is not None:
                cached["cache"] = "hit"
                return cached

        # Pull the three inputs in parallel. Each one degrades to
        # ``None`` on failure rather than 500'ing — the driver
        # ranker handles missing inputs by simply not firing the
        # relevant driver.
        results = await asyncio.gather(
            _safe_assemble(assembler, sym, fresh),
            _safe_sector_rank(app, sym, fresh, controller),
            _safe_row(controller, sym),
            return_exceptions=False,
        )
        analyzer_report, sector_rank, snapshot_row = results

        if analyzer_report is None:
            raise HTTPException(
                status_code=503,
                detail=f"Could not assemble analyzer report for {sym}.",
            )

        drivers = compute_key_drivers(
            sym,
            analyzer_report=analyzer_report,
            sector_rank=sector_rank,
            snapshot_row=snapshot_row,
        )
        payload: dict[str, Any] = jsonable_encoder(drivers.to_dict())
        payload["cache"] = "miss"
        # Expose which inputs were present so the UI can render
        # "Sector signal unavailable — add peers to your watchlist"
        # without re-fetching the sector endpoint to find out.
        payload["sources"] = {
            "analyzer": analyzer_report is not None,
            "sector": sector_rank is not None and sector_rank.get("available", False),
            "snapshot_row": snapshot_row is not None,
        }

        if cache_file is not None:
            try:
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
            except OSError as exc:
                log.warning(
                    "movement.cache.write_failed",
                    error=str(exc),
                    path=str(cache_file),
                )
        log.info(
            "movement.drivers.done",
            symbol=sym,
            driver_count=len(drivers.drivers),
            suppressed=drivers.suppressed,
        )
        return payload


# ---------------------------------------------------------------------------
# Per-source helpers
# ---------------------------------------------------------------------------


async def _safe_assemble(
    assembler: _Assembler, symbol: str, fresh: bool
) -> dict[str, Any] | None:
    """Wrap the analyzer assembler so a failure degrades to ``None``."""
    try:
        return await assembler(symbol, fresh)
    except Exception as exc:
        log.warning(
            "movement.analyzer.failed", symbol=symbol, error=str(exc)
        )
        return None


async def _safe_sector_rank(
    app: FastAPI, symbol: str, fresh: bool, controller: BaseController
) -> dict[str, Any] | None:
    """Run the sector ranker inline by composing the same calls the
    sector endpoint would. Returns ``None`` on any failure mode so
    the driver computation degrades cleanly.

    Inline instead of an HTTP call to ourselves — keeps the driver
    request a single round-trip from the caller's perspective.
    """
    try:
        # Late import to avoid a top-level dep on the sector module
        # (driver endpoint can run without it).
        from src.intelligence.analyzer.sector_medians import (
            normalize_sector_key,
        )
        from src.intelligence.sector_ranker import compute_sector_rankings

        assembler = _resolve_assembler(app)
        primary = await _safe_assemble(assembler, symbol, fresh)
        if primary is None:
            return None
        profile = primary.get("fundamentals_profile") or {}
        sector_raw = (
            profile.get("sector") if isinstance(profile, dict) else None
        )
        sector_key = normalize_sector_key(sector_raw)
        if sector_key is None:
            return {"available": False, "reason": "no sector"}
        peers = [
            s.upper() for s in controller.watchlist if s.upper() != symbol
        ]
        if not peers:
            return {"available": False, "reason": "no peers"}
        peer_results = await asyncio.gather(
            *(_safe_assemble(assembler, p, fresh) for p in peers),
            return_exceptions=False,
        )
        all_reports = [primary] + [r for r in peer_results if r is not None]
        rankings = compute_sector_rankings(all_reports)
        target = rankings.get(symbol)
        if target is None or target.cohort_size < 2:
            return {"available": False, "reason": "singleton cohort"}
        wire = target.to_dict()
        wire["available"] = True
        return wire
    except Exception as exc:
        log.warning("movement.sector.failed", symbol=symbol, error=str(exc))
        return None


async def _safe_row(
    controller: BaseController, symbol: str
) -> dict[str, Any] | None:
    """Pull the snapshot row for ``symbol`` as a dict.

    The snapshot already lives in controller memory; we extract the
    matching row and serialize the fields the driver computer needs.
    Returns ``None`` if the symbol isn't on the watchlist (degraded
    sentiment + news drivers but the analyzer-side ones still fire).
    """
    try:
        snap = await controller.fetch_snapshot()
    except Exception as exc:
        log.warning("movement.row.failed", symbol=symbol, error=str(exc))
        return None
    row = next((r for r in snap.rows if r.symbol == symbol), None)
    if row is None:
        return None
    return {
        "symbol": row.symbol,
        "sentiment_score": (
            float(row.sentiment_score)
            if row.sentiment_score is not None
            else None
        ),
        "num_news_articles": int(row.num_news_articles),
        "headlines": list(row.headlines),
        "last_price": (
            float(row.last_price) if row.last_price is not None else None
        ),
    }


def _resolve_assembler(app: FastAPI) -> _Assembler:
    assembler = getattr(app.state, "analyzer_assembler", None)
    if assembler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Key Drivers require the analyzer pipeline; register "
                "analyzer routes on this app before /api/movement."
            ),
        )
    return assembler  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(symbol: str, watchlist: list[str]) -> str:
    """Hash on symbol + watchlist so a watchlist change naturally invalidates."""
    fingerprint = {
        "symbol": symbol,
        "watchlist": sorted({s.upper() for s in watchlist}),
    }
    blob = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(blob, usedforsecurity=False).hexdigest()[:16]
    return f"{symbol}_{digest}.json"


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
            "movement.cache.read_failed",
            error=str(exc),
            path=str(path),
        )
        return None


__all__ = ["register_movement_routes"]
