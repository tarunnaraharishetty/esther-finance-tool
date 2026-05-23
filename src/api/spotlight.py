"""Watchlist Driver Spotlight endpoint.

``GET /api/spotlight`` scans every watchlist symbol's Key Drivers,
ranks them globally by salience, and returns the top N as a
fleet-level "what's notable across my watchlist right now" view.

Composes the per-symbol driver assembly from :mod:`src.api.movement`
without duplication — each symbol's drivers are computed using the
exact same pipeline the per-symbol ``/api/movement/{symbol}/drivers``
endpoint uses. That keeps the per-symbol and fleet-level views
consistent: a driver shown on the spotlight is the same driver the
trader sees when they click through.

Cache strategy
--------------
15-minute file cache keyed on the sorted watchlist + ``fresh`` flag.
Drivers shift intraday with sentiment + headlines, so a longer
window would stale the surface; shorter than 15 minutes would mean
the cross-symbol parallel fetch runs too often.
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

# Reuse the per-symbol assembly helpers from the movement module so the
# spotlight and the per-symbol drivers endpoint stay in lock-step.
from src.api.movement import _safe_assemble, _safe_row, _safe_sector_rank
from src.intelligence.driver_spotlight import compute_spotlight
from src.intelligence.movement_drivers import KeyDrivers, compute_key_drivers
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController

log = get_logger(__name__)


_Assembler = Callable[[str, bool], Awaitable[dict[str, Any]]]


# Match the per-symbol drivers cache TTL so the spotlight never lags
# more than one driver-cache cycle behind the underlying signals.
_CACHE_TTL_SECONDS = 15 * 60


def register_spotlight_routes(
    app: FastAPI,
    controller: BaseController,
    *,
    cache_dir: Path | None = None,
) -> None:
    """Attach ``/api/spotlight`` to ``app``.

    Requires :func:`register_analyzer_routes` and movement routes to
    have run first — the spotlight reads the assembler off
    ``app.state`` and reuses the per-symbol driver helpers. We 503
    with a clear message rather than silently mis-routing if either
    dependency is missing.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/api/spotlight")
    async def spotlight(fresh: bool = False) -> dict[str, Any]:
        """Return the cross-watchlist driver spotlight.

        Args:
            fresh: When ``True``, bypasses both the spotlight cache
                and the underlying per-symbol caches (forces a fresh
                fleet-wide assembly).
        """
        assembler = _resolve_assembler(app)
        watchlist = [s.upper() for s in controller.watchlist]
        if not watchlist:
            return _empty_response(reason="watchlist is empty")

        cache_file: Path | None = None
        if cache_dir is not None and not fresh:
            cache_file = cache_dir / _cache_key(watchlist)
            cached = _read_cache(cache_file)
            if cached is not None:
                cached["cache"] = "hit"
                return cached

        log.info(
            "spotlight.assemble.start",
            symbols=len(watchlist),
            fresh=fresh,
        )
        symbol_drivers = await _assemble_all(
            assembler, controller, app, watchlist, fresh
        )
        view = compute_spotlight(symbol_drivers)
        payload: dict[str, Any] = jsonable_encoder(view.to_dict())
        payload["watchlist_size"] = len(watchlist)
        payload["assembled_symbols"] = len(symbol_drivers)
        payload["cache"] = "miss"

        if cache_file is not None:
            try:
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
            except OSError as exc:
                log.warning(
                    "spotlight.cache.write_failed",
                    error=str(exc),
                    path=str(cache_file),
                )
        log.info(
            "spotlight.assemble.done",
            symbols=len(watchlist),
            assembled=len(symbol_drivers),
            entries=len(view.entries),
        )
        return payload


# ---------------------------------------------------------------------------
# Cross-symbol assembly
# ---------------------------------------------------------------------------


async def _assemble_all(
    assembler: _Assembler,
    controller: BaseController,
    app: FastAPI,
    watchlist: list[str],
    fresh: bool,
) -> dict[str, KeyDrivers]:
    """Run the per-symbol driver assembly for every watchlist symbol.

    Failures degrade per-symbol: a symbol whose analyzer fetch raises
    is dropped from the result dict (and logged). The composer only
    sees the symbols that produced usable drivers.
    """
    inputs = await asyncio.gather(
        *(
            _gather_one_symbol_inputs(
                assembler, app, controller, symbol, fresh
            )
            for symbol in watchlist
        ),
        return_exceptions=False,
    )
    out: dict[str, KeyDrivers] = {}
    for symbol, bundle in zip(watchlist, inputs, strict=True):
        analyzer_report, sector_rank, snapshot_row = bundle
        if analyzer_report is None:
            # No analyzer report → no driver basis → skip the symbol.
            # Trust, technical, calibration all require the analyzer
            # report; without it the panel would be empty anyway.
            continue
        out[symbol] = compute_key_drivers(
            symbol,
            analyzer_report=analyzer_report,
            sector_rank=sector_rank,
            snapshot_row=snapshot_row,
        )
    return out


async def _gather_one_symbol_inputs(
    assembler: _Assembler,
    app: FastAPI,
    controller: BaseController,
    symbol: str,
    fresh: bool,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Per-symbol parallel fetch — same trio as the movement endpoint."""
    results = await asyncio.gather(
        _safe_assemble(assembler, symbol, fresh),
        _safe_sector_rank(app, symbol, fresh, controller),
        _safe_row(controller, symbol),
        return_exceptions=False,
    )
    analyzer_report, sector_rank, snapshot_row = results
    return analyzer_report, sector_rank, snapshot_row


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


def _empty_response(*, reason: str) -> dict[str, Any]:
    """Wire shape for the "nothing to show" case.

    Used when the watchlist is empty. Frontend keys off
    ``entries == []`` to render the empty state; ``reason`` is the
    short text that goes under the empty placeholder.
    """
    return {
        "entries": [],
        "considered_symbols": 0,
        "surfaced_symbols": 0,
        "watchlist_size": 0,
        "assembled_symbols": 0,
        "reason": reason,
        "cache": "miss",
    }


def _resolve_assembler(app: FastAPI) -> _Assembler:
    assembler = getattr(app.state, "analyzer_assembler", None)
    if assembler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Spotlight requires the analyzer pipeline; register "
                "analyzer + movement routes on this app before "
                "/api/spotlight."
            ),
        )
    return assembler  # type: ignore[no-any-return]


def _cache_key(watchlist: list[str]) -> str:
    """Hash on the sorted watchlist so a membership change invalidates."""
    fingerprint = {"watchlist": sorted(set(watchlist))}
    blob = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(blob, usedforsecurity=False).hexdigest()[:16]
    return f"spotlight_{digest}.json"


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
            "spotlight.cache.read_failed",
            error=str(exc),
            path=str(path),
        )
        return None


__all__ = ["register_spotlight_routes"]
