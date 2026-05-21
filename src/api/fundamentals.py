"""Fundamentals API surface.

Exposes the orchestrated provider chain (Phase 1) over HTTP so the
frontend can fetch normalized company fundamentals without knowing
which upstream provider supplied them.

Routes
------
* ``GET /api/fundamentals/{symbol}`` — full normalized record. Cache-
  first: if a fresh JSON blob is on disk we return it instantly;
  otherwise we walk the provider chain, persist the result, and
  return it. ``?fresh=true`` bypasses the cache. The response always
  includes ``cache`` (``"hit"`` / ``"miss"``), ``cache_age_seconds``,
  ``primary_provider``, ``contributing_providers``, and ``health`` —
  the per-provider chain log from this fetch (empty array on cache
  hits since the chain wasn't walked).

* ``GET /api/fundamentals/{symbol}/health`` — chain-only probe. Walks
  the providers and returns the health log without persisting the
  fundamentals payload. Useful for diagnosing "which provider is the
  one returning data right now" without touching the cache.

* ``DELETE /api/fundamentals/{symbol}/cache`` — invalidate the cached
  blob for one symbol. Returns the number of files removed.

* ``GET /api/fundamentals/_/status`` — global status: configured
  providers, default chain order, cache directory.

The endpoint never raises a 5xx for "no provider worked". Chain
exhaustion is 503 with the health log in the response body so the
frontend can render "we tried 5 providers and got nothing" rather
than a generic crash.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

from src.config import get_settings
from src.intelligence.fundamentals import (
    FundamentalsResult,
    FundamentalsService,
    ProviderChainExhausted,
    ProviderName,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


def register_fundamentals_routes(
    app: FastAPI,
    *,
    cache_dir: Path,
    service: FundamentalsService | None = None,
) -> None:
    """Attach fundamentals routes to ``app``.

    Split out of :func:`create_app` so tests can mount it on a bare
    FastAPI app and pass a stubbed :class:`FundamentalsService` (no
    network, deterministic chain behavior).

    Args:
        app: The FastAPI app to extend.
        cache_dir: Directory where normalized fundamentals JSON blobs
            are persisted. Created on first use. Each symbol gets one
            file (``{cache_dir}/{SYMBOL}.json``) — fundamentals don't
            need the multi-mode fan-out the research cache uses.
        service: Optional pre-built :class:`FundamentalsService`. When
            ``None`` we lazily construct one from ``get_settings()`` on
            first call so importing this module doesn't require any
            API keys.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    # Lazy-built service so importing this module is side-effect free.
    # ``service_holder[0]`` is the cached instance; ``None`` means
    # "not built yet".
    service_holder: list[FundamentalsService | None] = [service]

    def _service() -> FundamentalsService:
        if service_holder[0] is None:
            service_holder[0] = FundamentalsService.from_settings(settings)
        return service_holder[0]

    @app.get("/api/fundamentals/{symbol}")
    async def fundamentals(
        symbol: str, fresh: bool = False
    ) -> dict[str, Any]:
        """Return normalized fundamentals for ``symbol``.

        On a cache hit the response is the persisted JSON blob with
        the cache-metadata fields stamped fresh. On a cache miss we
        walk the provider chain. If every provider fails we 503 with
        the health log so the frontend can show *why* nothing came
        back rather than rendering an empty card.
        """
        sym = symbol.upper()
        cache_file = cache_dir / f"{sym}.json"
        ttl_seconds = settings.fundamentals_cache_ttl_hours * 3600.0

        if not fresh and cache_file.exists():
            mtime = cache_file.stat().st_mtime
            age = time.time() - mtime
            if age < ttl_seconds:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                payload["cache"] = "hit"
                payload["cache_age_seconds"] = int(age)
                # Cache hit means we did not walk the chain this call;
                # the historical health log is part of the persisted
                # blob already. We surface it under ``health`` for
                # parity with the miss path.
                log.info(
                    "fundamentals.cache.hit",
                    symbol=sym,
                    age_seconds=int(age),
                )
                return payload

        log.info("fundamentals.fetch.start", symbol=sym, fresh=fresh)
        try:
            result = await _service().fetch(sym)
        except ProviderChainExhausted as exc:
            log.warning(
                "fundamentals.chain_exhausted",
                symbol=sym,
                errors=list(exc.errors),
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "message": str(exc),
                    "symbol": sym,
                    "errors": list(exc.errors),
                    "health": jsonable_encoder([asdict(h) for h in exc.health]),
                },
            ) from None

        payload = _serialize_result(result)
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

        # Stamp the cache-metadata after persisting so the file on disk
        # represents the canonical body; future reads can apply fresh
        # cache markers without rotting old "cache": "miss" labels.
        payload["cache"] = "miss"
        payload["cache_age_seconds"] = 0
        log.info(
            "fundamentals.fetch.ok",
            symbol=sym,
            primary_provider=result.fundamentals.primary_provider.value,
        )
        return payload

    @app.get("/api/fundamentals/{symbol}/health")
    async def fundamentals_health(symbol: str) -> dict[str, Any]:
        """Return only the per-provider health log for ``symbol``.

        Always walks the chain — never reads the cache. Useful for
        debugging "which provider just answered" without paying the
        cost of dumping the full normalized record. On chain
        exhaustion we still return 200 with the failure rows so a
        monitoring panel can poll this endpoint without alarm.
        """
        sym = symbol.upper()
        try:
            result = await _service().fetch(sym)
        except ProviderChainExhausted as exc:
            return {
                "symbol": sym,
                "primary_provider": None,
                "chain_exhausted": True,
                "errors": list(exc.errors),
                "health": jsonable_encoder([asdict(h) for h in exc.health]),
            }
        return {
            "symbol": sym,
            "primary_provider": result.fundamentals.primary_provider.value,
            "chain_exhausted": False,
            "errors": [],
            "health": jsonable_encoder([asdict(h) for h in result.health]),
        }

    @app.delete("/api/fundamentals/{symbol}/cache")
    async def clear_fundamentals_cache(symbol: str) -> dict[str, Any]:
        """Wipe the cached blob for one symbol.

        Frontend hooks this onto the "Refresh fundamentals" button. A
        delete followed by a GET forces a fresh chain walk.
        """
        sym = symbol.upper()
        cache_file = cache_dir / f"{sym}.json"
        removed = 0
        if cache_file.exists():
            cache_file.unlink()
            removed = 1
        return {"symbol": sym, "removed": removed}

    @app.get("/api/fundamentals/_/status")
    async def fundamentals_status() -> dict[str, Any]:
        """Report which providers are configured + chain order.

        Lets the frontend surface "Fundamentals limited — only Yahoo
        is configured" without making a real fetch. Cheap; never
        constructs the service unless it was already built.
        """
        provider_state = {
            ProviderName.FMP.value: settings.fmp_api_key is not None,
            ProviderName.FINNHUB.value: settings.finnhub_api_key is not None,
            ProviderName.ALPHA_VANTAGE.value: (
                settings.alphavantage_api_key is not None
            ),
            ProviderName.SEC_EDGAR.value: bool(
                settings.sec_edgar_user_agent.strip()
            ),
            ProviderName.YAHOO.value: True,
        }
        return {
            "chain_order": list(settings.fundamentals_provider_order),
            "providers_configured": provider_state,
            "cache_dir": str(cache_dir),
            "cache_ttl_hours": settings.fundamentals_cache_ttl_hours,
        }


def _serialize_result(result: FundamentalsResult) -> dict[str, Any]:
    """Convert a :class:`FundamentalsResult` to the JSON wire shape.

    We don't return the raw provider payload — it's huge and meant
    for the persistence layer (Phase 5 DB schema), not the frontend.
    The wire shape carries the normalized record plus the chain
    health log so clients can render provider-attribution UI.
    """
    fundamentals = result.fundamentals
    payload: dict[str, Any] = {
        "symbol": fundamentals.symbol,
        "fetched_at": fundamentals.fetched_at.isoformat(),
        "primary_provider": fundamentals.primary_provider.value,
        "contributing_providers": [
            p.value for p in fundamentals.contributing_providers
        ],
        "profile": jsonable_encoder(asdict(fundamentals.profile)),
        "income_statements": [
            jsonable_encoder(asdict(s)) for s in fundamentals.income_statements
        ],
        "balance_sheets": [
            jsonable_encoder(asdict(s)) for s in fundamentals.balance_sheets
        ],
        "cash_flows": [
            jsonable_encoder(asdict(s)) for s in fundamentals.cash_flows
        ],
        "key_ratios": jsonable_encoder(asdict(fundamentals.key_ratios)),
        "analyst_targets": (
            jsonable_encoder(asdict(fundamentals.analyst_targets))
            if fundamentals.analyst_targets is not None
            else None
        ),
        "warnings": list(fundamentals.warnings),
        "health": jsonable_encoder([asdict(h) for h in result.health]),
    }
    return payload


__all__ = ["register_fundamentals_routes"]
