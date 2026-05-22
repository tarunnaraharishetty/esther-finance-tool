"""Research-thesis API surface.

One endpoint today (``GET /api/research/{symbol}``) returning a fully
structured :class:`ResearchThesis`. File-backed cache lives under
``data/research_cache/`` keyed by a SHA1 of the inputs that would
change the report; cache hits skip the generator entirely.

The endpoint reads its data from the live ``controller`` instance —
same snapshot the dashboard renders — so the thesis is consistent with
what the trader sees on every other tab.

Two generator backends:

* :class:`~src.intelligence.research_thesis.TemplateThesisGenerator` —
  deterministic; always available; no API key.
* :class:`~src.intelligence.llm_research.LLMThesisGenerator` —
  Anthropic; only constructed when ``ANTHROPIC_API_KEY`` is set.

The endpoint prefers LLM mode when available and falls back to template
on (a) missing API key, (b) Anthropic auth / rate / API failures, or
(c) the ``?mode=template`` query override. On fallback the response
includes a ``warning`` field so the UI can surface what happened.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

from src.intelligence.research_thesis import (
    ResearchInput,
    TemplateThesisGenerator,
)
from src.intelligence.research_validator import (
    validate_thesis,
    validation_to_wire,
)
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController
    from src.dashboard.state import RecommendationRow
    from src.intelligence.llm_research import LLMThesisGenerator

    # Both backends expose the same shape: ``generate(ResearchInput) ->
    # ResearchThesis`` and ``model_id: str``. A Union is simpler than a
    # Protocol here — only this module needs to type them interchangeably.
    _ThesisGenerator = LLMThesisGenerator | TemplateThesisGenerator


log = get_logger(__name__)


# Cache TTL — 4 hours during what would be market hours, 24h otherwise.
# Kept simple here; a future refinement could parse America/New_York
# wall clock. The 4h baseline already de-duplicates intraday churn.
_CACHE_TTL_SECONDS = 4 * 3600


# Lazy singleton for the LLM generator. Built on first use so importing
# this module doesn't try to hit the Anthropic SDK / read settings.
_llm_generator_cache: _ThesisGenerator | None = None
_llm_init_attempted: bool = False
_llm_init_error: str | None = None


def _get_llm_generator() -> _ThesisGenerator | None:
    """Return the configured ``LLMThesisGenerator`` or ``None``.

    Caches both the success and the failure path — if the key is not
    set we don't keep re-checking on every request. A future config-
    reload mechanism can clear the singletons.
    """
    global _llm_generator_cache, _llm_init_attempted, _llm_init_error
    if _llm_init_attempted:
        return _llm_generator_cache
    _llm_init_attempted = True
    try:
        from src.intelligence.llm_research import LLMThesisGenerator

        _llm_generator_cache = LLMThesisGenerator()
        log.info(
            "research.llm.enabled",
            model=_llm_generator_cache.model_id,
        )
    except RuntimeError as e:
        # Most common: ANTHROPIC_API_KEY not configured. Record the
        # reason so the endpoint can surface it as a warning.
        _llm_init_error = str(e)
        _llm_generator_cache = None
        log.info("research.llm.disabled", reason=_llm_init_error)
    return _llm_generator_cache


def register_research_routes(
    app: FastAPI,
    controller: BaseController,
    *,
    cache_dir: Path,
) -> None:
    """Attach research routes to ``app``.

    Split out of ``create_app`` so the API surface can be wired
    incrementally and tests can mount it onto a bare FastAPI app.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/api/research/{symbol}")
    async def research_thesis(
        symbol: str, fresh: bool = False, mode: str = "template"
    ) -> dict[str, Any]:
        """Return a structured research thesis for ``symbol``.

        Query params:
            fresh: bypass the cache and force regeneration.
            mode: ``template`` (default — deterministic, no API call),
                ``llm`` (force Anthropic; 503 if not configured), or
                ``auto`` (LLM if available, else template + warning).
                Default is ``template`` so the page doesn't burn an
                Anthropic call on every load; flip to ``llm`` per
                request from the UI when the user explicitly asks for
                Claude-written prose.

        Behavior:
            * 404 if the symbol isn't on the active watchlist.
            * Cache hit → returns cached JSON (with ``cache: "hit"``).
            * Cache miss → builds a :class:`ResearchInput` from the
              latest snapshot, runs the selected generator, caches the
              result, and returns it.
            * On Anthropic auth / rate / API failures with ``mode=auto``,
              falls back to the template generator and stamps a
              ``warning`` field describing what failed.
        """
        sym = symbol.upper()
        snap = await controller.fetch_snapshot()
        row: RecommendationRow | None = next(
            (r for r in snap.rows if r.symbol == sym), None
        )
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Symbol '{sym}' is not on the active watchlist — "
                    "research is currently scoped to controller-tracked symbols."
                ),
            )

        generator, mode_used, warning = _select_generator(mode)
        cache_key = _cache_key_for(row, mode_used)
        cache_file = cache_dir / sym / f"{cache_key}.json"

        if not fresh and cache_file.exists():
            mtime = cache_file.stat().st_mtime
            if time.time() - mtime < _CACHE_TTL_SECONDS:
                log.info(
                    "research.cache.hit",
                    symbol=sym,
                    age_seconds=int(time.time() - mtime),
                    mode=mode_used,
                )
                cached_body: dict[str, Any] = json.loads(
                    cache_file.read_text(encoding="utf-8")
                )
                cached_body["cache"] = "hit"
                cached_body["cache_age_seconds"] = int(time.time() - mtime)
                cached_body["mode"] = mode_used
                if warning:
                    cached_body["warning"] = warning
                return cached_body

        log.info(
            "research.generate.start", symbol=sym, fresh=fresh, mode=mode_used
        )
        payload = ResearchInput(row=row, headlines=tuple(row.headlines))

        try:
            thesis = generator.generate(payload)
        except (
            anthropic.AuthenticationError,
            anthropic.RateLimitError,
            anthropic.APIStatusError,
            anthropic.APIConnectionError,
            ValueError,
        ) as e:
            # LLM failures degrade to template — we'd rather render a
            # report with a warning than 500 the page.
            if mode == "llm":
                raise HTTPException(
                    status_code=502,
                    detail=f"LLM research generation failed: {type(e).__name__}: {e}",
                ) from None
            log.warning(
                "research.llm.failed_fallback_template",
                symbol=sym,
                error=str(e),
                kind=type(e).__name__,
            )
            warning = (
                f"AI generation failed ({type(e).__name__}); rendering deterministic "
                "template instead."
            )
            generator = TemplateThesisGenerator()
            mode_used = "template"
            cache_key = _cache_key_for(row, mode_used)
            cache_file = cache_dir / sym / f"{cache_key}.json"
            thesis = generator.generate(payload)

        # Validator runs on every thesis (both LLM and template). The
        # template's numeric output always traces to the input row, so
        # in normal operation drops are zero; a non-zero drop count on
        # the template path is a regression signal worth surfacing.
        validated, report = validate_thesis(thesis, payload)
        encoded: dict[str, Any] = jsonable_encoder(asdict(validated))
        encoded["cache"] = "miss"
        encoded["cache_age_seconds"] = 0
        encoded["mode"] = mode_used
        encoded["validation"] = jsonable_encoder(validation_to_wire(report))
        if warning:
            encoded["warning"] = warning

        # Persist the canonical body without the cache-hit metadata so
        # subsequent reads can stamp fresh values.
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        persistable = {
            k: v
            for k, v in encoded.items()
            if k not in ("cache", "cache_age_seconds", "warning")
        }
        cache_file.write_text(json.dumps(persistable), encoding="utf-8")
        log.info(
            "research.generate.done",
            symbol=sym,
            mode=mode_used,
            cache_path=str(cache_file),
        )
        return encoded

    @app.delete("/api/research/{symbol}/cache")
    async def clear_research_cache(symbol: str) -> dict[str, Any]:
        """Wipe all cached theses for one symbol (template + LLM).

        Useful when the underlying data shape changes or the prompt
        revision shipped and the user wants a clean regenerate.
        """
        sym = symbol.upper()
        target = cache_dir / sym
        removed = 0
        if target.exists():
            for child in target.iterdir():
                child.unlink()
                removed += 1
        return {"symbol": sym, "removed": removed}

    @app.get("/api/research/_/status")
    async def research_status() -> dict[str, Any]:
        """Reports which generator the endpoint will pick by default.

        Lets the frontend surface "AI mode active" vs "template only"
        without making a real research call.
        """
        gen = _get_llm_generator()
        if gen is not None:
            return {
                "mode": "llm",
                "model": gen.model_id,
            }
        return {
            "mode": "template",
            "model": TemplateThesisGenerator.model_id,
            "reason": _llm_init_error,
        }


def _select_generator(
    mode: str,
) -> tuple[_ThesisGenerator, str, str | None]:
    """Pick the generator instance to use this request.

    Returns ``(generator, resolved_mode, warning_or_None)``. The
    ``warning`` is non-None when the caller asked for ``auto`` but the
    LLM path isn't available — surfacing the reason to the UI helps
    the user know they're seeing the fallback.
    """
    if mode == "template":
        return TemplateThesisGenerator(), "template", None
    llm = _get_llm_generator()
    if mode == "llm":
        if llm is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"LLM mode is not available: {_llm_init_error or 'unknown'}. "
                    "Set ANTHROPIC_API_KEY or call with ?mode=template."
                ),
            )
        return llm, "llm", None
    # auto
    if llm is not None:
        return llm, "llm", None
    return (
        TemplateThesisGenerator(),
        "template",
        f"LLM mode unavailable: {_llm_init_error or 'unknown'} — rendering deterministic template.",
    )


def _cache_key_for(row: RecommendationRow, mode: str) -> str:
    """Hash the inputs that would change the generated thesis.

    Quantize floats so trivial tick-to-tick drift doesn't bust the
    cache — a $0.01 price move shouldn't trigger a fresh report. The
    generator's ``model_id`` is folded in so LLM and template results
    cache to separate files.
    """
    if mode == "llm":
        llm = _get_llm_generator()
        model_version = (
            llm.model_id if llm is not None else TemplateThesisGenerator.model_id
        )
    else:
        model_version = TemplateThesisGenerator.model_id
    fingerprint = {
        "symbol": row.symbol,
        "action": row.action.value,
        "tier": row.tier.value,
        "price_bucket": round(row.last_price, 0),
        "rsi_bucket": round(row.rsi),
        "macd_bucket": round(row.macd, 1),
        "boll_bucket": round(row.bollinger, 1),
        "sent_bucket": round(row.sentiment_score, 1),
        "news_count": row.num_news_articles,
        "model_version": model_version,
    }
    blob = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob, usedforsecurity=False).hexdigest()[:16]


__all__ = ["register_research_routes"]
