"""Side-by-side stock comparison endpoints.

* ``GET /api/compare/{left}/{right}`` — structured per-metric verdicts
  (deterministic; runs the analyzer assembler for both sides).
* ``GET /api/compare/{left}/{right}/narrative`` — grounded AI narrative
  (optionally LLM-backed) consuming the structured comparison + both
  raw analyzer reports. Always runs through the claim validator.

Both endpoints reuse the analyzer pipeline rather than defining their
own. That keeps every comparison consistent with what the analyzer tab
renders for either symbol on its own — there's no case where the
analyzer says one thing and the comparison disagrees.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

from src.intelligence.comparison import ComparisonView, compare_reports
from src.intelligence.comparison_narrative import (
    ComparisonNarrator,
    LLMComparisonNarrator,
    NarrativeInput,
    TemplateComparisonNarrator,
)
from src.intelligence.comparison_narrative_validator import (
    narrative_validation_to_wire,
    validate_narrative,
)
from src.intelligence.trust_score import compute_research_trust_score
from src.utils.logging import get_logger

log = get_logger(__name__)


_Assembler = Callable[[str, bool], Awaitable[dict[str, Any]]]


# Narrative cache TTL — same 4-hour window as research. Two
# comparisons over the same pair of symbols within the window reuse
# the cached prose; the structured view itself is regenerated
# (cheap, deterministic, lives on the analyzer cache).
_NARRATIVE_TTL_SECONDS = 4 * 3600


# Lazy LLM narrator singleton — built on first use so importing this
# module doesn't touch the Anthropic SDK or read settings. Mirrors the
# pattern used in research.py for the same reason.
_llm_narrator_cache: LLMComparisonNarrator | None = None
_llm_init_attempted: bool = False
_llm_init_error: str | None = None


def _get_llm_narrator() -> LLMComparisonNarrator | None:
    global _llm_narrator_cache, _llm_init_attempted, _llm_init_error
    if _llm_init_attempted:
        return _llm_narrator_cache
    _llm_init_attempted = True
    try:
        _llm_narrator_cache = LLMComparisonNarrator()
        log.info(
            "compare_narrative.llm.enabled",
            model=_llm_narrator_cache.model_id,
        )
    except RuntimeError as e:
        _llm_init_error = str(e)
        _llm_narrator_cache = None
        log.info("compare_narrative.llm.disabled", reason=_llm_init_error)
    return _llm_narrator_cache


def register_compare_routes(
    app: FastAPI,
    *,
    narrative_cache_dir: Path | None = None,
) -> None:
    """Attach ``/api/compare/*`` routes to ``app``.

    Requires that :func:`register_analyzer_routes` has already run on
    the same app — the comparison endpoints read the assembler
    callable off ``app.state``. If the analyzer route isn't registered
    we 503 with a clear message rather than silently mis-routing.

    Args:
        app: FastAPI app to extend.
        narrative_cache_dir: Optional override for the narrative
            file cache. ``None`` disables caching (every request
            regenerates) — used by tests to avoid disk side-effects.
    """

    if narrative_cache_dir is not None:
        narrative_cache_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/api/compare/{left}/{right}")
    async def compare(
        left: str, right: str, fresh: bool = False
    ) -> dict[str, Any]:
        """Build a comparison view for two symbols.

        Both reports are assembled in parallel via the shared analyzer
        pipeline (cache-aware, same TTL semantics). When a symbol is
        not on the active watchlist the underlying assembler returns
        a report with ``warnings`` populated and most fields null —
        :func:`compare_reports` degrades cleanly in that case and
        surfaces the warning on the response.
        """
        l_sym = left.upper()
        r_sym = right.upper()
        if l_sym == r_sym:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Comparison requires two distinct symbols; "
                    f"got '{l_sym}' on both sides."
                ),
            )
        assembler = _resolve_assembler(app)
        left_report, right_report, merged_warnings = await _assemble_pair(
            assembler, l_sym, r_sym, fresh
        )
        view = compare_reports(left_report, right_report)
        merged = list(view.warnings) + merged_warnings
        payload = view.to_dict()
        payload["warnings"] = merged
        payload["left_report"] = left_report
        payload["right_report"] = right_report
        log.info(
            "compare.done",
            left=l_sym,
            right=r_sym,
            overall_winner=view.overall_winner,
            warning_count=len(merged),
        )
        return payload

    @app.get("/api/compare/{left}/{right}/narrative")
    async def compare_narrative(
        left: str,
        right: str,
        fresh: bool = False,
        mode: str = "auto",
    ) -> dict[str, Any]:
        """Return a grounded narrative for the two-symbol comparison.

        Query params:
            fresh: Bypass the narrative cache and regenerate.
            mode: ``template`` (deterministic, no API call) /
                ``llm`` (force Anthropic; 503 if not configured) /
                ``auto`` (LLM if available, else template + warning).

        Behavior:
            * Runs the analyzer pipeline for both symbols (same as
              ``/api/compare``).
            * Generates the narrative via the selected backend.
            * Always runs the claim validator — drops unsupported
              sentences and surfaces ``validation.drop_count`` so the
              frontend can render the trust signal.
            * Always returns a Trust Score grade (validator drop count
              is the load-bearing input on the research-style scorer).
        """
        l_sym = left.upper()
        r_sym = right.upper()
        if l_sym == r_sym:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Comparison narrative requires two distinct symbols; "
                    f"got '{l_sym}' on both sides."
                ),
            )
        assembler = _resolve_assembler(app)
        narrator, mode_used, warning = _select_narrator(mode)

        left_report, right_report, assembly_warnings = await _assemble_pair(
            assembler, l_sym, r_sym, fresh
        )
        view = compare_reports(left_report, right_report)
        payload = NarrativeInput(
            view=view, left_report=left_report, right_report=right_report
        )

        # Cache hit check happens *after* assembly: the assembler's own
        # cache is the slow part, and a stale narrative against fresh
        # data is misleading. Fingerprint the cache key on both reports
        # (their analyzer cache age effectively gates this too).
        cache_file: Path | None = None
        if narrative_cache_dir is not None and not fresh:
            cache_file = narrative_cache_dir / _narrative_cache_key(
                l_sym, r_sym, mode_used, left_report, right_report
            )
            cached = _read_cache(cache_file)
            if cached is not None:
                cached["cache"] = "hit"
                cached["mode"] = mode_used
                if warning:
                    cached["warning"] = warning
                return cached

        try:
            raw_narrative = narrator.generate(payload)
        except (
            anthropic.AuthenticationError,
            anthropic.RateLimitError,
            anthropic.APIStatusError,
            anthropic.APIConnectionError,
            ValueError,
        ) as e:
            if mode == "llm":
                raise HTTPException(
                    status_code=502,
                    detail=f"LLM narrative generation failed: {type(e).__name__}: {e}",
                ) from None
            log.warning(
                "compare_narrative.llm.failed_fallback_template",
                left=l_sym,
                right=r_sym,
                error=str(e),
                kind=type(e).__name__,
            )
            warning = (
                f"AI narrative failed ({type(e).__name__}); rendering "
                "deterministic template instead."
            )
            narrator = TemplateComparisonNarrator()
            mode_used = "template"
            raw_narrative = narrator.generate(payload)

        validated, validation_report = validate_narrative(
            raw_narrative, view, left_report, right_report
        )
        trust = compute_research_trust_score(
            validation_drop_count=validation_report.drop_count
        )

        encoded: dict[str, Any] = jsonable_encoder(validated.to_dict())
        encoded["validation"] = jsonable_encoder(
            narrative_validation_to_wire(validation_report)
        )
        encoded["trust_score"] = jsonable_encoder(trust.to_dict())
        encoded["mode"] = mode_used
        encoded["assembly_warnings"] = assembly_warnings
        if warning:
            encoded["warning"] = warning
        encoded["cache"] = "miss"

        if cache_file is not None:
            persistable = {
                k: v
                for k, v in encoded.items()
                if k not in ("cache", "warning")
            }
            try:
                cache_file.write_text(json.dumps(persistable), encoding="utf-8")
            except OSError as exc:
                log.warning(
                    "compare_narrative.cache.write_failed",
                    error=str(exc),
                    path=str(cache_file),
                )
        log.info(
            "compare_narrative.done",
            left=l_sym,
            right=r_sym,
            mode=mode_used,
            drop_count=validation_report.drop_count,
        )
        return encoded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _assemble_pair(
    assembler: _Assembler, left: str, right: str, fresh: bool
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Run the analyzer assembler for both sides in parallel.

    Returns ``(left_report, right_report, warnings)``. Each report
    may be ``None`` if assembly raised — the caller decides how to
    degrade. Warnings concatenate the per-side warnings the analyzer
    itself surfaced.
    """
    results = await asyncio.gather(
        assembler(left, fresh),
        assembler(right, fresh),
        return_exceptions=True,
    )
    left_report = _unwrap(results[0], left, side="left")
    right_report = _unwrap(results[1], right, side="right")
    warnings: list[str] = []
    if left_report is not None:
        warnings.extend(f"{left}: {w}" for w in left_report.get("warnings", []))
    if right_report is not None:
        warnings.extend(f"{right}: {w}" for w in right_report.get("warnings", []))
    return left_report, right_report, warnings


def _select_narrator(mode: str) -> tuple[ComparisonNarrator, str, str | None]:
    """Pick the narrator instance to use for this request."""
    if mode == "template":
        return TemplateComparisonNarrator(), "template", None
    llm = _get_llm_narrator()
    if mode == "llm":
        if llm is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"LLM narrative is not available: "
                    f"{_llm_init_error or 'unknown'}. Set ANTHROPIC_API_KEY "
                    "or call with ?mode=template."
                ),
            )
        return llm, "llm", None
    if llm is not None:
        return llm, "llm", None
    return (
        TemplateComparisonNarrator(),
        "template",
        f"LLM narrative unavailable: {_llm_init_error or 'unknown'} — "
        "rendering deterministic template.",
    )


def _narrative_cache_key(
    left: str,
    right: str,
    mode: str,
    left_report: dict[str, Any] | None,
    right_report: dict[str, Any] | None,
) -> str:
    """Hash the inputs that would change the generated narrative.

    Fingerprint includes the symbols, the mode, and each report's
    trust score + headline metrics — quantized loosely so a $0.01
    move in last_price doesn't bust the cache.
    """
    fingerprint = {
        "left": left,
        "right": right,
        "mode": mode,
        "left_trust": _report_fingerprint(left_report),
        "right_trust": _report_fingerprint(right_report),
    }
    blob = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(blob, usedforsecurity=False).hexdigest()[:16]
    return f"{left}_{right}_{mode}_{digest}.json"


def _report_fingerprint(report: dict[str, Any] | None) -> dict[str, Any]:
    """Minimal report fingerprint for the cache key (price-bucket'd)."""
    if report is None:
        return {"available": False}
    trust = report.get("trust_score") or {}
    return {
        "grade": trust.get("grade"),
        "score": round(float(trust.get("score") or 0), 0),
        "price_bucket": round(float(report.get("last_price") or 0), 0),
    }


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age >= _NARRATIVE_TTL_SECONDS:
        return None
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "compare_narrative.cache.read_failed",
            error=str(exc),
            path=str(path),
        )
        return None


def _resolve_assembler(app: FastAPI) -> _Assembler:
    assembler = getattr(app.state, "analyzer_assembler", None)
    if assembler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Comparison requires the analyzer pipeline; register "
                "analyzer routes on this app before /api/compare."
            ),
        )
    return assembler  # type: ignore[no-any-return]


def _unwrap(
    result: BaseException | dict[str, Any], symbol: str, *, side: str
) -> dict[str, Any] | None:
    if isinstance(result, BaseException):
        log.warning(
            "compare.assemble.failed",
            symbol=symbol,
            side=side,
            error=str(result),
        )
        return None
    return result


def view_for_test(left: dict[str, Any] | None, right: dict[str, Any] | None) -> ComparisonView:
    """Test-only re-export so test files don't need to import from
    ``src.intelligence.comparison`` separately."""
    return compare_reports(left, right)


__all__ = ["register_compare_routes"]
