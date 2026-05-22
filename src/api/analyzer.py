"""Financial Analyzer API surface (Phase 6).

Assembles every analyzer subsystem into one cached endpoint:

* ``GET /api/analyzer/{symbol}`` — full report (technicals + valuation
  + grounded explanation + overall analyzer scores).

The endpoint composes:

* OHLCV bars via :meth:`BaseController.fetch_bars` → normalized
  technical sub-scores from :mod:`src.intelligence.analyzer.technical`.
* Fundamentals from the Phase-1 :class:`FundamentalsService` → multi-
  method valuation from :mod:`src.intelligence.analyzer.valuation`.
* The last :class:`RecommendationRow` (already in the snapshot) for
  sentiment / news / last-price grounding signals.
* Grounded explanation from :mod:`src.intelligence.analyzer.explanation`
  — defaults to the deterministic template builder. The LLM-backed
  generator is opt-in via ``?mode=llm`` once Anthropic is configured.

The endpoint always returns 200 on partial data — if fundamentals are
unavailable the technical-only report is rendered; the
``warnings`` array tells the frontend which streams degraded.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from src.config import get_settings
from src.data.accuracy_store import AccuracyStore
from src.data.health_store import HealthStore
from src.data.retry_queue import RetryQueue
from src.intelligence.analyzer import (
    AnalyzerExplanation,
    AnalyzerInputs,
    CalibrationReading,
    ScenarioModel,
    build_grounded_explanation,
    build_scenarios,
    build_valuation,
    lookup_readings,
    score_technicals,
)
from src.intelligence.analyzer.technical import TechnicalScores
from src.intelligence.analyzer.valuation import ValuationEnsemble
from src.intelligence.calibration import CalibrationStore
from src.intelligence.fundamentals import (
    FundamentalsService,
    NormalizedFundamentals,
    ProviderChainExhausted,
)
from src.intelligence.trust_score import (
    TrustScore,
    compute_analyzer_trust_score,
)
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController

log = get_logger(__name__)


def register_analyzer_routes(
    app: FastAPI,
    controller: BaseController,
    *,
    cache_dir: Path,
    fundamentals_service: FundamentalsService | None = None,
    health_store: HealthStore | None = None,
    retry_queue: RetryQueue | None = None,
    calibration_store: CalibrationStore | None = None,
    accuracy_store: AccuracyStore | None = None,
) -> None:
    """Attach analyzer routes to ``app``.

    Args:
        app: FastAPI app to extend.
        controller: Shared :class:`BaseController`. Used to fetch
            bars + the latest :class:`RecommendationRow` for the
            requested symbol.
        cache_dir: Directory where assembled reports are persisted.
            One JSON file per symbol. Reuses the same file-cache shape
            as the fundamentals + research endpoints.
        fundamentals_service: Optional injected service for tests. In
            production we lazily build one from settings on first call.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    service_holder: list[FundamentalsService | None] = [fundamentals_service]

    def _service() -> FundamentalsService:
        cached = service_holder[0]
        if cached is None:
            cached = FundamentalsService.from_settings(
                settings,
                health_store=health_store,
                retry_queue=retry_queue,
                accuracy_store=accuracy_store,
            )
            service_holder[0] = cached
        return cached

    async def _assemble(symbol: str, fresh: bool) -> dict[str, Any]:
        """Cache-aware assembly for one symbol. Shared by the analyzer
        route and the comparison route."""
        return await assemble_analyzer_report(
            symbol,
            controller=controller,
            service=_service(),
            cache_dir=cache_dir,
            settings=settings,
            calibration_store=calibration_store,
            fresh=fresh,
        )

    # Stash the assembler on the FastAPI app state so the comparison
    # route can reach it without re-wiring fundamentals + calibration
    # by hand. ``app.state`` is the canonical FastAPI extension point;
    # routes that need the assembler look it up by attribute.
    app.state.analyzer_assembler = _assemble

    @app.get("/api/analyzer/{symbol}")
    async def analyzer(symbol: str, fresh: bool = False) -> dict[str, Any]:
        """Return the assembled Financial Analyzer report for ``symbol``.

        Cache-first; the cache TTL piggybacks on
        ``fundamentals_cache_ttl_hours`` since fundamentals are the
        slowest-moving input. Pass ``?fresh=true`` to force a rebuild.
        """
        return await _assemble(symbol.upper(), fresh)

    @app.delete("/api/analyzer/{symbol}/cache")
    async def clear_analyzer_cache(symbol: str) -> dict[str, Any]:
        sym = symbol.upper()
        cache_file = cache_dir / f"{sym}.json"
        removed = 1 if cache_file.exists() else 0
        if removed:
            cache_file.unlink()
        return {"symbol": sym, "removed": removed}


# ---- helpers -------------------------------------------------------------


async def assemble_analyzer_report(
    symbol: str,
    *,
    controller: BaseController,
    service: FundamentalsService,
    cache_dir: Path,
    settings: Any,
    calibration_store: CalibrationStore | None,
    fresh: bool = False,
) -> dict[str, Any]:
    """Assemble the analyzer report dict for ``symbol``, cache-first.

    Standalone so the comparison endpoint can call it for two symbols
    in parallel without re-implementing the cache + pipeline logic.
    Returns the wire dict — same shape ``/api/analyzer/{symbol}`` emits.

    On any cache miss this walks the full pipeline (controller bars,
    fundamentals chain, valuation ensemble, scenarios, calibrations,
    explanation, trust score) and persists the result. Pass
    ``fresh=True`` to bypass the cache; useful when the user clicks
    "regenerate".
    """
    sym = symbol.upper()
    cache_file = cache_dir / f"{sym}.json"
    ttl_seconds = settings.fundamentals_cache_ttl_hours * 3600.0

    if not fresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < ttl_seconds:
            cached_payload: dict[str, Any] = json.loads(
                cache_file.read_text(encoding="utf-8")
            )
            cached_payload["cache"] = "hit"
            cached_payload["cache_age_seconds"] = int(age)
            return cached_payload

    log.info("analyzer.assemble.start", symbol=sym, fresh=fresh)

    warnings: list[str] = []

    last_price, sentiment_score, headline_count, headlines = await _pull_row_context(
        controller, sym, warnings
    )

    bars_df = await controller.fetch_bars(sym)
    if bars_df is None or bars_df.empty:
        warnings.append("Technical scoring unavailable — no bars returned.")
        technicals: TechnicalScores | None = None
    else:
        technicals = score_technicals(bars_df)

    fundamentals, fundamentals_freshness, fundamentals_warnings = (
        await _pull_fundamentals(service, sym)
    )
    warnings.extend(fundamentals_warnings)

    valuation: ValuationEnsemble | None = None
    if fundamentals is not None:
        valuation = build_valuation(fundamentals)

    scenarios: ScenarioModel | None = None
    if bars_df is not None and not bars_df.empty and last_price is not None:
        scenarios = build_scenarios(
            current_price=last_price,
            ohlcv=bars_df,
            valuation=valuation,
        )

    calibrations = _lookup_calibration_readings(
        technicals, calibration_store, settings
    )

    volume_z, volume_spike = _volume_signals(technicals)

    inputs = AnalyzerInputs(
        symbol=sym,
        last_price=last_price,
        technicals=technicals,
        fundamentals=fundamentals,
        valuation=valuation,
        sentiment_score=sentiment_score,
        headline_count=headline_count,
        top_headlines=headlines,
        volume_z=volume_z,
        volume_spike_score=volume_spike,
    )

    explanation: AnalyzerExplanation = build_grounded_explanation(inputs)

    trust_score = compute_analyzer_trust_score(
        freshness=(
            fundamentals_freshness.get("freshness")
            if fundamentals_freshness is not None
            else None
        ),
        provider_confidence=(
            fundamentals_freshness.get("provider_confidence")
            if fundamentals_freshness is not None
            else None
        ),
        analyzer_confidence=(
            technicals.confidence_score if technicals is not None else None
        ),
        calibration_coverage=_calibration_coverage(calibrations),
        scenarios_available=(scenarios is not None),
    )

    payload = _serialize_report(
        symbol=sym,
        last_price=last_price,
        technicals=technicals,
        valuation=valuation,
        fundamentals=fundamentals,
        fundamentals_freshness=fundamentals_freshness,
        scenarios=scenarios,
        calibrations=calibrations,
        explanation=explanation,
        trust_score=trust_score,
        warnings=warnings,
    )

    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    payload["cache"] = "miss"
    payload["cache_age_seconds"] = 0
    log.info(
        "analyzer.assemble.ok",
        symbol=sym,
        warnings=len(warnings),
        has_technicals=technicals is not None,
        has_valuation=valuation is not None,
    )
    return payload


async def _pull_row_context(
    controller: BaseController, symbol: str, warnings: list[str]
) -> tuple[float | None, float | None, int | None, tuple[str, ...]]:
    """Read sentiment + news + last-price from the live snapshot.

    The analyzer doesn't have its own news pipeline — it leans on the
    same :class:`RecommendationRow` the rest of the dashboard already
    renders so the score is consistent across surfaces.

    Returns ``(last_price, sentiment_score, headline_count, headlines)``.
    Each field is ``None`` / empty when the row is missing or the
    underlying value isn't finite.
    """
    import math

    try:
        snap = await controller.fetch_snapshot()
    except Exception as exc:
        warnings.append(f"Could not read snapshot for sentiment grounding: {exc}")
        return None, None, None, ()

    row = next((r for r in snap.rows if r.symbol == symbol), None)
    if row is None:
        warnings.append(f"Symbol {symbol} is not on the active watchlist.")
        return None, None, None, ()

    last_price: float | None = (
        float(row.last_price)
        if row.last_price is not None and math.isfinite(row.last_price)
        else None
    )
    sentiment = float(row.sentiment_score) if math.isfinite(row.sentiment_score) else None
    return last_price, sentiment, int(row.num_news_articles), tuple(row.headlines)


async def _pull_fundamentals(
    service: FundamentalsService, symbol: str
) -> tuple[NormalizedFundamentals | None, dict[str, Any] | None, list[str]]:
    """Try the provider chain; degrade to ``(None, None, [warnings])`` on failure.

    Returns ``(fundamentals, freshness_block, warnings)``. The freshness
    block is the envelope's wire shape — a dict the analyzer report
    embeds so the UI can show staleness badges next to the
    fundamentals card.
    """
    try:
        result = await service.fetch(symbol)
    except ProviderChainExhausted as exc:
        return None, None, [
            f"Fundamentals chain exhausted: {', '.join(exc.errors) or 'no providers'}."
        ]
    envelope = result.envelope
    freshness_block: dict[str, Any] = {
        "as_of": envelope.as_of.isoformat(),
        "fetched_at": envelope.fetched_at.isoformat(),
        "freshness": envelope.freshness,
        "data_age_days": round(envelope.data_age.total_seconds() / 86400.0, 2),
        "source_chain": list(envelope.source_chain),
        "provider_confidence": round(envelope.provider_confidence, 3),
        # Surface the cross-provider divergence count directly so the
        # analyzer card can render a "providers disagree" chip without
        # cross-referencing the fundamentals endpoint.
        "divergence_count": len(result.divergences),
        "reconciliation_warning_count": len(result.reconciliation_warnings),
    }
    return result.fundamentals, freshness_block, []


def _volume_signals(
    technicals: TechnicalScores | None,
) -> tuple[float | None, float | None]:
    """Pull volume z + spike score out of the technical scoring set."""
    if technicals is None:
        return None, None
    return technicals.raw_volume_z, technicals.subscores.volume_spike_score


def _lookup_calibration_readings(
    technicals: TechnicalScores | None,
    store: CalibrationStore | None,
    settings: Any,
) -> tuple[CalibrationReading, ...]:
    """Load the calibration table and produce readings for this symbol.

    Reads the materialized bucket cache on every analyzer call. The
    table is small (one row per bucket × score × horizon × outcome)
    so the per-request cost is negligible. Returns an empty tuple
    when the store is not wired or the table is empty / unbuilt —
    the wire shape carries an empty array, never absent.
    """
    if store is None or technicals is None:
        return ()
    try:
        table = store.load_table()
    except Exception as exc:  # defensive — never break the analyzer
        log.warning("calibration.load.failed", error=str(exc))
        return ()
    return lookup_readings(
        technicals,
        table,
        horizon_days=settings.calibration_horizon_days,
        min_observations=settings.calibration_min_observations,
    )


def _calibration_reading_to_wire(reading: CalibrationReading) -> dict[str, Any]:
    """Serialize one :class:`CalibrationReading` to its JSON shape.

    Bucket fields are ``None`` when ``bucket_published`` is False so
    the UI never has to disambiguate "we have data but didn't publish"
    from "we have data but kept it private" — only the published
    case carries the numeric block.
    """
    bucket = reading.bucket
    return {
        "score_name": reading.score_name,
        "score_value": reading.score_value,
        "outcome_name": reading.outcome_name,
        "horizon_days": reading.horizon_days,
        "bucket_published": reading.bucket_published,
        "bucket_lo": bucket.bucket_lo if bucket is not None else None,
        "bucket_hi": bucket.bucket_hi if bucket is not None else None,
        "n_observations": bucket.n_observations if bucket is not None else None,
        "n_hits": bucket.n_hits if bucket is not None else None,
        "hit_rate": bucket.hit_rate if bucket is not None else None,
        "confidence_low": bucket.confidence_low if bucket is not None else None,
        "confidence_high": bucket.confidence_high if bucket is not None else None,
        "last_updated": (
            bucket.last_updated.isoformat() if bucket is not None else None
        ),
    }


def _calibration_coverage(
    calibrations: tuple[CalibrationReading, ...],
) -> float | None:
    """Fraction of analyzer score/outcome pairings with a published bucket.

    Returns ``None`` when the readings tuple is empty (store unwired,
    table never built, or no technical scores) so the trust score
    marks calibration as ``missing`` rather than fabricating a zero.
    """
    if not calibrations:
        return None
    published = sum(1 for r in calibrations if r.bucket_published)
    return published / len(calibrations)


def _serialize_report(
    *,
    symbol: str,
    last_price: float | None,
    technicals: TechnicalScores | None,
    valuation: ValuationEnsemble | None,
    fundamentals: NormalizedFundamentals | None,
    fundamentals_freshness: dict[str, Any] | None,
    scenarios: ScenarioModel | None,
    calibrations: tuple[CalibrationReading, ...],
    explanation: AnalyzerExplanation,
    trust_score: TrustScore,
    warnings: list[str],
) -> dict[str, Any]:
    """Render the wire shape.

    The frontend mirrors the structure 1:1 in ``web/src/lib/analyzer.ts``
    — any field added here needs a TS-side update for the strict
    compile to keep passing.
    """
    technical_score = _technical_overall(technicals)
    valuation_score = (
        valuation.confidence_score if valuation is not None else None
    )
    fundamental_score = _fundamental_overall(fundamentals)
    overall = _overall_analyzer_score(
        technical_score, fundamental_score, valuation_score
    )
    return {
        "symbol": symbol,
        "generated_at": explanation.generated_at,
        "last_price": last_price,
        "overall_analyzer_score": overall,
        "technical_score": technical_score,
        "fundamental_score": fundamental_score,
        "valuation_score": valuation_score,
        "technicals": (
            jsonable_encoder(technicals.to_dict()) if technicals is not None else None
        ),
        "valuation": (
            jsonable_encoder(valuation.to_dict()) if valuation is not None else None
        ),
        "fundamentals_profile": (
            jsonable_encoder(asdict(fundamentals.profile))
            if fundamentals is not None
            else None
        ),
        "fundamentals_freshness": fundamentals_freshness,
        "scenarios": (
            jsonable_encoder(scenarios.to_dict()) if scenarios is not None else None
        ),
        "calibrations": [
            _calibration_reading_to_wire(r) for r in calibrations
        ],
        "explanation": jsonable_encoder(explanation.to_dict()),
        "trust_score": jsonable_encoder(trust_score.to_dict()),
        "warnings": warnings,
    }


def _technical_overall(scores: TechnicalScores | None) -> float | None:
    """Composite tech-side score for the header ring."""
    if scores is None:
        return None
    if scores.overbought_score is None and scores.oversold_score is None:
        return None
    # Use the stronger directional read as the headline number — that's
    # what the header ring visually represents.
    candidates = [
        s
        for s in (scores.overbought_score, scores.oversold_score)
        if s is not None
    ]
    return max(candidates) if candidates else None


def _fundamental_overall(funds: NormalizedFundamentals | None) -> float | None:
    """Lightweight 0-100 score reflecting fundamentals coverage + health.

    Until we ship a proper fundamentals-strength score, the header ring
    consumes a rough coverage measure: how many key data points are
    populated, gated by basic profitability + leverage flags.
    """
    if funds is None:
        return None
    populated = 0
    total = 6
    if funds.profile.market_cap:
        populated += 1
    if funds.profile.sector:
        populated += 1
    if funds.latest_annual_income is not None and (
        funds.latest_annual_income.revenue or 0
    ) > 0:
        populated += 1
    if funds.latest_annual_income is not None and (
        funds.latest_annual_income.net_income or 0
    ) > 0:
        populated += 1
    if funds.latest_annual_cashflow is not None and (
        funds.latest_annual_cashflow.free_cash_flow or 0
    ) > 0:
        populated += 1
    if funds.key_ratios.debt_to_equity is not None and funds.key_ratios.debt_to_equity < 1.5:
        populated += 1
    return (populated / total) * 100.0


def _overall_analyzer_score(
    technical: float | None,
    fundamental: float | None,
    valuation: float | None,
) -> float | None:
    """Average of the three sub-scores, dropping ``None``s."""
    parts = [v for v in (technical, fundamental, valuation) if v is not None]
    if not parts:
        return None
    return sum(parts) / len(parts)


__all__ = ["assemble_analyzer_report", "register_analyzer_routes"]
