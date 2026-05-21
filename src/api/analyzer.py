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

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

from src.config import get_settings
from src.intelligence.analyzer import (
    AnalyzerExplanation,
    AnalyzerInputs,
    build_grounded_explanation,
    build_valuation,
    score_technicals,
)
from src.intelligence.analyzer.technical import TechnicalScores
from src.intelligence.analyzer.valuation import ValuationEnsemble
from src.intelligence.fundamentals import (
    FundamentalsService,
    NormalizedFundamentals,
    ProviderChainExhausted,
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
        if service_holder[0] is None:
            service_holder[0] = FundamentalsService.from_settings(settings)
        return service_holder[0]

    @app.get("/api/analyzer/{symbol}")
    async def analyzer(symbol: str, fresh: bool = False) -> dict[str, Any]:
        """Return the assembled Financial Analyzer report for ``symbol``.

        Cache-first; the cache TTL piggybacks on
        ``fundamentals_cache_ttl_hours`` since fundamentals are the
        slowest-moving input. Pass ``?fresh=true`` to force a rebuild.
        """
        sym = symbol.upper()
        cache_file = cache_dir / f"{sym}.json"
        ttl_seconds = settings.fundamentals_cache_ttl_hours * 3600.0

        if not fresh and cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < ttl_seconds:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                payload["cache"] = "hit"
                payload["cache_age_seconds"] = int(age)
                return payload

        log.info("analyzer.assemble.start", symbol=sym, fresh=fresh)

        # ---- gather inputs ----
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

        fundamentals, fundamentals_warnings = await _pull_fundamentals(
            _service(), sym
        )
        warnings.extend(fundamentals_warnings)

        valuation: ValuationEnsemble | None = None
        if fundamentals is not None:
            valuation = build_valuation(fundamentals)

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

        payload = _serialize_report(
            symbol=sym,
            last_price=last_price,
            technicals=technicals,
            valuation=valuation,
            fundamentals=fundamentals,
            explanation=explanation,
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

    @app.delete("/api/analyzer/{symbol}/cache")
    async def clear_analyzer_cache(symbol: str) -> dict[str, Any]:
        sym = symbol.upper()
        cache_file = cache_dir / f"{sym}.json"
        removed = 1 if cache_file.exists() else 0
        if removed:
            cache_file.unlink()
        return {"symbol": sym, "removed": removed}


# ---- helpers -------------------------------------------------------------


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
) -> tuple[NormalizedFundamentals | None, list[str]]:
    """Try the provider chain; degrade to None + a warning on failure."""
    try:
        result = await service.fetch(symbol)
    except ProviderChainExhausted as exc:
        return None, [
            f"Fundamentals chain exhausted: {', '.join(exc.errors) or 'no providers'}."
        ]
    return result.fundamentals, []


def _volume_signals(
    technicals: TechnicalScores | None,
) -> tuple[float | None, float | None]:
    """Pull volume z + spike score out of the technical scoring set."""
    if technicals is None:
        return None, None
    return technicals.raw_volume_z, technicals.subscores.volume_spike_score


def _serialize_report(
    *,
    symbol: str,
    last_price: float | None,
    technicals: TechnicalScores | None,
    valuation: ValuationEnsemble | None,
    fundamentals: NormalizedFundamentals | None,
    explanation: AnalyzerExplanation,
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
        "explanation": jsonable_encoder(explanation.to_dict()),
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


__all__ = ["register_analyzer_routes"]
