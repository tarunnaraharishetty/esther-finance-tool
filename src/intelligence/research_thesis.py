"""Institutional-style equity research thesis generator.

Produces a structured, multi-section research report for one symbol —
the kind of artifact a sell-side analyst would draft. The dataclass
shape is the contract; two backends produce it:

* :class:`TemplateThesisGenerator` — deterministic, derived from the
  snapshot row + headlines we already have. Always available, no
  network, no API key. The fallback when ANTHROPIC_API_KEY is unset
  and the default for the bundled demo experience.
* :class:`LLMThesisGenerator` — Anthropic-backed. Same dataclass
  output. Wires a structured XML-tagged prompt + parser so the model
  can't return free-form prose the UI can't render.

Both generators consume the same :class:`ResearchInput` so the data
fetch logic lives in one place (``research_data.build_input``) and the
generator is purely a transform.

Future seams (slot in without touching call sites):

- :data:`ResearchInput.fundamentals` is currently always ``None``. When
  a fundamentals provider lands, populate it and the LLM prompt picks
  it up automatically; the template fallback degrades gracefully.
- :class:`ResearchThesis.metrics` is a free-form list of metric cards
  the UI renders — adding fundamentals later just appends entries.
- :func:`generate_thesis` is the public surface so a future "compare
  two stocks" or "portfolio thesis" endpoint can reuse the same primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.strategy.base import RecommendationTier, SignalAction

if TYPE_CHECKING:
    from src.dashboard.state import RecommendationRow


# ---------------------------------------------------------------------------
# Dataclasses — the contract the frontend renders.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThesisSection:
    """One narrative section of the report.

    ``body`` is plain prose (the UI may split on double-newlines for
    paragraphs). ``bullets`` are short, parallel-structure items the
    UI renders as a list. Either may be empty; the section renderer
    skips the missing half.

    ``provenance`` maps each numeric token surviving validation to
    its source-field label (e.g. ``"row.last_price"``). Empty when
    the validator hasn't run yet or no tokens grounded. The UI
    renders provenance entries as hover tooltips on the prose.
    """

    title: str
    body: str = ""
    bullets: tuple[str, ...] = ()
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BullBearArgument:
    """One side of the bull-vs-bear stack.

    ``weight`` is the analyst's own ranking of how load-bearing this
    argument is, normalized to [0, 1] across the side it belongs to.
    The UI uses it for the bar length so the eye can compare arguments
    against each other at a glance.

    ``provenance`` mirrors :class:`ThesisSection` — numeric tokens in
    ``detail`` mapped to their source-field labels.
    """

    label: str
    weight: float  # 0..1
    detail: str
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Catalyst:
    """Upcoming event with an expected directional bias."""

    label: str
    when: str  # human phrase: "Next 30 days" / "Q1 2026" / "post-CPI"
    impact: str  # "bullish" | "bearish" | "uncertain"
    detail: str
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OutlookEntry:
    """One horizon of the AI investment outlook triad."""

    horizon: str  # "short" | "medium" | "long"
    bias: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0..1
    detail: str
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricEntry:
    """One numeric metric card.

    Designed to render in a compact grid. ``tone`` colors the cell:
    ``bull`` / ``bear`` / ``warn`` / ``None`` (neutral). ``delta``
    is an optional secondary line ("+12% YoY", "vs sector avg 14.2").

    ``provenance`` records the source label for the metric's
    ``value`` (the most important numeric on the card). Empty when
    the validator hasn't run.
    """

    label: str
    value: str
    delta: str | None = None
    tone: str | None = None
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchThesis:
    """Full report — every field renderable on its own.

    The UI walks fields top-down; nothing here is required to be
    present except ``symbol`` / ``rating`` / ``generated_at``. Missing
    sections simply collapse.
    """

    symbol: str
    generated_at: str  # ISO-8601 with timezone
    model: str  # "template-v1" or "claude-sonnet-4-6@research-v1"
    rating: str  # "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
    confidence: float  # 0..1
    tagline: str  # one-line gist
    company_overview: ThesisSection
    bull_thesis: tuple[BullBearArgument, ...] = ()
    bear_thesis: tuple[BullBearArgument, ...] = ()
    technical_analysis: ThesisSection = field(
        default_factory=lambda: ThesisSection(title="Technical Analysis")
    )
    fundamental_analysis: ThesisSection = field(
        default_factory=lambda: ThesisSection(title="Fundamental Analysis")
    )
    metrics: tuple[MetricEntry, ...] = ()
    sentiment_news: ThesisSection = field(
        default_factory=lambda: ThesisSection(title="Sentiment & News")
    )
    catalysts: tuple[Catalyst, ...] = ()
    risk_assessment: ThesisSection = field(
        default_factory=lambda: ThesisSection(title="Risk Assessment")
    )
    outlook: tuple[OutlookEntry, ...] = ()
    explainability: ThesisSection = field(
        default_factory=lambda: ThesisSection(title="Why this read")
    )
    data_sources: tuple[str, ...] = ()
    disclaimers: tuple[str, ...] = (
        "Decision-support output — research only, not investment advice.",
        "Esther does not submit orders. All ratings reflect signal-based reads, not certainty.",
    )


@dataclass(frozen=True)
class ResearchInput:
    """Everything the generator needs about one symbol.

    Snapshot fields are required (the controller already produces
    them). Adapter fields are optional — they all default to ``None``
    until a real provider lands.
    """

    row: RecommendationRow
    headlines: tuple[str, ...]
    # Adapter-provided extras — all None today; the seams are reserved.
    company_profile: CompanyProfile | None = None
    fundamentals: FundamentalsSnapshot | None = None
    earnings: EarningsSnapshot | None = None
    analyst_ratings: AnalystRatings | None = None
    insider_activity: InsiderActivity | None = None
    macro_context: MacroContext | None = None
    social_sentiment: SocialSentiment | None = None


# ---------------------------------------------------------------------------
# Adapter shapes — pluggable, no implementation yet.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompanyProfile:
    name: str
    sector: str
    industry: str
    summary: str  # 2-3 sentence company description
    revenue_drivers: tuple[str, ...] = ()
    competitive_moat: str | None = None


@dataclass(frozen=True)
class FundamentalsSnapshot:
    pe: float | None = None
    forward_pe: float | None = None
    peg: float | None = None
    ev_ebitda: float | None = None
    price_to_sales: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    fcf_yield: float | None = None
    revenue_growth_yoy: float | None = None
    eps_growth_yoy: float | None = None
    debt_to_equity: float | None = None


@dataclass(frozen=True)
class EarningsSnapshot:
    next_report: datetime | None = None
    last_eps_actual: float | None = None
    last_eps_estimate: float | None = None
    last_revenue_actual: float | None = None
    last_revenue_estimate: float | None = None
    surprise_streak: int = 0  # consecutive beats (positive) or misses (negative)


@dataclass(frozen=True)
class AnalystRatings:
    consensus: str | None = None  # "Buy" | "Hold" | "Sell"
    buy_count: int = 0
    hold_count: int = 0
    sell_count: int = 0
    avg_target: float | None = None
    high_target: float | None = None
    low_target: float | None = None


@dataclass(frozen=True)
class InsiderActivity:
    net_30d_usd: float | None = None  # net insider buy ($) over 30 days
    buys_30d: int = 0
    sells_30d: int = 0
    last_event: str | None = None  # "CEO bought 50K shares · 3 days ago"


@dataclass(frozen=True)
class MacroContext:
    regime: str | None = None  # "risk-on" | "risk-off" | "rotational"
    next_fed_event: str | None = None
    next_cpi: str | None = None
    sector_trend: str | None = None  # short prose


@dataclass(frozen=True)
class SocialSentiment:
    overall: float | None = None  # -1..1
    volume_24h: int | None = None
    notable_threads: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Generators.
# ---------------------------------------------------------------------------


class TemplateThesisGenerator:
    """Deterministic research thesis derived from snapshot data only.

    Always available — no API key, no network. Output is structurally
    identical to the LLM generator so the UI doesn't branch. Tone is
    factual ("RSI is at 62, climbing through the upper band") rather
    than the polished analyst prose the LLM mode produces.
    """

    model_id = "template-v1"

    def generate(self, payload: ResearchInput) -> ResearchThesis:
        row = payload.row
        rating, conf, tagline = _classify(row)
        bulls, bears = _split_arguments(row)
        return ResearchThesis(
            symbol=row.symbol,
            generated_at=datetime.now(UTC).isoformat(),
            model=self.model_id,
            rating=rating,
            confidence=conf,
            tagline=tagline,
            company_overview=_company_overview(payload),
            bull_thesis=bulls,
            bear_thesis=bears,
            technical_analysis=_technical_section(row),
            fundamental_analysis=_fundamental_section(payload),
            metrics=_metrics(payload),
            sentiment_news=_sentiment_section(payload),
            catalysts=_catalysts(payload),
            risk_assessment=_risk_section(row),
            outlook=_outlook_triad(row),
            explainability=_explainability(row, rating, conf),
            data_sources=_data_sources(payload),
        )


# ---------------------------------------------------------------------------
# Template helpers — pure functions over the input.
# ---------------------------------------------------------------------------


def _classify(row: RecommendationRow) -> tuple[str, float, str]:
    """Map (action, tier, confidence) → (rating, confidence, tagline)."""
    rating_from_tier = {
        RecommendationTier.STRONG_BUY: "strong_buy",
        RecommendationTier.BUY: "buy",
        RecommendationTier.HOLD: "hold",
        RecommendationTier.SELL: "sell",
        RecommendationTier.STRONG_SELL: "strong_sell",
    }
    rating = rating_from_tier.get(row.tier, "hold")
    conf = max(0.0, min(1.0, row.confidence))
    direction = (
        "constructive"
        if row.action == SignalAction.BUY
        else "cautious"
        if row.action == SignalAction.SELL
        else "balanced"
    )
    tagline = (
        f"{row.symbol} reads {direction} on the current tape — "
        f"{rating.replace('_', ' ')} at {int(conf * 100)}% confidence."
    )
    return rating, conf, tagline


def _company_overview(payload: ResearchInput) -> ThesisSection:
    profile = payload.company_profile
    if profile is not None:
        bullets = [
            f"Sector: {profile.sector}",
            f"Industry: {profile.industry}",
        ]
        if profile.competitive_moat:
            bullets.append(f"Moat: {profile.competitive_moat}")
        if profile.revenue_drivers:
            bullets.append(
                "Revenue drivers: " + ", ".join(profile.revenue_drivers)
            )
        return ThesisSection(
            title="Company Overview", body=profile.summary, bullets=tuple(bullets)
        )
    # No profile available — write a stub the user can recognize as
    # waiting on a data source rather than hallucinated content.
    return ThesisSection(
        title="Company Overview",
        body=(
            f"{payload.row.symbol} — company profile data is not yet wired into "
            "Esther. Sector, business model, and revenue-driver breakdown will "
            "populate when a fundamentals provider (Polygon, FMP, or AlphaVantage) "
            "is connected. The signal pipeline is still fully active."
        ),
    )


def _technical_section(row: RecommendationRow) -> ThesisSection:
    body_parts: list[str] = []
    bullets: list[str] = []

    # RSI commentary.
    if row.rsi >= 70:
        bullets.append(
            f"RSI {row.rsi:.1f} — overbought; momentum strong but vulnerable to a pullback."
        )
    elif row.rsi <= 30:
        bullets.append(
            f"RSI {row.rsi:.1f} — oversold; potential mean-reversion setup if confirmed."
        )
    elif row.rsi >= 55:
        bullets.append(f"RSI {row.rsi:.1f} — momentum tilts bullish, not yet stretched.")
    elif row.rsi <= 45:
        bullets.append(f"RSI {row.rsi:.1f} — momentum tilts bearish, not yet capitulation.")
    else:
        bullets.append(f"RSI {row.rsi:.1f} — neutral, no directional bias.")

    # MACD.
    if row.macd > 0.05:
        bullets.append(f"MACD {row.macd:+.2f} — above signal, histogram supportive.")
    elif row.macd < -0.05:
        bullets.append(f"MACD {row.macd:+.2f} — below signal, histogram weakening.")
    else:
        bullets.append(f"MACD {row.macd:+.2f} — flat against signal, no momentum edge.")

    # Bollinger.
    if row.bollinger > 0.5:
        bullets.append(
            f"Bollinger {row.bollinger:+.2f} — price riding the upper band; "
            "trend-continuation regime."
        )
    elif row.bollinger < -0.5:
        bullets.append(
            f"Bollinger {row.bollinger:+.2f} — price near the lower band; "
            "extreme reading on the downside."
        )
    else:
        bullets.append(
            f"Bollinger {row.bollinger:+.2f} — mid-band oscillation; "
            "no volatility regime breakout."
        )

    # Stability + quality reasoning.
    body_parts.append(
        f"Composite technical score is {row.technical_score:+.2f} on a "
        f"[-1, +1] scale. Signal stability registers as {row.stability}; "
        f"quality score {row.signal_quality}."
    )
    if row.quality_reasons:
        body_parts.append(
            "Quality flags from the engine: " + ", ".join(row.quality_reasons) + "."
        )

    return ThesisSection(
        title="Technical Analysis",
        body=" ".join(body_parts),
        bullets=tuple(bullets),
    )


def _fundamental_section(payload: ResearchInput) -> ThesisSection:
    fundamentals = payload.fundamentals
    if fundamentals is None:
        return ThesisSection(
            title="Fundamental Analysis",
            body=(
                "Fundamental data (P/E, FCF, margins, revenue growth) is not yet "
                "wired into Esther. When a fundamentals provider is connected, "
                "this section will populate with valuation multiples, margin "
                "trajectory, and balance-sheet posture. The technical and "
                "sentiment signal stacks below are unaffected."
            ),
        )
    bullets: list[str] = []
    if fundamentals.pe is not None:
        bullets.append(f"P/E (trailing): {fundamentals.pe:.1f}")
    if fundamentals.forward_pe is not None:
        bullets.append(f"P/E (forward): {fundamentals.forward_pe:.1f}")
    if fundamentals.revenue_growth_yoy is not None:
        bullets.append(f"Revenue growth YoY: {fundamentals.revenue_growth_yoy:+.1%}")
    if fundamentals.eps_growth_yoy is not None:
        bullets.append(f"EPS growth YoY: {fundamentals.eps_growth_yoy:+.1%}")
    if fundamentals.gross_margin is not None:
        bullets.append(f"Gross margin: {fundamentals.gross_margin:.1%}")
    if fundamentals.operating_margin is not None:
        bullets.append(f"Operating margin: {fundamentals.operating_margin:.1%}")
    if fundamentals.fcf_yield is not None:
        bullets.append(f"FCF yield: {fundamentals.fcf_yield:.2%}")
    if fundamentals.debt_to_equity is not None:
        bullets.append(f"Debt / Equity: {fundamentals.debt_to_equity:.2f}")
    return ThesisSection(
        title="Fundamental Analysis",
        body="Snapshot of headline valuation and growth metrics.",
        bullets=tuple(bullets),
    )


def _metrics(payload: ResearchInput) -> tuple[MetricEntry, ...]:
    """Compact card grid — mixes technical + (when available) fundamentals."""
    row = payload.row
    cards: list[MetricEntry] = [
        MetricEntry(
            label="Last Price", value=f"${row.last_price:,.2f}", tone=None
        ),
        MetricEntry(
            label="Confidence",
            value=f"{int(row.confidence * 100)}%",
            tone="bull"
            if row.action == SignalAction.BUY
            else "bear"
            if row.action == SignalAction.SELL
            else "warn",
        ),
        MetricEntry(
            label="Composite",
            value=f"{row.combined_score:+.2f}",
            tone=_tone_for_score(row.combined_score),
        ),
        MetricEntry(
            label="RSI",
            value=f"{row.rsi:.1f}",
            tone="bear"
            if row.rsi >= 70
            else "bull"
            if row.rsi <= 30
            else "warn"
            if 45 <= row.rsi <= 55
            else None,
        ),
        MetricEntry(
            label="MACD",
            value=f"{row.macd:+.2f}",
            tone=_tone_for_score(row.macd),
        ),
        MetricEntry(
            label="Sentiment",
            value=f"{row.sentiment_score:+.2f}",
            delta=f"{row.num_news_articles} headlines",
            tone=_tone_for_score(row.sentiment_score),
        ),
    ]
    fundamentals = payload.fundamentals
    if fundamentals is not None:
        if fundamentals.forward_pe is not None:
            cards.append(
                MetricEntry(label="Fwd P/E", value=f"{fundamentals.forward_pe:.1f}")
            )
        if fundamentals.revenue_growth_yoy is not None:
            cards.append(
                MetricEntry(
                    label="Rev Growth",
                    value=f"{fundamentals.revenue_growth_yoy:+.1%}",
                    tone="bull" if fundamentals.revenue_growth_yoy > 0 else "bear",
                )
            )
        if fundamentals.operating_margin is not None:
            cards.append(
                MetricEntry(
                    label="Op Margin",
                    value=f"{fundamentals.operating_margin:.1%}",
                )
            )
    return tuple(cards)


def _sentiment_section(payload: ResearchInput) -> ThesisSection:
    row = payload.row
    body_parts: list[str] = []
    if row.num_news_articles == 0:
        body_parts.append(
            "No headlines registered against this symbol in the current window — "
            "the signal stack is technical-only for now."
        )
    else:
        direction = (
            "positive"
            if row.sentiment_score > 0.15
            else "negative"
            if row.sentiment_score < -0.15
            else "neutral"
        )
        body_parts.append(
            f"FinBERT scores the {row.num_news_articles}-article window at "
            f"{row.sentiment_score:+.2f} ({direction})."
        )
    if payload.social_sentiment is not None and payload.social_sentiment.overall is not None:
        body_parts.append(
            f"Social sentiment runs {payload.social_sentiment.overall:+.2f} across "
            f"{payload.social_sentiment.volume_24h or 0} 24h posts."
        )
    bullets = tuple(f'"{h}"' for h in payload.headlines[:5])
    return ThesisSection(
        title="Sentiment & News", body=" ".join(body_parts), bullets=bullets
    )


def _catalysts(payload: ResearchInput) -> tuple[Catalyst, ...]:
    out: list[Catalyst] = []
    if payload.earnings is not None and payload.earnings.next_report is not None:
        out.append(
            Catalyst(
                label="Earnings",
                when=payload.earnings.next_report.strftime("%Y-%m-%d"),
                impact="uncertain",
                detail=(
                    "Next earnings release — guidance and beat/miss tend to drive "
                    "the largest single-session moves."
                ),
            )
        )
    if payload.macro_context is not None:
        if payload.macro_context.next_fed_event:
            out.append(
                Catalyst(
                    label="Fed",
                    when=payload.macro_context.next_fed_event,
                    impact="uncertain",
                    detail="Rate-path commentary — sets the broader risk regime.",
                )
            )
        if payload.macro_context.next_cpi:
            out.append(
                Catalyst(
                    label="CPI",
                    when=payload.macro_context.next_cpi,
                    impact="uncertain",
                    detail="Inflation print — drives sector rotation in/out of duration.",
                )
            )
    return tuple(out)


def _risk_section(row: RecommendationRow) -> ThesisSection:
    bullets: list[str] = []
    if row.confidence < 0.4:
        bullets.append("Low conviction — signal stack is not yet aligned.")
    if row.rsi >= 70 and row.action == SignalAction.BUY:
        bullets.append("Overbought tape — entries here carry mean-reversion risk.")
    if row.rsi <= 30 and row.action == SignalAction.SELL:
        bullets.append("Oversold tape — bearish entries risk a relief bounce.")
    if abs(row.combined_score) > 0.6 and row.stability == "noisy":
        bullets.append("Strong signal but noisy series — risk of a fast reversal.")
    if not bullets:
        bullets.append(
            "No specific risk flags from the current signal stack — apply standard "
            "position sizing and stop discipline."
        )
    return ThesisSection(
        title="Risk Assessment",
        body=(
            "Risk readout combines confidence level, technical extremes, and "
            "signal stability. Treat as a checklist, not a vetoing committee."
        ),
        bullets=tuple(bullets),
    )


def _outlook_triad(row: RecommendationRow) -> tuple[OutlookEntry, ...]:
    """Short / Medium / Long-term outlooks derived from current signal posture."""
    base_bias = (
        "bullish"
        if row.action == SignalAction.BUY
        else "bearish"
        if row.action == SignalAction.SELL
        else "neutral"
    )
    return (
        OutlookEntry(
            horizon="short",
            bias=base_bias,
            confidence=row.confidence,
            detail=(
                f"Next 1–5 sessions track the current technical posture "
                f"(composite {row.combined_score:+.2f})."
            ),
        ),
        OutlookEntry(
            horizon="medium",
            bias=base_bias if row.confidence >= 0.5 else "neutral",
            confidence=max(0.0, row.confidence - 0.15),
            detail=(
                "Multi-week read depends on whether the news flow confirms the "
                "technical direction. Watch for fresh catalysts."
            ),
        ),
        OutlookEntry(
            horizon="long",
            bias="neutral",
            confidence=0.3,
            detail=(
                "Long-term read requires fundamentals (revenue/margin trajectory) "
                "which aren't yet wired in. Treat the long-term column as "
                "informational only until that data lands."
            ),
        ),
    )


def _explainability(
    row: RecommendationRow, rating: str, conf: float
) -> ThesisSection:
    body = (
        f"This {rating.replace('_', ' ')} rating ({int(conf * 100)}% confidence) "
        "comes directly from the multi-signal engine, not a forecast model. "
        f"Composite score {row.combined_score:+.2f} weighs technical "
        f"({row.technical_score:+.2f}) and sentiment ({row.sentiment_score:+.2f}) "
        "contributors equally before tier promotion gates the final rating. "
        "Every input is observable in the panels above — no hidden variables, "
        "no future-leak."
    )
    return ThesisSection(title="Why this read", body=body)


def _data_sources(payload: ResearchInput) -> tuple[str, ...]:
    sources = ["Alpaca paper-feed OHLCV", "Esther signal engine (RSI/MACD/Bollinger)"]
    if payload.row.num_news_articles > 0:
        sources.append("Alpaca news API + FinBERT sentiment")
    if payload.fundamentals is not None:
        sources.append("Fundamentals provider")
    if payload.analyst_ratings is not None:
        sources.append("Analyst ratings provider")
    if payload.insider_activity is not None:
        sources.append("Insider activity provider")
    if payload.macro_context is not None:
        sources.append("Macro context provider")
    if payload.social_sentiment is not None:
        sources.append("Social sentiment provider")
    return tuple(sources)


def _split_arguments(
    row: RecommendationRow,
) -> tuple[tuple[BullBearArgument, ...], tuple[BullBearArgument, ...]]:
    """Decompose the multi-signal composite into ranked bull/bear arguments."""
    bull: list[BullBearArgument] = []
    bear: list[BullBearArgument] = []

    def push(score: float, name: str, bullish_detail: str, bearish_detail: str) -> None:
        weight = min(1.0, abs(score))
        if score > 0.15:
            bull.append(BullBearArgument(label=name, weight=weight, detail=bullish_detail))
        elif score < -0.15:
            bear.append(BullBearArgument(label=name, weight=weight, detail=bearish_detail))

    # RSI as a [-1, 1]-scaled contribution: 50 = neutral.
    rsi_score = (row.rsi - 50) / 50
    push(
        rsi_score,
        "RSI momentum",
        f"RSI at {row.rsi:.1f} — upside momentum building.",
        f"RSI at {row.rsi:.1f} — downside momentum extending.",
    )
    push(
        row.macd,
        "MACD",
        f"MACD {row.macd:+.2f} above signal — trend-following confirmation.",
        f"MACD {row.macd:+.2f} below signal — trend-following pressure.",
    )
    push(
        row.bollinger,
        "Bollinger posture",
        f"Bollinger {row.bollinger:+.2f} — riding upper band, trend continuation.",
        f"Bollinger {row.bollinger:+.2f} — near lower band, downside extension.",
    )
    push(
        row.sentiment_score,
        "News sentiment",
        f"FinBERT scores news flow {row.sentiment_score:+.2f} across "
        f"{row.num_news_articles} articles — supportive.",
        f"FinBERT scores news flow {row.sentiment_score:+.2f} across "
        f"{row.num_news_articles} articles — headwind.",
    )

    bull.sort(key=lambda a: a.weight, reverse=True)
    bear.sort(key=lambda a: a.weight, reverse=True)

    # Re-normalize each side to sum to 1.0 so the bar lengths read as
    # shares of the total argument weight, not raw indicator magnitudes.
    bull = _normalize_weights(bull)
    bear = _normalize_weights(bear)
    return tuple(bull), tuple(bear)


def _normalize_weights(args: list[BullBearArgument]) -> list[BullBearArgument]:
    total = sum(a.weight for a in args)
    if total <= 0:
        return args
    return [
        BullBearArgument(label=a.label, weight=a.weight / total, detail=a.detail)
        for a in args
    ]


def _tone_for_score(score: float) -> str | None:
    if score > 0.15:
        return "bull"
    if score < -0.15:
        return "bear"
    return None


# ---------------------------------------------------------------------------
# Public surface — `generate_thesis` is the one entry point callers use.
# ---------------------------------------------------------------------------


def generate_thesis(
    payload: ResearchInput, *, generator: TemplateThesisGenerator | None = None
) -> ResearchThesis:
    """Produce a thesis using the given generator (template by default).

    Wraps construction so future caller code stays stable when LLMThesisGenerator
    or a hybrid generator becomes the default.
    """
    gen = generator or TemplateThesisGenerator()
    return gen.generate(payload)


__all__ = [
    "AnalystRatings",
    "BullBearArgument",
    "Catalyst",
    "CompanyProfile",
    "EarningsSnapshot",
    "FundamentalsSnapshot",
    "InsiderActivity",
    "MacroContext",
    "MetricEntry",
    "OutlookEntry",
    "ResearchInput",
    "ResearchThesis",
    "SocialSentiment",
    "TemplateThesisGenerator",
    "ThesisSection",
    "generate_thesis",
]
