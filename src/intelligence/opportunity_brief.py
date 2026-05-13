"""Per-opportunity AI brief.

Generates a short prose explanation for one ranked opportunity — "why
is this symbol on the OPP list right now?" Same shape as
:class:`~src.intelligence.recap.LLMRecapGenerator` but at the
opportunity level instead of the watchlist level, with the same
anti-hallucination grounding rules.

Inputs (frozen :class:`OpportunityBriefContext`):
- The ranked opportunity itself: symbol, tier, composite score, rank,
  the seven driver scores, the three-axis profile, and the
  pre-computed rationale phrases the renderer already shows.
- The underlying market data: price, RSI/MACD/BB, sentiment, news count.
- Recent headlines (verbatim strings).
- Top-N membership history: streak + appearances within the rolling window.

Output: 3–5 sentences of plain prose explaining what's driving the
opportunity, framed as decision support — never execution advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import anthropic

from src.config import Settings, get_settings
from src.intelligence.grounding import GROUNDING_RULES_PER_SYMBOL
from src.intelligence.opportunities import rank_opportunities
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot

log = get_logger(__name__)


_SYSTEM_PROMPT = f"""You are Esther, an AI trading research assistant in a terminal dashboard. \
This task is the OPP brief: explain why this symbol is currently ranked among the top \
opportunities on the trader's watchlist. Decision support — never execution advice, never \
autonomous instruction.

You receive a structured fact set: the symbol's tier (e.g., STRONG BUY), a composite \
opportunity score in [0, 1], its rank within the top-N, per-driver scores across seven \
dimensions (technical_alignment, sentiment_alignment, confidence_acceleration, \
momentum_persistence, unusual_activity, reversal_strength, signal_quality_score), a \
three-axis signal profile (stability, trend, persistence), pre-computed rationale phrases \
the dashboard renders, the underlying market data (price, RSI, MACD, Bollinger, sentiment, \
news count), recent headlines, and how many of the last N ticks the symbol has held a \
top-N spot.

{GROUNDING_RULES_PER_SYMBOL}
- The composite score and per-driver scores are the engine's outputs. Discuss what they \
say; do not reinterpret or override them.
- The profile axes are deterministic classifications (stable vs noisy, strengthening vs \
weakening vs flat, persistent vs flipping). Use them as-is.
- The streak ("X of last N ticks in top-N") is a measured fact, not a prediction.

Structure your output (3–5 short sentences, plain prose, no markdown):

1. Open with the symbol, tier, rank, and composite score (e.g., "NVDA is the #1 \
opportunity at composite 0.82 — STRONG BUY tier.").
2. Cite the strongest one or two drivers and what they mean (e.g., "All three indicators \
align with the action and news sentiment matches direction over 8 articles.").
3. Frame the profile + streak (e.g., "The signal is stable, strengthening, and has held \
top-3 for 5 of the last 10 ticks.").
4. Optionally reference one verbatim grounded data point — a price, an indicator value, \
or a headline (quoted exactly).
5. Close with a watchful framing — what to keep an eye on, not what to do.

Avoid: price targets, stop-loss recommendations, imperative phrasing like "you should \
buy", hype, hedging boilerplate, generic disclaimers, and markdown formatting."""


@dataclass(frozen=True)
class OpportunityBriefContext:
    """Frozen facts captured from a snapshot for OPP brief generation.

    Constructed by :meth:`from_snapshot` from data already in the
    system; never holds free-form descriptions or out-of-band facts.
    The LLM literally cannot speculate about anything not in here.
    """

    # Identity + ranking.
    symbol: str
    tier_display: str  # e.g., "STRONG BUY"
    composite_score: float  # [0, 1]
    rank: int  # 1-indexed position within top-N

    # Seven drivers (each in [0, 1]).
    technical_alignment: float
    sentiment_alignment: float
    confidence_acceleration: float
    momentum_persistence: float
    unusual_activity: float
    reversal_strength: float
    signal_quality_score: float

    # Three-axis profile.
    stability: str  # "stable" | "noisy"
    trend: str  # "strengthening" | "weakening" | "flat"
    persistence: str  # "persistent" | "flipping"

    # Pre-computed rationale phrases (the OPP renderer already shows these).
    rationale: tuple[str, ...]

    # Underlying market data (grounded numbers the model can cite).
    confidence: float
    last_price: float
    rsi: float
    macd: float
    bollinger: float
    sentiment_score: float
    num_news_articles: int

    # Verbatim headlines for direct quoting.
    headlines: tuple[str, ...] = ()

    # Top-N membership history.
    streak: int = 0
    appearances: int = 0
    window: int = 0

    @classmethod
    def from_snapshot(
        cls,
        snapshot: DashboardSnapshot,
        symbol: str,
        *,
        n: int = 3,
        max_headlines: int = 5,
    ) -> OpportunityBriefContext | None:
        """Build a context for ``symbol`` if it currently ranks in top-N.

        Returns ``None`` when the symbol isn't in the snapshot, has an
        error, or doesn't make the top-N rank — the caller treats
        ``None`` as "no brief possible" rather than fabricating one.
        """
        ranked = rank_opportunities(snapshot, n=n)
        match = next(
            ((i, opp) for i, opp in enumerate(ranked) if opp.symbol == symbol),
            None,
        )
        if match is None:
            return None
        rank_idx, opp = match

        row = next(
            (r for r in snapshot.rows if r.symbol == symbol and not r.error),
            None,
        )
        if row is None:
            return None

        history = snapshot.opp_history.get(symbol)
        return cls(
            symbol=opp.symbol,
            tier_display=opp.tier.display,
            composite_score=opp.composite_score,
            rank=rank_idx + 1,
            technical_alignment=opp.technical_alignment,
            sentiment_alignment=opp.sentiment_alignment,
            confidence_acceleration=opp.confidence_acceleration,
            momentum_persistence=opp.momentum_persistence,
            unusual_activity=opp.unusual_activity,
            reversal_strength=opp.reversal_strength,
            signal_quality_score=opp.signal_quality_score,
            stability=opp.profile.stability,
            trend=opp.profile.trend,
            persistence=opp.profile.persistence,
            rationale=opp.rationale,
            confidence=row.confidence,
            last_price=row.last_price,
            rsi=row.rsi,
            macd=row.macd,
            bollinger=row.bollinger,
            sentiment_score=row.sentiment_score,
            num_news_articles=row.num_news_articles,
            headlines=tuple(row.headlines[:max_headlines]),
            streak=history.streak if history is not None else 0,
            appearances=history.appearances if history is not None else 0,
            window=history.window if history is not None else 0,
        )


class LLMOpportunityBriefer:
    """Anthropic-backed OPP briefer. Same wire shape as
    :class:`~src.intelligence.recap.LLMRecapGenerator` and
    :class:`~src.intelligence.llm_summary.LLMSummarizer`.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if client is not None:
            self.client = client
        else:
            key = self.settings.anthropic_api_key
            if key is None or not key.get_secret_value().strip():
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not configured. Set it in .env or pass an "
                    "explicit `client` argument to LLMOpportunityBriefer."
                )
            self.client = anthropic.Anthropic(api_key=key.get_secret_value())

    def brief(self, context: OpportunityBriefContext) -> str:
        """Render the OPP brief. Surfaces Anthropic SDK exceptions to
        the caller — the dashboard surfaces them as an error state."""
        user_message = _build_brief_user_message(context)
        log.info(
            "llm.opp_brief.start",
            symbol=context.symbol,
            tier=context.tier_display,
            composite=round(context.composite_score, 3),
            rank=context.rank,
        )
        response = self.client.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            system=_SYSTEM_PROMPT,
            output_config={"effort": "low"},
            cache_control={"type": "ephemeral"},
            messages=[{"role": "user", "content": user_message}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        log.info(
            "llm.opp_brief.done",
            symbol=context.symbol,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read=getattr(response.usage, "cache_read_input_tokens", 0),
            cache_write=getattr(response.usage, "cache_creation_input_tokens", 0),
        )
        return text


def _build_brief_user_message(ctx: OpportunityBriefContext) -> str:
    """Render the user message: structured, slot-filled, deterministic.

    Mirrors the per-symbol summary format so future-us can A/B prompts
    without surprising the model.
    """
    lines: list[str] = [
        f"Symbol: {ctx.symbol}",
        f"Tier: {ctx.tier_display}",
        f"Composite score: {ctx.composite_score:.2f} (rank #{ctx.rank} in top-N)",
        "",
        "Per-driver scores (each in [0, 1]):",
        f"- technical_alignment: {ctx.technical_alignment:.2f}",
        f"- sentiment_alignment: {ctx.sentiment_alignment:.2f}",
        f"- confidence_acceleration: {ctx.confidence_acceleration:.2f}",
        f"- momentum_persistence: {ctx.momentum_persistence:.2f}",
        f"- unusual_activity: {ctx.unusual_activity:.2f}",
        f"- reversal_strength: {ctx.reversal_strength:.2f}",
        f"- signal_quality_score: {ctx.signal_quality_score:.2f}",
        "",
        f"Profile: {ctx.stability} · {ctx.trend} · {ctx.persistence}",
    ]
    if ctx.window > 0:
        lines.append(
            f"Top-N membership: {ctx.streak} consecutive ticks "
            f"(appeared {ctx.appearances} of last {ctx.window})"
        )
    if ctx.rationale:
        lines.append("")
        lines.append("Pre-computed rationale phrases (already shown in the OPP line):")
        for phrase in ctx.rationale:
            lines.append(f"- {phrase}")

    lines.append("")
    lines.append("Underlying market data:")
    lines.append(f"- Confidence: {ctx.confidence:.2f}")
    lines.append(f"- Last price: ${ctx.last_price:,.2f}" if _is_finite(ctx.last_price) else "- Last price: n/a")
    lines.append(_indicator_line("RSI", ctx.rsi))
    lines.append(_indicator_line("MACD", ctx.macd))
    lines.append(_indicator_line("Bollinger", ctx.bollinger))
    lines.append(
        f"- News sentiment: {ctx.sentiment_score:+.2f} over {ctx.num_news_articles} "
        f"article{'s' if ctx.num_news_articles != 1 else ''}"
    )

    if ctx.headlines:
        lines.append("")
        lines.append("Recent headlines (verbatim — quote exactly or do not reference):")
        for i, headline in enumerate(ctx.headlines, start=1):
            lines.append(f'{i}. "{headline}"')

    return "\n".join(lines)


def _is_finite(x: float) -> bool:
    return x == x  # NaN != NaN


def _indicator_line(name: str, value: float) -> str:
    if not _is_finite(value):
        return f"- {name}: n/a"
    return f"- {name}: {value:+.2f}"


__all__ = ["LLMOpportunityBriefer", "OpportunityBriefContext"]
