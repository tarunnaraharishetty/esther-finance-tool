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
from src.intelligence.grounding import (
    GROUNDING_RULES_PER_SYMBOL,
    sanitize_prompt_value,
    sanitize_symbol,
)
from src.intelligence.opportunities import rank_opportunities
from src.intelligence.text_grounding import (
    build_corpus,
    build_default_allowance,
    validate_prose,
)
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot

log = get_logger(__name__)


# Few-shot examples below pad the system prefix past Opus 4.7's ~4K-token
# cache minimum AND anchor output across three distinct OPP shapes:
# top-of-stack alignment, fresh reversal, and high-conviction-with-noise.
# See shared/prompt-caching.md.
_FEW_SHOT_EXAMPLES = """EXAMPLES — these illustrate the structure and tone across three OPP \
shapes. Match the format: open with symbol + tier + rank + composite, cite the strongest \
one or two drivers with what they mean, frame the profile and streak, optionally cite one \
verbatim grounded data point, close with a watchful framing. Quote any headline verbatim.

--- Example 1 (Top-of-stack STRONG BUY, multi-driver alignment, sticky) ---

Input:
Symbol: NVDA
Tier: STRONG BUY
Composite score: 0.84 (rank #1 in top-N)

Per-driver scores (each in [0, 1]):
- technical_alignment: 1.00
- sentiment_alignment: 0.85
- confidence_acceleration: 0.55
- momentum_persistence: 0.70
- unusual_activity: 0.30
- reversal_strength: 0.00
- signal_quality_score: 1.00

Profile: stable · strengthening · persistent
Top-N membership: 7 consecutive ticks (appeared 9 of last 10)

Pre-computed rationale phrases (already shown in the OPP line):
- indicators aligned with action
- news sentiment matches direction
- momentum held 7 ticks
- high signal quality

Underlying market data:
- Confidence: 0.78
- Last price: $520.45
- RSI: +0.50
- MACD: +0.60
- Bollinger: +0.40
- News sentiment: +0.55 over 8 articles

Recent headlines (verbatim — quote exactly or do not reference):
1. "NVDA reports record quarterly revenue, beats estimates by 8%"
2. "Datacenter revenue growth accelerates on AI demand"

Output:
NVDA is the #1 opportunity at composite 0.84 — STRONG BUY tier. All three indicators \
align with the action and news sentiment matches direction over 8 articles, with \
momentum holding for 7 ticks. The signal is stable and strengthening, and has held a \
top-3 spot for 9 of the last 10 ticks. The headline "NVDA reports record quarterly \
revenue, beats estimates by 8%" is consistent with the technical setup. Worth watching \
whether the streak extends or starts to fade.

--- Example 2 (Fresh reversal, NEW entry to top-N) ---

Input:
Symbol: AAPL
Tier: BUY
Composite score: 0.62 (rank #2 in top-N)

Per-driver scores (each in [0, 1]):
- technical_alignment: 0.66
- sentiment_alignment: 0.40
- confidence_acceleration: 0.75
- momentum_persistence: 0.20
- unusual_activity: 0.15
- reversal_strength: 0.85
- signal_quality_score: 0.50

Profile: stable · strengthening · persistent
Top-N membership: 1 consecutive tick (appeared 1 of last 10)

Pre-computed rationale phrases (already shown in the OPP line):
- confidence rising in current run
- fresh reversal from 8-tick SELL

Underlying market data:
- Confidence: 0.61
- Last price: $185.20
- RSI: +0.30
- MACD: +0.40
- Bollinger: +0.10
- News sentiment: +0.20 over 3 articles

Recent headlines (verbatim — quote exactly or do not reference):
1. "Apple supplier signals improving production capacity for next quarter"
2. "Apple confirms expansion into a new wearable category"

Output:
AAPL is the #2 opportunity at composite 0.62 — BUY tier and brand new to the top-N this \
tick. The dominant drivers are reversal strength after an 8-tick SELL run and rising \
confidence in the current run, with technicals leaning into the new direction. The \
profile reads stable and strengthening, but with only one tick in the top-3 the staying \
power is unproven. Worth watching whether the reversal builds into a sustained move or \
fades back to the prior direction.

--- Example 3 (High conviction at the top, but noisy / flipping profile) ---

Input:
Symbol: TSLA
Tier: STRONG SELL
Composite score: 0.71 (rank #1 in top-N)

Per-driver scores (each in [0, 1]):
- technical_alignment: 1.00
- sentiment_alignment: 0.95
- confidence_acceleration: 0.50
- momentum_persistence: 0.30
- unusual_activity: 0.65
- reversal_strength: 0.00
- signal_quality_score: 1.00

Profile: noisy · weakening · flipping
Top-N membership: 2 consecutive ticks (appeared 4 of last 10)

Pre-computed rationale phrases (already shown in the OPP line):
- indicators aligned with action
- news sentiment matches direction
- unusual confidence swing
- high signal quality

Underlying market data:
- Confidence: 0.71
- Last price: $202.30
- RSI: -0.55
- MACD: -0.65
- Bollinger: -0.45
- News sentiment: -0.60 over 11 articles

Recent headlines (verbatim — quote exactly or do not reference):
1. "Tesla deliveries miss low end of analyst range"
2. "Tesla pricing cuts pressure margins, analyst notes"
3. "Multiple competitors expand EV lineups in Tesla's flagship segment"

Output:
TSLA is the #1 opportunity at composite 0.71 — STRONG SELL tier. All three indicators \
align with the action and news sentiment matches direction over 11 articles, with high \
signal quality and an unusual confidence swing. That said, the profile reads noisy, \
weakening, and flipping — the symbol has only held a top-3 spot for 4 of the last 10 \
ticks, so the conviction here is real but the durability is in question. The headline \
"Tesla deliveries miss low end of analyst range" reinforces the bearish read. Worth \
watching whether the signal stabilizes into a sustained run or chops back out of the \
top-N.

--- Example 4 (Multi-driver convergence, all seven drivers contributing) ---

Input:
Symbol: MSFT
Tier: STRONG BUY
Composite score: 0.79 (rank #1 in top-N)

Per-driver scores (each in [0, 1]):
- technical_alignment: 1.00
- sentiment_alignment: 0.78
- confidence_acceleration: 0.65
- momentum_persistence: 0.60
- unusual_activity: 0.55
- reversal_strength: 0.50
- signal_quality_score: 1.00

Profile: stable · strengthening · persistent
Top-N membership: 4 consecutive ticks (appeared 5 of last 10)

Pre-computed rationale phrases (already shown in the OPP line):
- indicators aligned with action
- news sentiment matches direction
- confidence rising in current run
- momentum held 4 ticks
- unusual confidence swing
- fresh reversal from 3-tick HOLD
- high signal quality

Underlying market data:
- Confidence: 0.74
- Last price: $432.10
- RSI: +0.42
- MACD: +0.55
- Bollinger: +0.30
- News sentiment: +0.60 over 7 articles

Recent headlines (verbatim — quote exactly or do not reference):
1. "Microsoft cloud-AI bookings hit fresh quarterly high"
2. "Microsoft expands enterprise Copilot offering, takes share in productivity tier"

Output:
MSFT is the #1 opportunity at composite 0.79 — STRONG BUY tier with all seven drivers \
contributing. Indicators align with the action, news sentiment matches direction over 7 \
articles, confidence is rising, momentum has held for 4 ticks, and the symbol just \
emerged from a brief HOLD into a fresh BUY. The profile reads stable, strengthening, \
and persistent, with a top-3 spot held for 5 of the last 10 ticks. The headline \
"Microsoft cloud-AI bookings hit fresh quarterly high" supports the technical setup. \
Worth watching whether the multi-driver alignment continues to broaden or whether one \
of the supporting drivers fades first.

--- Example 5 (Borderline composite, just above threshold) ---

Input:
Symbol: AMD
Tier: SELL
Composite score: 0.34 (rank #3 in top-N)

Per-driver scores (each in [0, 1]):
- technical_alignment: 0.66
- sentiment_alignment: 0.30
- confidence_acceleration: 0.50
- momentum_persistence: 0.20
- unusual_activity: 0.10
- reversal_strength: 0.15
- signal_quality_score: 0.50

Profile: stable · flat · persistent
Top-N membership: 1 consecutive tick (appeared 2 of last 10)

Pre-computed rationale phrases (already shown in the OPP line):
- indicators aligned with action

Underlying market data:
- Confidence: 0.52
- Last price: $148.75
- RSI: -0.30
- MACD: -0.35
- Bollinger: -0.10
- News sentiment: -0.20 over 4 articles

Recent headlines (verbatim — quote exactly or do not reference):
1. "AMD margin guidance comes in at the midpoint of analyst range"
2. "AMD reaffirms data-center pipeline ahead of next earnings cycle"

Output:
AMD sits at #3 in the OPP list at composite 0.34 — SELL tier and only barely above the \
ranking threshold. The dominant driver is technical alignment with the action; the \
other six drivers contribute little, and the profile is flat rather than strengthening. \
With only one tick in the top-3 and 2 of the last 10 historical, the staying power is \
limited. Worth watching whether more drivers light up to confirm the read or whether \
AMD slips out of the top-N as the technicals compress.

--- Example 6 (Confidence-acceleration driven, sparse rationale) ---

Input:
Symbol: GOOGL
Tier: BUY
Composite score: 0.51 (rank #2 in top-N)

Per-driver scores (each in [0, 1]):
- technical_alignment: 0.33
- sentiment_alignment: 0.20
- confidence_acceleration: 0.92
- momentum_persistence: 0.40
- unusual_activity: 0.45
- reversal_strength: 0.20
- signal_quality_score: 0.50

Profile: stable · strengthening · persistent
Top-N membership: 3 consecutive ticks (appeared 3 of last 10)

Pre-computed rationale phrases (already shown in the OPP line):
- confidence rising in current run

Underlying market data:
- Confidence: 0.62
- Last price: $172.40
- RSI: +0.20
- MACD: +0.25
- Bollinger: +0.15
- News sentiment: +0.10 over 3 articles

Recent headlines (verbatim — quote exactly or do not reference):
1. "Alphabet earnings preview: Street looking for cloud reacceleration"

Output:
GOOGL is the #2 opportunity at composite 0.51 — BUY tier, with the read driven \
predominantly by confidence acceleration at 0.92. The other drivers are quieter — \
technical alignment is partial, news sentiment is barely positive over 3 articles — so \
the OPP setup leans on the rate-of-confidence change rather than broad multi-driver \
agreement. The profile reads stable and strengthening across a 3-tick run in the top-3. \
Worth watching whether more drivers light up to confirm the rising-confidence read or \
whether the acceleration fades on its own.

--- Example 7 (High-conviction reversal with limited history) ---

Input:
Symbol: COIN
Tier: STRONG BUY
Composite score: 0.66 (rank #1 in top-N)

Per-driver scores (each in [0, 1]):
- technical_alignment: 0.66
- sentiment_alignment: 0.95
- confidence_acceleration: 0.85
- momentum_persistence: 0.20
- unusual_activity: 0.40
- reversal_strength: 0.95
- signal_quality_score: 1.00

Profile: stable · strengthening · persistent
Top-N membership: 1 consecutive tick (appeared 1 of last 10)

Pre-computed rationale phrases (already shown in the OPP line):
- news sentiment matches direction
- confidence rising in current run
- fresh reversal from 9-tick SELL
- high signal quality

Underlying market data:
- Confidence: 0.74
- Last price: $215.30
- RSI: +0.40
- MACD: +0.50
- Bollinger: +0.25
- News sentiment: +0.65 over 9 articles

Recent headlines (verbatim — quote exactly or do not reference):
1. "Coinbase secures regulatory clarity in two key European markets"
2. "Crypto trading volumes hit a six-month high, exchange operators benefit"
3. "Coinbase margin guidance lifts on the back of derivatives ramp"

Output:
COIN is the #1 opportunity at composite 0.66 — STRONG BUY tier and brand new to the \
top-N this tick. The dominant drivers are reversal strength after a 9-tick SELL run, \
news sentiment matching direction over 9 articles, and rising confidence in the new \
run. The profile reads stable and strengthening, but with only one tick in the top-3 \
the durability is unproven. The headline "Coinbase secures regulatory clarity in two \
key European markets" supports the reversal narrative. Worth watching whether the new \
direction sustains beyond the first few ticks of confirmation."""


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
buy", hype, hedging boilerplate, generic disclaimers, and markdown formatting.

{_FEW_SHOT_EXAMPLES}"""


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
            temperature=self.settings.llm_temperature,
            timeout=self.settings.llm_timeout_seconds,
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
        # Post-hoc grounding pass — drop any sentence whose numeric tokens
        # aren't in the input corpus. Same contract as LLMSummarizer +
        # LLMRecapGenerator + the compare narrative validator.
        validated, dropped = validate_prose(
            text,
            corpus=_build_brief_corpus(context),
            allowance=build_default_allowance(symbol=context.symbol),
            where="llm_opp_brief",
        )
        if dropped:
            log.warning(
                "llm.opp_brief.dropped_claims",
                symbol=context.symbol,
                drop_count=len(dropped),
                tokens=[t for c in dropped for t in c.unsupported_tokens],
            )
        return validated


def _build_brief_corpus(ctx: OpportunityBriefContext) -> str:
    """Assemble the substring corpus for validating an OPP brief.

    Includes every numeric scalar the prompt exposes to the model: the
    seven driver scores, composite + rank, the underlying indicators,
    sentiment + article count, top-N membership counters, pre-computed
    rationale phrases, and headlines verbatim.
    """
    parts: list[str] = [
        ctx.symbol,
        ctx.tier_display,
        f"{ctx.composite_score:.2f}",
        str(ctx.rank),
        # Seven drivers, each in the surface forms the prompt uses.
        f"{ctx.technical_alignment:.2f}",
        f"{ctx.sentiment_alignment:.2f}",
        f"{ctx.confidence_acceleration:.2f}",
        f"{ctx.momentum_persistence:.2f}",
        f"{ctx.unusual_activity:.2f}",
        f"{ctx.reversal_strength:.2f}",
        f"{ctx.signal_quality_score:.2f}",
        # Profile labels.
        ctx.stability,
        ctx.trend,
        ctx.persistence,
        # Underlying market data.
        f"{ctx.confidence:.2f}",
        f"{ctx.rsi:.2f}",
        f"{ctx.macd:+.2f}",
        f"{ctx.bollinger:+.2f}",
        f"{ctx.sentiment_score:+.2f}",
        str(ctx.num_news_articles),
        # Membership history counters.
        str(ctx.streak),
        str(ctx.appearances),
        str(ctx.window),
    ]
    if _is_finite(ctx.last_price):
        parts.append(f"{ctx.last_price:,.2f}")
        parts.append(f"{ctx.last_price:.2f}")
    parts.extend(ctx.rationale)
    parts.extend(ctx.headlines)
    return build_corpus(*parts)


def _build_brief_user_message(ctx: OpportunityBriefContext) -> str:
    """Render the user message: structured, slot-filled, deterministic.

    Mirrors the per-symbol summary format so future-us can A/B prompts
    without surprising the model.
    """
    safe_symbol = sanitize_symbol(ctx.symbol)
    lines: list[str] = [
        f"Symbol: {safe_symbol}",
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
            lines.append(f'{i}. "{sanitize_prompt_value(headline)}"')

    return "\n".join(lines)


def _is_finite(x: float) -> bool:
    return x == x  # NaN != NaN


def _indicator_line(name: str, value: float) -> str:
    if not _is_finite(value):
        return f"- {name}: n/a"
    return f"- {name}: {value:+.2f}"


__all__ = ["LLMOpportunityBriefer", "OpportunityBriefContext"]
