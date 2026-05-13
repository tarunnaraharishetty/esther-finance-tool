"""LLM-backed summarizer using the Anthropic API.

Implements the same :class:`~src.intelligence.summary.Summarizer` protocol
as :class:`~src.intelligence.summary.TemplateSummarizer`, so the consumer
side (dashboard, CLI) is unaware of which backend is in use.

This module is *the* place we talk to Anthropic. Everything else in
``src/intelligence/`` stays deterministic and offline-runnable so tests
can mock at the SDK boundary here without spreading network coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anthropic

from src.config import Settings, get_settings
from src.strategy.base import SignalAction
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.intelligence.explain import Explanation

log = get_logger(__name__)


# Stable across every call — anchored in the product framing from CLAUDE.md.
# Cached via prompt caching (the marker is set on messages.create). Note:
# Opus 4.7's minimum cacheable prefix is ~4K tokens, so this short system
# prompt currently won't actually hit the cache — the marker is in place so
# that as we add more guidance and few-shot examples here, caching kicks in
# automatically. See shared/prompt-caching.md.
_SYSTEM_PROMPT = """You are Esther, an AI trading research assistant in a terminal dashboard. \
Your role is to help a human trader interpret market signals — not to trade on their behalf, \
not to recommend specific actions to execute.

For each request you receive a structured analysis of one symbol: a directional read \
(BUY / HOLD / SELL), a confidence level (weak / moderate / high), a combined score in \
[-1, 1], and per-signal contributors (RSI, MACD, Bollinger Bands, news sentiment). \
You may also receive recent news headlines.

GROUNDING RULES — these are non-negotiable and override any instinct to be helpful:

- Only summarize information present in the user message. Do not invent specific numbers, \
prices, percentages, or events not in the data provided.
- If a fact would require external knowledge (recent earnings beat, analyst price target, \
company news beyond what's quoted) and that fact is not in the user message, do not \
include it. Stay with what the data shows.
- Quote headlines verbatim or do not reference them at all. Do not paraphrase a headline \
into a stronger or weaker claim than its literal text. Do not extrapolate causation from a \
headline.
- Describe the current state — do not predict where the symbol will go. No forecasts, no \
"likely to" language, no probability claims.

Write a tight 2–4 sentence brief that:

1. Opens with the symbol and the directional read in plain language (e.g., "AAPL leans \
bullish with high confidence").
2. Cites the one or two strongest contributors and what they actually mean for the symbol \
(e.g., "RSI is in the oversold zone and MACD is showing strong upward momentum").
3. Notes tension between signals if present (e.g., "bullish technicals are offset by \
negative news sentiment over 12 articles").
4. Closes by framing the output as decision support, not execution advice ("worth watching", \
"consider in the context of your existing positions"). Never tell the user what to do.

Avoid: price targets, stop-loss recommendations, imperative phrasing like "you should buy", \
hype, hedging boilerplate, and disclaimers. Avoid markdown — plain prose only, no bullets \
or headers."""


class LLMSummarizer:
    """Anthropic-backed implementation of the Summarizer protocol.

    Construct once at startup and reuse. The Anthropic client is cheap to
    keep around — connection pooling is handled inside the SDK.
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
                    "explicit `client` argument to LLMSummarizer."
                )
            self.client = anthropic.Anthropic(api_key=key.get_secret_value())

    def summarize(
        self,
        explanation: Explanation,
        *,
        headlines: list[str] | None = None,
    ) -> str:
        """Generate a 2–4 sentence brief for one symbol.

        Surfaces Anthropic SDK exceptions to the caller — the CLI wraps
        them in a user-friendly error message.
        """
        user_message = _build_user_message(explanation, headlines)
        log.info(
            "llm.summarize.start",
            symbol=explanation.symbol,
            action=explanation.action.value,
            num_headlines=len(headlines) if headlines else 0,
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
            "llm.summarize.done",
            symbol=explanation.symbol,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read=getattr(response.usage, "cache_read_input_tokens", 0),
            cache_write=getattr(response.usage, "cache_creation_input_tokens", 0),
        )
        return text


def _build_user_message(
    explanation: Explanation, headlines: list[str] | None
) -> str:
    """Render the structured Explanation as plain text for the user turn.

    Format kept stable so future-us can A/B against alternative wordings
    without surprising the model.
    """
    lines: list[str] = [
        f"Symbol: {explanation.symbol}",
        (
            f"Recommendation: {explanation.action.value.upper()} "
            f"(confidence: {explanation.confidence_label}, {explanation.confidence:.2f}; "
            f"combined score: {explanation.combined_score:+.2f})"
        ),
        "",
        "Signal contributors:",
    ]
    if explanation.contributors:
        for c in explanation.contributors:
            lines.append(f"- {c.name} ({c.score:+.2f}): {c.note}")
    else:
        lines.append("- (no contributing signals — likely insufficient data)")

    if headlines:
        lines.append("")
        lines.append("Recent headlines:")
        for i, headline in enumerate(headlines[:8], start=1):
            lines.append(f'{i}. "{headline}"')

    if explanation.action == SignalAction.HOLD:
        lines.append("")
        lines.append(
            "Note: action is HOLD — there is no directional edge here. Frame accordingly."
        )

    return "\n".join(lines)


__all__ = ["LLMSummarizer"]
