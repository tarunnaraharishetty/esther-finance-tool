"""Per-symbol summary generation.

The template implementation is deterministic and offline — it composes
prose from a structured :class:`~src.intelligence.explain.Explanation`.
A future LLM-backed implementation will satisfy the same
:class:`Summarizer` protocol and can be swapped in without UI changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.strategy.base import SignalAction

if TYPE_CHECKING:
    from src.data.models import NewsArticle
    from src.intelligence.explain import Explanation


class Summarizer(Protocol):
    """Produce a short prose brief for one symbol."""

    def summarize(
        self,
        explanation: Explanation,
        *,
        headlines: list[NewsArticle] | None = None,
    ) -> str: ...


class TemplateSummarizer:
    """Deterministic template-based summarizer.

    Useful for tests, offline mode, and as a fallback when the LLM
    backend is unavailable. Output is intentionally short — 2-4 sentences.
    """

    def summarize(
        self,
        explanation: Explanation,
        *,
        headlines: list[NewsArticle] | None = None,
    ) -> str:
        parts = [explanation.headline]

        if explanation.contributors:
            bullish = [c for c in explanation.contributors if c.direction == "bullish"]
            bearish = [c for c in explanation.contributors if c.direction == "bearish"]
            neutral = [c for c in explanation.contributors if c.direction == "neutral"]

            if bullish and bearish:
                parts.append(
                    f"Bullish reads from {_join_names(bullish)} are partially offset by "
                    f"bearish reads from {_join_names(bearish)}."
                )
            elif bullish:
                parts.append(f"Bullish across {_join_names(bullish)}.")
            elif bearish:
                parts.append(f"Bearish across {_join_names(bearish)}.")
            elif neutral:
                parts.append(f"Inputs from {_join_names(neutral)} are inconclusive.")

        if headlines:
            top = headlines[0]
            parts.append(f'Latest headline: "{top.headline}".')

        parts.append(_action_caveat(explanation.action))
        return " ".join(parts)


def _join_names(contributors: list) -> str:
    names = [c.name for c in contributors]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _action_caveat(action: SignalAction) -> str:
    return {
        SignalAction.BUY: "This is a research signal — confirm against your own setup before acting.",
        SignalAction.SELL: "This is a research signal — confirm against your own setup before acting.",
        SignalAction.HOLD: "Worth watching, but no clear entry or exit here.",
    }[action]
