"""Structured explanations for a single recommendation.

Translates the normalized scores produced by
:class:`src.strategy.recommendation.RecommendationEngine` into a human-
readable structure the UI can render verbatim or pipe into a summary
generator. Pure function — no side effects, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.strategy.base import SignalAction

# Score thresholds for translating a normalized [-1, 1] score into a label.
# Symmetric: positive = bullish, negative = bearish.
_STRONG = 0.6
_MODERATE = 0.3

# Confidence thresholds for the overall recommendation.
_CONF_HIGH = 0.6
_CONF_MODERATE = 0.3


@dataclass(frozen=True)
class Contributor:
    """One input to the combined score, with a human-readable note."""

    name: str  # "RSI", "MACD", "Bollinger", "News sentiment"
    score: float  # [-1, 1], positive = bullish
    note: str  # plain English

    @property
    def direction(self) -> str:
        if self.score >= _MODERATE:
            return "bullish"
        if self.score <= -_MODERATE:
            return "bearish"
        return "neutral"


@dataclass(frozen=True)
class Explanation:
    """Full structured "why" for a recommendation."""

    symbol: str
    action: SignalAction
    confidence: float  # [0, 1]
    confidence_label: str  # "high" | "moderate" | "weak"
    headline: str  # one-sentence verdict
    contributors: list[Contributor]
    combined_score: float  # [-1, 1]


def explain(
    *,
    symbol: str,
    action: SignalAction,
    confidence: float,
    combined_score: float,
    indicator_scores: dict[str, float],
    sentiment_score: float,
    num_news_articles: int,
) -> Explanation:
    """Build an :class:`Explanation` from recommendation scores.

    Takes structured inputs (not a recommendation object) so both
    :class:`~src.strategy.recommendation.TradingRecommendation` and
    :class:`~src.dashboard.state.RecommendationRow` can call it without
    cross-layer coupling.
    """
    contributors = _build_contributors(indicator_scores, sentiment_score, num_news_articles)
    confidence_label = _confidence_label(confidence)
    headline = _headline(symbol, action, confidence_label)
    return Explanation(
        symbol=symbol,
        action=action,
        confidence=confidence,
        confidence_label=confidence_label,
        headline=headline,
        contributors=contributors,
        combined_score=combined_score,
    )


def _build_contributors(
    indicator_scores: dict[str, float],
    sentiment_score: float,
    num_news_articles: int,
) -> list[Contributor]:
    out: list[Contributor] = []
    for key, label in (("rsi", "RSI"), ("macd", "MACD"), ("bollinger", "Bollinger")):
        if key in indicator_scores:
            score = indicator_scores[key]
            out.append(Contributor(name=label, score=score, note=_indicator_note(key, score)))
    if num_news_articles > 0:
        out.append(
            Contributor(
                name="News sentiment",
                score=sentiment_score,
                note=_sentiment_note(sentiment_score, num_news_articles),
            )
        )
    return out


def _indicator_note(key: str, score: float) -> str:
    strength = _strength_label(score)
    direction = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
    if abs(score) < _MODERATE:
        # Below the threshold for any directional read.
        return _flat_indicator_note(key)
    return {
        "rsi": _rsi_note(score, strength, direction),
        "macd": _macd_note(score, strength, direction),
        "bollinger": _bollinger_note(score, strength, direction),
    }.get(key, f"{strength} {direction}")


def _flat_indicator_note(key: str) -> str:
    return {
        "rsi": "RSI near midline — no directional read",
        "macd": "MACD line near zero — no clear momentum",
        "bollinger": "price near Bollinger midline — no mean-reversion edge",
    }.get(key, "neutral")


def _rsi_note(score: float, strength: str, direction: str) -> str:
    # Score derives from (midpoint - RSI) / half_range, so positive = oversold.
    zone = "oversold" if score > 0 else "overbought"
    return f"RSI in {zone} zone ({strength} {direction} signal)"


def _macd_note(score: float, strength: str, direction: str) -> str:
    return f"MACD line {direction} ({strength} momentum)"


def _bollinger_note(score: float, strength: str, direction: str) -> str:
    band = "lower band (mean-reversion upside)" if score > 0 else "upper band (mean-reversion downside)"
    return f"price near {band}, {strength} {direction} read"


def _sentiment_note(score: float, n: int) -> str:
    word = "article" if n == 1 else "articles"
    if abs(score) < 0.1:
        return f"news sentiment roughly balanced across {n} {word}"
    direction = "positive" if score > 0 else "negative"
    return f"news sentiment {direction} ({score:+.2f} avg across {n} {word})"


def _strength_label(score: float) -> str:
    mag = abs(score)
    if mag >= _STRONG:
        return "strong"
    if mag >= _MODERATE:
        return "moderate"
    return "weak"


def _confidence_label(confidence: float) -> str:
    if confidence >= _CONF_HIGH:
        return "high"
    if confidence >= _CONF_MODERATE:
        return "moderate"
    return "weak"


def _headline(symbol: str, action: SignalAction, confidence_label: str) -> str:
    verb = {
        SignalAction.BUY: "leans bullish",
        SignalAction.SELL: "leans bearish",
        SignalAction.HOLD: "shows no clear directional edge",
    }[action]
    if action == SignalAction.HOLD:
        return f"{symbol} {verb}."
    return f"{symbol} {verb} with {confidence_label} confidence."
