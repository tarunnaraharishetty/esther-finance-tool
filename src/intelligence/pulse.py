"""Market pulse — a one-glance read of the whole watchlist's state.

Pure data, no LLM, no state. Recomputed every tick from the existing
:class:`~src.dashboard.state.DashboardSnapshot`. The output is one
:class:`MarketPulse` value the dashboard renders as a single dense
line: sentiment direction · conviction tier · activity tier · plain-
English summary.

This is the ambient "weather report" for a trader's watchlist — meant
to be glanceable, not authoritative. The trader uses it to decide
whether to lean in or relax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.strategy.base import SignalAction

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot


# Thresholds — kept inline so the calibration is visible at the
# definition site rather than buried in a config file.
_CONVICTION_HIGH = 0.55
_CONVICTION_MODERATE = 0.30
_SENTIMENT_LEAN = 0.10


@dataclass(frozen=True)
class MarketPulse:
    """One-glance read of the watchlist.

    Three short tier labels + one summary sentence. Stable schema so
    the dashboard render code never branches on values it doesn't
    expect.
    """

    sentiment: str  # "bullish" | "bearish" | "mixed" | "neutral"
    conviction: str  # "strong" | "moderate" | "weak"
    activity: str  # "calm" | "active" | "volatile"
    summary: str  # plain-English read, one sentence

    @property
    def is_empty(self) -> bool:
        """No healthy rows yielded any pulse — render path skips."""
        return self.summary == ""


_EMPTY = MarketPulse(sentiment="neutral", conviction="weak", activity="calm", summary="")


def compute_pulse(snapshot: DashboardSnapshot) -> MarketPulse:
    """Derive a :class:`MarketPulse` from the current snapshot.

    Uses existing snapshot fields only — no fresh data fetches:
    - sentiment from the action mix + avg sentiment_score
    - conviction from avg confidence across healthy rows
    - activity from total signal_history episode count + recent_alerts
    """
    healthy = [r for r in snapshot.rows if not r.error]
    if not healthy:
        return _EMPTY

    sentiment = _classify_sentiment(healthy)
    conviction = _classify_conviction(healthy)
    activity = _classify_activity(snapshot, healthy)
    summary = _build_summary(sentiment, conviction, activity, healthy)

    return MarketPulse(
        sentiment=sentiment,
        conviction=conviction,
        activity=activity,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def _classify_sentiment(healthy: list) -> str:
    """Combine action mix bias with average news sentiment.

    Both must agree (within tolerance) for a directional read. Mixed
    signals → "mixed". Both flat → "neutral".
    """
    buys = sum(1 for r in healthy if r.action == SignalAction.BUY)
    sells = sum(1 for r in healthy if r.action == SignalAction.SELL)
    n = len(healthy)
    action_bias = (buys - sells) / n  # in [-1, +1]

    sent_rows = [r for r in healthy if r.num_news_articles > 0]
    avg_news_sent = (
        sum(r.sentiment_score for r in sent_rows) / len(sent_rows) if sent_rows else 0.0
    )

    action_bullish = action_bias > _SENTIMENT_LEAN
    action_bearish = action_bias < -_SENTIMENT_LEAN
    news_bullish = avg_news_sent > _SENTIMENT_LEAN
    news_bearish = avg_news_sent < -_SENTIMENT_LEAN

    if action_bullish and not action_bearish and not news_bearish:
        return "bullish"
    if action_bearish and not action_bullish and not news_bullish:
        return "bearish"
    # If they disagree directionally, that's mixed (interesting!).
    if (action_bullish and news_bearish) or (action_bearish and news_bullish):
        return "mixed"
    # Both flat-ish.
    return "neutral"


def _classify_conviction(healthy: list) -> str:
    avg_conf = sum(r.confidence for r in healthy) / len(healthy)
    if avg_conf >= _CONVICTION_HIGH:
        return "strong"
    if avg_conf >= _CONVICTION_MODERATE:
        return "moderate"
    return "weak"


def _classify_activity(snapshot: DashboardSnapshot, healthy: list) -> str:
    """Combine total episode count (across history) + recent alert
    count. More episodes = more flips = more activity."""
    total_episodes = 0
    for r in healthy:
        summary = snapshot.signal_history.get(r.symbol)
        if summary is not None:
            total_episodes += 1 + len(summary.recent)
    alert_count = len(snapshot.recent_alerts)

    # "volatile" when symbols have flipped a lot OR alerts have piled up.
    if total_episodes >= 2 * len(healthy) or alert_count >= 5:
        return "volatile"
    if total_episodes > len(healthy) or alert_count >= 1:
        return "active"
    return "calm"


def _build_summary(
    sentiment: str, conviction: str, activity: str, healthy: list
) -> str:
    """One-sentence English read.

    Phrasing is deliberately observational — no predictions, no
    "likely to" language. Same anti-hallucination contract the LLM
    modules follow, applied to deterministic text.
    """
    n = len(healthy)
    sym_word = "symbol" if n == 1 else "symbols"

    sentiment_phrase = {
        "bullish": "leaning bullish",
        "bearish": "leaning bearish",
        "mixed": "showing mixed signals",
        "neutral": "in neutral",
    }[sentiment]

    conviction_phrase = {
        "strong": "with strong conviction",
        "moderate": "with moderate conviction",
        "weak": "with weak conviction",
    }[conviction]

    activity_phrase = {
        "volatile": "and high activity",
        "active": "and moderate activity",
        "calm": "and quiet activity",
    }[activity]

    return f"{n} {sym_word} {sentiment_phrase} {conviction_phrase} {activity_phrase}."


__all__ = ["MarketPulse", "compute_pulse"]
