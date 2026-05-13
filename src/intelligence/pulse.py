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

from src.strategy.base import RecommendationTier, SignalAction

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot, RecommendationRow


# Thresholds — kept inline so the calibration is visible at the
# definition site rather than buried in a config file.
_CONVICTION_HIGH = 0.55
_CONVICTION_MODERATE = 0.30
_SENTIMENT_LEAN = 0.10
# Indicator/sentiment alignment threshold for breadth metrics. Same
# 0.1 floor used for the row-level "directional" check in `tier.py`.
_DIRECTIONAL_FLOOR = 0.10
# Max items in the strongest_symbols list — small enough to render
# in one line of the dashboard header.
_STRONGEST_LIMIT = 3


@dataclass(frozen=True)
class MarketPulse:
    """One-glance read of the watchlist.

    Three short tier labels + one summary sentence (the "headline" view),
    plus a set of structured breadth / intensity fields the dashboard
    can surface densely. Stable schema so the dashboard render code
    never branches on values it doesn't expect.

    All fields are derived per-tick from the snapshot. None of them
    require history beyond what's already on ``DashboardSnapshot``.
    """

    sentiment: str  # "bullish" | "bearish" | "mixed" | "neutral"
    conviction: str  # "strong" | "moderate" | "weak"
    activity: str  # "calm" | "active" | "volatile"
    summary: str  # plain-English read, one sentence

    # ---- Breadth / intensity (defaults preserve the old constructor
    # so existing callers and tests keep working unchanged) ----

    bullish_count: int = 0
    """Rows whose tier is BUY or STRONG_BUY (action direction +1)."""

    bearish_count: int = 0
    """Rows whose tier is SELL or STRONG_SELL (action direction -1)."""

    healthy_count: int = 0
    """Total non-error rows. ``bullish_count + bearish_count`` may be
    less than this when symbols are on HOLD."""

    momentum_breadth: float = 0.0
    """Fraction of healthy rows whose MACD score has the same sign as
    the action direction (BUY/SELL only, HOLD excluded). 0.0 when no
    directional rows exist."""

    sentiment_breadth: float = 0.0
    """Fraction of news-bearing healthy rows whose sentiment score has
    the same sign as the action direction. 0.0 when no news-bearing
    directional rows exist."""

    reversal_intensity: int = 0
    """Sum of ``(episode_count - 1)`` across all symbols in the
    snapshot's signal_history — i.e., total flips this session."""

    alert_intensity: int = 0
    """Length of the snapshot's recent_alerts window."""

    strongest_symbols: tuple[tuple[str, str], ...] = ()
    """Top ``_STRONGEST_LIMIT`` symbols on STRONG_BUY or STRONG_SELL
    tier, ranked by confidence descending. Each entry is
    ``(symbol, tier_display)`` — e.g., ``("NVDA", "STRONG BUY")``.
    Empty when no row qualifies."""

    @property
    def is_empty(self) -> bool:
        """No healthy rows yielded any pulse — render path skips."""
        return self.summary == ""


_EMPTY = MarketPulse(sentiment="neutral", conviction="weak", activity="calm", summary="")


def compute_pulse(snapshot: DashboardSnapshot) -> MarketPulse:
    """Derive a :class:`MarketPulse` from the current snapshot.

    Uses existing snapshot fields only — no fresh data fetches.
    """
    healthy = [r for r in snapshot.rows if not r.error]
    if not healthy:
        return _EMPTY

    sentiment = _classify_sentiment(healthy)
    conviction = _classify_conviction(healthy)
    activity = _classify_activity(snapshot, healthy)
    summary = _build_summary(sentiment, conviction, activity, healthy)

    bullish_count = sum(1 for r in healthy if r.action == SignalAction.BUY)
    bearish_count = sum(1 for r in healthy if r.action == SignalAction.SELL)
    momentum_breadth = _compute_momentum_breadth(healthy)
    sentiment_breadth = _compute_sentiment_breadth(healthy)
    reversal_intensity = _compute_reversal_intensity(snapshot)
    alert_intensity = len(snapshot.recent_alerts)
    strongest_symbols = _compute_strongest_symbols(healthy)

    return MarketPulse(
        sentiment=sentiment,
        conviction=conviction,
        activity=activity,
        summary=summary,
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        healthy_count=len(healthy),
        momentum_breadth=momentum_breadth,
        sentiment_breadth=sentiment_breadth,
        reversal_intensity=reversal_intensity,
        alert_intensity=alert_intensity,
        strongest_symbols=strongest_symbols,
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


# ---------------------------------------------------------------------------
# Breadth / intensity helpers (extension)
# ---------------------------------------------------------------------------


def _direction(action: SignalAction) -> int:
    """+1 bullish, -1 bearish, 0 neutral."""
    if action == SignalAction.BUY:
        return 1
    if action == SignalAction.SELL:
        return -1
    return 0


def _is_finite(x: float) -> bool:
    """NaN-safe finite check."""
    return x == x  # NaN != NaN; finite numerics return True


def _compute_momentum_breadth(healthy: list) -> float:
    """Fraction of directional rows whose MACD aligns with action.

    Returns 0.0 when no directional rows exist.
    """
    directional = [r for r in healthy if _direction(r.action) != 0]
    if not directional:
        return 0.0
    aligned = sum(
        1
        for r in directional
        if _is_finite(r.macd)
        and _direction(r.action) * r.macd > _DIRECTIONAL_FLOOR
    )
    return aligned / len(directional)


def _compute_sentiment_breadth(healthy: list) -> float:
    """Fraction of news-bearing directional rows whose sentiment aligns
    with action. Returns 0.0 when no qualifying rows exist."""
    candidates = [
        r for r in healthy if _direction(r.action) != 0 and r.num_news_articles > 0
    ]
    if not candidates:
        return 0.0
    aligned = sum(
        1
        for r in candidates
        if _direction(r.action) * r.sentiment_score > _DIRECTIONAL_FLOOR
    )
    return aligned / len(candidates)


def _compute_reversal_intensity(snapshot: DashboardSnapshot) -> int:
    """Total flips across the session — sum of (episode_count - 1) per
    symbol in signal_history.

    A symbol with 1 episode has flipped 0 times; with 3 episodes,
    flipped 2 times. Symbols absent from history contribute 0.
    """
    total = 0
    for summary in snapshot.signal_history.values():
        # episode_count = current + len(recent); flips = episode_count - 1
        total += len(summary.recent)
    return total


def _compute_strongest_symbols(
    healthy: list,
) -> tuple[tuple[str, str], ...]:
    """Top STRONG-tier rows by confidence (descending), capped at the
    module limit. Returns empty when no row qualifies — render layer
    omits the chunk."""
    strong = [r for r in healthy if r.tier.is_strong]
    if not strong:
        return ()
    strong.sort(key=lambda r: r.confidence, reverse=True)
    return tuple(
        (r.symbol, r.tier.display) for r in strong[:_STRONGEST_LIMIT]
    )


__all__ = ["MarketPulse", "compute_pulse"]
