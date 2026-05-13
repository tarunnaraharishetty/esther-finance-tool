"""Tests for src.intelligence.tier — recommendation tier promotion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.intelligence.history import SignalEpisode, SignalHistorySummary
from src.intelligence.tier import (
    STRONG_CONFIDENCE_FLOOR,
    promote_to_tier,
)
from src.strategy.base import RecommendationTier, SignalAction
from src.strategy.recommendation import TradingRecommendation


def _rec(
    *,
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.7,
    indicator_scores: dict[str, float] | None = None,
    sentiment_score: float = 0.5,
    technical_score: float = 0.5,
    combined_score: float = 0.5,
    num_news: int = 5,
) -> TradingRecommendation:
    return TradingRecommendation(
        symbol="AAPL",
        action=action,
        confidence=confidence,
        combined_score=combined_score,
        technical_score=technical_score,
        sentiment_score=sentiment_score,
        indicator_scores=indicator_scores
        if indicator_scores is not None
        else {"rsi": 0.4, "macd": 0.5, "bollinger": 0.4},
        reasoning="",
        timestamp=datetime.now(UTC),
        num_news_articles=num_news,
    )


def _episode(
    action: SignalAction,
    ticks: int = 5,
    conf_first: float = 0.5,
    conf_last: float = 0.5,
) -> SignalEpisode:
    return SignalEpisode(
        action=action,
        started_at=datetime(2026, 5, 13, tzinfo=UTC),
        last_seen_at=datetime(2026, 5, 13, tzinfo=UTC) + timedelta(seconds=ticks * 5),
        tick_count=ticks,
        confidence_first=conf_first,
        confidence_last=conf_last,
    )


def _stable_history(
    action: SignalAction = SignalAction.BUY,
    ticks: int = 5,
    conf_first: float = 0.6,
    conf_last: float = 0.65,
) -> SignalHistorySummary:
    """A history that should pass the stability gate: long-held single
    episode with rising/stable confidence."""
    return SignalHistorySummary(
        current=_episode(action, ticks=ticks, conf_first=conf_first, conf_last=conf_last),
        recent=(),
    )


# ---------------------------------------------------------------------------
# Promotion gate — STRONG only when ALL FOUR conditions hold
# ---------------------------------------------------------------------------


def test_strong_buy_when_all_gates_pass() -> None:
    """Aligned indicators + bullish sentiment + stable long episode +
    confidence above floor → STRONG_BUY."""
    rec = _rec(
        action=SignalAction.BUY,
        confidence=0.75,
        indicator_scores={"rsi": 0.5, "macd": 0.6, "bollinger": 0.4},
        sentiment_score=0.5,
        num_news=8,
    )
    promoted = promote_to_tier(rec, _stable_history())
    assert promoted.tier == RecommendationTier.STRONG_BUY
    assert promoted.signal_quality == "high"
    assert promoted.stability == "stable"


def test_strong_sell_when_all_gates_pass_bearish() -> None:
    rec = _rec(
        action=SignalAction.SELL,
        confidence=0.75,
        indicator_scores={"rsi": -0.5, "macd": -0.6, "bollinger": -0.4},
        sentiment_score=-0.5,
        num_news=8,
    )
    promoted = promote_to_tier(rec, _stable_history(action=SignalAction.SELL))
    assert promoted.tier == RecommendationTier.STRONG_SELL


def test_not_strong_when_confidence_below_floor() -> None:
    rec = _rec(
        action=SignalAction.BUY,
        confidence=STRONG_CONFIDENCE_FLOOR - 0.01,
        indicator_scores={"rsi": 0.5, "macd": 0.6, "bollinger": 0.4},
        sentiment_score=0.5,
        num_news=8,
    )
    promoted = promote_to_tier(rec, _stable_history())
    assert promoted.tier == RecommendationTier.BUY


def test_not_strong_when_signal_quality_only_moderate() -> None:
    """One indicator disagrees → quality not 'high' → no promotion."""
    rec = _rec(
        action=SignalAction.BUY,
        confidence=0.75,
        indicator_scores={"rsi": 0.5, "macd": -0.4, "bollinger": 0.4},
        sentiment_score=0.5,
        num_news=8,
    )
    promoted = promote_to_tier(rec, _stable_history())
    assert promoted.tier == RecommendationTier.BUY
    assert promoted.signal_quality in ("moderate", "low")


def test_not_strong_when_stability_only_moderate() -> None:
    """A recent flip → episode_count > 1 → stability drops below 'stable'."""
    rec = _rec(
        action=SignalAction.BUY,
        confidence=0.75,
        indicator_scores={"rsi": 0.5, "macd": 0.6, "bollinger": 0.4},
        sentiment_score=0.5,
        num_news=8,
    )
    flippy_history = SignalHistorySummary(
        current=_episode(SignalAction.BUY, ticks=2, conf_first=0.5, conf_last=0.6),
        recent=(
            _episode(SignalAction.HOLD),
            _episode(SignalAction.SELL),
            _episode(SignalAction.BUY),
        ),
    )
    promoted = promote_to_tier(rec, flippy_history)
    assert promoted.tier == RecommendationTier.BUY
    assert promoted.stability == "volatile"


def test_hold_never_promotes() -> None:
    """HOLD has no direction — STRONG isn't meaningful."""
    rec = _rec(action=SignalAction.HOLD, confidence=0.9)
    promoted = promote_to_tier(rec, _stable_history())
    assert promoted.tier == RecommendationTier.HOLD
    assert not promoted.tier.is_strong


# ---------------------------------------------------------------------------
# Signal-quality scoring
# ---------------------------------------------------------------------------


def test_quality_high_when_all_indicators_aligned_and_news_positive() -> None:
    rec = _rec(
        action=SignalAction.BUY,
        indicator_scores={"rsi": 0.5, "macd": 0.6, "bollinger": 0.4},
        sentiment_score=0.4,
        num_news=5,
    )
    promoted = promote_to_tier(rec, _stable_history())
    assert promoted.signal_quality == "high"
    assert "Strong technical alignment" in promoted.quality_reasons


def test_quality_low_when_sentiment_opposes_action() -> None:
    """Bullish action but news is bearish → quality should be low."""
    rec = _rec(
        action=SignalAction.BUY,
        indicator_scores={"rsi": 0.1, "macd": 0.05, "bollinger": 0.05},
        sentiment_score=-0.5,
        num_news=5,
    )
    promoted = promote_to_tier(rec, _stable_history())
    assert promoted.signal_quality == "low"


def test_quality_skips_sentiment_factor_when_no_news() -> None:
    """No-news sentiment is meaningless — it must not penalize or
    falsely lift the quality score."""
    aligned = _rec(
        action=SignalAction.BUY,
        indicator_scores={"rsi": 0.5, "macd": 0.6, "bollinger": 0.4},
        sentiment_score=0.0,
        num_news=0,
    )
    promoted = promote_to_tier(aligned, _stable_history())
    # All indicators aligned → quality should still be at least moderate.
    assert promoted.signal_quality in ("high", "moderate")


def test_quality_reasons_include_momentum_when_macd_strong() -> None:
    rec = _rec(
        action=SignalAction.BUY,
        indicator_scores={"rsi": 0.3, "macd": 0.7, "bollinger": 0.3},
        sentiment_score=0.4,
        num_news=5,
    )
    promoted = promote_to_tier(rec, _stable_history())
    assert any("momentum" in r.lower() for r in promoted.quality_reasons)


# ---------------------------------------------------------------------------
# Stability scoring (anti-flip safeguard)
# ---------------------------------------------------------------------------


def test_stability_stable_when_single_long_episode_with_rising_conf() -> None:
    history = SignalHistorySummary(
        current=_episode(SignalAction.BUY, ticks=8, conf_first=0.5, conf_last=0.7),
        recent=(),
    )
    promoted = promote_to_tier(_rec(), history)
    assert promoted.stability == "stable"
    reasons_text = "  ".join(promoted.quality_reasons).lower()
    assert "low reversal frequency" in reasons_text


def test_stability_volatile_with_many_recent_flips() -> None:
    """4+ episodes means the symbol has flipped 3+ times this session
    → volatile by construction."""
    history = SignalHistorySummary(
        current=_episode(SignalAction.BUY),
        recent=(
            _episode(SignalAction.HOLD),
            _episode(SignalAction.SELL),
            _episode(SignalAction.BUY),
        ),
    )
    promoted = promote_to_tier(_rec(), history)
    assert promoted.stability == "volatile"


def test_stability_with_no_history_is_middle_of_the_road() -> None:
    """First tick of the session shouldn't disqualify a STRONG
    promotion outright, but shouldn't slam through to 'stable' either."""
    promoted = promote_to_tier(_rec(), history=None)
    assert promoted.stability in ("stable", "moderate")


def test_stability_drops_when_confidence_is_falling() -> None:
    history = SignalHistorySummary(
        current=_episode(SignalAction.BUY, ticks=5, conf_first=0.7, conf_last=0.4),
        recent=(),
    )
    promoted = promote_to_tier(_rec(), history)
    # A long held episode with falling confidence shouldn't earn 'stable'.
    assert promoted.stability != "stable"


# ---------------------------------------------------------------------------
# Defaults / preservation
# ---------------------------------------------------------------------------


def test_promotion_preserves_existing_fields() -> None:
    """promote_to_tier returns a new instance with the same base
    recommendation data — only the tier fields change."""
    rec = _rec(action=SignalAction.BUY)
    promoted = promote_to_tier(rec, _stable_history())
    for field_name in (
        "symbol",
        "action",
        "confidence",
        "combined_score",
        "technical_score",
        "sentiment_score",
        "reasoning",
        "timestamp",
        "num_news_articles",
    ):
        assert getattr(rec, field_name) == getattr(promoted, field_name)


def test_promotion_is_idempotent_at_default_input() -> None:
    """Promoting a recommendation with no history twice should produce
    the same tier each time (no hidden state in the promoter)."""
    rec = _rec()
    a = promote_to_tier(rec)
    b = promote_to_tier(rec)
    assert a.tier == b.tier
    assert a.signal_quality == b.signal_quality
    assert a.stability == b.stability


def test_quality_reasons_are_observational_not_predictive() -> None:
    """Anti-hallucination guard on the tier reasons — same forbidden
    words pattern used on pulse + opportunity rationale."""
    rec = _rec(
        action=SignalAction.BUY,
        confidence=0.8,
        indicator_scores={"rsi": 0.5, "macd": 0.7, "bollinger": 0.4},
        sentiment_score=0.5,
        num_news=5,
    )
    promoted = promote_to_tier(rec, _stable_history())
    forbidden = ("likely", "will rise", "will fall", "expected to", "forecast")
    text = "  ".join(promoted.quality_reasons).lower()
    for word in forbidden:
        assert word not in text


# ---------------------------------------------------------------------------
# RecommendationTier helper-method coverage
# ---------------------------------------------------------------------------


def test_tier_is_strong_property() -> None:
    assert RecommendationTier.STRONG_BUY.is_strong
    assert RecommendationTier.STRONG_SELL.is_strong
    assert not RecommendationTier.BUY.is_strong
    assert not RecommendationTier.HOLD.is_strong
    assert not RecommendationTier.SELL.is_strong


def test_tier_direction_property() -> None:
    assert RecommendationTier.STRONG_BUY.direction == 1
    assert RecommendationTier.BUY.direction == 1
    assert RecommendationTier.HOLD.direction == 0
    assert RecommendationTier.SELL.direction == -1
    assert RecommendationTier.STRONG_SELL.direction == -1


def test_tier_from_action_maps_correctly() -> None:
    assert RecommendationTier.from_action(SignalAction.BUY) == RecommendationTier.BUY
    assert RecommendationTier.from_action(SignalAction.HOLD) == RecommendationTier.HOLD
    assert RecommendationTier.from_action(SignalAction.SELL) == RecommendationTier.SELL


def test_tier_display_is_human_friendly() -> None:
    assert RecommendationTier.STRONG_BUY.display == "STRONG BUY"
    assert RecommendationTier.STRONG_SELL.display == "STRONG SELL"
    assert RecommendationTier.HOLD.display == "HOLD"
