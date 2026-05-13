"""Tests for src.intelligence.opportunity_drilldown."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from src.dashboard.state import RecommendationRow
from src.intelligence.opportunities import RankedOpportunity
from src.intelligence.opportunity_drilldown import (
    DriverBreakdown,
    OpportunityDrilldown,
    build_drilldown,
)
from src.intelligence.opportunity_history import OpportunityHistory
from src.intelligence.signal_profile import SignalProfile
from src.strategy.base import RecommendationTier, SignalAction


def _row(
    *,
    symbol: str = "AAPL",
    action: SignalAction = SignalAction.BUY,
    num_news_articles: int = 0,
) -> RecommendationRow:
    return RecommendationRow(
        symbol=symbol,
        action=action,
        confidence=0.7,
        combined_score=0.5,
        technical_score=0.5,
        sentiment_score=0.4,
        rsi=math.nan,
        macd=math.nan,
        bollinger=math.nan,
        last_price=100.0,
        num_news_articles=num_news_articles,
        reasoning="",
        timestamp=datetime.now(UTC),
    )


def _profile(
    stability: str = "stable",
    trend: str = "strengthening",
    persistence: str = "persistent",
) -> SignalProfile:
    return SignalProfile(stability=stability, trend=trend, persistence=persistence)


def _opp(
    *,
    symbol: str = "AAPL",
    composite: float = 0.70,
    technical_alignment: float = 0.5,
    sentiment_alignment: float = 0.5,
    confidence_acceleration: float = 0.5,
    momentum_persistence: float = 0.5,
    unusual_activity: float = 0.0,
    reversal_strength: float = 0.0,
    signal_quality_score: float = 0.5,
    profile: SignalProfile | None = None,
    rationale: tuple[str, ...] = (),
    tier: RecommendationTier = RecommendationTier.BUY,
) -> RankedOpportunity:
    return RankedOpportunity(
        symbol=symbol,
        tier=tier,
        composite_score=composite,
        profile=profile or _profile(),
        rationale=rationale,
        technical_alignment=technical_alignment,
        sentiment_alignment=sentiment_alignment,
        confidence_acceleration=confidence_acceleration,
        momentum_persistence=momentum_persistence,
        unusual_activity=unusual_activity,
        reversal_strength=reversal_strength,
        signal_quality_score=signal_quality_score,
    )


# ---------------------------------------------------------------------------
# Schema + sort order
# ---------------------------------------------------------------------------


def test_build_drilldown_returns_seven_drivers() -> None:
    """The drilldown always carries one DriverBreakdown per driver key
    on RankedOpportunity, regardless of score magnitude."""
    drilldown = build_drilldown(_opp(), rank=1, history=None)
    keys = {d.key for d in drilldown.drivers}
    assert keys == {
        "technical_alignment",
        "sentiment_alignment",
        "confidence_acceleration",
        "momentum_persistence",
        "unusual_activity",
        "reversal_strength",
        "signal_quality_score",
    }
    assert len(drilldown.drivers) == 7


def test_drivers_sorted_strongest_first() -> None:
    """The drilldown sorts drivers by score descending so the render
    layer iterates strongest-to-weakest without reshuffling."""
    opp = _opp(
        technical_alignment=0.1,
        sentiment_alignment=0.9,
        confidence_acceleration=0.7,
        momentum_persistence=0.3,
        unusual_activity=0.0,
        reversal_strength=0.5,
        signal_quality_score=1.0,
    )
    drilldown = build_drilldown(opp, rank=1, history=None)
    scores = [d.score for d in drilldown.drivers]
    assert scores == sorted(scores, reverse=True)
    assert drilldown.drivers[0].key == "signal_quality_score"
    assert drilldown.drivers[-1].key == "unusual_activity"


def test_passes_through_rank_symbol_composite_and_history() -> None:
    """Identity fields land on the drilldown unchanged."""
    history = OpportunityHistory(streak=3, appearances=5, window=10)
    opp = _opp(symbol="NVDA", composite=0.82)
    drilldown = build_drilldown(opp, rank=2, history=history)
    assert drilldown.rank == 2
    assert drilldown.symbol == "NVDA"
    assert drilldown.composite_score == 0.82
    assert drilldown.history is history


def test_rationale_passed_through_verbatim() -> None:
    """Drilldown reuses the rationale tuple from the source opp — single
    source of truth for the header line and the drilldown block."""
    phrases = ("indicators aligned with action", "high signal quality")
    opp = _opp(rationale=phrases)
    drilldown = build_drilldown(opp, rank=1, history=None)
    assert drilldown.rationale == phrases


# ---------------------------------------------------------------------------
# Per-driver descriptors
# ---------------------------------------------------------------------------


def _driver(drilldown: OpportunityDrilldown, key: str) -> DriverBreakdown:
    return next(d for d in drilldown.drivers if d.key == key)


def test_descriptor_fires_at_floor() -> None:
    """A driver at exactly 0.60 produces its observational descriptor."""
    drilldown = build_drilldown(_opp(technical_alignment=0.60), rank=1, history=None)
    assert _driver(drilldown, "technical_alignment").descriptor == (
        "indicators aligned with action"
    )


def test_descriptor_silent_below_floor() -> None:
    """Below 0.60 the descriptor is empty — bar carries magnitude alone."""
    drilldown = build_drilldown(_opp(technical_alignment=0.59), rank=1, history=None)
    assert _driver(drilldown, "technical_alignment").descriptor == ""


def test_signal_quality_descriptor_only_at_high_tier() -> None:
    """signal_quality_score maps a categorical tier — only the high tier
    (1.0) earns the 'high signal quality' descriptor; the moderate tier
    (0.5) and low (0.0) stay quiet."""
    high = build_drilldown(_opp(signal_quality_score=1.0), rank=1, history=None)
    moderate = build_drilldown(_opp(signal_quality_score=0.5), rank=1, history=None)
    assert _driver(high, "signal_quality_score").descriptor == "high signal quality"
    assert _driver(moderate, "signal_quality_score").descriptor == ""


def test_each_driver_has_its_own_descriptor() -> None:
    """Saturate every driver — each one emits its dedicated phrase."""
    opp = _opp(
        technical_alignment=1.0,
        sentiment_alignment=1.0,
        confidence_acceleration=1.0,
        momentum_persistence=1.0,
        unusual_activity=1.0,
        reversal_strength=1.0,
        signal_quality_score=1.0,
    )
    drilldown = build_drilldown(opp, rank=1, history=None)
    descriptors = {d.key: d.descriptor for d in drilldown.drivers}
    assert descriptors["technical_alignment"] == "indicators aligned with action"
    assert descriptors["sentiment_alignment"] == "news sentiment matches direction"
    assert descriptors["confidence_acceleration"] == ("confidence rising in current run")
    assert descriptors["momentum_persistence"] == ("momentum holding across recent ticks")
    assert descriptors["unusual_activity"] == "unusual confidence swing"
    assert descriptors["reversal_strength"] == "reversal intensity elevated"
    assert descriptors["signal_quality_score"] == "high signal quality"


# ---------------------------------------------------------------------------
# Quality labels
# ---------------------------------------------------------------------------


def test_high_conviction_requires_both_quality_and_composite() -> None:
    """`high conviction` fires only when signal_quality is high AND
    composite clears 0.65 — either alone is not enough."""
    both = build_drilldown(_opp(signal_quality_score=1.0, composite=0.70), rank=1, history=None)
    assert "high conviction" in both.quality_labels

    quality_only = build_drilldown(
        _opp(signal_quality_score=1.0, composite=0.40), rank=1, history=None
    )
    assert "high conviction" not in quality_only.quality_labels

    composite_only = build_drilldown(
        _opp(signal_quality_score=0.5, composite=0.85), rank=1, history=None
    )
    assert "high conviction" not in composite_only.quality_labels


def test_building_momentum_requires_acceleration_and_persistence() -> None:
    """`building momentum` needs confidence accelerating AND momentum
    having actually run for a few ticks."""
    on = build_drilldown(
        _opp(confidence_acceleration=0.80, momentum_persistence=0.60),
        rank=1,
        history=None,
    )
    assert "building momentum" in on.quality_labels

    accel_only = build_drilldown(
        _opp(confidence_acceleration=0.90, momentum_persistence=0.20),
        rank=1,
        history=None,
    )
    assert "building momentum" not in accel_only.quality_labels


def test_unstable_choppy_fires_on_noisy_or_flipping_profile() -> None:
    """Noisy signal or a flipping symbol both earn the caveat label."""
    noisy = build_drilldown(_opp(profile=_profile(stability="noisy")), rank=1, history=None)
    flipping = build_drilldown(_opp(profile=_profile(persistence="flipping")), rank=1, history=None)
    clean = build_drilldown(_opp(), rank=1, history=None)
    assert "unstable / choppy" in noisy.quality_labels
    assert "unstable / choppy" in flipping.quality_labels
    assert "unstable / choppy" not in clean.quality_labels


def test_reversal_candidate_fires_at_floor() -> None:
    """`reversal candidate` is a single-driver gate at 0.60."""
    on = build_drilldown(_opp(reversal_strength=0.60), rank=1, history=None)
    off = build_drilldown(_opp(reversal_strength=0.59), rank=1, history=None)
    assert "reversal candidate" in on.quality_labels
    assert "reversal candidate" not in off.quality_labels


def test_sentiment_driven_fires_when_news_leads_without_technical() -> None:
    """`sentiment-driven` describes the case where news/sentiment is
    strong but the technicals are not — both gates must hold."""
    on = build_drilldown(
        _opp(sentiment_alignment=0.80, technical_alignment=0.30),
        rank=1,
        history=None,
    )
    off_tech_too_strong = build_drilldown(
        _opp(sentiment_alignment=0.80, technical_alignment=0.70),
        rank=1,
        history=None,
    )
    off_sent_too_weak = build_drilldown(
        _opp(sentiment_alignment=0.40, technical_alignment=0.20),
        rank=1,
        history=None,
    )
    assert "sentiment-driven" in on.quality_labels
    assert "sentiment-driven" not in off_tech_too_strong.quality_labels
    assert "sentiment-driven" not in off_sent_too_weak.quality_labels


def test_multiple_quality_labels_can_co_fire() -> None:
    """`high conviction` + `building momentum` should both appear when
    their gates independently hold; the caveat sits last when present."""
    opp = _opp(
        signal_quality_score=1.0,
        composite=0.85,
        confidence_acceleration=0.80,
        momentum_persistence=0.70,
        profile=_profile(stability="noisy"),
    )
    drilldown = build_drilldown(opp, rank=1, history=None)
    assert "high conviction" in drilldown.quality_labels
    assert "building momentum" in drilldown.quality_labels
    assert drilldown.quality_labels[-1] == "unstable / choppy"


def test_quiet_opportunity_has_no_quality_labels() -> None:
    """Composite low + drivers low + profile clean = no chips. Empty
    tuple lets the renderer skip the chip line entirely."""
    quiet = _opp(
        composite=0.20,
        technical_alignment=0.10,
        sentiment_alignment=0.10,
        confidence_acceleration=0.10,
        momentum_persistence=0.10,
        unusual_activity=0.0,
        reversal_strength=0.0,
        signal_quality_score=0.0,
    )
    drilldown = build_drilldown(quiet, rank=5, history=None)
    assert drilldown.quality_labels == ()


# ---------------------------------------------------------------------------
# State phrases (richer composite rationale)
# ---------------------------------------------------------------------------


def test_state_phrases_empty_without_row() -> None:
    """Action-aware phrases (breadth, persistence) need the row to
    derive direction; without it they don't fire. Momentum
    strengthening doesn't need the row, so it still can."""
    opp = _opp(confidence_acceleration=0.80, momentum_persistence=0.70)
    drilldown = build_drilldown(opp, rank=1, history=None)
    assert "momentum strengthening across recent ticks" in drilldown.state_phrases
    # No row → no breadth, no persistence phrase.
    assert all("breadth" not in p for p in drilldown.state_phrases)
    assert all("persistence" not in p for p in drilldown.state_phrases)


def test_momentum_strengthening_requires_both_accel_and_persistence() -> None:
    """The phrase distinguishes 'strengthening' from mere 'holding' —
    both confidence_acceleration and momentum_persistence must clear
    their floors together."""
    both = build_drilldown(
        _opp(confidence_acceleration=0.70, momentum_persistence=0.60),
        rank=1,
        history=None,
    )
    assert "momentum strengthening across recent ticks" in both.state_phrases

    persistence_only = build_drilldown(
        _opp(confidence_acceleration=0.40, momentum_persistence=0.90),
        rank=1,
        history=None,
    )
    assert "momentum strengthening across recent ticks" not in (persistence_only.state_phrases)


def test_sentiment_breadth_polarity_follows_action_buy() -> None:
    """BUY + strong sentiment alignment + multi-article support →
    'positive sentiment breadth'."""
    opp = _opp(sentiment_alignment=0.80)
    row = _row(action=SignalAction.BUY, num_news_articles=4)
    drilldown = build_drilldown(opp, rank=1, history=None, row=row)
    assert "positive sentiment breadth" in drilldown.state_phrases


def test_sentiment_breadth_polarity_follows_action_sell() -> None:
    """SELL + same gates → 'negative sentiment breadth'. Action
    direction is the only thing that changes the polarity word."""
    opp = _opp(sentiment_alignment=0.80)
    row = _row(action=SignalAction.SELL, num_news_articles=4)
    drilldown = build_drilldown(opp, rank=1, history=None, row=row)
    assert "negative sentiment breadth" in drilldown.state_phrases


def test_sentiment_breadth_silent_when_too_few_articles() -> None:
    """Strong alignment alone isn't 'breadth' — the multi-article
    floor (3) gates the phrase."""
    opp = _opp(sentiment_alignment=0.80)
    row = _row(action=SignalAction.BUY, num_news_articles=2)
    drilldown = build_drilldown(opp, rank=1, history=None, row=row)
    assert all("breadth" not in p for p in drilldown.state_phrases)


def test_stable_persistence_inlines_action_word() -> None:
    """When profile is stable + persistent and action is directional,
    the phrase carries the action word so the trader doesn't have to
    cross-reference."""
    opp = _opp(profile=_profile(stability="stable", persistence="persistent"))
    buy = build_drilldown(opp, rank=1, history=None, row=_row(action=SignalAction.BUY))
    sell = build_drilldown(opp, rank=1, history=None, row=_row(action=SignalAction.SELL))
    assert "stable BUY persistence" in buy.state_phrases
    assert "stable SELL persistence" in sell.state_phrases


def test_stable_persistence_silent_when_profile_isnt_clean() -> None:
    """Either a noisy or flipping profile suppresses the stable phrase
    — same axis the 'unstable / choppy' quality label keys off."""
    noisy = _opp(profile=_profile(stability="noisy"))
    flipping = _opp(profile=_profile(persistence="flipping"))
    row = _row(action=SignalAction.BUY)
    assert all(
        "persistence" not in p
        for p in build_drilldown(noisy, rank=1, history=None, row=row).state_phrases
    )
    assert all(
        "persistence" not in p
        for p in build_drilldown(flipping, rank=1, history=None, row=row).state_phrases
    )


def test_state_phrases_co_fire_on_strong_setup() -> None:
    """A genuinely strong setup — rising confidence + persistent run +
    aligned sentiment + multiple articles + clean profile — should
    produce all three new phrases at once."""
    opp = _opp(
        confidence_acceleration=0.80,
        momentum_persistence=0.70,
        sentiment_alignment=0.80,
        profile=_profile(stability="stable", persistence="persistent"),
    )
    row = _row(action=SignalAction.BUY, num_news_articles=5)
    drilldown = build_drilldown(opp, rank=1, history=None, row=row)
    assert "momentum strengthening across recent ticks" in drilldown.state_phrases
    assert "positive sentiment breadth" in drilldown.state_phrases
    assert "stable BUY persistence" in drilldown.state_phrases
