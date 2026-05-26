"""Tests for the hysteresis band on :class:`ConfidenceThresholdRule`.

Without the band, a confidence that wiggles 0.005 across 0.60 fires
every tick; with the band, only material crossings count. Pins the
contract so a future regression here will trip an obvious test.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.dashboard.state import RecommendationRow
from src.intelligence.alerts import ConfidenceThresholdRule
from src.strategy.base import RecommendationTier, SignalAction


def _row(confidence: float) -> RecommendationRow:
    return RecommendationRow(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=confidence,
        combined_score=0.5,
        technical_score=0.3,
        sentiment_score=0.2,
        rsi=55.0,
        macd=0.5,
        bollinger=0.0,
        last_price=100.0,
        num_news_articles=3,
        reasoning="test",
        timestamp=datetime.now(UTC),
        tier=RecommendationTier.BUY,
    )


def test_small_wiggle_across_threshold_does_not_fire() -> None:
    """A 0.005 move across 0.60 must not fire when band=0.02."""
    rule = ConfidenceThresholdRule(threshold=0.6, band=0.02)
    previous = _row(0.598)
    current = _row(0.602)
    assert rule.evaluate(current, previous) is None


def test_material_up_crossing_fires() -> None:
    """A move from 0.55 to 0.65 crosses 0.60 by 0.05 — well above the band."""
    rule = ConfidenceThresholdRule(threshold=0.6, band=0.02)
    previous = _row(0.55)
    current = _row(0.65)
    alert = rule.evaluate(current, previous)
    assert alert is not None
    assert "↑" in alert.message


def test_material_down_crossing_fires() -> None:
    rule = ConfidenceThresholdRule(threshold=0.6, band=0.02)
    previous = _row(0.65)
    current = _row(0.55)
    alert = rule.evaluate(current, previous)
    assert alert is not None
    assert "↓" in alert.message


def test_at_threshold_exactly_does_not_fire_without_band_clearance() -> None:
    """Landing exactly on the threshold doesn't clear the band — no alert."""
    rule = ConfidenceThresholdRule(threshold=0.6, band=0.02)
    previous = _row(0.59)
    current = _row(0.60)
    assert rule.evaluate(current, previous) is None
