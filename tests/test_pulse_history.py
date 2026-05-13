"""Tests for src.intelligence.pulse_history."""

from __future__ import annotations

import pytest

from src.intelligence.pulse import MarketPulse
from src.intelligence.pulse_history import PulseHistory, PulseHistoryTracker


def _pulse(
    *,
    momentum_breadth: float = 0.5,
    sentiment_breadth: float = 0.5,
    bullish_count: int = 2,
    bearish_count: int = 1,
    healthy_count: int = 5,
    reversal_intensity: int = 0,
    alert_intensity: int = 0,
    sentiment: str = "bullish",
    conviction: str = "moderate",
    activity: str = "active",
    summary: str = "5 symbols leaning bullish with moderate conviction and moderate activity.",
) -> MarketPulse:
    return MarketPulse(
        sentiment=sentiment,
        conviction=conviction,
        activity=activity,
        summary=summary,
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        healthy_count=healthy_count,
        momentum_breadth=momentum_breadth,
        sentiment_breadth=sentiment_breadth,
        reversal_intensity=reversal_intensity,
        alert_intensity=alert_intensity,
    )


def _empty_pulse() -> MarketPulse:
    return MarketPulse(sentiment="neutral", conviction="weak", activity="calm", summary="")


def test_tracker_records_pulses_and_summary_reflects_them() -> None:
    tracker = PulseHistoryTracker(window=5)
    tracker.record(_pulse(momentum_breadth=0.3))
    tracker.record(_pulse(momentum_breadth=0.5))
    tracker.record(_pulse(momentum_breadth=0.8))
    summary = tracker.summary()
    assert summary.momentum_breadth == (0.3, 0.5, 0.8)
    assert summary.length == 3
    assert summary.window == 5


def test_tracker_skips_empty_pulses_so_history_starts_at_first_useful_frame() -> None:
    """An empty pulse (no healthy rows) doesn't carry signal — record it
    and the sparkline opens with a meaningless zero. Tracker filters
    these so the first meaningful frame is the first entry."""
    tracker = PulseHistoryTracker(window=5)
    tracker.record(_empty_pulse())
    tracker.record(_empty_pulse())
    tracker.record(_pulse(momentum_breadth=0.6))
    summary = tracker.summary()
    assert summary.length == 1
    assert summary.momentum_breadth == (0.6,)


def test_tracker_window_caps_buffer_size() -> None:
    tracker = PulseHistoryTracker(window=3)
    for v in (0.1, 0.2, 0.3, 0.4, 0.5):
        tracker.record(_pulse(momentum_breadth=v))
    summary = tracker.summary()
    # Oldest two dropped; rightmost is the most-recent.
    assert summary.momentum_breadth == (0.3, 0.4, 0.5)
    assert summary.length == 3


def test_tracker_invalid_window_rejected() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        PulseHistoryTracker(window=0)


def test_tracker_summary_oldest_first_rightmost_is_most_recent() -> None:
    """The series order matters for the sparkline render (left=old,
    right=new). Pin the contract."""
    tracker = PulseHistoryTracker(window=5)
    tracker.record(_pulse(momentum_breadth=0.1, alert_intensity=1))
    tracker.record(_pulse(momentum_breadth=0.9, alert_intensity=4))
    summary = tracker.summary()
    assert summary.momentum_breadth[0] == 0.1  # oldest
    assert summary.momentum_breadth[-1] == 0.9  # most recent
    assert summary.alert_intensity[0] == 1
    assert summary.alert_intensity[-1] == 4


def test_summary_carries_categorical_series_too() -> None:
    """Numeric series drive the sparkline render today, but the
    categorical series (sentiment / conviction / activity) are kept
    around for future consumers (e.g. recap prose)."""
    tracker = PulseHistoryTracker(window=3)
    tracker.record(_pulse(sentiment="bullish", conviction="strong", activity="calm"))
    tracker.record(_pulse(sentiment="bullish", conviction="moderate", activity="active"))
    tracker.record(_pulse(sentiment="mixed", conviction="weak", activity="volatile"))
    summary = tracker.summary()
    assert summary.sentiment == ("bullish", "bullish", "mixed")
    assert summary.conviction == ("strong", "moderate", "weak")
    assert summary.activity == ("calm", "active", "volatile")


def test_has_trend_property() -> None:
    """has_trend is False until the buffer holds at least 2 entries —
    the renderer uses this to suppress the HIST line on tick 1 (a
    one-character sparkline conveys nothing)."""
    tracker = PulseHistoryTracker(window=5)
    assert not tracker.summary().has_trend  # empty
    tracker.record(_pulse())
    assert not tracker.summary().has_trend  # 1 entry
    tracker.record(_pulse())
    assert tracker.summary().has_trend  # 2 entries


def test_empty_tracker_summary_is_empty_pulse_history() -> None:
    """Summary on an empty tracker returns a PulseHistory with empty
    tuples — not None — so the renderer can call .has_trend etc.
    without a None check."""
    tracker = PulseHistoryTracker(window=5)
    summary = tracker.summary()
    assert isinstance(summary, PulseHistory)
    assert summary.length == 0
    assert summary.momentum_breadth == ()
    assert summary.window == 5
