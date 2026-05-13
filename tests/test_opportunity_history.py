"""Tests for src.intelligence.opportunity_history."""

from __future__ import annotations

import pytest

from src.intelligence.opportunity_history import (
    OpportunityHistory,
    OpportunityMembershipTracker,
)


def test_first_tick_present_symbol_is_new() -> None:
    """A symbol present on the very first call gets streak=1, appearances=1
    — that's the 'NEW' state for a freshly-launched dashboard."""
    tracker = OpportunityMembershipTracker(window=10)
    tracker.record(["NVDA", "AAPL"])
    summary = tracker.summary_for("NVDA")
    assert summary == OpportunityHistory(streak=1, appearances=1, window=10)


def test_consecutive_presence_bumps_streak() -> None:
    tracker = OpportunityMembershipTracker(window=10)
    for _ in range(3):
        tracker.record(["NVDA"])
    assert tracker.summary_for("NVDA") == OpportunityHistory(
        streak=3, appearances=3, window=10
    )


def test_absence_resets_streak_but_keeps_appearances() -> None:
    """Streak counts trailing presence only; appearances spans the
    whole window. A symbol in 3 of last 4 ticks (with a gap) has
    streak=1, appearances=3."""
    tracker = OpportunityMembershipTracker(window=10)
    tracker.record(["NVDA"])
    tracker.record(["NVDA"])
    tracker.record([])  # absent
    tracker.record(["NVDA"])
    assert tracker.summary_for("NVDA") == OpportunityHistory(
        streak=1, appearances=3, window=10
    )


def test_symbol_never_present_returns_none() -> None:
    tracker = OpportunityMembershipTracker(window=10)
    tracker.record(["NVDA"])
    assert tracker.summary_for("AAPL") is None


def test_symbol_pruned_after_full_window_absence() -> None:
    """A symbol absent for the entire window is dropped from memory.

    If it returns later, it reads as a fresh NEW entry — there's no
    way (and no need) to know it was ever here before."""
    tracker = OpportunityMembershipTracker(window=3)
    tracker.record(["NVDA"])  # NVDA: [T]
    tracker.record([])         # NVDA: [T, F]
    tracker.record([])         # NVDA: [T, F, F]
    tracker.record([])         # NVDA: [F, F, F] → pruned
    assert tracker.summary_for("NVDA") is None
    # Returning later reads as brand new.
    tracker.record(["NVDA"])
    assert tracker.summary_for("NVDA") == OpportunityHistory(
        streak=1, appearances=1, window=3
    )


def test_window_caps_appearances_count() -> None:
    """Appearances is bounded by window size — older entries fall off."""
    tracker = OpportunityMembershipTracker(window=3)
    for _ in range(10):
        tracker.record(["NVDA"])
    assert tracker.summary_for("NVDA") == OpportunityHistory(
        streak=3, appearances=3, window=3
    )


def test_record_handles_set_iterable_input() -> None:
    """Caller may pass a set, list, tuple, generator — anything iterable."""
    tracker = OpportunityMembershipTracker(window=5)
    tracker.record({"NVDA", "AAPL"})  # set
    tracker.record(("AAPL", "MSFT"))  # tuple
    tracker.record(s for s in ["NVDA"])  # generator
    assert tracker.summary_for("NVDA") is not None
    assert tracker.summary_for("MSFT") is not None


def test_tracked_symbols_lists_currently_held() -> None:
    tracker = OpportunityMembershipTracker(window=3)
    tracker.record(["NVDA", "AAPL"])
    tracker.record(["AAPL", "MSFT"])
    assert set(tracker.tracked_symbols()) == {"NVDA", "AAPL", "MSFT"}


def test_invalid_window_rejected() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        OpportunityMembershipTracker(window=0)


def test_streak_zero_for_currently_absent_tracked_symbol() -> None:
    """A symbol that's tracked but absent THIS tick has streak=0 even
    though earlier appearances are still in the window."""
    tracker = OpportunityMembershipTracker(window=5)
    tracker.record(["NVDA"])  # present
    tracker.record(["NVDA"])  # present
    tracker.record([])         # absent now
    # NVDA still tracked (window has Trues), but streak resets to 0.
    summary = tracker.summary_for("NVDA")
    assert summary is not None
    assert summary.streak == 0
    assert summary.appearances == 2
