"""Tests for src.intelligence.history."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.intelligence.history import SignalHistory
from src.strategy.base import SignalAction


def _ts(offset_minutes: int) -> datetime:
    return datetime(2026, 5, 13, 14, 0, tzinfo=UTC) + timedelta(minutes=offset_minutes)


# ---------------------------------------------------------------------------
# Episode building
# ---------------------------------------------------------------------------


def test_first_record_starts_an_episode() -> None:
    h = SignalHistory()
    h.record("AAPL", SignalAction.BUY, 0.5, _ts(0))
    summary = h.summary_for("AAPL")
    assert summary is not None
    assert summary.current.action == SignalAction.BUY
    assert summary.current.tick_count == 1
    assert summary.current.confidence_first == 0.5
    assert summary.current.confidence_last == 0.5
    assert summary.recent == ()


def test_same_action_extends_current_episode() -> None:
    h = SignalHistory()
    h.record("AAPL", SignalAction.BUY, 0.5, _ts(0))
    h.record("AAPL", SignalAction.BUY, 0.6, _ts(5))
    h.record("AAPL", SignalAction.BUY, 0.7, _ts(10))
    summary = h.summary_for("AAPL")
    assert summary is not None
    assert summary.current.tick_count == 3
    assert summary.current.confidence_first == 0.5
    assert summary.current.confidence_last == 0.7
    assert summary.recent == ()  # only one episode total


def test_action_flip_closes_episode_and_starts_new() -> None:
    h = SignalHistory()
    h.record("AAPL", SignalAction.BUY, 0.5, _ts(0))
    h.record("AAPL", SignalAction.BUY, 0.6, _ts(5))
    h.record("AAPL", SignalAction.HOLD, 0.2, _ts(10))
    summary = h.summary_for("AAPL")
    assert summary is not None
    assert summary.current.action == SignalAction.HOLD
    assert summary.current.tick_count == 1
    assert len(summary.recent) == 1
    assert summary.recent[0].action == SignalAction.BUY
    assert summary.recent[0].tick_count == 2


def test_summary_recent_is_most_recent_first() -> None:
    h = SignalHistory()
    # BUY, then HOLD, then SELL, then BUY again — 4 episodes total.
    h.record("AAPL", SignalAction.BUY, 0.5, _ts(0))
    h.record("AAPL", SignalAction.HOLD, 0.1, _ts(5))
    h.record("AAPL", SignalAction.SELL, -0.5, _ts(10))
    h.record("AAPL", SignalAction.BUY, 0.6, _ts(15))
    summary = h.summary_for("AAPL")
    assert summary is not None
    assert summary.current.action == SignalAction.BUY
    assert [ep.action for ep in summary.recent] == [
        SignalAction.SELL,
        SignalAction.HOLD,
        SignalAction.BUY,
    ]


def test_summary_recent_capped_by_kwarg() -> None:
    h = SignalHistory()
    actions = [
        SignalAction.BUY,
        SignalAction.HOLD,
        SignalAction.SELL,
        SignalAction.HOLD,
        SignalAction.BUY,
        SignalAction.HOLD,
    ]
    for i, action in enumerate(actions):
        h.record("AAPL", action, 0.5, _ts(i))
    summary = h.summary_for("AAPL", recent=2)
    assert summary is not None
    assert summary.current.action == SignalAction.HOLD
    assert len(summary.recent) == 2


def test_summary_returns_none_for_unrecorded_symbol() -> None:
    assert SignalHistory().summary_for("ZZZZ") is None


# ---------------------------------------------------------------------------
# Confidence trend
# ---------------------------------------------------------------------------


def test_confidence_trend_rising() -> None:
    h = SignalHistory()
    h.record("AAPL", SignalAction.BUY, 0.3, _ts(0))
    h.record("AAPL", SignalAction.BUY, 0.7, _ts(5))
    summary = h.summary_for("AAPL")
    assert summary is not None
    assert summary.current.confidence_trend == "rising"


def test_confidence_trend_falling() -> None:
    h = SignalHistory()
    h.record("AAPL", SignalAction.BUY, 0.7, _ts(0))
    h.record("AAPL", SignalAction.BUY, 0.2, _ts(5))
    summary = h.summary_for("AAPL")
    assert summary is not None
    assert summary.current.confidence_trend == "falling"


def test_confidence_trend_flat_within_eps() -> None:
    h = SignalHistory()
    h.record("AAPL", SignalAction.BUY, 0.50, _ts(0))
    h.record("AAPL", SignalAction.BUY, 0.51, _ts(5))
    summary = h.summary_for("AAPL")
    assert summary is not None
    assert summary.current.confidence_trend == "flat"


# ---------------------------------------------------------------------------
# Per-symbol isolation + bound
# ---------------------------------------------------------------------------


def test_symbols_are_isolated() -> None:
    h = SignalHistory()
    h.record("AAPL", SignalAction.BUY, 0.5, _ts(0))
    h.record("MSFT", SignalAction.SELL, -0.5, _ts(0))
    aapl = h.summary_for("AAPL")
    msft = h.summary_for("MSFT")
    assert aapl is not None and msft is not None
    assert aapl.current.action == SignalAction.BUY
    assert msft.current.action == SignalAction.SELL


def test_max_episodes_per_symbol_drops_oldest() -> None:
    """When the cap is hit, oldest episodes drop. Current must always remain."""
    h = SignalHistory(max_episodes_per_symbol=3)
    # 5 alternating episodes — should keep only the last 3.
    actions = [
        SignalAction.BUY,
        SignalAction.HOLD,
        SignalAction.SELL,
        SignalAction.BUY,
        SignalAction.HOLD,
    ]
    for i, action in enumerate(actions):
        h.record("AAPL", action, 0.5, _ts(i))
    eps = h.episodes_for("AAPL")
    assert len(eps) == 3
    assert [ep.action for ep in eps] == [
        SignalAction.SELL,
        SignalAction.BUY,
        SignalAction.HOLD,
    ]
