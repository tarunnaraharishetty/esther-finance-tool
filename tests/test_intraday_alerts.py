"""Tests for src.intelligence.intraday_alerts (MT2 phase 2c)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from src.dashboard.state import DashboardSnapshot, RecommendationRow
from src.data.models import TimeFrame
from src.intelligence.intraday_alerts import (
    IntradayMomentumCollapseRule,
    IntradayOpportunityEntryRule,
    IntradayReversalAccelerationRule,
    RapidConfidenceDecayRule,
    TimeframeDisagreementRule,
)
from src.intelligence.pulse_history import PulseHistory
from src.strategy.base import SignalAction
from src.strategy.multi_timeframe import IntradayRead


def _row(
    symbol: str,
    *,
    daily: SignalAction = SignalAction.HOLD,
    intraday: SignalAction | None = SignalAction.HOLD,
    intraday_conf: float = 0.5,
    intraday_technical: float = 0.0,
) -> RecommendationRow:
    intraday_read = (
        IntradayRead(
            timeframe=TimeFrame.MIN_15,
            action=intraday,
            confidence=intraday_conf,
            combined_score=intraday_technical,
            technical_score=intraday_technical,
        )
        if intraday is not None
        else None
    )
    return RecommendationRow(
        symbol=symbol,
        action=daily,
        confidence=0.5,
        combined_score=0.0,
        technical_score=0.0,
        sentiment_score=0.0,
        rsi=math.nan,
        macd=math.nan,
        bollinger=math.nan,
        last_price=100.0,
        num_news_articles=0,
        reasoning="",
        timestamp=datetime.now(UTC),
        intraday=intraday_read,
    )


# ---------------------------------------------------------------------------
# IntradayReversalAccelerationRule
# ---------------------------------------------------------------------------


def test_reversal_acceleration_fires_on_flip_with_rising_confidence() -> None:
    rule = IntradayReversalAccelerationRule()
    prev = _row("AAPL", intraday=SignalAction.SELL, intraday_conf=0.30)
    curr = _row("AAPL", intraday=SignalAction.BUY, intraday_conf=0.55)
    alert = rule.evaluate(curr, prev)
    assert alert is not None
    assert "flipped SELL → BUY" in alert.message


def test_reversal_acceleration_skips_when_action_unchanged() -> None:
    rule = IntradayReversalAccelerationRule()
    prev = _row("AAPL", intraday=SignalAction.BUY, intraday_conf=0.30)
    curr = _row("AAPL", intraday=SignalAction.BUY, intraday_conf=0.55)
    assert rule.evaluate(curr, prev) is None


def test_reversal_acceleration_skips_when_conf_rise_below_delta() -> None:
    rule = IntradayReversalAccelerationRule(delta=0.20)
    prev = _row("AAPL", intraday=SignalAction.SELL, intraday_conf=0.30)
    curr = _row("AAPL", intraday=SignalAction.BUY, intraday_conf=0.40)
    assert rule.evaluate(curr, prev) is None


# ---------------------------------------------------------------------------
# IntradayMomentumCollapseRule
# ---------------------------------------------------------------------------


def test_momentum_collapse_fires_on_sharp_drop() -> None:
    rule = IntradayMomentumCollapseRule()
    prev = _row("AAPL", intraday=SignalAction.BUY, intraday_conf=0.75)
    curr = _row("AAPL", intraday=SignalAction.BUY, intraday_conf=0.40)
    alert = rule.evaluate(curr, prev)
    assert alert is not None
    assert "collapsed 0.75" in alert.message


def test_momentum_collapse_skips_when_drop_below_delta() -> None:
    rule = IntradayMomentumCollapseRule(delta=0.30)
    prev = _row("AAPL", intraday=SignalAction.BUY, intraday_conf=0.60)
    curr = _row("AAPL", intraday=SignalAction.BUY, intraday_conf=0.50)
    assert rule.evaluate(curr, prev) is None


# ---------------------------------------------------------------------------
# TimeframeDisagreementRule
# ---------------------------------------------------------------------------


def test_timeframe_disagreement_fires_on_transition_into_conflict() -> None:
    rule = TimeframeDisagreementRule()
    prev = _row("AAPL", daily=SignalAction.BUY, intraday=SignalAction.BUY)
    curr = _row("AAPL", daily=SignalAction.BUY, intraday=SignalAction.SELL)
    alert = rule.evaluate(curr, prev)
    assert alert is not None
    assert "daily BUY" in alert.message
    assert "intraday SELL" in alert.message


def test_timeframe_disagreement_quiet_on_sustained_conflict() -> None:
    """A symbol that was already in conflict last tick shouldn't
    re-emit the alert each tick — the rule fires on the transition
    edge, not the membership."""
    rule = TimeframeDisagreementRule()
    prev = _row("AAPL", daily=SignalAction.BUY, intraday=SignalAction.SELL)
    curr = _row("AAPL", daily=SignalAction.BUY, intraday=SignalAction.SELL)
    assert rule.evaluate(curr, prev) is None


def test_timeframe_disagreement_skips_aligned_directions() -> None:
    rule = TimeframeDisagreementRule()
    prev = _row("AAPL", daily=SignalAction.BUY, intraday=SignalAction.BUY)
    curr = _row("AAPL", daily=SignalAction.BUY, intraday=SignalAction.BUY)
    assert rule.evaluate(curr, prev) is None


# ---------------------------------------------------------------------------
# IntradayOpportunityEntryRule
# ---------------------------------------------------------------------------


def _snap_with_intraday_buy(symbol: str, technical: float) -> DashboardSnapshot:
    row = _row(
        symbol,
        daily=SignalAction.BUY,
        intraday=SignalAction.BUY,
        intraday_conf=0.7,
        intraday_technical=technical,
    )
    # Patch the intraday signal history so the ranker has a friendly
    # tick count + confidence rise.
    from src.intelligence.history import SignalEpisode, SignalHistorySummary

    history = SignalHistorySummary(
        current=SignalEpisode(
            action=SignalAction.BUY,
            started_at=datetime.now(UTC),
            last_seen_at=datetime.now(UTC),
            tick_count=4,
            confidence_first=0.5,
            confidence_last=0.7,
        ),
        recent=(),
    )
    return DashboardSnapshot(
        tick=1,
        rows=[row],
        intraday_signal_history={symbol: history},
    )


def test_intraday_opp_entry_baselines_then_fires_on_new_entries() -> None:
    """First tick establishes baseline; subsequent ticks emit one
    alert per new top-N member, matching the daily OPP entry rule."""
    rule = IntradayOpportunityEntryRule(n=3, min_composite=0.10)

    # First tick — baseline only, no alerts.
    snap_a = _snap_with_intraday_buy("AAPL", technical=0.6)
    assert rule.evaluate_snapshot(snap_a) == []

    # Second tick adds a new symbol — alert fires on the new entry.
    snap_b = DashboardSnapshot(
        tick=2,
        rows=[
            _row(
                "AAPL",
                daily=SignalAction.BUY,
                intraday=SignalAction.BUY,
                intraday_conf=0.7,
                intraday_technical=0.6,
            ),
            _row(
                "NVDA",
                daily=SignalAction.BUY,
                intraday=SignalAction.BUY,
                intraday_conf=0.7,
                intraday_technical=0.7,
            ),
        ],
        intraday_signal_history=snap_a.intraday_signal_history,
    )
    alerts = rule.evaluate_snapshot(snap_b)
    assert [a.symbol for a in alerts] == ["NVDA"]


# ---------------------------------------------------------------------------
# RapidConfidenceDecayRule
# ---------------------------------------------------------------------------


def _pulse_history_with_breadth(*values: float) -> PulseHistory:
    """Build a PulseHistory carrying a custom momentum-breadth series.
    Other series are zeroed since the decay rule only reads breadth."""
    n = len(values)
    return PulseHistory(
        momentum_breadth=values,
        sentiment_breadth=(0.0,) * n,
        bullish_count=(0,) * n,
        bearish_count=(0,) * n,
        reversal_intensity=(0,) * n,
        alert_intensity=(0,) * n,
        sentiment=("neutral",) * n,
        conviction=("weak",) * n,
        activity=("calm",) * n,
        window=20,
    )


def test_rapid_decay_fires_when_breadth_drops_below_ratio() -> None:
    """Latest breadth value at 50% or less of the earlier-window peak
    (and the earlier peak is non-zero) fires the alert."""
    rule = RapidConfidenceDecayRule(window=4, ratio=0.50)
    history = _pulse_history_with_breadth(0.8, 0.7, 0.6, 0.3)
    snap = DashboardSnapshot(tick=1, rows=[], intraday_pulse_history=history)
    alerts = rule.evaluate_snapshot(snap)
    assert len(alerts) == 1
    assert "decayed" in alerts[0].message


def test_rapid_decay_does_not_refire_on_continued_decay() -> None:
    """Once the rule fires, it won't re-emit until the breadth
    recovers — keeps the alerts pane from churning on a sustained
    decay."""
    rule = RapidConfidenceDecayRule(window=4, ratio=0.50)
    history = _pulse_history_with_breadth(0.8, 0.7, 0.6, 0.3)
    snap = DashboardSnapshot(tick=1, rows=[], intraday_pulse_history=history)
    rule.evaluate_snapshot(snap)
    # Same shape next tick — already-fired flag suppresses re-emit.
    assert rule.evaluate_snapshot(snap) == []


def test_rapid_decay_skipped_when_history_too_short() -> None:
    rule = RapidConfidenceDecayRule(window=4, ratio=0.50)
    history = _pulse_history_with_breadth(0.8, 0.7)
    snap = DashboardSnapshot(tick=1, rows=[], intraday_pulse_history=history)
    assert rule.evaluate_snapshot(snap) == []
