"""Tests for src.intelligence.timeframe_compare (MT2 phase 2c)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from src.dashboard.state import DashboardSnapshot, RecommendationRow
from src.data.models import TimeFrame
from src.intelligence.history import SignalEpisode, SignalHistorySummary
from src.intelligence.timeframe_compare import (
    TimeframeStance,
    aggregate_counts,
    compare_timeframes,
)
from src.strategy.base import SignalAction
from src.strategy.multi_timeframe import IntradayRead


def _row(
    symbol: str,
    *,
    daily: SignalAction,
    intraday: SignalAction | None,
) -> RecommendationRow:
    intraday_read = (
        IntradayRead(
            timeframe=TimeFrame.MIN_15,
            action=intraday,
            confidence=0.5,
            combined_score=0.0,
            technical_score=0.0,
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


def _snap(
    rows: list[RecommendationRow],
    *,
    intraday_history: dict[str, SignalHistorySummary] | None = None,
) -> DashboardSnapshot:
    return DashboardSnapshot(
        tick=1,
        rows=rows,
        intraday_signal_history=intraday_history or {},
    )


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


def test_aligned_bullish_when_both_buy() -> None:
    """Both timeframes pointing at BUY → ``aligned_bullish``."""
    snap = _snap([_row("AAPL", daily=SignalAction.BUY, intraday=SignalAction.BUY)])
    stances = compare_timeframes(snap)
    assert stances["AAPL"].category == "aligned_bullish"


def test_aligned_bearish_when_both_sell() -> None:
    snap = _snap([_row("AAPL", daily=SignalAction.SELL, intraday=SignalAction.SELL)])
    assert compare_timeframes(snap)["AAPL"].category == "aligned_bearish"


def test_conflict_when_buy_vs_sell() -> None:
    snap = _snap([_row("AAPL", daily=SignalAction.BUY, intraday=SignalAction.SELL)])
    assert compare_timeframes(snap)["AAPL"].category == "conflict"


def test_intraday_only_when_daily_hold_intraday_directional() -> None:
    snap = _snap([_row("AAPL", daily=SignalAction.HOLD, intraday=SignalAction.BUY)])
    assert compare_timeframes(snap)["AAPL"].category == "intraday_only"


def test_daily_only_when_intraday_hold_daily_directional() -> None:
    snap = _snap([_row("AAPL", daily=SignalAction.SELL, intraday=SignalAction.HOLD)])
    assert compare_timeframes(snap)["AAPL"].category == "daily_only"


def test_neutral_when_both_hold() -> None:
    snap = _snap([_row("AAPL", daily=SignalAction.HOLD, intraday=SignalAction.HOLD)])
    assert compare_timeframes(snap)["AAPL"].category == "neutral"


def test_rows_without_intraday_are_omitted() -> None:
    """Rows where row.intraday is None can't be compared — there's
    no intraday side. They're omitted from the result."""
    snap = _snap([_row("AAPL", daily=SignalAction.BUY, intraday=None)])
    assert compare_timeframes(snap) == {}


# ---------------------------------------------------------------------------
# Phrases derived from intraday history
# ---------------------------------------------------------------------------


def _episode(
    action: SignalAction,
    ticks: int,
    conf_first: float,
    conf_last: float,
) -> SignalEpisode:
    return SignalEpisode(
        action=action,
        started_at=datetime(2026, 5, 14, tzinfo=UTC),
        last_seen_at=datetime(2026, 5, 14, tzinfo=UTC) + timedelta(seconds=ticks * 5),
        tick_count=ticks,
        confidence_first=conf_first,
        confidence_last=conf_last,
    )


def test_strengthening_phrase_fires_on_rising_intraday_confidence() -> None:
    history = SignalHistorySummary(
        current=_episode(SignalAction.BUY, ticks=4, conf_first=0.3, conf_last=0.6),
        recent=(),
    )
    snap = _snap(
        [_row("AAPL", daily=SignalAction.BUY, intraday=SignalAction.BUY)],
        intraday_history={"AAPL": history},
    )
    phrases = compare_timeframes(snap)["AAPL"].phrases
    assert "strengthening intraday momentum" in phrases


def test_weakening_phrase_fires_on_falling_intraday_confidence() -> None:
    history = SignalHistorySummary(
        current=_episode(SignalAction.BUY, ticks=4, conf_first=0.7, conf_last=0.4),
        recent=(),
    )
    snap = _snap(
        [_row("AAPL", daily=SignalAction.BUY, intraday=SignalAction.BUY)],
        intraday_history={"AAPL": history},
    )
    phrases = compare_timeframes(snap)["AAPL"].phrases
    assert "weakening intraday conviction" in phrases


def test_reversal_against_daily_trend_phrase_on_fresh_conflict_flip() -> None:
    """Conflict + fresh intraday flip (current tick_count low + prior
    episode opposite direction) → 'intraday reversal against daily
    trend'."""
    history = SignalHistorySummary(
        current=_episode(SignalAction.SELL, ticks=2, conf_first=0.4, conf_last=0.5),
        recent=(_episode(SignalAction.BUY, ticks=6, conf_first=0.5, conf_last=0.7),),
    )
    snap = _snap(
        [_row("AAPL", daily=SignalAction.BUY, intraday=SignalAction.SELL)],
        intraday_history={"AAPL": history},
    )
    phrases = compare_timeframes(snap)["AAPL"].phrases
    assert "intraday reversal against daily trend" in phrases


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def test_aggregate_counts_returns_category_totals() -> None:
    """Counts roll up by category, mirroring the labels the ALIGN
    header chip surfaces."""
    stances = {
        "AAPL": TimeframeStance(
            symbol="AAPL",
            daily_action=SignalAction.BUY,
            intraday_action=SignalAction.BUY,
            category="aligned_bullish",
        ),
        "MSFT": TimeframeStance(
            symbol="MSFT",
            daily_action=SignalAction.SELL,
            intraday_action=SignalAction.BUY,
            category="conflict",
        ),
        "NVDA": TimeframeStance(
            symbol="NVDA",
            daily_action=SignalAction.BUY,
            intraday_action=SignalAction.BUY,
            category="aligned_bullish",
        ),
    }
    counts = aggregate_counts(stances)
    assert counts["aligned_bullish"] == 2
    assert counts["conflict"] == 1
