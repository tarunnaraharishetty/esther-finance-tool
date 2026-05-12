"""Tests for src.intelligence.watchlist."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from src.dashboard.state import DashboardSnapshot, RecommendationRow
from src.intelligence.watchlist import (
    action_breakdown,
    diff_snapshots,
    top_movers,
)
from src.strategy.base import SignalAction


def _row(
    symbol: str,
    *,
    action: SignalAction = SignalAction.HOLD,
    confidence: float = 0.3,
    combined: float | None = None,
    error: str | None = None,
) -> RecommendationRow:
    return RecommendationRow(
        symbol=symbol,
        action=action,
        confidence=confidence,
        combined_score=combined if combined is not None else confidence,
        technical_score=0.0,
        sentiment_score=0.0,
        rsi=math.nan,
        macd=math.nan,
        bollinger=math.nan,
        last_price=100.0,
        num_news_articles=0,
        reasoning="",
        timestamp=datetime.now(UTC),
        error=error,
    )


def _snap(rows: list[RecommendationRow], *, tick: int = 1) -> DashboardSnapshot:
    return DashboardSnapshot(tick=tick, rows=rows, events=[], timestamp=datetime.now(UTC))


def test_diff_returns_empty_on_first_snapshot() -> None:
    snap = _snap([_row("AAPL")])
    assert diff_snapshots(snap, None) == []


def test_diff_detects_action_change() -> None:
    prev = _snap([_row("AAPL", action=SignalAction.HOLD)])
    curr = _snap([_row("AAPL", action=SignalAction.BUY)], tick=2)
    changes = diff_snapshots(curr, prev)
    assert len(changes) == 1
    assert changes[0].kind == "action_changed"
    assert "AAPL" in changes[0].description


def test_diff_detects_confidence_jump_only_when_action_unchanged() -> None:
    prev = _snap([_row("AAPL", action=SignalAction.BUY, confidence=0.3)])
    curr = _snap([_row("AAPL", action=SignalAction.BUY, confidence=0.7)], tick=2)
    changes = diff_snapshots(curr, prev)
    assert len(changes) == 1
    assert changes[0].kind == "confidence_jump"


def test_diff_ignores_small_confidence_moves() -> None:
    prev = _snap([_row("AAPL", action=SignalAction.BUY, confidence=0.50)])
    curr = _snap([_row("AAPL", action=SignalAction.BUY, confidence=0.55)], tick=2)
    assert diff_snapshots(curr, prev) == []


def test_diff_marks_new_and_removed_symbols() -> None:
    prev = _snap([_row("AAPL"), _row("MSFT")])
    curr = _snap([_row("AAPL"), _row("NVDA")], tick=2)
    changes = {(c.symbol, c.kind) for c in diff_snapshots(curr, prev)}
    assert ("NVDA", "new") in changes
    assert ("MSFT", "removed") in changes


def test_diff_skips_error_rows() -> None:
    prev = _snap([_row("AAPL", action=SignalAction.BUY)])
    curr = _snap([_row("AAPL", action=SignalAction.HOLD, error="fetch failed")], tick=2)
    assert diff_snapshots(curr, prev) == []


def test_top_movers_ranks_by_abs_combined_score() -> None:
    snap = _snap(
        [
            _row("AAPL", combined=0.2),
            _row("MSFT", combined=-0.8),
            _row("NVDA", combined=0.5),
            _row("TSLA", combined=-0.1),
        ]
    )
    top = top_movers(snap, n=2)
    assert [r.symbol for r in top] == ["MSFT", "NVDA"]


def test_top_movers_excludes_error_rows() -> None:
    snap = _snap(
        [
            _row("AAPL", combined=0.9, error="boom"),
            _row("MSFT", combined=0.3),
        ]
    )
    top = top_movers(snap, n=5)
    assert [r.symbol for r in top] == ["MSFT"]


def test_action_breakdown_counts_each_action() -> None:
    snap = _snap(
        [
            _row("AAPL", action=SignalAction.BUY),
            _row("MSFT", action=SignalAction.BUY),
            _row("NVDA", action=SignalAction.HOLD),
            _row("TSLA", action=SignalAction.SELL),
        ]
    )
    counts = action_breakdown(snap)
    assert counts[SignalAction.BUY] == 2
    assert counts[SignalAction.HOLD] == 1
    assert counts[SignalAction.SELL] == 1


def test_action_breakdown_treats_error_rows_as_hold() -> None:
    snap = _snap(
        [
            _row("AAPL", action=SignalAction.BUY, error="boom"),
            _row("MSFT", action=SignalAction.BUY),
        ]
    )
    counts = action_breakdown(snap)
    assert counts[SignalAction.BUY] == 1
    assert counts[SignalAction.HOLD] == 1
