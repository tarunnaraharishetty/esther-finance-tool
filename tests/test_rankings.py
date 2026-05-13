"""Tests for src.intelligence.rankings."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from src.dashboard.state import DashboardSnapshot, RecommendationRow
from src.intelligence.history import (
    SignalEpisode,
    SignalHistorySummary,
)
from src.intelligence.rankings import compute
from src.strategy.base import SignalAction


def _row(
    symbol: str,
    *,
    action: SignalAction = SignalAction.HOLD,
    confidence: float = 0.3,
    combined: float = 0.0,
    sentiment: float = 0.0,
    news: int = 0,
    macd: float = 0.0,
    error: str | None = None,
) -> RecommendationRow:
    return RecommendationRow(
        symbol=symbol,
        action=action,
        confidence=confidence,
        combined_score=combined,
        technical_score=0.0,
        sentiment_score=sentiment,
        rsi=math.nan,
        macd=macd,
        bollinger=math.nan,
        last_price=100.0,
        num_news_articles=news,
        reasoning="",
        timestamp=datetime.now(UTC),
        error=error,
    )


def _snap(
    rows: list[RecommendationRow],
    *,
    history: dict[str, SignalHistorySummary] | None = None,
) -> DashboardSnapshot:
    return DashboardSnapshot(
        tick=1,
        rows=rows,
        events=[],
        signal_history=history or {},
        timestamp=datetime.now(UTC),
    )


def _episode(
    action: SignalAction,
    ticks: int = 1,
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


# ---------------------------------------------------------------------------
# strongest_momentum
# ---------------------------------------------------------------------------


def test_momentum_ranks_by_abs_macd() -> None:
    snap = _snap([
        _row("AAPL", macd=0.10),
        _row("NVDA", macd=-0.80),
        _row("MSFT", macd=0.40),
    ])
    ranks = compute(snap, n=3)
    assert ranks.strongest_momentum == (
        ("NVDA", -0.80),
        ("MSFT", 0.40),
        ("AAPL", 0.10),
    )


def test_momentum_excludes_error_rows() -> None:
    snap = _snap([
        _row("AAPL", macd=0.50, error="boom"),
        _row("NVDA", macd=0.20),
    ])
    assert compute(snap, n=3).strongest_momentum == (("NVDA", 0.20),)


def test_momentum_handles_nan_macd() -> None:
    snap = _snap([
        _row("AAPL", macd=math.nan),
        _row("NVDA", macd=0.30),
    ])
    ranks = compute(snap, n=3).strongest_momentum
    # NaN-MACD rows score 0, not NaN — they may or may not appear, but
    # NVDA must be first.
    assert ranks[0] == ("NVDA", 0.30)


# ---------------------------------------------------------------------------
# strongest_sentiment
# ---------------------------------------------------------------------------


def test_sentiment_ranks_by_abs() -> None:
    snap = _snap([
        _row("AAPL", sentiment=0.6, news=5),
        _row("NVDA", sentiment=-0.7, news=3),
        _row("MSFT", sentiment=0.2, news=2),
    ])
    ranks = compute(snap, n=3).strongest_sentiment
    assert ranks == (("NVDA", -0.7), ("AAPL", 0.6), ("MSFT", 0.2))


def test_sentiment_excludes_zero_news_symbols() -> None:
    """A symbol with sentiment_score=0.5 but no articles is misleading;
    it shouldn't rank."""
    snap = _snap([
        _row("AAPL", sentiment=0.9, news=0),  # excluded
        _row("NVDA", sentiment=0.3, news=2),
    ])
    assert compute(snap, n=3).strongest_sentiment == (("NVDA", 0.3),)


# ---------------------------------------------------------------------------
# highest_confidence
# ---------------------------------------------------------------------------


def test_confidence_ranks_largest_first() -> None:
    snap = _snap([
        _row("AAPL", confidence=0.3),
        _row("NVDA", confidence=0.8),
        _row("MSFT", confidence=0.5),
    ])
    assert compute(snap, n=3).highest_confidence == (
        ("NVDA", 0.8),
        ("MSFT", 0.5),
        ("AAPL", 0.3),
    )


def test_confidence_excludes_error_rows() -> None:
    snap = _snap([
        _row("AAPL", confidence=0.9, error="boom"),
        _row("NVDA", confidence=0.3),
    ])
    assert compute(snap, n=3).highest_confidence == (("NVDA", 0.3),)


# ---------------------------------------------------------------------------
# biggest_reversals
# ---------------------------------------------------------------------------


def test_reversals_require_action_flip_to_opposite_direction() -> None:
    """BUY→SELL or SELL→BUY counts. BUY→HOLD does not (HOLD has dir=0)."""
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(SignalAction.BUY, conf_last=0.7),
            recent=(_episode(SignalAction.SELL),),
        ),
        "MSFT": SignalHistorySummary(
            current=_episode(SignalAction.BUY, conf_last=0.5),
            recent=(_episode(SignalAction.HOLD),),  # HOLD->BUY doesn't count
        ),
        "NVDA": SignalHistorySummary(
            current=_episode(SignalAction.SELL, conf_last=0.4),
            recent=(_episode(SignalAction.BUY),),
        ),
    }
    snap = _snap(
        [
            _row("AAPL", action=SignalAction.BUY, confidence=0.7),
            _row("MSFT", action=SignalAction.BUY, confidence=0.5),
            _row("NVDA", action=SignalAction.SELL, confidence=0.4),
        ],
        history=history,
    )
    ranks = compute(snap, n=3).biggest_reversals
    symbols = [sym for sym, _ in ranks]
    assert "MSFT" not in symbols
    assert "AAPL" in symbols and "NVDA" in symbols


def test_reversals_sign_carries_direction() -> None:
    """Positive score = bearish-to-bullish (current is BUY).
    Negative score = bullish-to-bearish (current is SELL)."""
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(SignalAction.BUY, conf_last=0.7),
            recent=(_episode(SignalAction.SELL),),
        ),
        "NVDA": SignalHistorySummary(
            current=_episode(SignalAction.SELL, conf_last=0.6),
            recent=(_episode(SignalAction.BUY),),
        ),
    }
    snap = _snap(
        [
            _row("AAPL", action=SignalAction.BUY, confidence=0.7),
            _row("NVDA", action=SignalAction.SELL, confidence=0.6),
        ],
        history=history,
    )
    ranks = dict(compute(snap, n=3).biggest_reversals)
    assert ranks["AAPL"] > 0  # bearish -> bullish
    assert ranks["NVDA"] < 0  # bullish -> bearish


def test_reversals_empty_without_history() -> None:
    snap = _snap([_row("AAPL", action=SignalAction.BUY)])
    assert compute(snap, n=3).biggest_reversals == ()


# ---------------------------------------------------------------------------
# unusual_movers
# ---------------------------------------------------------------------------


def test_unusual_rewards_long_episodes_with_big_confidence_swing() -> None:
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(
                SignalAction.BUY, ticks=10, conf_first=0.3, conf_last=0.7
            ),
            recent=(),
        ),
        "NVDA": SignalHistorySummary(
            current=_episode(
                SignalAction.BUY, ticks=2, conf_first=0.3, conf_last=0.7
            ),
            recent=(),
        ),
    }
    snap = _snap(
        [
            _row("AAPL", action=SignalAction.BUY),
            _row("NVDA", action=SignalAction.BUY),
        ],
        history=history,
    )
    ranks = compute(snap, n=3).unusual_movers
    # AAPL: |0.7-0.3| * 10 = 4.0; NVDA: 0.4 * 2 = 0.8. AAPL should rank first.
    assert ranks[0][0] == "AAPL"


def test_unusual_excludes_first_tick_episodes() -> None:
    """A 1-tick episode has zero confidence delta and shouldn't qualify."""
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(SignalAction.BUY, ticks=1, conf_first=0.5, conf_last=0.5),
            recent=(),
        ),
    }
    snap = _snap([_row("AAPL", action=SignalAction.BUY)], history=history)
    assert compute(snap, n=3).unusual_movers == ()


# ---------------------------------------------------------------------------
# most_volatile
# ---------------------------------------------------------------------------


def test_volatile_counts_episodes() -> None:
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(SignalAction.BUY),
            recent=(
                _episode(SignalAction.HOLD),
                _episode(SignalAction.SELL),
                _episode(SignalAction.BUY),
            ),
        ),
        "NVDA": SignalHistorySummary(
            current=_episode(SignalAction.HOLD),
            recent=(_episode(SignalAction.BUY),),
        ),
    }
    snap = _snap(
        [
            _row("AAPL", action=SignalAction.BUY),
            _row("NVDA", action=SignalAction.HOLD),
        ],
        history=history,
    )
    ranks = compute(snap, n=3).most_volatile
    assert ranks == (("AAPL", 4.0), ("NVDA", 2.0))


def test_volatile_excludes_single_episode_symbols() -> None:
    """No flips = not volatile."""
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(SignalAction.BUY, ticks=5),
            recent=(),
        ),
    }
    snap = _snap([_row("AAPL", action=SignalAction.BUY)], history=history)
    assert compute(snap, n=3).most_volatile == ()


# ---------------------------------------------------------------------------
# Empty snapshot
# ---------------------------------------------------------------------------


def test_empty_snapshot_returns_all_empty_sections() -> None:
    snap = _snap([])
    ranks = compute(snap, n=3)
    assert ranks.strongest_momentum == ()
    assert ranks.strongest_sentiment == ()
    assert ranks.highest_confidence == ()
    assert ranks.biggest_reversals == ()
    assert ranks.unusual_movers == ()
    assert ranks.most_volatile == ()


def test_n_caps_each_section() -> None:
    rows = [_row(f"S{i}", confidence=i / 10) for i in range(1, 9)]
    snap = _snap(rows)
    ranks = compute(snap, n=3)
    assert len(ranks.highest_confidence) == 3
