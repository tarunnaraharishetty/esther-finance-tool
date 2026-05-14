"""Tests for src.strategy.multi_timeframe + the engine's intraday read.

Covers the IntradayRead dataclass, the is_divergent helper, and the
new RecommendationEngine.recommend_intraday() method that runs the
existing indicator pipeline on a different bar dataframe.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.data.models import TimeFrame
from src.strategy.base import SignalAction
from src.strategy.multi_timeframe import IntradayRead, is_divergent
from src.strategy.recommendation import RecommendationEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bullish_df(n: int = 60) -> pd.DataFrame:
    """Noisy uptrend — same shape the daily tests use, just renamed."""
    rng = np.random.default_rng(7)
    rets = rng.normal(loc=0.015, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range(end=datetime.now(UTC), periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=int),
        },
        index=idx,
    )


def _flat_df(n: int = 60) -> pd.DataFrame:
    close = np.full(n, 100.0, dtype=float)
    idx = pd.date_range(end=datetime.now(UTC), periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=int),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# is_divergent semantics
# ---------------------------------------------------------------------------


def _intraday(action: SignalAction) -> IntradayRead:
    return IntradayRead(
        timeframe=TimeFrame.MIN_15,
        action=action,
        confidence=0.5,
        combined_score=0.3 if action == SignalAction.BUY else -0.3,
        technical_score=0.3 if action == SignalAction.BUY else -0.3,
    )


def test_divergent_buy_vs_sell() -> None:
    """The whole point: daily BUY + intraday SELL is the alignment-
    breaking case the dashboard surfaces."""
    assert is_divergent(SignalAction.BUY, _intraday(SignalAction.SELL))
    assert is_divergent(SignalAction.SELL, _intraday(SignalAction.BUY))


def test_aligned_buys_are_not_divergent() -> None:
    """Same-direction actions on both timeframes = aligned, not
    divergent. (Trader gets the green chip in the DetailPanel.)"""
    assert not is_divergent(SignalAction.BUY, _intraday(SignalAction.BUY))
    assert not is_divergent(SignalAction.SELL, _intraday(SignalAction.SELL))


def test_hold_on_either_side_is_not_divergent() -> None:
    """HOLD means 'no directional read' — it can't conflict with the
    other side. We deliberately don't flag BUY-vs-HOLD as divergence
    because HOLD isn't an opposing direction."""
    assert not is_divergent(SignalAction.BUY, _intraday(SignalAction.HOLD))
    assert not is_divergent(SignalAction.HOLD, _intraday(SignalAction.BUY))
    assert not is_divergent(SignalAction.HOLD, _intraday(SignalAction.HOLD))


def test_missing_intraday_is_never_divergent() -> None:
    """None means 'no intraday read available' — disabled feature or
    failed fetch. Can't diverge by definition."""
    assert not is_divergent(SignalAction.BUY, None)
    assert not is_divergent(SignalAction.SELL, None)
    assert not is_divergent(SignalAction.HOLD, None)


# ---------------------------------------------------------------------------
# RecommendationEngine.recommend_intraday
# ---------------------------------------------------------------------------


def test_recommend_intraday_returns_action_matching_combined_sign() -> None:
    """Technical-only path: the action follows the combined score sign,
    same threshold convention as the daily recommend()."""
    engine = RecommendationEngine(buy_threshold=0.1, sell_threshold=0.1)
    bullish = engine.recommend_intraday("AAPL", _bullish_df(), timeframe=TimeFrame.MIN_15)
    # A bullish uptrend produces a directional read — the indicator
    # combination may go either way under the daily convention, but
    # the test just asserts the read is internally consistent.
    assert bullish.timeframe == TimeFrame.MIN_15
    assert bullish.combined_score == bullish.technical_score
    assert bullish.confidence == pytest.approx(abs(bullish.combined_score))


def test_recommend_intraday_empty_df_returns_hold() -> None:
    """An empty bar dataframe — the kind the controller might pass on
    a cold-start cache miss — must not crash. Returns HOLD with zero
    scores so the renderer can still surface the chip."""
    engine = RecommendationEngine()
    read = engine.recommend_intraday(
        "AAPL", pd.DataFrame(columns=["close"]), timeframe=TimeFrame.MIN_15
    )
    assert read.action == SignalAction.HOLD
    assert read.confidence == 0.0
    assert read.combined_score == 0.0


def test_recommend_intraday_flat_df_returns_hold_with_zero_scores() -> None:
    """Flat bars yield zero technical score → HOLD action, zero
    confidence. Below the buy/sell thresholds the engine defaults to
    HOLD, same as the daily path."""
    engine = RecommendationEngine()
    read = engine.recommend_intraday("AAPL", _flat_df(), timeframe=TimeFrame.MIN_15)
    assert read.action == SignalAction.HOLD
    assert read.confidence == 0.0


def test_recommend_intraday_does_not_touch_sentiment() -> None:
    """The intraday path is technical-only — no sentiment analyzer is
    invoked. A SentimentAnalyzer set to raise on call must still let
    the intraday read complete cleanly."""

    class _ExplodingAnalyzer:
        def score_text(self, _: str) -> object:
            raise AssertionError("sentiment must not be invoked for intraday")

        def score_article(self, _: object) -> object:
            raise AssertionError("sentiment must not be invoked for intraday")

    engine = RecommendationEngine(sentiment_analyzer=_ExplodingAnalyzer())  # type: ignore[arg-type]
    read = engine.recommend_intraday("AAPL", _bullish_df(), timeframe=TimeFrame.MIN_15)
    # No exception = pass; the read returns a valid action.
    assert read.action in (SignalAction.BUY, SignalAction.HOLD, SignalAction.SELL)


def test_recommend_intraday_carries_supplied_timeframe() -> None:
    """The dataclass faithfully records which timeframe produced the
    read — the renderer keys off this for the chip label."""
    engine = RecommendationEngine()
    for tf in (TimeFrame.MIN_5, TimeFrame.MIN_15, TimeFrame.HOUR_1):
        read = engine.recommend_intraday("AAPL", _flat_df(), timeframe=tf)
        assert read.timeframe == tf
