"""Tests for RecommendationBacktest using deterministic synthetic OHLCV.

No network, no FinBERT — uses a stub sentiment analyzer or runs purely
technical.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.backtesting.recommendation_backtest import (
    BacktestRunConfig,
    Position,
    RecommendationBacktest,
)
from src.data.models import NewsArticle
from src.sentiment.analyzer import SentimentAnalyzer, SentimentLabel, SentimentScore
from src.strategy.recommendation import IndicatorWeights, RecommendationEngine


class _NeutralSentiment(SentimentAnalyzer):
    """Skip FinBERT entirely — every article scores neutral / zero."""

    def __init__(self) -> None:
        pass

    def score_text(self, _text: str) -> SentimentScore:  # type: ignore[override]
        return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

    def score_article(self, _article: NewsArticle) -> SentimentScore:  # type: ignore[override]
        return self.score_text("")


def _engine_technical_only() -> RecommendationEngine:
    """Engine with sentiment weight zero — purely technical decisions."""
    return RecommendationEngine(
        technical_weight=1.0,
        sentiment_weight=0.0,
        sentiment_analyzer=_NeutralSentiment(),
    )


def _ohlcv(
    n: int, *, drift: float, noise: float = 0.01, seed: int = 7, start: float = 100.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=noise, size=n)
    close = start * np.exp(np.cumsum(rets))
    idx = pd.date_range(end=datetime.now(UTC), periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=int),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Plumbing / structure
# ---------------------------------------------------------------------------


def test_run_requires_enough_bars() -> None:
    bt = RecommendationBacktest(_engine_technical_only(), BacktestRunConfig(lookback_bars=60))
    short_df = _ohlcv(40, drift=0.0)
    with pytest.raises(ValueError, match="need >"):
        bt.run(short_df)


def test_run_validates_columns() -> None:
    bt = RecommendationBacktest(_engine_technical_only())
    bad = pd.DataFrame({"close": [100, 101]}, index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="missing columns"):
        bt.run(bad)


def test_equity_curve_length_matches_input() -> None:
    bt = RecommendationBacktest(_engine_technical_only())
    df = _ohlcv(180, drift=0.001)
    result = bt.run(df, symbol="TEST")
    assert len(result.equity_curve) == len(df)
    assert len(result.returns) == len(df)
    assert len(result.actions) == len(df)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_flat_market_results_in_few_trades() -> None:
    """No drift, no noise → engine rarely sees a trade-worthy signal."""
    bt = RecommendationBacktest(
        _engine_technical_only(), BacktestRunConfig(commission_pct=0.0)
    )
    df = _ohlcv(180, drift=0.0, noise=0.0001)
    result = bt.run(df, symbol="FLAT")
    # Allow up to a handful of trades from indicator edge effects, but should
    # be much fewer than re-entering every bar.
    assert result.trades <= 4
    # Equity should be very close to starting cash.
    assert abs(result.final_value - bt.config.starting_cash) < 1_000


def test_long_only_avoids_full_drawdown_in_bear_market() -> None:
    """In a downtrend the long-only strategy should beat buy-and-hold."""
    bt = RecommendationBacktest(
        _engine_technical_only(),
        BacktestRunConfig(commission_pct=0.0, allow_short=False),
    )
    df = _ohlcv(200, drift=-0.012, noise=0.01, seed=21)
    result = bt.run(df, symbol="BEAR")
    assert result.buy_and_hold_return_pct < -20  # market genuinely fell
    # Long-only should not be substantially worse than buy-and-hold; usually better.
    assert result.alpha_pct > 0, (
        f"long-only lost more than buy-and-hold "
        f"(strategy={result.total_return_pct:.1f}%, bh={result.buy_and_hold_return_pct:.1f}%)"
    )


def test_trades_count_increases_with_commissions_applied() -> None:
    bt_no_comm = RecommendationBacktest(
        _engine_technical_only(), BacktestRunConfig(commission_pct=0.0)
    )
    bt_high_comm = RecommendationBacktest(
        _engine_technical_only(), BacktestRunConfig(commission_pct=0.02)
    )
    df = _ohlcv(200, drift=0.003, noise=0.02, seed=3)
    r_no = bt_no_comm.run(df, symbol="X")
    r_hi = bt_high_comm.run(df, symbol="X")
    # Same trade-decision sequence → same number of trades; high commission only
    # reduces the equity curve.
    assert r_no.trades == r_hi.trades
    assert r_hi.final_value <= r_no.final_value


def test_positions_only_take_long_or_flat_when_short_disallowed() -> None:
    bt = RecommendationBacktest(
        _engine_technical_only(), BacktestRunConfig(allow_short=False)
    )
    df = _ohlcv(180, drift=-0.01, noise=0.012, seed=5)
    result = bt.run(df, symbol="X")
    seen = set(result.positions.unique())
    assert Position.SHORT.value not in seen


def test_short_appears_when_allowed_in_bear_market() -> None:
    """With trend-dominant weights, a SELL emerges in a strong downtrend
    and (allow_short=True) the backtester actually opens a short."""
    engine = RecommendationEngine(
        technical_weight=1.0,
        sentiment_weight=0.0,
        # MACD line dominates; mean-reversion components are zeroed so a
        # sustained downtrend reliably yields SELL signals.
        indicator_weights=IndicatorWeights(rsi=0.0, macd=1.0, bollinger=0.0),
        sentiment_analyzer=_NeutralSentiment(),
    )
    bt = RecommendationBacktest(
        engine,
        BacktestRunConfig(allow_short=True, commission_pct=0.0, min_confidence=0.0),
    )
    df = _ohlcv(200, drift=-0.012, noise=0.01, seed=21)
    result = bt.run(df, symbol="X")
    assert Position.SHORT.value in set(result.positions.unique())


def test_min_confidence_filters_low_confidence_signals() -> None:
    """Raising min_confidence should reduce the trade count."""
    df = _ohlcv(200, drift=0.005, noise=0.02, seed=9)
    base = RecommendationBacktest(
        _engine_technical_only(),
        BacktestRunConfig(min_confidence=0.0, commission_pct=0.0),
    ).run(df, symbol="X")
    strict = RecommendationBacktest(
        _engine_technical_only(),
        BacktestRunConfig(min_confidence=0.9, commission_pct=0.0),
    ).run(df, symbol="X")
    assert strict.trades <= base.trades


def test_buy_and_hold_return_matches_simple_ratio() -> None:
    bt = RecommendationBacktest(_engine_technical_only())
    df = _ohlcv(180, drift=0.001, noise=0.0001, seed=1)
    result = bt.run(df, symbol="X")
    expected = (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1.0) * 100.0
    assert result.buy_and_hold_return_pct == pytest.approx(expected, abs=1e-6)


def test_alpha_equals_strategy_minus_buy_and_hold() -> None:
    bt = RecommendationBacktest(_engine_technical_only())
    df = _ohlcv(180, drift=0.002, noise=0.01, seed=4)
    result = bt.run(df, symbol="X")
    assert result.alpha_pct == pytest.approx(
        result.total_return_pct - result.buy_and_hold_return_pct, abs=1e-6
    )


def test_max_drawdown_non_negative() -> None:
    bt = RecommendationBacktest(_engine_technical_only())
    df = _ohlcv(200, drift=0.002, noise=0.015, seed=11)
    result = bt.run(df, symbol="X")
    assert result.max_drawdown_pct >= 0.0
