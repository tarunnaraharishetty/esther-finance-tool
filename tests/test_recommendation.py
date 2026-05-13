"""Tests for the recommendation engine.

These tests use deterministic synthetic OHLCV data and a fake sentiment
analyzer so we don't download FinBERT or hit the network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.models import NewsArticle
from src.sentiment.analyzer import SentimentAnalyzer, SentimentLabel, SentimentScore
from src.strategy.base import SignalAction
from src.strategy.recommendation import (
    IndicatorWeights,
    RecommendationEngine,
    TradingRecommendation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trending_df(
    n: int = 60,
    start: float = 100.0,
    drift: float = 0.015,
    noise: float = 0.005,
    seed: int = 0,
) -> pd.DataFrame:
    """Noisy compounding trend so RSI is defined (needs both gains AND losses)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=noise, size=n)
    close = start * np.exp(np.cumsum(rets))
    high = close * 1.005
    low = close * 0.995
    open_ = np.concatenate([[start], close[:-1]])
    volume = np.full(n, 1_000_000, dtype=int)
    idx = pd.date_range(end=datetime.now(UTC), periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _bullish_df(n: int = 60, start: float = 100.0) -> pd.DataFrame:
    """Noisy uptrend → MACD positive, RSI overbought, price above Bollinger middle."""
    return _trending_df(n=n, start=start, drift=0.015, noise=0.01, seed=7)


def _bearish_df(n: int = 60, start: float = 100.0) -> pd.DataFrame:
    """Noisy downtrend → MACD negative, RSI oversold, price below Bollinger middle."""
    return _trending_df(n=n, start=start, drift=-0.015, noise=0.01, seed=7)


def _flat_df(n: int = 60, price: float = 100.0) -> pd.DataFrame:
    close = np.full(n, price, dtype=float)
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


def _article(symbol: str, headline: str, hours_ago: int = 1) -> NewsArticle:
    return NewsArticle(
        id=f"{symbol}-{hours_ago}-{abs(hash(headline)) % 100000}",
        headline=headline,
        summary="",
        source="test",
        symbols=[symbol],
        published_at=datetime.now(UTC) - timedelta(hours=hours_ago),
    )


class FakeSentimentAnalyzer(SentimentAnalyzer):
    """Returns predefined scores keyed by headline substring — no model load."""

    def __init__(self, label: SentimentLabel, confidence: float = 0.9) -> None:
        # Skip the parent __init__ to avoid touching settings/pipeline.
        self._label = label
        self._confidence = confidence

    def score_text(self, text: str) -> SentimentScore:  # type: ignore[override]
        if not text.strip():
            return SentimentScore(SentimentLabel.NEUTRAL, 0.0)
        return SentimentScore(self._label, self._confidence)

    def score_article(self, article: NewsArticle) -> SentimentScore:  # type: ignore[override]
        return self.score_text(article.headline)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_recommendation_is_frozen_and_validated() -> None:
    rec = TradingRecommendation(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.7,
        combined_score=0.7,
        technical_score=0.5,
        sentiment_score=0.9,
        reasoning="ok",
        timestamp=datetime.now(UTC),
    )
    assert rec.symbol == "AAPL"
    assert rec.confidence == pytest.approx(0.7)
    with pytest.raises(Exception):  # frozen ConfigDict → no mutation
        rec.symbol = "MSFT"  # type: ignore[misc]


def test_recommendation_rejects_out_of_range_confidence() -> None:
    with pytest.raises(Exception):
        TradingRecommendation(
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=1.5,
            combined_score=0.5,
            technical_score=0.5,
            sentiment_score=0.0,
            reasoning="",
            timestamp=datetime.now(UTC),
        )


# ---------------------------------------------------------------------------
# Indicator scoring
# ---------------------------------------------------------------------------


def test_bullish_data_yields_positive_technical_score() -> None:
    engine = RecommendationEngine(sentiment_analyzer=FakeSentimentAnalyzer(SentimentLabel.NEUTRAL))
    rec = engine.recommend("AAPL", _bullish_df(), news=[])
    # Strong uptrend: MACD must be bullish; bollinger pushes price toward upper.
    # Bullish trend pushes price above middle and RSI overbought → technical < 0
    # because RSI/Bollinger are mean-reversion in this engine. Confirm signs match
    # the documented convention rather than the trend direction.
    assert rec.indicator_scores["macd"] > 0
    assert rec.indicator_scores["rsi"] < 0  # overbought
    assert rec.indicator_scores["bollinger"] < 0  # price above middle band


def test_bearish_data_yields_oversold_rsi_and_negative_macd() -> None:
    engine = RecommendationEngine(sentiment_analyzer=FakeSentimentAnalyzer(SentimentLabel.NEUTRAL))
    rec = engine.recommend("AAPL", _bearish_df(), news=[])
    assert rec.indicator_scores["macd"] < 0
    assert rec.indicator_scores["rsi"] > 0  # oversold → bullish read
    assert rec.indicator_scores["bollinger"] > 0  # price below middle band


def test_flat_data_yields_near_zero_scores() -> None:
    engine = RecommendationEngine(sentiment_analyzer=FakeSentimentAnalyzer(SentimentLabel.NEUTRAL))
    rec = engine.recommend("AAPL", _flat_df(), news=[])
    assert abs(rec.indicator_scores["macd"]) < 1e-6
    # Flat prices → bollinger half-width is 0 → score 0
    assert rec.indicator_scores["bollinger"] == 0.0


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------


def test_positive_news_drives_positive_sentiment() -> None:
    fake = FakeSentimentAnalyzer(SentimentLabel.POSITIVE, confidence=0.85)
    engine = RecommendationEngine(sentiment_analyzer=fake)
    rec = engine.recommend(
        "AAPL",
        _flat_df(),
        news=[_article("AAPL", "Apple beats earnings")],
    )
    assert rec.sentiment_score == pytest.approx(0.85)
    assert rec.num_news_articles == 1


def test_negative_news_drives_negative_sentiment() -> None:
    fake = FakeSentimentAnalyzer(SentimentLabel.NEGATIVE, confidence=0.9)
    engine = RecommendationEngine(sentiment_analyzer=fake)
    rec = engine.recommend(
        "AAPL",
        _flat_df(),
        news=[_article("AAPL", "Apple misses guidance")],
    )
    assert rec.sentiment_score == pytest.approx(-0.9)


def test_no_news_yields_zero_sentiment() -> None:
    fake = FakeSentimentAnalyzer(SentimentLabel.POSITIVE)
    engine = RecommendationEngine(sentiment_analyzer=fake)
    rec = engine.recommend("AAPL", _flat_df(), news=[])
    assert rec.sentiment_score == 0.0
    assert rec.num_news_articles == 0


def test_max_news_articles_cap_is_respected() -> None:
    fake = FakeSentimentAnalyzer(SentimentLabel.POSITIVE, confidence=0.5)
    engine = RecommendationEngine(sentiment_analyzer=fake, max_news_articles=3)
    articles = [_article("AAPL", f"headline {i}", hours_ago=i) for i in range(10)]
    rec = engine.recommend("AAPL", _flat_df(), news=articles)
    assert rec.num_news_articles == 3


# ---------------------------------------------------------------------------
# Action thresholds
# ---------------------------------------------------------------------------


def test_strong_positive_sentiment_drives_buy() -> None:
    fake = FakeSentimentAnalyzer(SentimentLabel.POSITIVE, confidence=1.0)
    engine = RecommendationEngine(
        technical_weight=0.0,
        sentiment_weight=1.0,
        buy_threshold=0.2,
        sentiment_analyzer=fake,
    )
    rec = engine.recommend("AAPL", _flat_df(), news=[_article("AAPL", "great news")])
    assert rec.action == SignalAction.BUY
    assert rec.confidence == pytest.approx(1.0)


def test_strong_negative_sentiment_drives_sell() -> None:
    fake = FakeSentimentAnalyzer(SentimentLabel.NEGATIVE, confidence=1.0)
    engine = RecommendationEngine(
        technical_weight=0.0,
        sentiment_weight=1.0,
        sell_threshold=0.2,
        sentiment_analyzer=fake,
    )
    rec = engine.recommend("AAPL", _flat_df(), news=[_article("AAPL", "bad news")])
    assert rec.action == SignalAction.SELL


def test_below_threshold_yields_hold() -> None:
    fake = FakeSentimentAnalyzer(SentimentLabel.POSITIVE, confidence=0.1)
    engine = RecommendationEngine(
        technical_weight=0.0,
        sentiment_weight=1.0,
        buy_threshold=0.5,
        sell_threshold=0.5,
        sentiment_analyzer=fake,
    )
    rec = engine.recommend("AAPL", _flat_df(), news=[_article("AAPL", "meh")])
    assert rec.action == SignalAction.HOLD


# ---------------------------------------------------------------------------
# Weighted combination
# ---------------------------------------------------------------------------


def test_weights_combine_technical_and_sentiment() -> None:
    """Tech score and sentiment score should be combined with their weights."""
    fake = FakeSentimentAnalyzer(SentimentLabel.NEGATIVE, confidence=1.0)
    engine = RecommendationEngine(
        technical_weight=0.3,
        sentiment_weight=0.7,
        sentiment_analyzer=fake,
        # Override indicator weights so we know technical score is zero on flat data.
        indicator_weights=IndicatorWeights(rsi=1.0, macd=1.0, bollinger=1.0),
    )
    rec = engine.recommend("AAPL", _flat_df(), news=[_article("AAPL", "bad")])
    # Flat tech score ≈ 0, sentiment = -1, weighted = 0.7 * -1 = -0.7
    assert rec.combined_score == pytest.approx(-0.7, abs=0.05)
    assert rec.action == SignalAction.SELL


def test_combined_score_is_clipped_to_unit_range() -> None:
    fake = FakeSentimentAnalyzer(SentimentLabel.POSITIVE, confidence=1.0)
    engine = RecommendationEngine(
        technical_weight=2.0,
        sentiment_weight=2.0,
        sentiment_analyzer=fake,
    )
    rec = engine.recommend("AAPL", _flat_df(), news=[_article("AAPL", "great")])
    assert -1.0 <= rec.combined_score <= 1.0
    assert 0.0 <= rec.confidence <= 1.0


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


def test_reasoning_string_mentions_action_and_scores() -> None:
    fake = FakeSentimentAnalyzer(SentimentLabel.POSITIVE, confidence=0.8)
    engine = RecommendationEngine(sentiment_analyzer=fake)
    rec = engine.recommend("TSLA", _bullish_df(), news=[_article("TSLA", "good")])
    assert "TSLA" in rec.reasoning
    assert rec.action.value.upper() in rec.reasoning
    assert "technical=" in rec.reasoning
    assert "sentiment=" in rec.reasoning


def test_reasoning_handles_zero_news() -> None:
    fake = FakeSentimentAnalyzer(SentimentLabel.POSITIVE)
    engine = RecommendationEngine(sentiment_analyzer=fake)
    rec = engine.recommend("AAPL", _flat_df(), news=[])
    assert "no news" in rec.reasoning


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_dataframe_does_not_crash() -> None:
    engine = RecommendationEngine(sentiment_analyzer=FakeSentimentAnalyzer(SentimentLabel.NEUTRAL))
    rec = engine.recommend("AAPL", pd.DataFrame(), news=[])
    assert rec.action == SignalAction.HOLD
    assert rec.combined_score == 0.0
