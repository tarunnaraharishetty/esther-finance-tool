"""Tests for src.intelligence.explain."""

from __future__ import annotations

from src.intelligence.explain import explain
from src.strategy.base import SignalAction


def _explain_buy(**overrides: object) -> object:
    kwargs: dict[str, object] = dict(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.65,
        combined_score=0.65,
        indicator_scores={"rsi": 0.5, "macd": 0.7, "bollinger": 0.4},
        sentiment_score=0.5,
        num_news_articles=8,
    )
    kwargs.update(overrides)
    return explain(**kwargs)  # type: ignore[arg-type]


def test_explain_returns_one_contributor_per_indicator_plus_sentiment() -> None:
    e = _explain_buy()
    names = [c.name for c in e.contributors]
    assert names == ["RSI", "MACD", "Bollinger", "News sentiment"]


def test_explain_skips_sentiment_when_no_news() -> None:
    e = _explain_buy(sentiment_score=0.0, num_news_articles=0)
    assert all(c.name != "News sentiment" for c in e.contributors)


def test_explain_confidence_labels() -> None:
    assert _explain_buy(confidence=0.8).confidence_label == "high"
    assert _explain_buy(confidence=0.4).confidence_label == "moderate"
    assert _explain_buy(confidence=0.1).confidence_label == "weak"


def test_explain_headline_buy_includes_confidence_label() -> None:
    e = _explain_buy(confidence=0.7)
    assert "bullish" in e.headline.lower()
    assert "high" in e.headline.lower()
    assert "AAPL" in e.headline


def test_explain_headline_hold_omits_confidence_phrase() -> None:
    e = _explain_buy(action=SignalAction.HOLD, confidence=0.1)
    assert "no clear" in e.headline.lower()
    # HOLD headline shouldn't claim bullish/bearish.
    assert "bullish" not in e.headline.lower()
    assert "bearish" not in e.headline.lower()


def test_explain_contributor_directions_match_score_sign() -> None:
    e = _explain_buy(
        indicator_scores={"rsi": 0.5, "macd": -0.6, "bollinger": 0.05},
        sentiment_score=-0.4,
    )
    by_name = {c.name: c for c in e.contributors}
    assert by_name["RSI"].direction == "bullish"
    assert by_name["MACD"].direction == "bearish"
    assert by_name["Bollinger"].direction == "neutral"
    assert by_name["News sentiment"].direction == "bearish"


def test_explain_rsi_note_mentions_oversold_when_positive() -> None:
    e = _explain_buy(indicator_scores={"rsi": 0.7, "macd": 0.0, "bollinger": 0.0})
    rsi = next(c for c in e.contributors if c.name == "RSI")
    assert "oversold" in rsi.note.lower()


def test_explain_rsi_note_mentions_overbought_when_negative() -> None:
    e = _explain_buy(indicator_scores={"rsi": -0.7, "macd": 0.0, "bollinger": 0.0})
    rsi = next(c for c in e.contributors if c.name == "RSI")
    assert "overbought" in rsi.note.lower()


def test_explain_sentiment_note_mentions_article_count() -> None:
    e = _explain_buy(num_news_articles=12, sentiment_score=0.5)
    sent = next(c for c in e.contributors if c.name == "News sentiment")
    assert "12" in sent.note


def test_explain_skips_indicators_not_in_scores() -> None:
    e = _explain_buy(indicator_scores={"rsi": 0.5})
    names = [c.name for c in e.contributors]
    assert "RSI" in names
    assert "MACD" not in names
    assert "Bollinger" not in names
