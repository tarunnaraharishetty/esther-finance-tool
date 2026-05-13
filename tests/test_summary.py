"""Tests for src.intelligence.summary.TemplateSummarizer."""

from __future__ import annotations

from src.intelligence.explain import explain
from src.intelligence.summary import TemplateSummarizer
from src.strategy.base import SignalAction


def _buy_explanation() -> object:
    return explain(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.65,
        combined_score=0.65,
        indicator_scores={"rsi": 0.5, "macd": 0.7, "bollinger": 0.4},
        sentiment_score=0.5,
        num_news_articles=8,
    )


def _mixed_explanation() -> object:
    return explain(
        symbol="MSFT",
        action=SignalAction.HOLD,
        confidence=0.1,
        combined_score=0.05,
        indicator_scores={"rsi": 0.5, "macd": -0.6, "bollinger": 0.05},
        sentiment_score=0.0,
        num_news_articles=0,
    )


def test_summary_is_deterministic() -> None:
    s = TemplateSummarizer()
    a = s.summarize(_buy_explanation())
    b = s.summarize(_buy_explanation())
    assert a == b


def test_summary_includes_symbol_and_action() -> None:
    s = TemplateSummarizer()
    text = s.summarize(_buy_explanation())
    assert "AAPL" in text
    assert "bullish" in text.lower()


def test_summary_lists_mixed_inputs() -> None:
    s = TemplateSummarizer()
    text = s.summarize(_mixed_explanation())
    # MSFT has bullish RSI + bearish MACD + neutral Bollinger → expect the
    # "bullish ... offset by bearish ..." phrasing.
    assert "offset" in text.lower() or ("bullish" in text.lower() and "bearish" in text.lower())


def test_summary_quotes_latest_headline_when_provided() -> None:
    s = TemplateSummarizer()
    text = s.summarize(
        _buy_explanation(),
        headlines=["AAPL announces record Q3 revenue"],
    )
    assert "record Q3 revenue" in text


def test_summary_action_caveat_for_hold() -> None:
    s = TemplateSummarizer()
    text = s.summarize(_mixed_explanation())
    # HOLD caveat shouldn't say "confirm against your own setup".
    assert "no clear entry or exit" in text.lower()


def test_summary_action_caveat_for_buy() -> None:
    s = TemplateSummarizer()
    text = s.summarize(_buy_explanation())
    assert "research signal" in text.lower()
