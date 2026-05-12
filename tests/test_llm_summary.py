"""Tests for src.intelligence.llm_summary.LLMSummarizer.

Mocks the Anthropic client at the SDK boundary — no network in unit tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.data.models import NewsArticle
from src.intelligence.explain import explain
from src.intelligence.llm_summary import LLMSummarizer
from src.strategy.base import SignalAction


def _mock_client(text: str = "AAPL leans bullish with high confidence.") -> MagicMock:
    """Build an Anthropic client whose messages.create returns a fixed text response."""
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=45,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    client = MagicMock()
    client.messages.create = MagicMock(return_value=response)
    return client


def _explanation(action: SignalAction = SignalAction.BUY) -> object:
    return explain(
        symbol="AAPL",
        action=action,
        confidence=0.65,
        combined_score=0.65,
        indicator_scores={"rsi": 0.5, "macd": 0.7, "bollinger": 0.4},
        sentiment_score=0.5,
        num_news_articles=8,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            LLMSummarizer()
    finally:
        settings_mod.get_settings.cache_clear()


def test_constructor_accepts_explicit_client() -> None:
    # Even with no API key set, an explicit client should bypass the env check.
    client = _mock_client()
    summarizer = LLMSummarizer(client=client)
    assert summarizer.client is client


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


def test_summarize_calls_messages_create_with_expected_shape() -> None:
    client = _mock_client()
    summarizer = LLMSummarizer(client=client)
    summarizer.summarize(_explanation())

    client.messages.create.assert_called_once()
    kwargs = client.messages.create.call_args.kwargs
    # Decision-support defaults
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["max_tokens"] == 1024
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    # System prompt frames the role correctly
    assert "research assistant" in kwargs["system"].lower()
    # The user-turn message is a single string
    msg = kwargs["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], str)


def test_summarize_user_message_includes_symbol_action_and_contributors() -> None:
    client = _mock_client()
    summarizer = LLMSummarizer(client=client)
    summarizer.summarize(_explanation())

    user_text = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "AAPL" in user_text
    assert "BUY" in user_text
    assert "RSI" in user_text
    assert "MACD" in user_text
    assert "Bollinger" in user_text
    assert "News sentiment" in user_text


def test_summarize_user_message_lists_headlines_when_provided() -> None:
    client = _mock_client()
    summarizer = LLMSummarizer(client=client)
    headlines = [
        NewsArticle(
            id="h1",
            headline="AAPL beats earnings",
            source="x",
            symbols=["AAPL"],
            published_at=datetime.now(UTC),
        ),
        NewsArticle(
            id="h2",
            headline="Apple supplier deal expands",
            source="x",
            symbols=["AAPL"],
            published_at=datetime.now(UTC),
        ),
    ]
    summarizer.summarize(_explanation(), headlines=headlines)
    user_text = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "AAPL beats earnings" in user_text
    assert "Apple supplier deal expands" in user_text


def test_summarize_hold_includes_no_directional_edge_note() -> None:
    client = _mock_client()
    summarizer = LLMSummarizer(client=client)
    summarizer.summarize(_explanation(action=SignalAction.HOLD))
    user_text = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "HOLD" in user_text
    # The user-message tail nudges the model away from directional framing.
    assert "no directional edge" in user_text.lower()


def test_summarize_returns_concatenated_text_blocks() -> None:
    # If the API ever splits the response across multiple text blocks, we
    # should concatenate them rather than dropping the tail.
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="AAPL leans bullish. "),
            SimpleNamespace(type="text", text="Technicals confirm."),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    client = MagicMock()
    client.messages.create = MagicMock(return_value=response)
    summarizer = LLMSummarizer(client=client)
    out = summarizer.summarize(_explanation())
    assert out == "AAPL leans bullish. Technicals confirm."


def test_summarize_strips_whitespace() -> None:
    client = _mock_client(text="   AAPL leans bullish.\n\n   ")
    summarizer = LLMSummarizer(client=client)
    out = summarizer.summarize(_explanation())
    assert out == "AAPL leans bullish."


def test_summarize_propagates_sdk_exceptions() -> None:
    import anthropic

    client = MagicMock()
    # Constructing a real APIError requires a Response — easier to just patch
    # raise the bare class for the test, since we only care that the exception
    # propagates rather than being swallowed.
    client.messages.create = MagicMock(
        side_effect=anthropic.RateLimitError(
            "rate limited", response=MagicMock(status_code=429), body=None
        )
    )
    summarizer = LLMSummarizer(client=client)
    with pytest.raises(anthropic.RateLimitError):
        summarizer.summarize(_explanation())
