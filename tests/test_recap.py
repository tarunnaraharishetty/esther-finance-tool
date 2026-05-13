"""Tests for src.intelligence.recap — watchlist-wide AI recap."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.dashboard.state import DashboardSnapshot, RecommendationRow
from src.intelligence.alerts import Alert
from src.intelligence.history import SignalEpisode, SignalHistorySummary
from src.intelligence.recap import (
    LLMRecapGenerator,
    RecapContext,
    _build_recap_user_message,
)
from src.strategy.base import SignalAction


def _row(
    symbol: str,
    *,
    action: SignalAction = SignalAction.HOLD,
    confidence: float = 0.3,
    combined: float = 0.0,
    sentiment: float = 0.0,
    news: int = 0,
    headlines: tuple[str, ...] = (),
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
        headlines=headlines,
        error=error,
    )


def _episode(action: SignalAction, ticks: int = 1) -> SignalEpisode:
    return SignalEpisode(
        action=action,
        started_at=datetime(2026, 5, 13, tzinfo=UTC),
        last_seen_at=datetime(2026, 5, 13, tzinfo=UTC) + timedelta(seconds=ticks * 5),
        tick_count=ticks,
        confidence_first=0.5,
        confidence_last=0.5,
    )


def _snap(
    rows: list[RecommendationRow],
    *,
    history: dict[str, SignalHistorySummary] | None = None,
    alerts: list[Alert] | None = None,
) -> DashboardSnapshot:
    return DashboardSnapshot(
        tick=42,
        rows=rows,
        events=[],
        alerts=alerts or [],
        signal_history=history or {},
        timestamp=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# RecapContext.from_snapshot
# ---------------------------------------------------------------------------


def test_context_captures_action_mix() -> None:
    snap = _snap([
        _row("AAPL", action=SignalAction.BUY),
        _row("MSFT", action=SignalAction.BUY),
        _row("NVDA", action=SignalAction.SELL),
    ])
    ctx = RecapContext.from_snapshot(snap)
    assert ctx.action_mix[SignalAction.BUY] == 2
    assert ctx.action_mix[SignalAction.SELL] == 1
    assert ctx.action_mix[SignalAction.HOLD] == 0


def test_context_captures_recent_flips_longest_first() -> None:
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(SignalAction.BUY, ticks=1),
            recent=(_episode(SignalAction.HOLD, ticks=10),),
        ),
        "NVDA": SignalHistorySummary(
            current=_episode(SignalAction.SELL, ticks=1),
            recent=(_episode(SignalAction.BUY, ticks=2),),
        ),
    }
    snap = _snap(
        [
            _row("AAPL", action=SignalAction.BUY),
            _row("NVDA", action=SignalAction.SELL),
        ],
        history=history,
    )
    ctx = RecapContext.from_snapshot(snap)
    assert len(ctx.recent_flips) == 2
    # AAPL's prior run was 10 ticks vs NVDA's 2; AAPL ranks first.
    sym, prior, current, ticks = ctx.recent_flips[0]
    assert sym == "AAPL"
    assert prior == SignalAction.HOLD
    assert current == SignalAction.BUY
    assert ticks == 10


def test_context_skips_non_flip_history() -> None:
    """If current action equals the most recent episode action (no flip),
    don't list it."""
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(SignalAction.BUY, ticks=5),
            recent=(_episode(SignalAction.BUY, ticks=3),),
        ),
    }
    snap = _snap([_row("AAPL", action=SignalAction.BUY)], history=history)
    assert RecapContext.from_snapshot(snap).recent_flips == ()


def test_context_captures_headlines_per_symbol() -> None:
    snap = _snap([
        _row("AAPL", news=2, headlines=("AAPL beats earnings", "Apple supplier deal")),
        _row("NVDA", news=0),  # no headlines
    ])
    ctx = RecapContext.from_snapshot(snap)
    assert ctx.headlines_by_symbol == {
        "AAPL": ("AAPL beats earnings", "Apple supplier deal"),
    }


def test_context_counts_alerts_by_severity() -> None:
    fired = datetime(2026, 5, 13, 14, 0, tzinfo=UTC)
    alerts = [
        Alert(symbol="AAPL", rule="action_changed", severity="warn",
              message="x", fired_at=fired),
        Alert(symbol="MSFT", rule="confidence_threshold", severity="info",
              message="x", fired_at=fired),
        Alert(symbol="NVDA", rule="confidence_threshold", severity="critical",
              message="x", fired_at=fired),
    ]
    snap = _snap([_row("AAPL")], alerts=alerts)
    ctx = RecapContext.from_snapshot(snap)
    assert ctx.alert_counts == {"warn": 1, "info": 1, "critical": 1}


def test_context_watchlist_preserves_order() -> None:
    snap = _snap([_row("NVDA"), _row("AAPL"), _row("MSFT")])
    ctx = RecapContext.from_snapshot(snap)
    assert ctx.watchlist == ("NVDA", "AAPL", "MSFT")


# ---------------------------------------------------------------------------
# _build_recap_user_message
# ---------------------------------------------------------------------------


def test_user_message_lists_action_mix_and_watchlist() -> None:
    snap = _snap([
        _row("AAPL", action=SignalAction.BUY),
        _row("NVDA", action=SignalAction.SELL),
    ])
    text = _build_recap_user_message(RecapContext.from_snapshot(snap))
    assert "Watchlist: AAPL, NVDA" in text
    assert "1 BUY" in text
    assert "1 SELL" in text


def test_user_message_only_includes_populated_sections() -> None:
    """Empty ranked sections must not produce empty labels in the prompt.
    The LLM would otherwise see 'Top momentum:' followed by nothing and
    could be tempted to fill it in."""
    snap = _snap([_row("AAPL", action=SignalAction.HOLD)])  # no data
    text = _build_recap_user_message(RecapContext.from_snapshot(snap))
    # No section label without data.
    assert "Top sentiment:" not in text
    assert "Biggest reversals:" not in text


def test_user_message_quotes_headlines_verbatim() -> None:
    snap = _snap([
        _row(
            "AAPL",
            news=2,
            headlines=("AAPL beats earnings expectations",),
        ),
    ])
    text = _build_recap_user_message(RecapContext.from_snapshot(snap))
    # Literal quoted form so the LLM can quote it back without paraphrasing.
    assert '"AAPL beats earnings expectations"' in text


def test_user_message_lists_flips_with_durations() -> None:
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(SignalAction.BUY, ticks=1),
            recent=(_episode(SignalAction.HOLD, ticks=7),),
        ),
    }
    snap = _snap([_row("AAPL", action=SignalAction.BUY)], history=history)
    text = _build_recap_user_message(RecapContext.from_snapshot(snap))
    assert "was HOLD for 7 ticks" in text
    assert "now BUY" in text


# ---------------------------------------------------------------------------
# LLMRecapGenerator
# ---------------------------------------------------------------------------


def _mock_anthropic_client(text: str = "The watchlist is leaning bullish.") -> MagicMock:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=300,
            output_tokens=80,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    client = MagicMock()
    client.messages.create = MagicMock(return_value=response)
    return client


def test_constructor_raises_without_api_key() -> None:
    """conftest sets ANTHROPIC_API_KEY="" — recap should raise the same
    way LLMSummarizer does."""
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        LLMRecapGenerator()


def test_constructor_accepts_explicit_client() -> None:
    gen = LLMRecapGenerator(client=_mock_anthropic_client())
    assert gen.client is not None


def test_generate_calls_messages_create_with_expected_shape() -> None:
    client = _mock_anthropic_client()
    gen = LLMRecapGenerator(client=client)
    snap = _snap([_row("AAPL", action=SignalAction.BUY)])
    context = RecapContext.from_snapshot(snap)
    gen.generate(context)

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert "recap" in kwargs["system"].lower()


def test_generate_returns_text() -> None:
    client = _mock_anthropic_client("The watchlist mix is balanced.")
    gen = LLMRecapGenerator(client=client)
    snap = _snap([_row("AAPL")])
    text = gen.generate(RecapContext.from_snapshot(snap))
    assert text == "The watchlist mix is balanced."


# ---------------------------------------------------------------------------
# Hallucination guard — same regression-test pattern as test_llm_summary.py
# ---------------------------------------------------------------------------


def test_system_prompt_contains_grounding_rules() -> None:
    """The recap prompt's anti-invention clauses must survive refactors."""
    client = _mock_anthropic_client()
    gen = LLMRecapGenerator(client=client)
    snap = _snap([_row("AAPL")])
    gen.generate(RecapContext.from_snapshot(snap))
    system = client.messages.create.call_args.kwargs["system"]
    assert "Only summarize information present in the user message" in system
    assert "Do not invent" in system
    assert "Quote headlines verbatim" in system
    assert "do not predict" in system.lower()
    # Recap-specific: the LLM must NOT bring up symbols outside the watchlist.
    assert "ONLY the symbols in the data" in system
