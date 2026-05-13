"""Tests for src.intelligence.opportunity_brief — per-OPP AI brief."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.dashboard.state import DashboardSnapshot, RecommendationRow
from src.intelligence.opportunity_brief import (
    LLMOpportunityBriefer,
    OpportunityBriefContext,
    _build_brief_user_message,
)
from src.intelligence.opportunity_history import OpportunityHistory
from src.strategy.base import RecommendationTier, SignalAction


def _opp_row(
    symbol: str,
    *,
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.7,
    technical: float = 0.5,
    sentiment: float = 0.5,
    news: int = 3,
    rsi: float = 0.5,
    macd: float = 0.5,
    bollinger: float = 0.5,
    last_price: float = 150.0,
    headlines: tuple[str, ...] = (),
    tier: RecommendationTier | None = None,
    error: str | None = None,
) -> RecommendationRow:
    return RecommendationRow(
        symbol=symbol,
        action=action,
        confidence=confidence,
        combined_score=confidence if action == SignalAction.BUY else -confidence,
        technical_score=technical,
        sentiment_score=sentiment,
        rsi=rsi,
        macd=macd,
        bollinger=bollinger,
        last_price=last_price,
        num_news_articles=news,
        reasoning="",
        timestamp=datetime.now(UTC),
        headlines=headlines,
        tier=tier if tier is not None else RecommendationTier.from_action(action),
        signal_quality="high",
        stability="stable",
        error=error,
    )


def _hold_row(symbol: str) -> RecommendationRow:
    return RecommendationRow(
        symbol=symbol,
        action=SignalAction.HOLD,
        confidence=0.2,
        combined_score=0.0,
        technical_score=0.0,
        sentiment_score=0.0,
        rsi=math.nan, macd=math.nan, bollinger=math.nan,
        last_price=100.0,
        num_news_articles=0,
        reasoning="",
        timestamp=datetime.now(UTC),
        tier=RecommendationTier.HOLD,
    )


def _snap(
    rows: list[RecommendationRow],
    *,
    opp_history: dict[str, OpportunityHistory] | None = None,
) -> DashboardSnapshot:
    return DashboardSnapshot(
        tick=42,
        rows=rows,
        events=[],
        opp_history=opp_history or {},
        timestamp=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# OpportunityBriefContext.from_snapshot
# ---------------------------------------------------------------------------


def test_from_snapshot_returns_context_for_top_ranked_symbol() -> None:
    snap = _snap([_opp_row("NVDA"), _opp_row("AAPL")])
    ctx = OpportunityBriefContext.from_snapshot(snap, "NVDA")
    assert ctx is not None
    assert ctx.symbol == "NVDA"
    assert ctx.composite_score > 0.0
    assert ctx.rank >= 1


def test_from_snapshot_returns_none_for_unranked_hold_symbol() -> None:
    """HOLD never ranks — from_snapshot must return None instead of
    fabricating a context."""
    snap = _snap([_opp_row("NVDA"), _hold_row("SPY")])
    assert OpportunityBriefContext.from_snapshot(snap, "SPY") is None


def test_from_snapshot_returns_none_for_unknown_symbol() -> None:
    snap = _snap([_opp_row("NVDA")])
    assert OpportunityBriefContext.from_snapshot(snap, "GOOG") is None


def test_from_snapshot_returns_none_for_error_row() -> None:
    snap = _snap([_opp_row("NVDA"), _opp_row("AAPL", error="fetch failed")])
    assert OpportunityBriefContext.from_snapshot(snap, "AAPL") is None


def test_from_snapshot_carries_opp_history_when_present() -> None:
    history = {"NVDA": OpportunityHistory(streak=5, appearances=8, window=10)}
    snap = _snap([_opp_row("NVDA")], opp_history=history)
    ctx = OpportunityBriefContext.from_snapshot(snap, "NVDA")
    assert ctx is not None
    assert ctx.streak == 5
    assert ctx.appearances == 8
    assert ctx.window == 10


def test_from_snapshot_history_defaults_to_zero_when_absent() -> None:
    """No history entry → zeros, not None — keeps the dataclass simple
    and the prompt builder doesn't have to special-case None."""
    snap = _snap([_opp_row("NVDA")])
    ctx = OpportunityBriefContext.from_snapshot(snap, "NVDA")
    assert ctx is not None
    assert ctx.streak == 0
    assert ctx.appearances == 0
    assert ctx.window == 0


def test_from_snapshot_truncates_headlines_to_max() -> None:
    headlines = tuple(f"headline {i}" for i in range(10))
    snap = _snap([_opp_row("NVDA", headlines=headlines)])
    ctx = OpportunityBriefContext.from_snapshot(snap, "NVDA", max_headlines=3)
    assert ctx is not None
    assert len(ctx.headlines) == 3
    assert ctx.headlines == ("headline 0", "headline 1", "headline 2")


def test_from_snapshot_rank_reflects_position_in_top_n() -> None:
    """rank is 1-indexed and mirrors position in the rank_opportunities output."""
    snap = _snap([
        _opp_row("AAPL", confidence=0.5, technical=0.3, sentiment=0.3),
        _opp_row("NVDA", confidence=0.85, technical=0.6, sentiment=0.6, news=8),
        _opp_row("MSFT", confidence=0.6, technical=0.4, sentiment=0.4),
    ])
    # NVDA's drivers are strongest; should be rank 1.
    nvda = OpportunityBriefContext.from_snapshot(snap, "NVDA")
    assert nvda is not None and nvda.rank == 1


# ---------------------------------------------------------------------------
# _build_brief_user_message
# ---------------------------------------------------------------------------


def _ctx(**overrides) -> OpportunityBriefContext:
    """Default fully-populated context for prompt-builder tests."""
    base = dict(
        symbol="NVDA",
        tier_display="STRONG BUY",
        composite_score=0.82,
        rank=1,
        technical_alignment=1.0,
        sentiment_alignment=0.7,
        confidence_acceleration=0.6,
        momentum_persistence=0.5,
        unusual_activity=0.3,
        reversal_strength=0.0,
        signal_quality_score=1.0,
        stability="stable",
        trend="strengthening",
        persistence="persistent",
        rationale=("indicators aligned", "news matches direction"),
        confidence=0.78,
        last_price=520.45,
        rsi=0.5, macd=0.6, bollinger=0.4,
        sentiment_score=0.55,
        num_news_articles=8,
        headlines=("NVDA beats earnings",),
        streak=5, appearances=7, window=10,
    )
    base.update(overrides)
    return OpportunityBriefContext(**base)


def test_user_message_lists_symbol_tier_and_composite() -> None:
    text = _build_brief_user_message(_ctx())
    assert "Symbol: NVDA" in text
    assert "Tier: STRONG BUY" in text
    assert "0.82" in text
    assert "rank #1" in text


def test_user_message_lists_all_seven_drivers_with_scores() -> None:
    text = _build_brief_user_message(_ctx())
    for driver in (
        "technical_alignment",
        "sentiment_alignment",
        "confidence_acceleration",
        "momentum_persistence",
        "unusual_activity",
        "reversal_strength",
        "signal_quality_score",
    ):
        assert driver in text


def test_user_message_includes_profile_axes() -> None:
    text = _build_brief_user_message(_ctx())
    assert "stable" in text
    assert "strengthening" in text
    assert "persistent" in text


def test_user_message_includes_history_when_window_positive() -> None:
    text = _build_brief_user_message(_ctx(streak=5, appearances=7, window=10))
    assert "5 consecutive ticks" in text
    assert "7 of last 10" in text


def test_user_message_omits_history_when_window_zero() -> None:
    """Skip the history line entirely when the tracker hasn't accumulated
    a window — otherwise the LLM sees a meaningless '0 of last 0' line."""
    text = _build_brief_user_message(_ctx(streak=0, appearances=0, window=0))
    assert "consecutive ticks" not in text


def test_user_message_quotes_headlines_verbatim() -> None:
    text = _build_brief_user_message(_ctx(
        headlines=("NVDA earnings beat by 12%",)
    ))
    # Literal quoted form so the LLM can echo it back without paraphrase.
    assert '"NVDA earnings beat by 12%"' in text


def test_user_message_renders_nan_indicator_as_na() -> None:
    """NaN means the indicator wasn't computed — must surface as 'n/a'
    so the LLM doesn't invent a value."""
    text = _build_brief_user_message(_ctx(rsi=math.nan))
    assert "RSI: n/a" in text


def test_user_message_renders_finite_indicator_with_sign() -> None:
    text = _build_brief_user_message(_ctx(macd=-0.42))
    assert "MACD: -0.42" in text


def test_user_message_renders_nan_price_as_na() -> None:
    text = _build_brief_user_message(_ctx(last_price=math.nan))
    assert "Last price: n/a" in text


def test_user_message_omits_rationale_when_empty() -> None:
    text = _build_brief_user_message(_ctx(rationale=()))
    assert "rationale phrases" not in text


# ---------------------------------------------------------------------------
# LLMOpportunityBriefer
# ---------------------------------------------------------------------------


def _mock_anthropic_client(text: str = "NVDA is the top opportunity.") -> MagicMock:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=400,
            output_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    client = MagicMock()
    client.messages.create = MagicMock(return_value=response)
    return client


def test_constructor_raises_without_api_key() -> None:
    """conftest sets ANTHROPIC_API_KEY=""; the briefer should raise like
    LLMSummarizer / LLMRecapGenerator do."""
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        LLMOpportunityBriefer()


def test_constructor_accepts_explicit_client() -> None:
    briefer = LLMOpportunityBriefer(client=_mock_anthropic_client())
    assert briefer.client is not None


def test_brief_calls_messages_create_with_expected_shape() -> None:
    client = _mock_anthropic_client()
    briefer = LLMOpportunityBriefer(client=client)
    briefer.brief(_ctx())

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    # System prompt must announce the OPP-brief role.
    assert "OPP brief" in kwargs["system"] or "opportunity" in kwargs["system"].lower()


def test_brief_returns_text() -> None:
    client = _mock_anthropic_client("NVDA leads at composite 0.82.")
    briefer = LLMOpportunityBriefer(client=client)
    assert briefer.brief(_ctx()) == "NVDA leads at composite 0.82."


# ---------------------------------------------------------------------------
# Hallucination guard — same shape as test_llm_summary / test_recap
# ---------------------------------------------------------------------------


def test_system_prompt_contains_grounding_rules() -> None:
    """The OPP brief prompt's anti-invention clauses must survive
    refactors. Same regression-guard pattern as the other LLM modules."""
    client = _mock_anthropic_client()
    briefer = LLMOpportunityBriefer(client=client)
    briefer.brief(_ctx())
    system = client.messages.create.call_args.kwargs["system"]
    assert "Only summarize information present in the user message" in system
    assert "Do not invent" in system
    assert "Quote headlines verbatim" in system
    assert "do not predict" in system.lower()
    # OPP-specific: the engine's outputs are authoritative.
    assert "engine's outputs" in system or "deterministic classifications" in system
