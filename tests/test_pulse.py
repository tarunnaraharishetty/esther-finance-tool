"""Tests for src.intelligence.pulse."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from src.dashboard.state import DashboardSnapshot, RecommendationRow
from src.intelligence.alerts import Alert
from src.intelligence.history import SignalEpisode, SignalHistorySummary
from src.intelligence.pulse import compute_pulse
from src.strategy.base import SignalAction


def _row(
    symbol: str = "AAPL",
    *,
    action: SignalAction = SignalAction.HOLD,
    confidence: float = 0.3,
    sentiment: float = 0.0,
    news: int = 0,
    error: str | None = None,
) -> RecommendationRow:
    return RecommendationRow(
        symbol=symbol,
        action=action,
        confidence=confidence,
        combined_score=0.0,
        technical_score=0.0,
        sentiment_score=sentiment,
        rsi=math.nan,
        macd=math.nan,
        bollinger=math.nan,
        last_price=100.0,
        num_news_articles=news,
        reasoning="",
        timestamp=datetime.now(UTC),
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
    alerts: tuple[Alert, ...] = (),
) -> DashboardSnapshot:
    return DashboardSnapshot(
        tick=1,
        rows=rows,
        events=[],
        alerts=list(alerts),
        recent_alerts=alerts,
        signal_history=history or {},
        timestamp=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------


def test_pulse_empty_snapshot_is_marked_empty() -> None:
    pulse = compute_pulse(_snap([]))
    assert pulse.is_empty


def test_pulse_all_errored_rows_is_empty() -> None:
    pulse = compute_pulse(_snap([_row(error="boom"), _row("MSFT", error="boom")]))
    assert pulse.is_empty


def test_pulse_excludes_error_rows_from_calculations() -> None:
    pulse = compute_pulse(
        _snap(
            [
                _row("AAPL", action=SignalAction.BUY, confidence=0.7),
                _row("MSFT", error="boom"),
            ]
        )
    )
    # Should derive from AAPL alone, not be empty.
    assert not pulse.is_empty
    assert pulse.sentiment == "bullish"


# ---------------------------------------------------------------------------
# Sentiment classification
# ---------------------------------------------------------------------------


def test_pulse_bullish_when_action_mix_skews_buy_and_news_neutral() -> None:
    pulse = compute_pulse(
        _snap(
            [
                _row(f"S{i}", action=SignalAction.BUY, confidence=0.5)
                for i in range(3)
            ]
            + [_row("X", action=SignalAction.HOLD, confidence=0.2)]
        )
    )
    assert pulse.sentiment == "bullish"


def test_pulse_bearish_when_action_mix_skews_sell() -> None:
    pulse = compute_pulse(
        _snap(
            [_row(f"S{i}", action=SignalAction.SELL, confidence=0.5) for i in range(3)]
            + [_row("X", action=SignalAction.HOLD, confidence=0.2)]
        )
    )
    assert pulse.sentiment == "bearish"


def test_pulse_mixed_when_action_bullish_but_news_bearish() -> None:
    """A subtle but trader-relevant case: technicals say buy, news says
    sell. The pulse should flag that disagreement, not pick one side."""
    pulse = compute_pulse(
        _snap(
            [
                _row("AAPL", action=SignalAction.BUY, confidence=0.6,
                     sentiment=-0.5, news=5),
                _row("MSFT", action=SignalAction.BUY, confidence=0.6,
                     sentiment=-0.4, news=5),
            ]
        )
    )
    assert pulse.sentiment == "mixed"


def test_pulse_neutral_when_action_balanced_and_news_quiet() -> None:
    pulse = compute_pulse(
        _snap(
            [
                _row("AAPL", action=SignalAction.BUY, confidence=0.3),
                _row("MSFT", action=SignalAction.SELL, confidence=0.3),
            ]
        )
    )
    assert pulse.sentiment == "neutral"


# ---------------------------------------------------------------------------
# Conviction classification
# ---------------------------------------------------------------------------


def test_pulse_strong_conviction_when_avg_confidence_high() -> None:
    pulse = compute_pulse(
        _snap([_row(f"S{i}", confidence=0.7) for i in range(3)])
    )
    assert pulse.conviction == "strong"


def test_pulse_moderate_conviction_at_middle_band() -> None:
    pulse = compute_pulse(
        _snap([_row(f"S{i}", confidence=0.4) for i in range(3)])
    )
    assert pulse.conviction == "moderate"


def test_pulse_weak_conviction_when_avg_confidence_low() -> None:
    pulse = compute_pulse(
        _snap([_row(f"S{i}", confidence=0.1) for i in range(3)])
    )
    assert pulse.conviction == "weak"


# ---------------------------------------------------------------------------
# Activity classification
# ---------------------------------------------------------------------------


def test_pulse_calm_with_no_history_or_alerts() -> None:
    pulse = compute_pulse(_snap([_row("AAPL"), _row("MSFT")]))
    assert pulse.activity == "calm"


def test_pulse_active_when_some_alerts_fire() -> None:
    fired = datetime.now(UTC)
    alerts = (
        Alert(symbol="AAPL", rule="action_changed", severity="warn",
              message="x", fired_at=fired),
    )
    pulse = compute_pulse(_snap([_row("AAPL"), _row("MSFT")], alerts=alerts))
    assert pulse.activity == "active"


def test_pulse_volatile_when_lots_of_flips() -> None:
    """Many episodes per symbol = high churn = volatile."""
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(SignalAction.BUY),
            recent=(
                _episode(SignalAction.HOLD),
                _episode(SignalAction.SELL),
                _episode(SignalAction.BUY),
            ),
        ),
    }
    pulse = compute_pulse(_snap([_row("AAPL")], history=history))
    assert pulse.activity == "volatile"


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------


def test_pulse_summary_mentions_symbol_count() -> None:
    pulse = compute_pulse(
        _snap(
            [_row("AAPL"), _row("MSFT"), _row("NVDA")],
        )
    )
    assert "3 symbols" in pulse.summary


def test_pulse_summary_is_singular_for_one_symbol() -> None:
    pulse = compute_pulse(_snap([_row("AAPL")]))
    assert "1 symbol " in pulse.summary  # space-suffixed avoids "1 symbols"


def test_pulse_summary_is_observational_not_predictive() -> None:
    """Anti-hallucination: the pulse summary describes state, doesn't
    forecast. This guards against future edits that introduce
    'likely to' or 'expected to' phrasing."""
    pulse = compute_pulse(_snap([_row("AAPL", action=SignalAction.BUY, confidence=0.8)]))
    forbidden = ("likely", "will rise", "will fall", "expected to", "forecast")
    lower = pulse.summary.lower()
    for word in forbidden:
        assert word not in lower, f"summary should not contain '{word}': {pulse.summary}"
