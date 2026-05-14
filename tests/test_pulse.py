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
            [_row(f"S{i}", action=SignalAction.BUY, confidence=0.5) for i in range(3)]
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
                _row("AAPL", action=SignalAction.BUY, confidence=0.6, sentiment=-0.5, news=5),
                _row("MSFT", action=SignalAction.BUY, confidence=0.6, sentiment=-0.4, news=5),
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
    pulse = compute_pulse(_snap([_row(f"S{i}", confidence=0.7) for i in range(3)]))
    assert pulse.conviction == "strong"


def test_pulse_moderate_conviction_at_middle_band() -> None:
    pulse = compute_pulse(_snap([_row(f"S{i}", confidence=0.4) for i in range(3)]))
    assert pulse.conviction == "moderate"


def test_pulse_weak_conviction_when_avg_confidence_low() -> None:
    pulse = compute_pulse(_snap([_row(f"S{i}", confidence=0.1) for i in range(3)]))
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
        Alert(symbol="AAPL", rule="action_changed", severity="warn", message="x", fired_at=fired),
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


# ---------------------------------------------------------------------------
# Breadth / intensity extensions
# ---------------------------------------------------------------------------


def _row_full(
    symbol: str,
    *,
    action: SignalAction = SignalAction.HOLD,
    confidence: float = 0.3,
    macd: float = 0.0,
    sentiment: float = 0.0,
    news: int = 0,
    tier: object = None,
    error: str | None = None,
) -> RecommendationRow:
    """Like _row but exposes macd + tier for breadth tests."""
    from src.strategy.base import RecommendationTier

    return RecommendationRow(
        symbol=symbol,
        action=action,
        confidence=confidence,
        combined_score=0.0,
        technical_score=0.0,
        sentiment_score=sentiment,
        rsi=math.nan,
        macd=macd,
        bollinger=math.nan,
        last_price=100.0,
        num_news_articles=news,
        reasoning="",
        timestamp=datetime.now(UTC),
        tier=tier if tier is not None else RecommendationTier.from_action(action),
        error=error,
    )


def test_pulse_counts_bullish_and_bearish_rows() -> None:
    pulse = compute_pulse(
        _snap(
            [
                _row("AAPL", action=SignalAction.BUY),
                _row("MSFT", action=SignalAction.BUY),
                _row("NVDA", action=SignalAction.SELL),
                _row("TSLA", action=SignalAction.HOLD),
                _row("SPY", action=SignalAction.HOLD),
            ]
        )
    )
    assert pulse.bullish_count == 2
    assert pulse.bearish_count == 1
    assert pulse.healthy_count == 5


def test_pulse_counts_exclude_error_rows() -> None:
    pulse = compute_pulse(
        _snap(
            [
                _row("AAPL", action=SignalAction.BUY),
                _row("MSFT", action=SignalAction.BUY, error="boom"),
            ]
        )
    )
    assert pulse.bullish_count == 1
    assert pulse.healthy_count == 1


def test_momentum_breadth_full_alignment_returns_one() -> None:
    """All BUY rows have positive MACD → breadth = 1.0."""
    pulse = compute_pulse(
        _snap(
            [
                _row_full("AAPL", action=SignalAction.BUY, macd=0.5),
                _row_full("MSFT", action=SignalAction.BUY, macd=0.4),
                _row_full("NVDA", action=SignalAction.SELL, macd=-0.3),
            ]
        )
    )
    assert pulse.momentum_breadth == 1.0


def test_momentum_breadth_half_aligned() -> None:
    pulse = compute_pulse(
        _snap(
            [
                _row_full("AAPL", action=SignalAction.BUY, macd=0.5),  # aligned
                _row_full("MSFT", action=SignalAction.BUY, macd=-0.4),  # opposed
            ]
        )
    )
    assert pulse.momentum_breadth == 0.5


def test_momentum_breadth_excludes_hold_rows() -> None:
    """HOLD has no direction to align with — those rows shouldn't be
    in the denominator."""
    pulse = compute_pulse(
        _snap(
            [
                _row_full("AAPL", action=SignalAction.BUY, macd=0.5),
                _row_full("MSFT", action=SignalAction.HOLD, macd=0.5),
                _row_full("NVDA", action=SignalAction.HOLD, macd=-0.5),
            ]
        )
    )
    # Only AAPL counts in the directional pool; it's aligned → 1.0.
    assert pulse.momentum_breadth == 1.0


def test_momentum_breadth_handles_nan_macd_as_unaligned() -> None:
    pulse = compute_pulse(
        _snap(
            [
                _row_full("AAPL", action=SignalAction.BUY, macd=math.nan),
                _row_full("MSFT", action=SignalAction.BUY, macd=0.4),
            ]
        )
    )
    # 1 of 2 is aligned (NaN doesn't qualify).
    assert pulse.momentum_breadth == 0.5


def test_momentum_breadth_zero_when_no_directional_rows() -> None:
    pulse = compute_pulse(_snap([_row_full("AAPL", action=SignalAction.HOLD, macd=0.5)]))
    assert pulse.momentum_breadth == 0.0


def test_sentiment_breadth_only_counts_news_bearing_rows() -> None:
    """A bullish row with no news shouldn't be in the denominator —
    the sentiment score is meaningless without articles."""
    pulse = compute_pulse(
        _snap(
            [
                _row_full("AAPL", action=SignalAction.BUY, sentiment=0.5, news=5),
                _row_full("MSFT", action=SignalAction.BUY, sentiment=0.5, news=0),
            ]
        )
    )
    # Only AAPL counts; aligned → 1.0.
    assert pulse.sentiment_breadth == 1.0


def test_sentiment_breadth_partial_alignment() -> None:
    pulse = compute_pulse(
        _snap(
            [
                _row_full("AAPL", action=SignalAction.BUY, sentiment=0.5, news=5),
                _row_full("MSFT", action=SignalAction.BUY, sentiment=-0.5, news=5),
            ]
        )
    )
    assert pulse.sentiment_breadth == 0.5


def test_sentiment_breadth_zero_when_no_news() -> None:
    pulse = compute_pulse(
        _snap([_row_full("AAPL", action=SignalAction.BUY, sentiment=0.5, news=0)])
    )
    assert pulse.sentiment_breadth == 0.0


def test_reversal_intensity_sums_flips_across_symbols() -> None:
    """A symbol with 3 episodes has flipped twice; sum across symbols."""
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(SignalAction.BUY),
            recent=(_episode(SignalAction.HOLD), _episode(SignalAction.SELL)),
        ),
        "MSFT": SignalHistorySummary(
            current=_episode(SignalAction.HOLD),
            recent=(_episode(SignalAction.BUY),),
        ),
    }
    pulse = compute_pulse(_snap([_row("AAPL"), _row("MSFT")], history=history))
    # AAPL: 2 flips, MSFT: 1 flip → total 3
    assert pulse.reversal_intensity == 3


def test_reversal_intensity_zero_without_history() -> None:
    pulse = compute_pulse(_snap([_row("AAPL")]))
    assert pulse.reversal_intensity == 0


def test_alert_intensity_is_length_of_recent_alerts() -> None:
    fired = datetime.now(UTC)
    alerts = tuple(
        Alert(
            symbol="AAPL",
            rule="action_changed",
            severity="warn",
            message="x",
            fired_at=fired,
        )
        for _ in range(7)
    )
    pulse = compute_pulse(_snap([_row("AAPL")], alerts=alerts))
    assert pulse.alert_intensity == 7


def test_strongest_symbols_lists_only_strong_tier_rows() -> None:
    from src.strategy.base import RecommendationTier

    pulse = compute_pulse(
        _snap(
            [
                _row_full(
                    "NVDA",
                    action=SignalAction.BUY,
                    confidence=0.85,
                    tier=RecommendationTier.STRONG_BUY,
                ),
                _row_full("AAPL", action=SignalAction.BUY, confidence=0.6),
                _row_full(
                    "TSLA",
                    action=SignalAction.SELL,
                    confidence=0.72,
                    tier=RecommendationTier.STRONG_SELL,
                ),
            ]
        )
    )
    syms = {sym for sym, _ in pulse.strongest_symbols}
    assert syms == {"NVDA", "TSLA"}
    # AAPL is BUY (not STRONG) → excluded.
    assert "AAPL" not in syms


def test_strongest_symbols_ranked_by_confidence_descending() -> None:
    from src.strategy.base import RecommendationTier

    pulse = compute_pulse(
        _snap(
            [
                _row_full(
                    "AAPL",
                    action=SignalAction.BUY,
                    confidence=0.66,
                    tier=RecommendationTier.STRONG_BUY,
                ),
                _row_full(
                    "NVDA",
                    action=SignalAction.BUY,
                    confidence=0.88,
                    tier=RecommendationTier.STRONG_BUY,
                ),
                _row_full(
                    "MSFT",
                    action=SignalAction.BUY,
                    confidence=0.71,
                    tier=RecommendationTier.STRONG_BUY,
                ),
            ]
        )
    )
    order = [sym for sym, _ in pulse.strongest_symbols]
    assert order == ["NVDA", "MSFT", "AAPL"]


def test_strongest_symbols_capped_at_module_limit() -> None:
    from src.intelligence.pulse import _STRONGEST_LIMIT
    from src.strategy.base import RecommendationTier

    rows = [
        _row_full(
            f"S{i}",
            action=SignalAction.BUY,
            confidence=0.7 + i * 0.01,
            tier=RecommendationTier.STRONG_BUY,
        )
        for i in range(_STRONGEST_LIMIT + 5)
    ]
    pulse = compute_pulse(_snap(rows))
    assert len(pulse.strongest_symbols) == _STRONGEST_LIMIT


def test_strongest_symbols_empty_when_no_strong_tier() -> None:
    pulse = compute_pulse(
        _snap(
            [
                _row("AAPL", action=SignalAction.BUY, confidence=0.5),
                _row("MSFT", action=SignalAction.SELL, confidence=0.4),
            ]
        )
    )
    assert pulse.strongest_symbols == ()


def test_strongest_symbols_carries_tier_display_strings() -> None:
    """The render layer reads the tier display string directly — so the
    tuple must carry the right human-readable label."""
    from src.strategy.base import RecommendationTier

    pulse = compute_pulse(
        _snap(
            [
                _row_full(
                    "NVDA",
                    action=SignalAction.BUY,
                    confidence=0.9,
                    tier=RecommendationTier.STRONG_BUY,
                ),
                _row_full(
                    "TSLA",
                    action=SignalAction.SELL,
                    confidence=0.7,
                    tier=RecommendationTier.STRONG_SELL,
                ),
            ]
        )
    )
    labels = dict(pulse.strongest_symbols)
    assert labels["NVDA"] == "STRONG BUY"
    assert labels["TSLA"] == "STRONG SELL"


def test_empty_snapshot_returns_zero_breadth_and_empty_strongest() -> None:
    """The is_empty fast-path bypasses all new fields — they should
    keep their default zero / empty values."""
    pulse = compute_pulse(_snap([]))
    assert pulse.is_empty
    assert pulse.bullish_count == 0
    assert pulse.bearish_count == 0
    assert pulse.healthy_count == 0
    assert pulse.momentum_breadth == 0.0
    assert pulse.sentiment_breadth == 0.0
    assert pulse.reversal_intensity == 0
    assert pulse.alert_intensity == 0
    assert pulse.strongest_symbols == ()


# ---------------------------------------------------------------------------
# compute_pulse_intraday (MT2 phase 2c)
# ---------------------------------------------------------------------------


def test_compute_pulse_intraday_empty_when_no_intraday_reads() -> None:
    """A snapshot whose rows lack IntradayRead returns the empty
    pulse sentinel — same null semantics as the daily pulse on an
    empty watchlist."""
    from src.intelligence.pulse import compute_pulse_intraday

    snap = DashboardSnapshot(
        tick=1,
        rows=[
            _row("AAPL", action=SignalAction.BUY),
            _row("MSFT", action=SignalAction.SELL),
        ],
    )
    pulse = compute_pulse_intraday(snap)
    assert pulse.is_empty


def test_compute_pulse_intraday_uses_intraday_actions() -> None:
    """Pulse breadth / bullish / bearish counts come from
    row.intraday.action, not row.action."""
    from src.data.models import TimeFrame
    from src.intelligence.pulse import compute_pulse_intraday
    from src.strategy.multi_timeframe import IntradayRead

    intraday_buy = IntradayRead(
        timeframe=TimeFrame.MIN_15,
        action=SignalAction.BUY,
        confidence=0.7,
        combined_score=0.5,
        technical_score=0.5,
    )
    intraday_sell = IntradayRead(
        timeframe=TimeFrame.MIN_15,
        action=SignalAction.SELL,
        confidence=0.6,
        combined_score=-0.4,
        technical_score=-0.4,
    )
    # Daily actions are flipped to confirm the helper reads from
    # intraday, not daily.
    rows = [
        RecommendationRow(
            symbol="AAPL",
            action=SignalAction.SELL,
            confidence=0.5,
            combined_score=0.0,
            technical_score=0.0,
            sentiment_score=0.0,
            rsi=math.nan,
            macd=math.nan,
            bollinger=math.nan,
            last_price=100.0,
            num_news_articles=0,
            reasoning="",
            timestamp=datetime.now(UTC),
            intraday=intraday_buy,
        ),
        RecommendationRow(
            symbol="MSFT",
            action=SignalAction.BUY,
            confidence=0.5,
            combined_score=0.0,
            technical_score=0.0,
            sentiment_score=0.0,
            rsi=math.nan,
            macd=math.nan,
            bollinger=math.nan,
            last_price=100.0,
            num_news_articles=0,
            reasoning="",
            timestamp=datetime.now(UTC),
            intraday=intraday_sell,
        ),
    ]
    snap = DashboardSnapshot(tick=1, rows=rows)
    pulse = compute_pulse_intraday(snap)
    assert pulse.bullish_count == 1  # one intraday BUY (AAPL)
    assert pulse.bearish_count == 1  # one intraday SELL (MSFT)
    assert pulse.healthy_count == 2
