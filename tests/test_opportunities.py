"""Tests for src.intelligence.opportunities."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from src.dashboard.state import DashboardSnapshot, RecommendationRow
from src.intelligence.history import SignalEpisode, SignalHistorySummary
from src.intelligence.opportunities import detect_opportunities
from src.strategy.base import SignalAction


def _row(
    symbol: str = "AAPL",
    *,
    action: SignalAction = SignalAction.HOLD,
    confidence: float = 0.3,
    technical: float = 0.0,
    sentiment: float = 0.0,
    news: int = 0,
    rsi: float = math.nan,
    macd: float = math.nan,
    bollinger: float = math.nan,
    error: str | None = None,
) -> RecommendationRow:
    return RecommendationRow(
        symbol=symbol,
        action=action,
        confidence=confidence,
        combined_score=0.0,
        technical_score=technical,
        sentiment_score=sentiment,
        rsi=rsi,
        macd=macd,
        bollinger=bollinger,
        last_price=100.0,
        num_news_articles=news,
        reasoning="",
        timestamp=datetime.now(UTC),
        error=error,
    )


def _episode(
    action: SignalAction,
    ticks: int = 1,
    conf_first: float = 0.5,
    conf_last: float = 0.5,
) -> SignalEpisode:
    return SignalEpisode(
        action=action,
        started_at=datetime(2026, 5, 13, tzinfo=UTC),
        last_seen_at=datetime(2026, 5, 13, tzinfo=UTC) + timedelta(seconds=ticks * 5),
        tick_count=ticks,
        confidence_first=conf_first,
        confidence_last=conf_last,
    )


def _snap(
    rows: list[RecommendationRow],
    *,
    history: dict[str, SignalHistorySummary] | None = None,
) -> DashboardSnapshot:
    return DashboardSnapshot(
        tick=1,
        rows=rows,
        events=[],
        signal_history=history or {},
        timestamp=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def test_convergence_fires_when_all_indicators_and_sentiment_aligned() -> None:
    snap = _snap([
        _row(
            "AAPL",
            action=SignalAction.BUY,
            confidence=0.6,
            rsi=0.5, macd=0.4, bollinger=0.3,
            sentiment=0.4, news=5,
        ),
    ])
    opps = detect_opportunities(snap, n=3)
    kinds = {o.kind for o in opps}
    assert "convergence" in kinds


def test_convergence_requires_news_for_sentiment_alignment() -> None:
    """Sentiment without articles doesn't count — even a positive
    score is meaningless without grounding."""
    snap = _snap([
        _row(
            "AAPL",
            action=SignalAction.BUY,
            confidence=0.6,
            rsi=0.5, macd=0.4, bollinger=0.3,
            sentiment=0.4, news=0,  # no news
        ),
    ])
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "convergence" not in kinds


def test_convergence_skipped_on_hold_action() -> None:
    snap = _snap([
        _row(
            "AAPL",
            action=SignalAction.HOLD,
            rsi=0.5, macd=0.4, bollinger=0.3,
            sentiment=0.4, news=5,
        ),
    ])
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "convergence" not in kinds


def test_convergence_skipped_when_one_indicator_disagrees() -> None:
    snap = _snap([
        _row(
            "AAPL",
            action=SignalAction.BUY,
            rsi=0.5, macd=-0.4, bollinger=0.3,  # MACD disagrees
            sentiment=0.4, news=5,
        ),
    ])
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "convergence" not in kinds


def test_convergence_handles_nan_indicators_as_non_aligned() -> None:
    snap = _snap([
        _row(
            "AAPL",
            action=SignalAction.BUY,
            rsi=math.nan, macd=0.4, bollinger=0.3,  # RSI not computed
            sentiment=0.4, news=5,
        ),
    ])
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "convergence" not in kinds


def test_convergence_score_rewards_strong_alignment() -> None:
    """Larger magnitudes should produce higher scores."""
    weak = _snap([
        _row(
            "AAPL", action=SignalAction.BUY,
            rsi=0.25, macd=0.25, bollinger=0.25,
            sentiment=0.25, news=5,
        ),
    ])
    strong = _snap([
        _row(
            "MSFT", action=SignalAction.BUY,
            rsi=0.8, macd=0.8, bollinger=0.8,
            sentiment=0.8, news=5,
        ),
    ])
    weak_score = detect_opportunities(weak)[0].score
    strong_score = detect_opportunities(strong)[0].score
    assert strong_score > weak_score


# ---------------------------------------------------------------------------
# Reversal
# ---------------------------------------------------------------------------


def test_reversal_fires_on_buy_to_sell_with_rising_confidence() -> None:
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(
                SignalAction.SELL, conf_first=0.3, conf_last=0.6
            ),
            recent=(_episode(SignalAction.BUY, ticks=5),),
        ),
    }
    snap = _snap(
        [_row("AAPL", action=SignalAction.SELL, confidence=0.6)],
        history=history,
    )
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "reversal" in kinds


def test_reversal_skipped_when_confidence_not_rising() -> None:
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(
                SignalAction.SELL, conf_first=0.6, conf_last=0.6
            ),
            recent=(_episode(SignalAction.BUY, ticks=5),),
        ),
    }
    snap = _snap(
        [_row("AAPL", action=SignalAction.SELL, confidence=0.6)],
        history=history,
    )
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "reversal" not in kinds


def test_reversal_skipped_when_prior_was_hold() -> None:
    """Exiting HOLD isn't a reversal — it's a trigger. HOLD has no
    direction to reverse from."""
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(
                SignalAction.BUY, conf_first=0.3, conf_last=0.6
            ),
            recent=(_episode(SignalAction.HOLD, ticks=5),),
        ),
    }
    snap = _snap(
        [_row("AAPL", action=SignalAction.BUY, confidence=0.6)],
        history=history,
    )
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "reversal" not in kinds


def test_reversal_score_scales_with_prior_duration() -> None:
    """A flip after 10 ticks of BUY > a flip after 1 tick of BUY."""
    def _snap_for(prior_ticks: int, sym: str) -> DashboardSnapshot:
        history = {
            sym: SignalHistorySummary(
                current=_episode(
                    SignalAction.SELL, conf_first=0.3, conf_last=0.6
                ),
                recent=(_episode(SignalAction.BUY, ticks=prior_ticks),),
            ),
        }
        return _snap(
            [_row(sym, action=SignalAction.SELL, confidence=0.6)],
            history=history,
        )

    short = detect_opportunities(_snap_for(1, "A"))[0]
    long_ = detect_opportunities(_snap_for(10, "B"))[0]
    assert long_.score > short.score


def test_reversal_skipped_without_history() -> None:
    snap = _snap([_row("AAPL", action=SignalAction.BUY, confidence=0.7)])
    assert detect_opportunities(snap) == [] or all(
        o.kind != "reversal" for o in detect_opportunities(snap)
    )


# ---------------------------------------------------------------------------
# High conviction
# ---------------------------------------------------------------------------


def test_high_conviction_fires_with_aligned_tech_and_sentiment() -> None:
    snap = _snap([
        _row(
            "AAPL",
            action=SignalAction.BUY,
            confidence=0.65,
            technical=0.4,
            sentiment=0.4,
            news=5,
        ),
    ])
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "high_conviction" in kinds


def test_high_conviction_requires_sentiment_to_match_direction() -> None:
    """Bullish action + bearish sentiment = NOT high conviction.
    That's a tension case, not an aligned setup."""
    snap = _snap([
        _row(
            "AAPL",
            action=SignalAction.BUY,
            confidence=0.7,
            technical=0.5,
            sentiment=-0.5,  # disagrees with action
            news=5,
        ),
    ])
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "high_conviction" not in kinds


def test_high_conviction_requires_news() -> None:
    snap = _snap([
        _row(
            "AAPL",
            action=SignalAction.BUY,
            confidence=0.7,
            technical=0.5,
            sentiment=0.5,
            news=0,  # no news to ground sentiment
        ),
    ])
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "high_conviction" not in kinds


def test_high_conviction_skipped_when_confidence_below_threshold() -> None:
    snap = _snap([
        _row(
            "AAPL",
            action=SignalAction.BUY,
            confidence=0.4,  # below the 0.55 threshold
            technical=0.5,
            sentiment=0.5,
            news=5,
        ),
    ])
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "high_conviction" not in kinds


# ---------------------------------------------------------------------------
# Ranking + cap + error rows
# ---------------------------------------------------------------------------


def test_n_caps_total_opportunities() -> None:
    """Build five rows each qualifying for convergence; assert N caps."""
    rows = [
        _row(
            f"S{i}",
            action=SignalAction.BUY,
            confidence=0.6,
            rsi=0.5, macd=0.5, bollinger=0.5,
            sentiment=0.5, news=5,
        )
        for i in range(5)
    ]
    snap = _snap(rows)
    assert len(detect_opportunities(snap, n=3)) == 3
    assert len(detect_opportunities(snap, n=5)) == 5


def test_error_rows_are_excluded() -> None:
    snap = _snap([
        _row(
            "AAPL", action=SignalAction.BUY, confidence=0.7,
            rsi=0.5, macd=0.5, bollinger=0.5, sentiment=0.5, news=5,
            error="boom",
        ),
    ])
    assert detect_opportunities(snap) == []


def test_empty_snapshot_returns_empty_list() -> None:
    assert detect_opportunities(_snap([])) == []


def test_same_symbol_kind_pair_appears_once() -> None:
    """Even if scoring produces the same kind twice for one symbol,
    dedup keeps only the highest-scored one."""
    snap = _snap([
        _row(
            "AAPL", action=SignalAction.BUY, confidence=0.6,
            rsi=0.5, macd=0.5, bollinger=0.5, sentiment=0.5, news=5,
        ),
    ])
    opps = detect_opportunities(snap, n=10)
    keys = [(o.symbol, o.kind) for o in opps]
    assert len(keys) == len(set(keys))


def test_rationale_is_observational_not_predictive() -> None:
    """Anti-hallucination: opportunity rationales describe data, don't
    forecast. Catches future edits that introduce 'likely' or 'will'
    phrasing."""
    snap = _snap([
        _row(
            "AAPL", action=SignalAction.BUY, confidence=0.7,
            rsi=0.6, macd=0.6, bollinger=0.6, sentiment=0.5, news=5,
            technical=0.6,
        ),
    ])
    forbidden = ("likely", "will rise", "will fall", "expected to", "forecast")
    for opp in detect_opportunities(snap, n=10):
        for word in forbidden:
            assert word not in opp.rationale.lower(), (
                f"{opp.kind} rationale should not contain '{word}': {opp.rationale}"
            )
