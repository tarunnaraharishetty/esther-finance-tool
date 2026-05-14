"""Tests for src.intelligence.opportunities."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from src.dashboard.state import DashboardSnapshot, RecommendationRow
from src.intelligence.history import SignalEpisode, SignalHistorySummary
from src.intelligence.opportunities import (
    detect_opportunities,
    rank_opportunities,
)
from src.strategy.base import RecommendationTier, SignalAction


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
    intraday: object | None = None,
) -> RecommendationRow:
    from src.strategy.multi_timeframe import IntradayRead

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
        intraday=intraday if isinstance(intraday, IntradayRead) else None,
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
    intraday_history: dict[str, SignalHistorySummary] | None = None,
) -> DashboardSnapshot:
    return DashboardSnapshot(
        tick=1,
        rows=rows,
        events=[],
        signal_history=history or {},
        intraday_signal_history=intraday_history or {},
        timestamp=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def test_convergence_fires_when_all_indicators_and_sentiment_aligned() -> None:
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                confidence=0.6,
                rsi=0.5,
                macd=0.4,
                bollinger=0.3,
                sentiment=0.4,
                news=5,
            ),
        ]
    )
    opps = detect_opportunities(snap, n=3)
    kinds = {o.kind for o in opps}
    assert "convergence" in kinds


def test_convergence_requires_news_for_sentiment_alignment() -> None:
    """Sentiment without articles doesn't count — even a positive
    score is meaningless without grounding."""
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                confidence=0.6,
                rsi=0.5,
                macd=0.4,
                bollinger=0.3,
                sentiment=0.4,
                news=0,  # no news
            ),
        ]
    )
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "convergence" not in kinds


def test_convergence_skipped_on_hold_action() -> None:
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.HOLD,
                rsi=0.5,
                macd=0.4,
                bollinger=0.3,
                sentiment=0.4,
                news=5,
            ),
        ]
    )
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "convergence" not in kinds


def test_convergence_skipped_when_one_indicator_disagrees() -> None:
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                rsi=0.5,
                macd=-0.4,
                bollinger=0.3,  # MACD disagrees
                sentiment=0.4,
                news=5,
            ),
        ]
    )
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "convergence" not in kinds


def test_convergence_handles_nan_indicators_as_non_aligned() -> None:
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                rsi=math.nan,
                macd=0.4,
                bollinger=0.3,  # RSI not computed
                sentiment=0.4,
                news=5,
            ),
        ]
    )
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "convergence" not in kinds


def test_convergence_score_rewards_strong_alignment() -> None:
    """Larger magnitudes should produce higher scores."""
    weak = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                rsi=0.25,
                macd=0.25,
                bollinger=0.25,
                sentiment=0.25,
                news=5,
            ),
        ]
    )
    strong = _snap(
        [
            _row(
                "MSFT",
                action=SignalAction.BUY,
                rsi=0.8,
                macd=0.8,
                bollinger=0.8,
                sentiment=0.8,
                news=5,
            ),
        ]
    )
    weak_score = detect_opportunities(weak)[0].score
    strong_score = detect_opportunities(strong)[0].score
    assert strong_score > weak_score


# ---------------------------------------------------------------------------
# Reversal
# ---------------------------------------------------------------------------


def test_reversal_fires_on_buy_to_sell_with_rising_confidence() -> None:
    history = {
        "AAPL": SignalHistorySummary(
            current=_episode(SignalAction.SELL, conf_first=0.3, conf_last=0.6),
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
            current=_episode(SignalAction.SELL, conf_first=0.6, conf_last=0.6),
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
            current=_episode(SignalAction.BUY, conf_first=0.3, conf_last=0.6),
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
                current=_episode(SignalAction.SELL, conf_first=0.3, conf_last=0.6),
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
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                confidence=0.65,
                technical=0.4,
                sentiment=0.4,
                news=5,
            ),
        ]
    )
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "high_conviction" in kinds


def test_high_conviction_requires_sentiment_to_match_direction() -> None:
    """Bullish action + bearish sentiment = NOT high conviction.
    That's a tension case, not an aligned setup."""
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                confidence=0.7,
                technical=0.5,
                sentiment=-0.5,  # disagrees with action
                news=5,
            ),
        ]
    )
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "high_conviction" not in kinds


def test_high_conviction_requires_news() -> None:
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                confidence=0.7,
                technical=0.5,
                sentiment=0.5,
                news=0,  # no news to ground sentiment
            ),
        ]
    )
    kinds = {o.kind for o in detect_opportunities(snap)}
    assert "high_conviction" not in kinds


def test_high_conviction_skipped_when_confidence_below_threshold() -> None:
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                confidence=0.4,  # below the 0.55 threshold
                technical=0.5,
                sentiment=0.5,
                news=5,
            ),
        ]
    )
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
            rsi=0.5,
            macd=0.5,
            bollinger=0.5,
            sentiment=0.5,
            news=5,
        )
        for i in range(5)
    ]
    snap = _snap(rows)
    assert len(detect_opportunities(snap, n=3)) == 3
    assert len(detect_opportunities(snap, n=5)) == 5


def test_error_rows_are_excluded() -> None:
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                confidence=0.7,
                rsi=0.5,
                macd=0.5,
                bollinger=0.5,
                sentiment=0.5,
                news=5,
                error="boom",
            ),
        ]
    )
    assert detect_opportunities(snap) == []


def test_empty_snapshot_returns_empty_list() -> None:
    assert detect_opportunities(_snap([])) == []


def test_same_symbol_kind_pair_appears_once() -> None:
    """Even if scoring produces the same kind twice for one symbol,
    dedup keeps only the highest-scored one."""
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                confidence=0.6,
                rsi=0.5,
                macd=0.5,
                bollinger=0.5,
                sentiment=0.5,
                news=5,
            ),
        ]
    )
    opps = detect_opportunities(snap, n=10)
    keys = [(o.symbol, o.kind) for o in opps]
    assert len(keys) == len(set(keys))


def test_rationale_is_observational_not_predictive() -> None:
    """Anti-hallucination: opportunity rationales describe data, don't
    forecast. Catches future edits that introduce 'likely' or 'will'
    phrasing."""
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                confidence=0.7,
                rsi=0.6,
                macd=0.6,
                bollinger=0.6,
                sentiment=0.5,
                news=5,
                technical=0.6,
            ),
        ]
    )
    forbidden = ("likely", "will rise", "will fall", "expected to", "forecast")
    for opp in detect_opportunities(snap, n=10):
        for word in forbidden:
            assert word not in opp.rationale.lower(), (
                f"{opp.kind} rationale should not contain '{word}': {opp.rationale}"
            )


# ---------------------------------------------------------------------------
# rank_opportunities (composite-score ranking)
# ---------------------------------------------------------------------------


def _row_full(
    symbol: str,
    *,
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.6,
    technical: float = 0.4,
    sentiment: float = 0.4,
    news: int = 5,
    rsi: float = 0.4,
    macd: float = 0.5,
    bollinger: float = 0.3,
    tier: RecommendationTier | None = None,
    signal_quality: str = "moderate",
    stability: str = "stable",
    error: str | None = None,
):
    """Helper that produces a richly-populated RecommendationRow for
    composite-score tests."""
    return RecommendationRow(
        symbol=symbol,
        action=action,
        confidence=confidence,
        combined_score=technical,
        technical_score=technical,
        sentiment_score=sentiment,
        rsi=rsi,
        macd=macd,
        bollinger=bollinger,
        last_price=100.0,
        num_news_articles=news,
        reasoning="",
        timestamp=datetime.now(UTC),
        tier=tier if tier is not None else RecommendationTier.from_action(action),
        signal_quality=signal_quality,
        stability=stability,
        error=error,
    )


def test_rank_skips_hold_rows() -> None:
    """HOLD has no direction — nothing to rank."""
    snap = _snap([_row_full("AAPL", action=SignalAction.HOLD)])
    assert rank_opportunities(snap) == []


def test_rank_skips_error_rows() -> None:
    snap = _snap([_row_full("AAPL", action=SignalAction.BUY, error="boom")])
    assert rank_opportunities(snap) == []


def test_rank_returns_directional_rows_ordered_by_composite() -> None:
    """A higher-quality row should rank above a lower-quality row of
    the same kind."""
    snap = _snap(
        [
            _row_full(
                "WEAK",
                action=SignalAction.BUY,
                confidence=0.3,
                technical=0.1,
                sentiment=0.1,
                news=0,
                rsi=0.05,
                macd=0.05,
                bollinger=0.05,
                signal_quality="low",
            ),
            _row_full(
                "STRONG",
                action=SignalAction.BUY,
                confidence=0.85,
                technical=0.7,
                sentiment=0.7,
                news=8,
                rsi=0.6,
                macd=0.7,
                bollinger=0.5,
                tier=RecommendationTier.STRONG_BUY,
                signal_quality="high",
            ),
        ]
    )
    ranked = rank_opportunities(snap)
    assert [r.symbol for r in ranked] == ["STRONG", "WEAK"]
    assert ranked[0].composite_score > ranked[1].composite_score


def test_rank_carries_through_tier_field() -> None:
    snap = _snap(
        [
            _row_full(
                "NVDA",
                tier=RecommendationTier.STRONG_BUY,
                action=SignalAction.BUY,
                confidence=0.8,
                signal_quality="high",
            ),
        ]
    )
    ranked = rank_opportunities(snap)
    assert len(ranked) == 1
    assert ranked[0].tier == RecommendationTier.STRONG_BUY


def test_rank_caps_at_n() -> None:
    rows = [
        _row_full(f"S{i}", action=SignalAction.BUY, confidence=0.5 + i * 0.05) for i in range(8)
    ]
    snap = _snap(rows)
    assert len(rank_opportunities(snap, n=3)) == 3
    assert len(rank_opportunities(snap, n=8)) == 8


def test_rank_composite_in_unit_interval() -> None:
    rows = [
        _row_full(
            "S",
            action=SignalAction.BUY,
            confidence=0.95,
            technical=0.95,
            sentiment=0.95,
            news=10,
            rsi=0.9,
            macd=0.95,
            bollinger=0.9,
            signal_quality="high",
        ),
    ]
    snap = _snap(rows)
    ranked = rank_opportunities(snap)
    assert 0.0 <= ranked[0].composite_score <= 1.0


def test_rank_attaches_profile_for_each_entry() -> None:
    from src.intelligence.signal_profile import SignalProfile

    snap = _snap(
        [
            _row_full(
                "AAPL",
                action=SignalAction.BUY,
                stability="stable",
            ),
        ]
    )
    ranked = rank_opportunities(snap)
    assert isinstance(ranked[0].profile, SignalProfile)
    assert ranked[0].profile.stability == "stable"


def test_technical_alignment_drops_with_opposing_indicators() -> None:
    """A BUY row where one indicator opposes should score lower on the
    technical_alignment driver than a fully-aligned BUY."""
    aligned = _row_full(
        "ALIGN",
        action=SignalAction.BUY,
        rsi=0.5,
        macd=0.6,
        bollinger=0.4,
    )
    opposed = _row_full(
        "OPPOSE",
        action=SignalAction.BUY,
        rsi=0.5,
        macd=-0.6,
        bollinger=0.4,  # MACD opposes
    )
    snap = _snap([aligned, opposed])
    ranked = {r.symbol: r for r in rank_opportunities(snap)}
    assert ranked["ALIGN"].technical_alignment > ranked["OPPOSE"].technical_alignment


def test_sentiment_alignment_zero_without_news() -> None:
    """Sentiment without articles isn't grounded — driver scores 0."""
    snap = _snap(
        [
            _row_full(
                "AAPL",
                action=SignalAction.BUY,
                sentiment=0.7,
                news=0,
            ),
        ]
    )
    ranked = rank_opportunities(snap)
    assert ranked[0].sentiment_alignment == 0.0


def test_confidence_acceleration_high_when_conf_rising() -> None:
    """A row with a history showing rising confidence in the current
    episode should score higher on the acceleration driver than one
    with falling confidence."""
    history = {
        "RISING": SignalHistorySummary(
            current=SignalEpisode(
                action=SignalAction.BUY,
                started_at=datetime(2026, 5, 13, tzinfo=UTC),
                last_seen_at=datetime(2026, 5, 13, tzinfo=UTC),
                tick_count=5,
                confidence_first=0.4,
                confidence_last=0.8,
            ),
            recent=(),
        ),
        "FALLING": SignalHistorySummary(
            current=SignalEpisode(
                action=SignalAction.BUY,
                started_at=datetime(2026, 5, 13, tzinfo=UTC),
                last_seen_at=datetime(2026, 5, 13, tzinfo=UTC),
                tick_count=5,
                confidence_first=0.8,
                confidence_last=0.4,
            ),
            recent=(),
        ),
    }
    snap = _snap(
        [
            _row_full("RISING", action=SignalAction.BUY),
            _row_full("FALLING", action=SignalAction.BUY),
        ],
        history=history,
    )
    ranked = {r.symbol: r for r in rank_opportunities(snap)}
    assert ranked["RISING"].confidence_acceleration > ranked["FALLING"].confidence_acceleration


def test_momentum_persistence_scales_with_episode_duration() -> None:
    history = {
        "SHORT": SignalHistorySummary(
            current=SignalEpisode(
                action=SignalAction.BUY,
                started_at=datetime(2026, 5, 13, tzinfo=UTC),
                last_seen_at=datetime(2026, 5, 13, tzinfo=UTC),
                tick_count=1,
                confidence_first=0.5,
                confidence_last=0.5,
            ),
            recent=(),
        ),
        "LONG": SignalHistorySummary(
            current=SignalEpisode(
                action=SignalAction.BUY,
                started_at=datetime(2026, 5, 13, tzinfo=UTC),
                last_seen_at=datetime(2026, 5, 13, tzinfo=UTC),
                tick_count=10,
                confidence_first=0.5,
                confidence_last=0.5,
            ),
            recent=(),
        ),
    }
    snap = _snap(
        [
            _row_full("SHORT", action=SignalAction.BUY),
            _row_full("LONG", action=SignalAction.BUY),
        ],
        history=history,
    )
    ranked = {r.symbol: r for r in rank_opportunities(snap)}
    assert ranked["LONG"].momentum_persistence > ranked["SHORT"].momentum_persistence


def test_reversal_strength_requires_opposite_prior_action() -> None:
    """No reversal score when the prior episode was the same action."""
    history = {
        "AAPL": SignalHistorySummary(
            current=SignalEpisode(
                action=SignalAction.BUY,
                started_at=datetime(2026, 5, 13, tzinfo=UTC),
                last_seen_at=datetime(2026, 5, 13, tzinfo=UTC),
                tick_count=3,
                confidence_first=0.5,
                confidence_last=0.6,
            ),
            recent=(
                SignalEpisode(
                    action=SignalAction.BUY,  # same direction
                    started_at=datetime(2026, 5, 13, tzinfo=UTC),
                    last_seen_at=datetime(2026, 5, 13, tzinfo=UTC),
                    tick_count=5,
                    confidence_first=0.4,
                    confidence_last=0.5,
                ),
            ),
        ),
    }
    snap = _snap(
        [_row_full("AAPL", action=SignalAction.BUY)],
        history=history,
    )
    ranked = rank_opportunities(snap)
    assert ranked[0].reversal_strength == 0.0


def test_reversal_strength_nonzero_when_flipping_from_opposite() -> None:
    history = {
        "AAPL": SignalHistorySummary(
            current=SignalEpisode(
                action=SignalAction.BUY,
                started_at=datetime(2026, 5, 13, tzinfo=UTC),
                last_seen_at=datetime(2026, 5, 13, tzinfo=UTC),
                tick_count=2,
                confidence_first=0.5,
                confidence_last=0.7,
            ),
            recent=(
                SignalEpisode(
                    action=SignalAction.SELL,  # opposite — flipped from this
                    started_at=datetime(2026, 5, 13, tzinfo=UTC),
                    last_seen_at=datetime(2026, 5, 13, tzinfo=UTC),
                    tick_count=8,
                    confidence_first=0.6,
                    confidence_last=0.4,
                ),
            ),
        ),
    }
    snap = _snap(
        [
            _row_full("AAPL", action=SignalAction.BUY, confidence=0.7),
        ],
        history=history,
    )
    ranked = rank_opportunities(snap)
    assert ranked[0].reversal_strength > 0.0


def test_signal_quality_score_maps_label_to_number() -> None:
    """high=1.0, moderate=0.5, low=0.0 — used for weighting in the
    composite."""
    snap = _snap(
        [
            _row_full("HIGH", action=SignalAction.BUY, signal_quality="high"),
            _row_full("MOD", action=SignalAction.BUY, signal_quality="moderate"),
            _row_full("LOW", action=SignalAction.BUY, signal_quality="low"),
        ]
    )
    by_sym = {r.symbol: r for r in rank_opportunities(snap)}
    assert by_sym["HIGH"].signal_quality_score == 1.0
    assert by_sym["MOD"].signal_quality_score == 0.5
    assert by_sym["LOW"].signal_quality_score == 0.0


def test_rationale_includes_strongest_driver_phrases() -> None:
    """When a driver scores above the rationale floor, its phrase
    should appear in the rationale tuple."""
    snap = _snap(
        [
            _row_full(
                "AAPL",
                action=SignalAction.BUY,
                confidence=0.8,
                technical=0.7,
                sentiment=0.7,
                news=10,
                rsi=0.6,
                macd=0.7,
                bollinger=0.6,
                tier=RecommendationTier.STRONG_BUY,
                signal_quality="high",
            ),
        ]
    )
    ranked = rank_opportunities(snap)
    rationale_text = " ".join(ranked[0].rationale).lower()
    # At least one of these should appear given the highly-aligned inputs.
    assert any(
        phrase in rationale_text
        for phrase in (
            "indicators aligned",
            "news sentiment matches",
            "high signal quality",
        )
    )


def test_rank_rationale_is_observational_not_predictive() -> None:
    """Anti-hallucination guard — same forbidden words pattern."""
    snap = _snap(
        [
            _row_full(
                "AAPL",
                action=SignalAction.BUY,
                confidence=0.85,
                technical=0.7,
                sentiment=0.7,
                news=8,
                rsi=0.6,
                macd=0.7,
                bollinger=0.5,
                tier=RecommendationTier.STRONG_BUY,
                signal_quality="high",
            ),
        ]
    )
    forbidden = ("likely", "will rise", "will fall", "expected to", "forecast")
    for ranked in rank_opportunities(snap):
        text = " ".join(ranked.rationale).lower()
        for word in forbidden:
            assert word not in text, f"rationale should not contain '{word}': {ranked.rationale}"


# ---------------------------------------------------------------------------
# rank_opportunities_intraday (MT2 phase 2b)
# ---------------------------------------------------------------------------


def _intraday(
    action: SignalAction = SignalAction.BUY,
    *,
    confidence: float = 0.6,
    technical_score: float = 0.4,
) -> object:
    from src.data.models import TimeFrame
    from src.strategy.multi_timeframe import IntradayRead

    return IntradayRead(
        timeframe=TimeFrame.MIN_15,
        action=action,
        confidence=confidence,
        combined_score=technical_score,
        technical_score=technical_score,
    )


def test_intraday_ranker_skips_rows_without_intraday_read() -> None:
    """Rows where row.intraday is None can't be ranked — phase 2a's
    `_record_intraday_history` skips them and so does the ranker."""
    from src.intelligence.opportunities import rank_opportunities_intraday

    snap = _snap(
        [
            _row("AAPL", action=SignalAction.BUY, intraday=None),
            _row("MSFT", action=SignalAction.SELL, intraday=None),
        ]
    )
    assert rank_opportunities_intraday(snap) == []


def test_intraday_ranker_returns_directional_intraday_rows() -> None:
    """Only intraday BUY / SELL rows participate — intraday HOLD is
    skipped, matching the daily ranker's contract."""
    from src.intelligence.opportunities import rank_opportunities_intraday

    snap = _snap(
        [
            _row("AAPL", action=SignalAction.BUY, intraday=_intraday(SignalAction.BUY)),
            _row("MSFT", action=SignalAction.BUY, intraday=_intraday(SignalAction.HOLD)),
            _row("NVDA", action=SignalAction.SELL, intraday=_intraday(SignalAction.SELL)),
        ]
    )
    symbols = [opp.symbol for opp in rank_opportunities_intraday(snap)]
    assert "AAPL" in symbols
    assert "NVDA" in symbols
    assert "MSFT" not in symbols  # intraday HOLD


def test_intraday_ranker_uses_intraday_action_for_tier() -> None:
    """The tier on a ranked intraday opportunity reflects the
    intraday action, not the daily one — that's the whole point of
    the separate ranker."""
    from src.intelligence.opportunities import rank_opportunities_intraday
    from src.strategy.base import RecommendationTier

    # Daily action is BUY, intraday action is SELL → ranked tier
    # should reflect SELL.
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                intraday=_intraday(SignalAction.SELL, technical_score=-0.5),
            ),
        ]
    )
    ranked = rank_opportunities_intraday(snap)
    assert len(ranked) == 1
    assert ranked[0].tier == RecommendationTier.SELL


def test_intraday_ranker_uses_intraday_history_for_drivers() -> None:
    """Intraday-specific drivers (momentum_persistence,
    confidence_acceleration) key off the intraday history.
    With a multi-tick rising-confidence intraday episode the
    momentum_persistence score should clearly favor that symbol
    over one without history."""
    from src.intelligence.opportunities import rank_opportunities_intraday

    rich = SignalHistorySummary(
        current=_episode(SignalAction.BUY, ticks=6, conf_first=0.4, conf_last=0.8),
        recent=(),
    )
    sparse = SignalHistorySummary(
        current=_episode(SignalAction.BUY, ticks=1, conf_first=0.4, conf_last=0.4),
        recent=(),
    )
    snap = _snap(
        [
            _row(
                "AAPL",
                action=SignalAction.BUY,
                intraday=_intraday(SignalAction.BUY, technical_score=0.4),
            ),
            _row(
                "MSFT",
                action=SignalAction.BUY,
                intraday=_intraday(SignalAction.BUY, technical_score=0.4),
            ),
        ],
        intraday_history={"AAPL": rich, "MSFT": sparse},
    )
    ranked = {opp.symbol: opp for opp in rank_opportunities_intraday(snap)}
    assert ranked["AAPL"].momentum_persistence > ranked["MSFT"].momentum_persistence
    assert ranked["AAPL"].composite_score > ranked["MSFT"].composite_score
