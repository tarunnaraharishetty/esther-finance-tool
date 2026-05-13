"""Tests for src.intelligence.signal_profile."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from src.dashboard.state import RecommendationRow
from src.intelligence.history import SignalEpisode, SignalHistorySummary
from src.intelligence.signal_profile import (
    compute_signal_profile,
)
from src.strategy.base import RecommendationTier, SignalAction


def _row(
    *,
    action: SignalAction = SignalAction.BUY,
    stability: str = "stable",
    confidence: float = 0.5,
    tier: RecommendationTier | None = None,
) -> RecommendationRow:
    return RecommendationRow(
        symbol="AAPL",
        action=action,
        confidence=confidence,
        combined_score=0.5,
        technical_score=0.5,
        sentiment_score=0.0,
        rsi=math.nan, macd=math.nan, bollinger=math.nan,
        last_price=100.0,
        num_news_articles=0,
        reasoning="",
        timestamp=datetime.now(UTC),
        tier=tier if tier is not None else RecommendationTier.from_action(action),
        stability=stability,
    )


def _episode(
    action: SignalAction = SignalAction.BUY,
    ticks: int = 5,
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


def _hist(
    current: SignalEpisode | None = None,
    recent: tuple[SignalEpisode, ...] = (),
) -> SignalHistorySummary:
    return SignalHistorySummary(current=current or _episode(), recent=recent)


# ---------------------------------------------------------------------------
# Stability axis — binary collapse of the 3-tier promote_to_tier output
# ---------------------------------------------------------------------------


def test_stability_stable_passes_through() -> None:
    profile = compute_signal_profile(_row(stability="stable"), _hist())
    assert profile.stability == "stable"


def test_stability_moderate_is_noisy() -> None:
    """Moderate is uncertain — for the trader's 'follow it?' question
    that's noise, not signal."""
    profile = compute_signal_profile(_row(stability="moderate"), _hist())
    assert profile.stability == "noisy"


def test_stability_volatile_is_noisy() -> None:
    profile = compute_signal_profile(_row(stability="volatile"), _hist())
    assert profile.stability == "noisy"


# ---------------------------------------------------------------------------
# Trend axis — confidence trend within current episode
# ---------------------------------------------------------------------------


def test_trend_strengthening_when_conf_rising() -> None:
    profile = compute_signal_profile(
        _row(),
        _hist(current=_episode(ticks=5, conf_first=0.4, conf_last=0.7)),
    )
    assert profile.trend == "strengthening"


def test_trend_weakening_when_conf_falling() -> None:
    profile = compute_signal_profile(
        _row(),
        _hist(current=_episode(ticks=5, conf_first=0.7, conf_last=0.4)),
    )
    assert profile.trend == "weakening"


def test_trend_flat_when_conf_steady() -> None:
    profile = compute_signal_profile(
        _row(),
        _hist(current=_episode(ticks=5, conf_first=0.5, conf_last=0.55)),
    )
    assert profile.trend == "flat"


def test_trend_flat_when_episode_too_young() -> None:
    """Episode of 2 ticks isn't enough to call a trend yet."""
    profile = compute_signal_profile(
        _row(),
        _hist(current=_episode(ticks=2, conf_first=0.3, conf_last=0.8)),
    )
    assert profile.trend == "flat"


def test_trend_flat_when_no_history() -> None:
    profile = compute_signal_profile(_row(), None)
    assert profile.trend == "flat"


# ---------------------------------------------------------------------------
# Persistence axis — episode count
# ---------------------------------------------------------------------------


def test_persistence_persistent_with_one_episode() -> None:
    profile = compute_signal_profile(_row(), _hist())  # 1 episode default
    assert profile.persistence == "persistent"


def test_persistence_persistent_with_two_episodes() -> None:
    profile = compute_signal_profile(
        _row(),
        _hist(current=_episode(), recent=(_episode(),)),
    )
    assert profile.persistence == "persistent"


def test_persistence_flipping_with_three_episodes() -> None:
    profile = compute_signal_profile(
        _row(),
        _hist(current=_episode(), recent=(_episode(), _episode())),
    )
    assert profile.persistence == "flipping"


def test_persistence_persistent_when_no_history() -> None:
    """First tick of the session — no evidence of flipping yet."""
    profile = compute_signal_profile(_row(), None)
    assert profile.persistence == "persistent"


# ---------------------------------------------------------------------------
# Frozen-dataclass / schema invariants
# ---------------------------------------------------------------------------


def test_profile_is_a_three_field_frozen_dataclass() -> None:
    profile = compute_signal_profile(_row(), _hist())
    # Set of value-domains is constrained so render code never branches
    # on unexpected strings.
    assert profile.stability in ("stable", "noisy")
    assert profile.trend in ("strengthening", "weakening", "flat")
    assert profile.persistence in ("persistent", "flipping")


def test_profile_immutable() -> None:
    import pytest

    profile = compute_signal_profile(_row(), _hist())
    with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
        profile.stability = "very noisy"  # type: ignore[misc]


def test_profile_equal_when_inputs_equal() -> None:
    """Pure function — same inputs produce equal profiles."""
    row = _row()
    history = _hist()
    a = compute_signal_profile(row, history)
    b = compute_signal_profile(row, history)
    assert a == b
