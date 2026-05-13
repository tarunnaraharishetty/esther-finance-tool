"""Tests for src.intelligence.multi_timeframe.

Grounded measurable logic only — every test asserts a derived value of
signed close-price comparisons. No prediction language anywhere in the
output domain; the anti-hallucination test at the bottom enforces the
absence of forbidden words on every field that's a string.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.intelligence.multi_timeframe import (
    MultiTimeframeView,
    TimeframeTrend,
    compute_mtf_view,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _df(closes: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV df from a close-price series.

    The compute function only reads ``close``, but the row is shaped like
    the real market-data dataframe so callers don't have to special-case.
    """
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        }
    )


def _linear(start: float, end: float, n: int) -> list[float]:
    """Monotonic linear close series — deterministic, easy to reason about."""
    step = (end - start) / max(1, n - 1)
    return [start + step * i for i in range(n)]


# ---------------------------------------------------------------------------
# Insufficient-data fallbacks
# ---------------------------------------------------------------------------


def test_returns_none_when_df_too_short() -> None:
    """Fewer than the short window (5 bars) → unrenderable."""
    view = compute_mtf_view(_df([100.0, 101.0, 102.0, 103.0]), "AAPL")
    assert view is None


def test_returns_none_when_df_is_empty() -> None:
    view = compute_mtf_view(pd.DataFrame(), "AAPL")
    assert view is None


def test_returns_none_when_no_close_column() -> None:
    df = pd.DataFrame({"open": [1.0] * 10, "high": [1.0] * 10})
    view = compute_mtf_view(df, "AAPL")
    assert view is None


def test_short_df_uses_available_bars_for_long_horizon() -> None:
    """30 bars: long horizon (60) falls back to 30; medium + short fit."""
    closes = _linear(100.0, 110.0, 30)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.short.bars_used == 5
    assert view.medium.bars_used == 20
    assert view.long.bars_used == 30


def test_short_df_at_exact_minimum() -> None:
    """5 bars (short minimum): all three horizons use those 5 bars."""
    closes = _linear(100.0, 105.0, 5)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.short.bars_used == 5
    assert view.medium.bars_used == 5
    assert view.long.bars_used == 5


def test_full_window_uses_each_horizons_canonical_size() -> None:
    """60+ bars: short uses 5, medium 20, long 60."""
    closes = _linear(100.0, 110.0, 80)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.short.bars_used == 5
    assert view.medium.bars_used == 20
    assert view.long.bars_used == 60


# ---------------------------------------------------------------------------
# Direction classification
# ---------------------------------------------------------------------------


def test_aligned_bullish_when_all_horizons_rise_steeply() -> None:
    """Steady +40% over 60 bars: last 5 bars +2.5%, last 20 +10%, last 60
    +40% — all comfortably past their neutral thresholds (1.5/3/5%)."""
    closes = _linear(100.0, 140.0, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.short.direction == "bullish"
    assert view.medium.direction == "bullish"
    assert view.long.direction == "bullish"
    assert view.alignment == "aligned_bullish"


def test_aligned_bearish_when_all_horizons_fall_steeply() -> None:
    """Symmetric to the bullish case — same move, opposite sign."""
    closes = _linear(140.0, 100.0, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.short.direction == "bearish"
    assert view.medium.direction == "bearish"
    assert view.long.direction == "bearish"
    assert view.alignment == "aligned_bearish"


def test_neutral_direction_when_pct_change_below_threshold() -> None:
    """+0.5% over 60 bars: under every neutral threshold (1.5 / 3 / 5)."""
    closes = _linear(100.0, 100.5, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.short.direction == "neutral"
    assert view.medium.direction == "neutral"
    assert view.long.direction == "neutral"
    assert view.alignment == "mixed"


def test_short_term_reversal_against_long_uptrend() -> None:
    """Long uptrend (+15%), then sharp pullback at the end.

    Last 5 bars drop ~3% — short bearish; first 55 bars climb steadily
    making long bullish. Trader's textbook 'short-term pullback in an
    uptrend' read.
    """
    closes = _linear(100.0, 115.0, 55) + _linear(115.0, 111.5, 5)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.short.direction == "bearish"
    assert view.long.direction == "bullish"
    assert view.alignment == "short_reversal"


def test_short_term_reversal_against_long_downtrend() -> None:
    closes = _linear(120.0, 102.0, 55) + _linear(102.0, 105.5, 5)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.short.direction == "bullish"
    assert view.long.direction == "bearish"
    assert view.alignment == "short_reversal"


def test_mixed_alignment_when_one_horizon_is_neutral() -> None:
    """Strong long-term trend, but the last 5 bars sit flat at the top
    (less than 1.5% range). Long bullish, short neutral → mixed."""
    closes = _linear(100.0, 115.0, 55) + [115.0] * 5
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.long.direction == "bullish"
    assert view.short.direction == "neutral"
    # Short is neutral so it cannot be a "short_reversal" (requires
    # short and long to be opposite non-neutrals).
    assert view.alignment == "mixed"


# ---------------------------------------------------------------------------
# Strength scoring
# ---------------------------------------------------------------------------


def test_strength_saturates_at_one_for_extreme_moves() -> None:
    """+50% over 60 bars: well past long saturation (20%)."""
    closes = _linear(100.0, 150.0, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.long.strength == pytest.approx(1.0)


def test_strength_in_range_for_moderate_moves() -> None:
    """+10% over 60 bars: half of long saturation → strength 0.5."""
    closes = _linear(100.0, 110.0, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert 0.4 < view.long.strength < 0.6


def test_strength_is_zero_for_zero_delta() -> None:
    closes = [100.0] * 60
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.short.strength == 0.0
    assert view.medium.strength == 0.0
    assert view.long.strength == 0.0


def test_strength_uses_absolute_value_for_bearish_moves() -> None:
    """-20% should saturate just like +20%."""
    closes = _linear(120.0, 96.0, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.long.strength == pytest.approx(1.0)


def test_delta_pct_signed_correctly() -> None:
    """+10% rise → +10 delta_pct."""
    closes = _linear(100.0, 110.0, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.long.delta_pct == pytest.approx(10.0, abs=0.5)


def test_delta_pct_negative_for_decline() -> None:
    closes = _linear(110.0, 100.0, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.long.delta_pct < 0


# ---------------------------------------------------------------------------
# Alignment strength
# ---------------------------------------------------------------------------


def test_alignment_strength_averages_three_horizons_when_aligned() -> None:
    """Steady +40% rise gives all three horizons bullish direction;
    alignment strength is the mean of the three horizon strengths."""
    closes = _linear(100.0, 140.0, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    expected = (view.short.strength + view.medium.strength + view.long.strength) / 3
    assert view.alignment_strength == pytest.approx(expected)


def test_alignment_strength_averages_only_short_and_long_for_reversal() -> None:
    closes = _linear(100.0, 115.0, 55) + _linear(115.0, 111.5, 5)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    expected = (view.short.strength + view.long.strength) / 2
    assert view.alignment_strength == pytest.approx(expected)


def test_alignment_strength_zero_when_all_neutral() -> None:
    closes = [100.0] * 60
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.alignment_strength == 0.0


# ---------------------------------------------------------------------------
# Convenience properties
# ---------------------------------------------------------------------------


def test_is_aligned_property() -> None:
    closes = _linear(100.0, 140.0, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.is_aligned is True
    assert view.is_conflicting is False


def test_is_conflicting_property() -> None:
    closes = _linear(100.0, 115.0, 55) + _linear(115.0, 111.5, 5)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.is_conflicting is True
    assert view.is_aligned is False


# ---------------------------------------------------------------------------
# Schema invariants — render code relies on these
# ---------------------------------------------------------------------------


def test_horizon_labels_are_constrained() -> None:
    closes = _linear(100.0, 110.0, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    assert view.short.horizon == "short"
    assert view.medium.horizon == "medium"
    assert view.long.horizon == "long"


def test_direction_labels_constrained_to_three_values() -> None:
    closes = _linear(100.0, 115.0, 55) + _linear(115.0, 111.5, 5)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    for t in (view.short, view.medium, view.long):
        assert t.direction in ("bullish", "bearish", "neutral")


def test_alignment_labels_constrained_to_four_values() -> None:
    """Sample a few scenarios; assert alignment is always one of the four."""
    scenarios = [
        _linear(100.0, 120.0, 60),  # aligned bull
        _linear(120.0, 100.0, 60),  # aligned bear
        _linear(100.0, 115.0, 55) + _linear(115.0, 111.5, 5),  # short_reversal
        [100.0] * 60,  # mixed (all neutral)
        _linear(100.0, 115.0, 55) + [115.0] * 5,  # mixed (short neutral)
    ]
    for closes in scenarios:
        view = compute_mtf_view(_df(closes), "AAPL")
        assert view is not None
        assert view.alignment in (
            "aligned_bullish",
            "aligned_bearish",
            "short_reversal",
            "mixed",
        )


def test_view_is_frozen() -> None:
    closes = _linear(100.0, 110.0, 60)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
        view.alignment = "anything else"  # type: ignore[misc]


def test_view_carries_symbol_unchanged() -> None:
    closes = _linear(100.0, 110.0, 60)
    view = compute_mtf_view(_df(closes), "TSLA")
    assert view is not None
    assert view.symbol == "TSLA"


def test_strength_field_in_unit_interval() -> None:
    closes = _linear(100.0, 200.0, 60)  # +100%, well past saturation
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    for t in (view.short, view.medium, view.long):
        assert 0.0 <= t.strength <= 1.0
    assert 0.0 <= view.alignment_strength <= 1.0


# ---------------------------------------------------------------------------
# Pure-function contract
# ---------------------------------------------------------------------------


def test_compute_is_pure_same_inputs_same_output() -> None:
    closes = _linear(100.0, 115.0, 60)
    a = compute_mtf_view(_df(closes), "AAPL")
    b = compute_mtf_view(_df(closes), "AAPL")
    assert a == b


def test_zero_start_price_does_not_crash() -> None:
    """Defensive: a degenerate start=0 (shouldn't happen with real
    equities) should not divide-by-zero — returns 0% delta."""
    closes = [0.0] + _linear(0.0, 10.0, 59)
    closes[0] = 0.0
    view = compute_mtf_view(_df(closes), "BUG")
    assert view is not None
    assert math.isfinite(view.long.delta_pct)


# ---------------------------------------------------------------------------
# Anti-hallucination — string fields carry no prediction language
# ---------------------------------------------------------------------------


_FORBIDDEN_WORDS = (
    "likely",
    "will rise",
    "will fall",
    "expected to",
    "forecast",
    "predict",
)


def test_no_prediction_language_in_string_fields() -> None:
    """The module exposes only structured labels — none of them should
    contain prediction language. Same regression guard pattern used in
    pulse / opportunities / tier."""
    closes = _linear(100.0, 115.0, 55) + _linear(115.0, 111.5, 5)
    view = compute_mtf_view(_df(closes), "AAPL")
    assert view is not None
    strings = (
        view.alignment,
        view.short.direction,
        view.short.horizon,
        view.medium.direction,
        view.medium.horizon,
        view.long.direction,
        view.long.horizon,
    )
    joined = " ".join(strings).lower()
    for word in _FORBIDDEN_WORDS:
        assert word not in joined, f"forbidden word {word!r} in {joined!r}"


def test_dataclass_field_set_is_stable() -> None:
    """If a future change adds a field to TimeframeTrend or
    MultiTimeframeView, the dashboard render code must be updated to
    handle it. This test fails to draw attention to that."""
    tf_fields = set(TimeframeTrend.__dataclass_fields__.keys())
    assert tf_fields == {
        "horizon",
        "direction",
        "strength",
        "delta_pct",
        "bars_used",
    }
    view_fields = set(MultiTimeframeView.__dataclass_fields__.keys())
    assert view_fields == {
        "symbol",
        "short",
        "medium",
        "long",
        "alignment",
        "alignment_strength",
    }
