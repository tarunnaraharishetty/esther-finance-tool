"""Tests for src.intelligence.pulse_evolution.

Covers the regime classifier, each pattern detector, and the
indeterminate / empty-history edge cases. Inputs are constructed as
PulseHistory dataclasses directly so the tests stay independent of
the controller's recording path.
"""

from __future__ import annotations

from src.intelligence.pulse_evolution import (
    PulseEvolution,
    TrajectoryPattern,
    evolve,
)
from src.intelligence.pulse_history import PulseHistory


def _history(
    *,
    length: int,
    momentum_breadth: tuple[float, ...] | None = None,
    sentiment_breadth: tuple[float, ...] | None = None,
    bullish_count: tuple[int, ...] | None = None,
    bearish_count: tuple[int, ...] | None = None,
    reversal_intensity: tuple[int, ...] | None = None,
    alert_intensity: tuple[int, ...] | None = None,
    window: int = 20,
) -> PulseHistory:
    """Build a PulseHistory of exactly ``length`` ticks with sensible
    defaults for series the test doesn't customize."""
    zeros_f = (0.0,) * length
    zeros_i = (0,) * length
    return PulseHistory(
        momentum_breadth=momentum_breadth or zeros_f,
        sentiment_breadth=sentiment_breadth or zeros_f,
        bullish_count=bullish_count or zeros_i,
        bearish_count=bearish_count or zeros_i,
        reversal_intensity=reversal_intensity or zeros_i,
        alert_intensity=alert_intensity or zeros_i,
        sentiment=("neutral",) * length,
        conviction=("weak",) * length,
        activity=("calm",) * length,
        window=window,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_none_history_returns_indeterminate() -> None:
    """No history at all = indeterminate, no patterns. The renderer
    treats that as 'skip the chip'."""
    result = evolve(None)
    assert result.regime == "indeterminate"
    assert result.patterns == ()


def test_short_history_returns_indeterminate() -> None:
    """Two ticks is enough to render a sparkline but too thin for a
    regime call — the classifier wants at least 3 ticks."""
    history = _history(length=2, bullish_count=(3, 3))
    assert evolve(history).regime == "indeterminate"


def test_history_with_no_directional_counts_is_indeterminate() -> None:
    """Even with enough length, a HOLD-heavy window with zero
    bull/bear counts can't be classified as either side."""
    history = _history(length=6)
    assert evolve(history).regime == "indeterminate"


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


def test_dominant_bullish_window_classifies_risk_on() -> None:
    """When bullish counts dominate the window, the regime is risk-on."""
    history = _history(
        length=6,
        bullish_count=(5, 5, 4, 5, 6, 5),
        bearish_count=(0, 0, 1, 0, 0, 1),
    )
    assert evolve(history).regime == "risk-on"


def test_dominant_bearish_window_classifies_risk_off() -> None:
    """The mirror case — bearish counts dominate → risk-off."""
    history = _history(
        length=6,
        bullish_count=(0, 1, 0, 1, 0, 0),
        bearish_count=(5, 4, 6, 5, 5, 5),
    )
    assert evolve(history).regime == "risk-off"


def test_balanced_window_classifies_mixed() -> None:
    """When bullish and bearish counts sit close to parity, the regime
    is mixed (neither side commands the tilt)."""
    history = _history(
        length=6,
        bullish_count=(3, 3, 3, 3, 3, 3),
        bearish_count=(3, 3, 3, 3, 3, 3),
    )
    assert evolve(history).regime == "mixed"


def test_regime_only_reads_trajectory_window_not_full_history() -> None:
    """Long-history sessions ignore old ticks — the regime reflects
    the recent trajectory only. A bearish opening with a bullish
    recent run reads as risk-on."""
    history = _history(
        length=20,
        # First 14 ticks bearish, last 6 ticks bullish — recent window
        # wins.
        bullish_count=(0,) * 14 + (5, 5, 5, 5, 5, 5),
        bearish_count=(5,) * 14 + (0, 0, 0, 0, 0, 0),
    )
    assert evolve(history).regime == "risk-on"


# ---------------------------------------------------------------------------
# Pattern: firming / fading
# ---------------------------------------------------------------------------


def _pattern_names(evolution: PulseEvolution) -> tuple[str, ...]:
    return tuple(p.name for p in evolution.patterns)


def test_momentum_breadth_firming_pattern() -> None:
    """A rising-mean momentum series fires the firming pattern with
    grounding text containing the window length."""
    history = _history(
        length=6,
        bullish_count=(3, 3, 3, 3, 3, 3),  # keep regime out of indeterminate
        momentum_breadth=(0.1, 0.15, 0.2, 0.6, 0.65, 0.7),
    )
    evolution = evolve(history)
    assert "momentum_breadth_firming" in _pattern_names(evolution)
    firming = next(p for p in evolution.patterns if p.name == "momentum_breadth_firming")
    assert "last 6 ticks" in firming.detail


def test_momentum_breadth_fading_pattern() -> None:
    """Mirror case: a falling-mean series fires fading."""
    history = _history(
        length=6,
        bullish_count=(3, 3, 3, 3, 3, 3),
        momentum_breadth=(0.7, 0.65, 0.6, 0.2, 0.15, 0.1),
    )
    evolution = evolve(history)
    assert "momentum_breadth_fading" in _pattern_names(evolution)


def test_no_breadth_pattern_when_delta_inside_floor() -> None:
    """A flat-ish series (delta < 0.15) shouldn't fire either firming
    or fading — the trader doesn't want chips on noise."""
    history = _history(
        length=6,
        bullish_count=(3, 3, 3, 3, 3, 3),
        momentum_breadth=(0.40, 0.42, 0.41, 0.44, 0.43, 0.45),
    )
    evolution = evolve(history)
    names = _pattern_names(evolution)
    assert "momentum_breadth_firming" not in names
    assert "momentum_breadth_fading" not in names


def test_sentiment_breadth_firming_pattern() -> None:
    """Sentiment series fires its own dedicated label."""
    history = _history(
        length=6,
        bullish_count=(3, 3, 3, 3, 3, 3),
        sentiment_breadth=(0.2, 0.2, 0.25, 0.6, 0.65, 0.7),
    )
    evolution = evolve(history)
    assert "sentiment_breadth_firming" in _pattern_names(evolution)


# ---------------------------------------------------------------------------
# Pattern: alert spike
# ---------------------------------------------------------------------------


def test_alert_intensity_spike_pattern() -> None:
    """A late-window alert count well above the early max (and above
    the absolute floor of 3) fires the spike pattern with numeric
    detail grounding."""
    history = _history(
        length=6,
        bullish_count=(3, 3, 3, 3, 3, 3),
        alert_intensity=(0, 0, 1, 4, 5, 6),
    )
    evolution = evolve(history)
    assert "alert_intensity_spiking" in _pattern_names(evolution)
    spike = next(p for p in evolution.patterns if p.name == "alert_intensity_spiking")
    # Detail mentions raw counts so the chip is self-justifying.
    assert "6" in spike.detail


def test_alert_spike_skipped_below_absolute_floor() -> None:
    """A small ratio (0 → 2) clears the ratio gate but not the
    absolute floor — we don't want chips on 2-alert blips."""
    history = _history(
        length=6,
        bullish_count=(3, 3, 3, 3, 3, 3),
        alert_intensity=(0, 0, 0, 0, 0, 2),
    )
    evolution = evolve(history)
    assert "alert_intensity_spiking" not in _pattern_names(evolution)


# ---------------------------------------------------------------------------
# Pattern: reversal cluster
# ---------------------------------------------------------------------------


def test_reversal_cluster_pattern() -> None:
    """A late-window cluster of reversals (sum >= 3, late > 2x early)
    fires the cluster pattern."""
    history = _history(
        length=6,
        bullish_count=(3, 3, 3, 3, 3, 3),
        reversal_intensity=(0, 1, 0, 2, 2, 3),
    )
    evolution = evolve(history)
    assert "reversal_cluster" in _pattern_names(evolution)


def test_reversal_cluster_skipped_below_floor() -> None:
    """Below the cluster floor (sum 3) the pattern stays quiet even
    when the ratio test passes."""
    history = _history(
        length=6,
        bullish_count=(3, 3, 3, 3, 3, 3),
        reversal_intensity=(0, 0, 0, 0, 1, 1),
    )
    evolution = evolve(history)
    assert "reversal_cluster" not in _pattern_names(evolution)


# ---------------------------------------------------------------------------
# Multi-pattern co-firing
# ---------------------------------------------------------------------------


def test_multiple_patterns_co_fire_when_independent_gates_hold() -> None:
    """A risk-on window with simultaneously rising momentum breadth
    AND an alert spike should produce both patterns in the output."""
    history = _history(
        length=6,
        bullish_count=(4, 4, 4, 5, 5, 5),
        bearish_count=(0, 0, 0, 0, 0, 0),
        momentum_breadth=(0.1, 0.15, 0.2, 0.6, 0.7, 0.8),
        alert_intensity=(0, 0, 0, 4, 4, 5),
    )
    evolution = evolve(history)
    names = _pattern_names(evolution)
    assert "momentum_breadth_firming" in names
    assert "alert_intensity_spiking" in names
    assert evolution.regime == "risk-on"


def test_pattern_objects_carry_label_and_detail() -> None:
    """The dataclass exposes both a machine name and a human label
    + numeric detail string — the dashboard renders label + detail
    inline, the recap stitches `<label> (<detail>)`."""
    pattern = TrajectoryPattern(
        name="x", label="momentum breadth firming", detail="across last 6 ticks"
    )
    assert pattern.label == "momentum breadth firming"
    assert pattern.detail == "across last 6 ticks"
