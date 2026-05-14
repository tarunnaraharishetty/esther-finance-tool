"""Trajectory-level read of the pulse history.

Single-tick :class:`~src.intelligence.pulse.MarketPulse` answers "what
is the mood right now?" The rolling
:class:`~src.intelligence.pulse_history.PulseHistory` answers "is the
mood building or fading?" This module synthesizes both:

* **Regime classifier** — collapses recent ticks into one of four
  fixed labels (``risk-on`` / ``risk-off`` / ``mixed`` /
  ``indeterminate``) based on the directional tilt of the
  bullish/bearish counts across the window.
* **Pattern detectors** — observational phrases like "momentum
  breadth firming across last 6 ticks" or "alert intensity spiking".
  Each fires from a hard numeric gate over half-window means; no
  forecasts, no inferred causation.

Pure data, no I/O, no LLM. The output feeds the dashboard header
(short chip rendering) and the recap-brief input (structured prose
context). The LLM never sees raw history numbers — it consumes
deterministic labels this module already grounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.intelligence.pulse_history import PulseHistory


# Minimum history length before regime classification is meaningful.
# Two ticks is enough to render a sparkline but too thin for a regime
# call; three lets the half-window math operate on at least one entry
# per half.
_MIN_REGIME_TICKS = 3

# Window over which the trajectory is read. We never use more than this
# many recent ticks even if the history holds more — the trader cares
# about "what is happening now" not "what happened 20 ticks ago".
_TRAJECTORY_WINDOW = 6

# Regime classification gates. Tilt is the normalized signed ratio of
# (bull - bear) / (bull + bear) across the window. ±0.4 separates
# "directional" from "mixed".
_REGIME_RISK_ON_TILT = 0.40
_REGIME_RISK_OFF_TILT = -0.40

# Pattern detection gates. Firming/fading compares mean of the latter
# half vs the former half of the window. ``BREADTH_DELTA_FLOOR`` is a
# 0-1 scale move; ``SPIKE_RATIO`` and ``CLUSTER_FLOOR`` operate on raw
# integer counts.
_BREADTH_DELTA_FLOOR = 0.15
_SPIKE_RATIO = 2.0
_SPIKE_LATE_FLOOR = 3
_CLUSTER_LATE_FLOOR = 3
_CLUSTER_RATIO = 2.0
_MIN_PATTERN_TICKS = 4  # need 2 + 2 across the half-window split


@dataclass(frozen=True)
class TrajectoryPattern:
    """One observed pattern over the pulse trajectory.

    ``name`` is a stable machine identifier (used in cache signatures
    and tests). ``label`` is the human-readable phrase the dashboard
    surfaces. ``detail`` carries the numeric grounding ("across last
    N ticks", "now N vs prior M") so the chip is self-justifying.
    """

    name: str
    label: str
    detail: str


@dataclass(frozen=True)
class PulseEvolution:
    """Synthesized read of the pulse trajectory.

    ``regime`` is always one of the four fixed labels — the renderer
    branches on it for color, the recap LLM consumes it as prose
    context. ``patterns`` is empty when nothing crosses its gate;
    quiet sessions stay quiet.
    """

    regime: str  # "risk-on" | "risk-off" | "mixed" | "indeterminate"
    patterns: tuple[TrajectoryPattern, ...]


def evolve(history: PulseHistory | None) -> PulseEvolution:
    """Derive regime + patterns from the rolling pulse history.

    Returns a ``PulseEvolution`` with ``regime="indeterminate"`` and
    no patterns when history is missing or has fewer than the
    classification floor. Callers can render unconditionally —
    "indeterminate" + empty patterns means "skip the chip".
    """
    if history is None or history.length < _MIN_REGIME_TICKS:
        return PulseEvolution(regime="indeterminate", patterns=())
    regime = _classify_regime(history)
    patterns = _detect_patterns(history)
    return PulseEvolution(regime=regime, patterns=patterns)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


def _classify_regime(history: PulseHistory) -> str:
    """One of ``risk-on`` / ``risk-off`` / ``mixed`` / ``indeterminate``.

    Computes the net bull/bear tilt across the trajectory window. A
    zero-directional window (every tick HOLD-heavy with no directional
    counts) returns ``indeterminate`` rather than ``mixed`` — there's
    no signal either way to call mixed.
    """
    window = min(history.length, _TRAJECTORY_WINDOW)
    recent_bull = sum(history.bullish_count[-window:])
    recent_bear = sum(history.bearish_count[-window:])
    total = recent_bull + recent_bear
    if total == 0:
        return "indeterminate"
    tilt = (recent_bull - recent_bear) / total
    if tilt >= _REGIME_RISK_ON_TILT:
        return "risk-on"
    if tilt <= _REGIME_RISK_OFF_TILT:
        return "risk-off"
    return "mixed"


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def _detect_patterns(history: PulseHistory) -> tuple[TrajectoryPattern, ...]:
    """Run each pattern detector in sequence; return the firing ones.

    Order is intentional — strongest signals first so the renderer
    can show the most informative chip when limited by horizontal
    space.
    """
    if history.length < _MIN_PATTERN_TICKS:
        return ()
    window = min(history.length, _TRAJECTORY_WINDOW)
    out: list[TrajectoryPattern] = []

    # Momentum breadth — firming or fading.
    momentum_pattern = _firming_or_fading_pattern(
        history.momentum_breadth, window, axis_label="momentum breadth"
    )
    if momentum_pattern is not None:
        out.append(momentum_pattern)

    # Sentiment breadth — same shape but on the sentiment series.
    sentiment_pattern = _firming_or_fading_pattern(
        history.sentiment_breadth, window, axis_label="sentiment breadth"
    )
    if sentiment_pattern is not None:
        out.append(sentiment_pattern)

    # Alert intensity spike — late max materially above early max.
    spike = _spike_pattern(
        history.alert_intensity,
        window,
        name="alert_intensity_spiking",
        label="alert intensity spiking",
    )
    if spike is not None:
        out.append(spike)

    # Reversal cluster — late sum elevated above early sum.
    cluster = _cluster_pattern(
        history.reversal_intensity,
        window,
        name="reversal_cluster",
        label="reversal cluster",
    )
    if cluster is not None:
        out.append(cluster)

    return tuple(out)


def _firming_or_fading_pattern(
    series: tuple[float, ...],
    window: int,
    *,
    axis_label: str,
) -> TrajectoryPattern | None:
    """Return ``"<axis> firming"`` or ``"<axis> fading"`` when the
    latter half of the window meaningfully outpaces the former half
    (or vice versa). ``None`` when the delta sits inside the floor.
    """
    early_mean, late_mean = _half_window_means(series, window)
    if early_mean is None or late_mean is None:
        return None
    delta = late_mean - early_mean
    if delta >= _BREADTH_DELTA_FLOOR:
        return TrajectoryPattern(
            name=f"{axis_label.replace(' ', '_')}_firming",
            label=f"{axis_label} firming",
            detail=f"across last {window} ticks",
        )
    if delta <= -_BREADTH_DELTA_FLOOR:
        return TrajectoryPattern(
            name=f"{axis_label.replace(' ', '_')}_fading",
            label=f"{axis_label} fading",
            detail=f"across last {window} ticks",
        )
    return None


def _spike_pattern(
    series: tuple[int, ...],
    window: int,
    *,
    name: str,
    label: str,
) -> TrajectoryPattern | None:
    """Return a spike pattern when the latter-half max is materially
    above the former-half max AND clears the absolute floor.

    The absolute floor matters because a 1 → 2 jump triples on ratio
    but isn't worth surfacing; the trader cares about meaningful
    counts.
    """
    early_max, late_max = _half_window_maxes(series, window)
    if early_max is None or late_max is None:
        return None
    if late_max < _SPIKE_LATE_FLOOR:
        return None
    if late_max < early_max * _SPIKE_RATIO:
        return None
    return TrajectoryPattern(
        name=name,
        label=label,
        detail=f"latest {late_max} vs prior {early_max}",
    )


def _cluster_pattern(
    series: tuple[int, ...],
    window: int,
    *,
    name: str,
    label: str,
) -> TrajectoryPattern | None:
    """Return a cluster pattern when the latter half's sum is
    meaningfully above the former half's sum AND clears the absolute
    floor."""
    early_sum, late_sum = _half_window_sums(series, window)
    if early_sum is None or late_sum is None:
        return None
    if late_sum < _CLUSTER_LATE_FLOOR:
        return None
    if late_sum < early_sum * _CLUSTER_RATIO:
        return None
    return TrajectoryPattern(
        name=name,
        label=label,
        detail=f"latest {late_sum} vs prior {early_sum}",
    )


# ---------------------------------------------------------------------------
# Half-window aggregations
# ---------------------------------------------------------------------------


def _half_window_means(series: tuple[float, ...], window: int) -> tuple[float | None, float | None]:
    """Return ``(early_mean, late_mean)`` over the trailing ``window``."""
    early, late = _half_window_slices(series, window)
    if not early or not late:
        return None, None
    return sum(early) / len(early), sum(late) / len(late)


def _half_window_maxes(series: tuple[int, ...], window: int) -> tuple[int | None, int | None]:
    early, late = _half_window_slices(series, window)
    if not early or not late:
        return None, None
    return max(early), max(late)


def _half_window_sums(series: tuple[int, ...], window: int) -> tuple[int | None, int | None]:
    early, late = _half_window_slices(series, window)
    if not early or not late:
        return None, None
    return sum(early), sum(late)


def _half_window_slices[NumericT: (int, float)](
    series: tuple[NumericT, ...], window: int
) -> tuple[tuple[NumericT, ...], tuple[NumericT, ...]]:
    """Split the trailing ``window`` of ``series`` into halves.

    Odd window sizes assign the extra element to the early half so
    "late" is always at least as fresh as the half-window boundary.
    Generic over numeric element types so int/float series each
    preserve their element type through the split.
    """
    trailing = series[-window:]
    if len(trailing) < 2:
        return (), ()
    mid = (len(trailing) + 1) // 2
    return tuple(trailing[:mid]), tuple(trailing[mid:])


__all__ = ["PulseEvolution", "TrajectoryPattern", "evolve"]
