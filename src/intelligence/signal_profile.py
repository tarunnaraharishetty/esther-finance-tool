"""Three-axis signal classification — "trend or noise?" at a glance.

Answers the discretionary trader's first-screen question: is this
symbol's signal worth following, or is it choppy noise that should be
discounted?

Pure data. Recomputed per tick from a row + its
:class:`~src.intelligence.history.SignalHistorySummary`. No new state
on the controller, no new field on the snapshot — the dashboard
computes profiles on render where it needs them.

Three orthogonal axes, each binary or trinary:

* ``stability``    — ``"stable"`` vs ``"noisy"``. Collapses the
  promote_to_tier 3-tier (stable / moderate / volatile) into the
  trader-relevant binary "can I trust this?" read.
* ``trend``        — ``"strengthening"`` / ``"weakening"`` / ``"flat"``.
  Derived from confidence delta within the current episode. Short
  episodes (under 3 ticks) default to "flat" — not enough data yet.
* ``persistence``  — ``"persistent"`` vs ``"flipping"``. Persistent
  means the symbol's been in 1-2 episodes this session; flipping
  means 3+.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dashboard.state import RecommendationRow
    from src.intelligence.history import SignalHistorySummary


# Calibration — inlined at the call site for visibility.
_TREND_RISING_DELTA = 0.10  # min positive conf delta to call "strengthening"
_TREND_FALLING_DELTA = -0.10  # max negative conf delta to call "weakening"
_TREND_MIN_TICKS = 3  # episode must run this many ticks before we judge trend
_PERSISTENCE_MAX_EPISODES = 2  # 1-2 episodes = persistent; 3+ = flipping


@dataclass(frozen=True)
class SignalProfile:
    """Three-axis classification of a single symbol's signal.

    Stable schema so the dashboard render code never branches on
    values it doesn't expect.
    """

    stability: str  # "stable" | "noisy"
    trend: str  # "strengthening" | "weakening" | "flat"
    persistence: str  # "persistent" | "flipping"


def compute_signal_profile(
    row: RecommendationRow,
    history: SignalHistorySummary | None,
) -> SignalProfile:
    """Derive a :class:`SignalProfile` from a row + its history.

    Both inputs are already on the snapshot; this function does no
    side-effects and reads no global state.
    """
    return SignalProfile(
        stability=_classify_stability(row),
        trend=_classify_trend(history),
        persistence=_classify_persistence(history),
    )


def _classify_stability(row: RecommendationRow) -> str:
    """Collapse promote_to_tier's 3-tier stability into the binary
    trader read.

    ``row.stability`` is already populated by
    :func:`src.intelligence.tier.promote_to_tier`. We treat anything
    short of "stable" as noise — moderate is too uncertain for the
    "follow it" question, volatile is obviously noise.
    """
    return "stable" if row.stability == "stable" else "noisy"


def _classify_trend(history: SignalHistorySummary | None) -> str:
    """Confidence trend within the current episode.

    Returns ``"flat"`` when there's no history or the current episode
    is too young to judge. Otherwise uses the confidence delta.
    """
    if history is None or history.current.tick_count < _TREND_MIN_TICKS:
        return "flat"
    delta = history.current.confidence_last - history.current.confidence_first
    if delta >= _TREND_RISING_DELTA:
        return "strengthening"
    if delta <= _TREND_FALLING_DELTA:
        return "weakening"
    return "flat"


def _classify_persistence(history: SignalHistorySummary | None) -> str:
    """Persistence = how many episodes this session.

    1-2 episodes counts as persistent (the current run is the only
    one, or there's one prior — settled-ish behavior). 3+ episodes
    means the symbol's bouncing.

    Without history (first tick), default to "persistent" — we have
    no evidence of flipping yet.
    """
    if history is None:
        return "persistent"
    episode_count = 1 + len(history.recent)
    return (
        "persistent"
        if episode_count <= _PERSISTENCE_MAX_EPISODES
        else "flipping"
    )


__all__ = ["SignalProfile", "compute_signal_profile"]
