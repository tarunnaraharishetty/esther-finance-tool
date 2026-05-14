"""Multi-timeframe intelligence — secondary intraday read alongside daily.

The full pipeline (RecommendationEngine, signal history, opportunities,
pulse, alerts) runs on a single timeframe — daily bars by default.
This module supplies a small, technical-only "intraday read" that
runs alongside per tick so the trader sees alignment vs divergence
between the daily recommendation and an intraday view:

    DAILY     BUY  conf 0.78
    INTRADAY  HOLD conf 0.32   ← divergent — heads up

The intraday read is intentionally narrower than the daily one:

* Indicators only — sentiment is timeframe-agnostic (news weighting is
  the same regardless of bar timeframe), so re-running it would
  duplicate work without changing the result.
* No alerts / opportunities / pulse — Phase 1 surfaces alignment
  context only; full intraday trackers are a Phase 2 lift that
  requires a SessionStore schema bump and parallel state.
* Pure compute over the existing indicator scoring path — no new
  Alpaca calls beyond the second bar fetch the controller does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.data.models import TimeFrame
from src.strategy.base import SignalAction

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class IntradayRead:
    """One symbol's intraday read at a specific timeframe.

    Carries only what the renderer needs to communicate "what does the
    intraday view say, and how does it compare to the daily action?"
    All fields derive from the same indicator-scoring path the daily
    pipeline uses — only the input dataframe differs.
    """

    timeframe: TimeFrame
    action: SignalAction
    confidence: float  # [0, 1]
    combined_score: float  # [-1, 1]
    technical_score: float  # [-1, 1]


def is_divergent(
    daily_action: SignalAction,
    intraday: IntradayRead | None,
) -> bool:
    """True when the intraday read materially disagrees with the daily.

    Materially: BUY vs SELL (sign flip). BUY-vs-HOLD or SELL-vs-HOLD
    is *not* counted — HOLD on either side means "no directional read",
    which doesn't divergence-conflict with a directional read on the
    other timeframe.

    Returns ``False`` when ``intraday`` is ``None`` — symbols without
    an intraday read can't diverge by definition.
    """
    if intraday is None:
        return False
    if daily_action == SignalAction.HOLD or intraday.action == SignalAction.HOLD:
        return False
    return daily_action != intraday.action


__all__ = ["IntradayRead", "is_divergent"]
