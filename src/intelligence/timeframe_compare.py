"""Multi-timeframe comparison — daily vs intraday stance per symbol.

Pure data. Consumes the snapshot's daily ``rows`` + ``signal_history``
and the parallel ``intraday_signal_history`` to produce a fixed-label
``TimeframeStance`` for each symbol that has data on both sides.

The dashboard surfaces these as an ALIGN chip in the header and the
side-by-side Daily / Intraday Intelligence blocks in the DetailPanel.
The recap brief consumes them as structured prose context.

Every category label is deterministic — derived from action-pair
direction comparison + a small set of numeric gates over the intraday
history. No predictions, no inferred causation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.strategy.base import SignalAction

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot


# Confidence-rise thresholds for the descriptor phrases. Aligned with
# the existing pulse-evolution and signal-profile gates so the dashboard
# uses one shared notion of "rising / falling fast".
_INTRADAY_RISING_DELTA = 0.10
_INTRADAY_FALLING_DELTA = -0.10
_MIN_HISTORY_TICKS = 2


@dataclass(frozen=True)
class TimeframeStance:
    """Per-symbol read of daily-vs-intraday alignment.

    ``category`` is one of:

    * ``"aligned_bullish"`` — both timeframes BUY.
    * ``"aligned_bearish"`` — both timeframes SELL.
    * ``"conflict"`` — directional disagreement (daily BUY ↔ intraday
      SELL, or vice versa).
    * ``"intraday_only"`` — daily is HOLD, intraday is directional.
    * ``"daily_only"`` — daily is directional, intraday is HOLD.
    * ``"neutral"`` — both HOLD.

    ``phrases`` carries optional observational descriptors keyed off
    intraday-history deltas — e.g. ``"strengthening intraday
    momentum"`` when intraday confidence has been rising inside the
    current intraday episode.

    ``alignment_label`` is the trader-facing summary string — one of
    a fixed set: ``"aligned bullish"`` / ``"aligned bearish"`` /
    ``"strengthening continuation"`` / ``"momentum conflict"`` /
    ``"intraday reversal"`` / ``"short-term pullback"`` /
    ``"intraday only"`` / ``"neutral"``. Deterministic mapping from
    ``(category, phrases)`` — see :func:`_alignment_label`.
    """

    symbol: str
    daily_action: SignalAction
    intraday_action: SignalAction
    category: str
    phrases: tuple[str, ...] = ()
    alignment_label: str = "neutral"


_DIRECTIONAL = {SignalAction.BUY, SignalAction.SELL}


def _classify(daily: SignalAction, intraday: SignalAction) -> str:
    if daily == intraday and daily == SignalAction.BUY:
        return "aligned_bullish"
    if daily == intraday and daily == SignalAction.SELL:
        return "aligned_bearish"
    if daily in _DIRECTIONAL and intraday in _DIRECTIONAL and daily != intraday:
        return "conflict"
    if daily == SignalAction.HOLD and intraday in _DIRECTIONAL:
        return "intraday_only"
    if daily in _DIRECTIONAL and intraday == SignalAction.HOLD:
        return "daily_only"
    return "neutral"


def compare_timeframes(snapshot: DashboardSnapshot) -> dict[str, TimeframeStance]:
    """Return a per-symbol ``TimeframeStance`` for every row with an
    intraday read.

    Rows without an ``IntradayRead`` are omitted — there's no
    intraday side to compare against. Error rows are also omitted.
    """
    out: dict[str, TimeframeStance] = {}
    for row in snapshot.rows:
        if row.error or row.intraday is None:
            continue
        category = _classify(row.action, row.intraday.action)
        phrases = _phrases_for(row, category, snapshot)
        out[row.symbol] = TimeframeStance(
            symbol=row.symbol,
            daily_action=row.action,
            intraday_action=row.intraday.action,
            category=category,
            phrases=phrases,
            alignment_label=_alignment_label(category, phrases),
        )
    return out


def _alignment_label(category: str, phrases: tuple[str, ...]) -> str:
    """Deterministic mapping from ``(category, phrases)`` to a fixed
    trader-facing label. Order matters — more specific labels
    (combinations of category + phrase) win over the generic
    category-only labels.
    """
    strengthening = "strengthening intraday momentum" in phrases
    intraday_flip = "intraday reversal against daily trend" in phrases
    if category in ("aligned_bullish", "aligned_bearish") and strengthening:
        return "strengthening continuation"
    if category == "aligned_bullish":
        return "aligned bullish"
    if category == "aligned_bearish":
        return "aligned bearish"
    if category == "conflict" and intraday_flip:
        return "intraday reversal"
    if category == "conflict":
        return "momentum conflict"
    if category == "daily_only":
        return "short-term pullback"
    if category == "intraday_only":
        return "intraday only"
    return "neutral"


def _phrases_for(
    row: object,
    category: str,
    snapshot: DashboardSnapshot,
) -> tuple[str, ...]:
    """Observational descriptors derived from intraday history deltas.

    Phrases describe the *current measurable state* — "strengthening",
    "weakening", "reversal against daily trend". No forecasts.
    """
    symbol = row.symbol  # type: ignore[attr-defined]
    intraday_summary = snapshot.intraday_signal_history.get(symbol)
    phrases: list[str] = []
    if intraday_summary is None or intraday_summary.current.tick_count < _MIN_HISTORY_TICKS:
        return ()
    delta = intraday_summary.current.confidence_last - intraday_summary.current.confidence_first
    if delta >= _INTRADAY_RISING_DELTA:
        phrases.append("strengthening intraday momentum")
    elif delta <= _INTRADAY_FALLING_DELTA:
        phrases.append("weakening intraday conviction")
    # "Reversal against daily trend" fires when the intraday side just
    # flipped (most-recent intraday episode is brief AND the daily
    # action is non-HOLD opposite of the new intraday direction).
    if (
        category == "conflict"
        and intraday_summary.current.tick_count <= 3
        and intraday_summary.recent
    ):
        prior = intraday_summary.recent[0]
        if prior.action != intraday_summary.current.action:
            phrases.append("intraday reversal against daily trend")
    return tuple(phrases)


def aggregate_counts(stances: dict[str, TimeframeStance]) -> dict[str, int]:
    """Roll the per-symbol stances up into category counts for the
    ALIGN header chip."""
    counts: dict[str, int] = {}
    for stance in stances.values():
        counts[stance.category] = counts.get(stance.category, 0) + 1
    return counts


__all__ = [
    "TimeframeStance",
    "aggregate_counts",
    "alignment_label",
    "compare_timeframes",
]


# Re-export the helper under a public name so downstream tests can
# exercise the label mapping directly without going through a full
# snapshot.
def alignment_label(category: str, phrases: tuple[str, ...] = ()) -> str:
    return _alignment_label(category, phrases)
