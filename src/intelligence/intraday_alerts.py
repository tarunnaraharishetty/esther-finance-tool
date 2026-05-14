"""Intraday-specific alert rules (MT2 phase 2c).

Five rules that fire from the per-row ``IntradayRead`` + parallel
intraday signal/opp/pulse trackers. They flow through the existing
:class:`~src.intelligence.alerts.AlertEngine` and
:class:`~src.intelligence.alert_prioritizer.AlertPrioritizer`
unchanged — these are just new ``Rule`` / ``SnapshotRule``
implementations.

All rules are deterministic:

* Each gate is a numeric comparison over existing fields.
* Messages are template strings with numeric facts.
* No forecast words ("will", "expected", "likely") anywhere.

Cooldowns are added to ``_DEFAULT_COOLDOWNS`` in
:mod:`src.intelligence.alert_prioritizer` so each rule debounces
sensibly out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.intelligence.alerts import Alert
from src.strategy.base import SignalAction

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot, RecommendationRow


# ---------------------------------------------------------------------------
# Per-row rules
# ---------------------------------------------------------------------------


@dataclass
class IntradayReversalAccelerationRule:
    """Fires when intraday action just flipped AND intraday confidence
    rose meaningfully across the new (current) intraday episode.

    The "just flipped" check is by-direction: previous row's intraday
    action differs from current's. The acceleration check uses the
    intraday history's current-episode confidence delta to confirm
    the flip is gaining steam rather than fading.
    """

    delta: float = 0.15
    severity: str = "warn"
    name: str = "intraday_reversal_acceleration"

    def evaluate(
        self,
        current: RecommendationRow,
        previous: RecommendationRow | None,
    ) -> Alert | None:
        if (
            previous is None
            or current.error
            or previous.error
            or current.intraday is None
            or previous.intraday is None
        ):
            return None
        if current.intraday.action == previous.intraday.action:
            return None
        if current.intraday.action == SignalAction.HOLD:
            return None
        # Confirm acceleration — current confidence is materially
        # above the previous intraday confidence in the same direction.
        delta = current.intraday.confidence - previous.intraday.confidence
        if delta < self.delta:
            return None
        return Alert(
            symbol=current.symbol,
            rule=self.name,
            severity=self.severity,
            message=(
                f"intraday flipped {previous.intraday.action.value.upper()} → "
                f"{current.intraday.action.value.upper()} with conf rising "
                f"{previous.intraday.confidence:.2f} → "
                f"{current.intraday.confidence:.2f}"
            ),
            fired_at=datetime.now(UTC),
        )


@dataclass
class IntradayMomentumCollapseRule:
    """Fires when intraday confidence falls by at least ``delta`` in
    one tick. Targets the case where the intraday view rapidly
    discounts the prior conviction.
    """

    delta: float = 0.25
    severity: str = "warn"
    name: str = "intraday_momentum_collapse"

    def evaluate(
        self,
        current: RecommendationRow,
        previous: RecommendationRow | None,
    ) -> Alert | None:
        if (
            previous is None
            or current.error
            or previous.error
            or current.intraday is None
            or previous.intraday is None
        ):
            return None
        drop = previous.intraday.confidence - current.intraday.confidence
        if drop < self.delta:
            return None
        return Alert(
            symbol=current.symbol,
            rule=self.name,
            severity=self.severity,
            message=(
                f"intraday confidence collapsed {previous.intraday.confidence:.2f} → "
                f"{current.intraday.confidence:.2f} ({drop:+.2f})"
            ),
            fired_at=datetime.now(UTC),
        )


@dataclass
class TimeframeDisagreementRule:
    """Fires when daily-vs-intraday actions transition into a
    directional conflict (BUY ↔ SELL) for this symbol.

    Only emits on the transition tick — symbols that stay in conflict
    keep quiet. Membership is tracked per-symbol via the previous-row
    diff inherited from the engine.
    """

    severity: str = "warn"
    name: str = "timeframe_disagreement"

    def evaluate(
        self,
        current: RecommendationRow,
        previous: RecommendationRow | None,
    ) -> Alert | None:
        if current.error or current.intraday is None:
            return None
        if not _is_conflict(current.action, current.intraday.action):
            return None
        # First-sight conflict (no previous row OR previous wasn't a
        # conflict) is what we surface; sustained conflict stays
        # quiet so the alerts pane doesn't churn.
        if (
            previous is not None
            and previous.intraday is not None
            and _is_conflict(previous.action, previous.intraday.action)
        ):
            return None
        return Alert(
            symbol=current.symbol,
            rule=self.name,
            severity=self.severity,
            message=(
                f"timeframe disagreement: daily {current.action.value.upper()} "
                f"vs intraday {current.intraday.action.value.upper()} "
                f"(intraday conf {current.intraday.confidence:.2f})"
            ),
            fired_at=datetime.now(UTC),
        )


def _is_conflict(daily: SignalAction, intraday: SignalAction) -> bool:
    """True when both sides are directional and disagree."""
    return (
        daily in (SignalAction.BUY, SignalAction.SELL)
        and intraday in (SignalAction.BUY, SignalAction.SELL)
        and daily != intraday
    )


# ---------------------------------------------------------------------------
# Snapshot-level rules
# ---------------------------------------------------------------------------


@dataclass
class IntradayOpportunityEntryRule:
    """Intraday counterpart to
    :class:`~src.intelligence.alerts.OpportunityEntryRule`. Fires when
    a symbol enters the intraday top-N composite ranking.
    """

    n: int = 3
    min_composite: float = 0.30
    severity: str = "info"
    name: str = "intraday_opportunity_entry"

    _previous_top: set[str] | None = field(default=None, init=False, repr=False)

    def evaluate_snapshot(self, snapshot: DashboardSnapshot) -> list[Alert]:
        from src.intelligence.opportunities import rank_opportunities_intraday

        ranked = rank_opportunities_intraday(snapshot, n=self.n)
        qualifying = [opp for opp in ranked if opp.composite_score >= self.min_composite]
        current_top = {opp.symbol for opp in qualifying}

        prior = self._previous_top
        self._previous_top = current_top

        if prior is None:
            return []

        now = datetime.now(UTC)
        new_entries = [opp for opp in qualifying if opp.symbol not in prior]
        return [
            Alert(
                symbol=opp.symbol,
                rule=self.name,
                severity=self.severity,
                message=(
                    f"entered intraday top-{self.n} at "
                    f"{opp.composite_score:.2f} ({opp.tier.display})"
                ),
                fired_at=now,
            )
            for opp in new_entries
        ]


@dataclass
class RapidConfidenceDecayRule:
    """Fires when intraday confidence is below ``ratio`` of where it
    was ``window`` ticks ago, using the intraday pulse history's
    series of recent breadth values as a proxy.

    Specifically, we compare the most-recent intraday momentum-
    breadth value against the earlier-window max — a sharp drop from
    a recent peak signals the intraday read is losing steam fast.

    Snapshot-level rather than per-row because the diagnosis keys
    off the pulse trajectory rather than per-symbol state.
    """

    window: int = 4
    ratio: float = 0.50
    severity: str = "info"
    name: str = "rapid_confidence_decay"

    _previous_alerted: bool = field(default=False, init=False, repr=False)

    def evaluate_snapshot(self, snapshot: DashboardSnapshot) -> list[Alert]:
        history = snapshot.intraday_pulse_history
        if history is None or history.length < self.window:
            self._previous_alerted = False
            return []
        recent = history.momentum_breadth[-self.window :]
        latest = recent[-1]
        earlier = recent[:-1]
        if not earlier:
            return []
        peak = max(earlier)
        if peak <= 0:
            self._previous_alerted = False
            return []
        decay_threshold = peak * self.ratio
        if latest >= decay_threshold:
            self._previous_alerted = False
            return []
        # Don't re-fire on the same continuing decay — the pulse-
        # history rule fires once on entry into decay state.
        if self._previous_alerted:
            return []
        self._previous_alerted = True
        return [
            Alert(
                symbol="MARKET",
                rule=self.name,
                severity=self.severity,
                message=(
                    f"intraday momentum breadth decayed {peak:.2f} → "
                    f"{latest:.2f} over {self.window} ticks"
                ),
                fired_at=datetime.now(UTC),
            )
        ]


__all__ = [
    "IntradayMomentumCollapseRule",
    "IntradayOpportunityEntryRule",
    "IntradayReversalAccelerationRule",
    "RapidConfidenceDecayRule",
    "TimeframeDisagreementRule",
]
