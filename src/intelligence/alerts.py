"""Alert rules that fire when something noteworthy changes between ticks.

Rules consume the *current* :class:`~src.dashboard.state.RecommendationRow`
and the *previous* one (or ``None`` on first sight) and return an
:class:`Alert` if they fire. Stateless — composition with state happens
in :class:`AlertEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.dashboard.state import RecommendationRow


@dataclass(frozen=True)
class Alert:
    """A single alert that fired for one symbol."""

    symbol: str
    rule: str
    severity: str  # "info" | "warn" | "critical"
    message: str
    fired_at: datetime


class Rule(Protocol):
    """A pluggable rule. Stateless: receives current + previous row."""

    name: str

    def evaluate(
        self,
        current: RecommendationRow,
        previous: RecommendationRow | None,
    ) -> Alert | None: ...


# ---------------------------------------------------------------------------
# Concrete rules
# ---------------------------------------------------------------------------


@dataclass
class ActionChangedRule:
    """Fires when the recommended action changes (BUY → HOLD, etc.)."""

    severity: str = "warn"
    name: str = "action_changed"

    def evaluate(
        self,
        current: RecommendationRow,
        previous: RecommendationRow | None,
    ) -> Alert | None:
        if previous is None or current.error or previous.error:
            return None
        if current.action == previous.action:
            return None
        return Alert(
            symbol=current.symbol,
            rule=self.name,
            severity=self.severity,
            message=(
                f"recommendation flipped {previous.action.value.upper()} → "
                f"{current.action.value.upper()} (conf {current.confidence:.2f})"
            ),
            fired_at=datetime.now(UTC),
        )


@dataclass
class ConfidenceThresholdRule:
    """Fires when confidence crosses ``threshold`` (in either direction)."""

    threshold: float = 0.6
    severity: str = "info"
    name: str = "confidence_threshold"

    def evaluate(
        self,
        current: RecommendationRow,
        previous: RecommendationRow | None,
    ) -> Alert | None:
        if previous is None or current.error or previous.error:
            return None
        crossed_up = previous.confidence < self.threshold <= current.confidence
        crossed_down = current.confidence < self.threshold <= previous.confidence
        if not (crossed_up or crossed_down):
            return None
        direction = "↑" if crossed_up else "↓"
        return Alert(
            symbol=current.symbol,
            rule=self.name,
            severity=self.severity,
            message=(
                f"confidence crossed {self.threshold:.2f} {direction} "
                f"(now {current.confidence:.2f})"
            ),
            fired_at=datetime.now(UTC),
        )


@dataclass
class SentimentShiftRule:
    """Fires when the sentiment score moves by at least ``delta`` since last look."""

    delta: float = 0.4
    severity: str = "info"
    name: str = "sentiment_shift"

    def evaluate(
        self,
        current: RecommendationRow,
        previous: RecommendationRow | None,
    ) -> Alert | None:
        if previous is None or current.error or previous.error:
            return None
        move = current.sentiment_score - previous.sentiment_score
        if abs(move) < self.delta:
            return None
        return Alert(
            symbol=current.symbol,
            rule=self.name,
            severity=self.severity,
            message=(
                f"news sentiment shifted {move:+.2f} "
                f"({previous.sentiment_score:+.2f} → {current.sentiment_score:+.2f})"
            ),
            fired_at=datetime.now(UTC),
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AlertEngine:
    """Holds last-seen state per symbol and applies a rule set per tick."""

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules: list[Rule] = list(rules) if rules else _default_rules()
        self._previous: dict[str, RecommendationRow] = {}

    def evaluate(self, rows: list[RecommendationRow]) -> list[Alert]:
        """Run every rule against each row; update internal state."""
        alerts: list[Alert] = []
        for row in rows:
            prev = self._previous.get(row.symbol)
            for rule in self.rules:
                alert = rule.evaluate(row, prev)
                if alert is not None:
                    alerts.append(alert)
            # Only update state for healthy rows so an error row doesn't blow
            # away the last-good baseline a rule needs to compare against.
            if not row.error:
                self._previous[row.symbol] = row
        return alerts


def _default_rules() -> list[Rule]:
    return [
        ActionChangedRule(),
        ConfidenceThresholdRule(),
        SentimentShiftRule(),
    ]
