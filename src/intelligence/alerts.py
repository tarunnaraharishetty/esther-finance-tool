"""Alert rules that fire when something noteworthy changes between ticks.

Rules consume the *current* :class:`~src.dashboard.state.RecommendationRow`
and the *previous* one (or ``None`` on first sight) and return an
:class:`Alert` if they fire. Stateless — composition with state happens
in :class:`AlertEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import yaml

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot, RecommendationRow


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


class SnapshotRule(Protocol):
    """A rule that evaluates the whole snapshot rather than one row.

    Used for cross-row signals — e.g. "this symbol just entered the
    top-N ranked opportunities" depends on every other row's composite
    score, not just one row's history. May fire for zero, one, or
    multiple symbols in a single tick, so the contract returns a list.
    """

    name: str

    def evaluate_snapshot(self, snapshot: DashboardSnapshot) -> list[Alert]: ...


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
    """Fires when confidence crosses ``threshold`` with at least ``band`` magnitude.

    The ``band`` parameter is a hysteresis device: the up-crossing only
    fires when ``current >= threshold + band`` (and the previous tick
    was below ``threshold``); the down-crossing requires
    ``current <= threshold - band``. Without this band, a confidence
    that wiggles by 0.005 across 0.60 would fire every tick that hopped
    the boundary — the cooldown filter would then mute most of them but
    the alert log would still show flapping intent.
    """

    threshold: float = 0.6
    band: float = 0.02
    severity: str = "info"
    name: str = "confidence_threshold"

    def evaluate(
        self,
        current: RecommendationRow,
        previous: RecommendationRow | None,
    ) -> Alert | None:
        if previous is None or current.error or previous.error:
            return None
        # Up-crossing: previous below threshold, current ≥ threshold+band.
        up_floor = self.threshold + self.band
        # Down-crossing: previous at or above threshold, current ≤ threshold-band.
        down_ceil = self.threshold - self.band
        crossed_up = previous.confidence < self.threshold <= current.confidence and current.confidence >= up_floor
        crossed_down = (
            current.confidence < self.threshold <= previous.confidence
            and current.confidence <= down_ceil
        )
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


@dataclass
class TierChangedRule:
    """Fires when the 5-tier recommendation shifts WITHIN the same base
    action — i.e. a promote/demote like ``BUY -> STRONG_BUY`` or
    ``STRONG_SELL -> SELL``.

    Tier changes that come with an action flip (``BUY -> SELL``) are
    already covered by :class:`ActionChangedRule`; this rule deliberately
    skips them to avoid double-alerting on the same event. The result:
    each rule emits distinct information.
    """

    severity: str = "info"
    name: str = "tier_changed"

    def evaluate(
        self,
        current: RecommendationRow,
        previous: RecommendationRow | None,
    ) -> Alert | None:
        if previous is None or current.error or previous.error:
            return None
        if current.tier == previous.tier:
            return None
        # Base action change is ActionChangedRule's territory.
        if current.action != previous.action:
            return None
        return Alert(
            symbol=current.symbol,
            rule=self.name,
            severity=self.severity,
            message=(
                f"tier {previous.tier.display} -> {current.tier.display} "
                f"(conf {current.confidence:.2f})"
            ),
            fired_at=datetime.now(UTC),
        )


@dataclass
class OpportunityEntryRule:
    """Fires when a symbol newly enters the top-N ranked opportunities.

    Snapshot-level rule: each tick it recomputes the top-N composite
    ranking and compares membership against the prior tick's top-N. Any
    symbol that's in this tick's top-N but wasn't in the last one
    triggers an alert. Symbols that stay in the top-N tick-over-tick
    stay silent (the alert is the *entry event*, not the membership).

    Skipped on the first tick of a session — there's no prior set to
    diff against, and emitting "everything is new!" the first time the
    dashboard opens would be noise rather than signal.

    Symbols below ``min_composite`` are filtered before comparison so a
    quiet entry at composite 0.05 doesn't trip the alert.
    """

    n: int = 3
    """Top-N to consider — should match what the dashboard renders."""

    min_composite: float = 0.30
    """Composite floor; entries below this don't fire (cuts noise)."""

    severity: str = "info"
    name: str = "opportunity_entry"

    # Sentinel ``None`` means "first tick, no baseline yet."
    _previous_top: set[str] | None = field(default=None, init=False, repr=False)

    def evaluate_snapshot(self, snapshot: DashboardSnapshot) -> list[Alert]:
        # Lazy import: opportunities -> dashboard.state -> alerts would
        # otherwise create a cycle at import time.
        from src.intelligence.opportunities import rank_opportunities

        ranked = rank_opportunities(snapshot, n=self.n)
        # Apply the composite floor before computing membership so a
        # quiet entry doesn't churn the prior set either.
        qualifying = [opp for opp in ranked if opp.composite_score >= self.min_composite]
        current_top = {opp.symbol for opp in qualifying}

        prior = self._previous_top
        self._previous_top = current_top

        if prior is None:
            # First tick — baseline only, no alerts.
            return []

        now = datetime.now(UTC)
        new_entries = [opp for opp in qualifying if opp.symbol not in prior]
        return [
            Alert(
                symbol=opp.symbol,
                rule=self.name,
                severity=self.severity,
                message=(
                    f"entered top-{self.n} opportunities at "
                    f"{opp.composite_score:.2f} ({opp.tier.display})"
                ),
                fired_at=now,
            )
            for opp in new_entries
        ]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AlertEngine:
    """Holds last-seen state per symbol and applies a rule set per tick.

    Two rule families:

    * ``rules`` — per-row, stateless rules that compare current vs
      previous :class:`RecommendationRow` (the engine holds the previous
      row state).
    * ``snapshot_rules`` — snapshot-level rules that need the whole
      :class:`DashboardSnapshot` and hold their own cross-tick state.

    Both families produce :class:`Alert` instances that flow into the
    prioritizer together — cooldowns, composites, and the per-tick cap
    apply uniformly across both.
    """

    def __init__(
        self,
        rules: list[Rule] | None = None,
        snapshot_rules: list[SnapshotRule] | None = None,
    ) -> None:
        # `rules is None` means "use defaults"; an explicit empty list means
        # "no rules" (useful for tests and for users who want a silent dashboard).
        self.rules: list[Rule] = _default_rules() if rules is None else list(rules)
        self.snapshot_rules: list[SnapshotRule] = (
            _default_snapshot_rules()
            if snapshot_rules is None
            else list(snapshot_rules)
        )
        self._previous: dict[str, RecommendationRow] = {}

    def evaluate(self, rows: list[RecommendationRow]) -> list[Alert]:
        """Run every per-row rule against each row; update internal state."""
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

    def evaluate_snapshot(self, snapshot: DashboardSnapshot) -> list[Alert]:
        """Run every snapshot-level rule against the whole snapshot.

        Each rule may emit zero, one, or many alerts. Snapshot rules
        own their own state; the engine just dispatches.
        """
        alerts: list[Alert] = []
        for rule in self.snapshot_rules:
            alerts.extend(rule.evaluate_snapshot(snapshot))
        return alerts


def _default_rules() -> list[Rule]:
    return [
        ActionChangedRule(),
        ConfidenceThresholdRule(),
        SentimentShiftRule(),
        TierChangedRule(),
    ]


def _default_snapshot_rules() -> list[SnapshotRule]:
    return [OpportunityEntryRule()]


# ---------------------------------------------------------------------------
# YAML configuration
# ---------------------------------------------------------------------------


_RULE_REGISTRY: dict[str, type[Rule]] = {
    "action_changed": ActionChangedRule,
    "confidence_threshold": ConfidenceThresholdRule,
    "sentiment_shift": SentimentShiftRule,
    "tier_changed": TierChangedRule,
}

_SNAPSHOT_RULE_REGISTRY: dict[str, type[SnapshotRule]] = {
    "opportunity_entry": OpportunityEntryRule,
}

_VALID_SEVERITIES = {"info", "warn", "critical"}


def load_rules_from_yaml(path: Path) -> list[Rule]:
    """Parse ``config/alerts.yaml`` into a list of Rule instances.

    Expected schema::

        rules:
          - type: action_changed
            severity: warn
          - type: confidence_threshold
            threshold: 0.6
            severity: critical

    Unknown rule types and unknown rule kwargs raise ``ValueError`` so a
    typo in the user's config fails loudly at startup, not silently
    later.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"alerts config must be a mapping, got {type(raw).__name__}")
    rules_raw = raw.get("rules", [])
    if not isinstance(rules_raw, list):
        raise ValueError("alerts config: 'rules' must be a list")

    out: list[Rule] = []
    for i, entry in enumerate(rules_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"rules[{i}] must be a mapping")
        kwargs = dict(entry)
        rtype = kwargs.pop("type", None)
        if rtype is None:
            raise ValueError(f"rules[{i}] is missing 'type'")
        rule_cls = _RULE_REGISTRY.get(rtype)
        if rule_cls is None:
            valid = sorted(_RULE_REGISTRY.keys())
            raise ValueError(
                f"rules[{i}]: unknown rule type {rtype!r} (valid: {valid})"
            )
        if "severity" in kwargs and kwargs["severity"] not in _VALID_SEVERITIES:
            raise ValueError(
                f"rules[{i}]: severity must be one of "
                f"{sorted(_VALID_SEVERITIES)}, got {kwargs['severity']!r}"
            )
        try:
            out.append(rule_cls(**kwargs))
        except TypeError as e:
            raise ValueError(f"rules[{i}] ({rtype}): {e}") from e
    return out


def load_snapshot_rules_from_yaml(path: Path) -> list[SnapshotRule]:
    """Parse an optional ``snapshot_rules:`` block from ``alerts.yaml``.

    Schema mirrors the per-row ``rules:`` block; valid types come from
    :data:`_SNAPSHOT_RULE_REGISTRY`. Returns an empty list when the
    block is absent, which leaves :class:`AlertEngine` free to fall back
    to :func:`_default_snapshot_rules`.

    Example::

        snapshot_rules:
          - type: opportunity_entry
            n: 5
            min_composite: 0.25
            severity: warn
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"alerts config must be a mapping, got {type(raw).__name__}")
    rules_raw = raw.get("snapshot_rules", [])
    if not isinstance(rules_raw, list):
        raise ValueError("alerts config: 'snapshot_rules' must be a list")

    out: list[SnapshotRule] = []
    for i, entry in enumerate(rules_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"snapshot_rules[{i}] must be a mapping")
        kwargs = dict(entry)
        rtype = kwargs.pop("type", None)
        if rtype is None:
            raise ValueError(f"snapshot_rules[{i}] is missing 'type'")
        rule_cls = _SNAPSHOT_RULE_REGISTRY.get(rtype)
        if rule_cls is None:
            valid = sorted(_SNAPSHOT_RULE_REGISTRY.keys())
            raise ValueError(
                f"snapshot_rules[{i}]: unknown rule type {rtype!r} (valid: {valid})"
            )
        if "severity" in kwargs and kwargs["severity"] not in _VALID_SEVERITIES:
            raise ValueError(
                f"snapshot_rules[{i}]: severity must be one of "
                f"{sorted(_VALID_SEVERITIES)}, got {kwargs['severity']!r}"
            )
        try:
            out.append(rule_cls(**kwargs))
        except TypeError as e:
            raise ValueError(f"snapshot_rules[{i}] ({rtype}): {e}") from e
    return out


def load_prioritizer_config_from_yaml(path: Path) -> object:
    """Parse an optional ``prioritizer:`` block from ``alerts.yaml``.

    Returns a :class:`PrioritizerConfig` (imported lazily to avoid a
    cycle). If the block is absent, returns the default config so the
    dashboard wiring stays backward-compatible with old YAML files.

    Schema::

        prioritizer:
          cooldowns:
            confidence_threshold: 300
            sentiment_shift: 600
          composites:
            - name: high_conviction_reversal
              requires: [action_changed, sentiment_shift]
              severity: critical
              message: "{symbol}: action flip + sentiment swing"
          max_per_tick: 10        # optional
    """
    from src.intelligence.alert_prioritizer import (
        CompositeRule,
        PrioritizerConfig,
        default_config,
    )

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"alerts config must be a mapping, got {type(raw).__name__}"
        )
    prio_raw = raw.get("prioritizer")
    if prio_raw is None:
        return default_config()
    if not isinstance(prio_raw, dict):
        raise ValueError("alerts config: 'prioritizer' must be a mapping")

    # Cooldowns
    cooldowns_raw = prio_raw.get("cooldowns", {}) or {}
    if not isinstance(cooldowns_raw, dict):
        raise ValueError("prioritizer.cooldowns must be a mapping")
    cooldowns: dict[str, int] = {}
    for rule_name, secs in cooldowns_raw.items():
        if not isinstance(secs, int) or secs < 0:
            raise ValueError(
                f"prioritizer.cooldowns.{rule_name}: must be a non-negative int"
            )
        cooldowns[str(rule_name)] = secs

    # Composites
    composites_raw = prio_raw.get("composites", []) or []
    if not isinstance(composites_raw, list):
        raise ValueError("prioritizer.composites must be a list")
    composites: list[CompositeRule] = []
    for i, entry in enumerate(composites_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"prioritizer.composites[{i}] must be a mapping")
        name = entry.get("name")
        requires = entry.get("requires")
        severity = entry.get("severity")
        message = entry.get("message")
        if not name or not isinstance(name, str):
            raise ValueError(f"prioritizer.composites[{i}]: 'name' required (string)")
        if not isinstance(requires, list) or not requires:
            raise ValueError(
                f"prioritizer.composites[{i}]: 'requires' must be a non-empty list"
            )
        if severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"prioritizer.composites[{i}]: severity must be one of "
                f"{sorted(_VALID_SEVERITIES)}, got {severity!r}"
            )
        if not message or not isinstance(message, str):
            raise ValueError(
                f"prioritizer.composites[{i}]: 'message' required (string)"
            )
        composites.append(
            CompositeRule(
                name=name,
                requires=tuple(str(r) for r in requires),
                severity=severity,
                message=message,
            )
        )

    # Optional cap
    cap = prio_raw.get("max_per_tick")
    if cap is not None and (not isinstance(cap, int) or cap <= 0):
        raise ValueError("prioritizer.max_per_tick must be a positive integer or omitted")

    return PrioritizerConfig(
        cooldowns=cooldowns,
        composites=tuple(composites),
        max_per_tick=cap,
    )
