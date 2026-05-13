"""Alert prioritization + session-scoped alert state.

The existing :class:`~src.intelligence.alerts.AlertEngine` fires every
matching rule on every tick. For a multi-symbol watchlist that produces
a stream that is noisy in two specific ways:

1. The same `(symbol, rule)` can fire repeatedly when a signal hovers
   near a threshold.
2. Multiple correlated rules can fire for the same symbol in the same
   tick (an action flip + a sentiment swing usually mean the same
   thing), each appearing as a separate item in the alerts pane.

This module addresses both without changing the existing rule API:

- :class:`AlertState` keeps a bounded session-wide log and a
  ``(symbol, rule)`` last-fired map.
- :class:`AlertPrioritizer` applies a four-stage pipeline:
  cooldown filter → composite detection → severity sort → optional cap.

Both types are small, in-memory, per-session, and free of any LLM
calls. The composite ``Alert`` message is a fixed format string —
nothing is model-generated.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from src.intelligence.alerts import Alert
from src.utils.logging import get_logger

log = get_logger(__name__)

_SEVERITY_RANK = {"critical": 0, "warn": 1, "info": 2}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AlertState:
    """Session-scoped log of fired alerts.

    Bounded — the deque drops oldest entries past ``max_history``. The
    ``(symbol, rule)`` last-fired map mirrors the log; it isn't pruned
    on its own (the map is at most ``len(rules) * len(symbols)``, which
    is tiny in practice).
    """

    def __init__(self, max_history: int = 200) -> None:
        self._log: deque[Alert] = deque(maxlen=max_history)
        self._last_fired: dict[tuple[str, str], datetime] = {}

    def record(self, alerts: Iterable[Alert]) -> None:
        for alert in alerts:
            self._log.append(alert)
            self._last_fired[(alert.symbol, alert.rule)] = alert.fired_at

    def last_fired(self, symbol: str, rule_name: str) -> datetime | None:
        return self._last_fired.get((symbol, rule_name))

    def recent(self, n: int = 20) -> tuple[Alert, ...]:
        """Most-recent-first slice of the session log."""
        items = list(self._log)
        return tuple(reversed(items[-n:]))

    def __len__(self) -> int:
        return len(self._log)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeRule:
    """If ``requires`` rule_names all fire for the same symbol in the
    same tick, emit one composite alert with this severity."""

    name: str
    requires: tuple[str, ...]
    severity: str
    message: str  # plain text; ``{symbol}`` substitutes


@dataclass(frozen=True)
class PrioritizerConfig:
    """User-tunable knobs for the prioritizer."""

    cooldowns: dict[str, int] = field(default_factory=dict)
    """Per-rule cooldown in seconds. Missing keys default to 0 (no cooldown)."""

    composites: tuple[CompositeRule, ...] = ()
    """Composite-alert definitions, evaluated in order."""

    max_per_tick: int | None = None
    """Cap on alerts emitted per tick. ``None`` = no cap."""


# Defaults applied when no config is provided. Conservative — don't drop
# real transitions, do debounce noisy threshold/sentiment rules.
_DEFAULT_COOLDOWNS = {
    "action_changed": 0,
    "confidence_threshold": 300,
    "sentiment_shift": 600,
    # Tier changes can oscillate around the promotion gate when one
    # quality driver hovers near a threshold; 5 min suppresses the noise.
    "tier_changed": 300,
}


def default_config() -> PrioritizerConfig:
    """Reasonable defaults for users who haven't customized alerts.yaml."""
    return PrioritizerConfig(cooldowns=dict(_DEFAULT_COOLDOWNS))


# ---------------------------------------------------------------------------
# Prioritizer
# ---------------------------------------------------------------------------


class AlertPrioritizer:
    """Stateless pipeline that turns raw rule firings into a curated list.

    The four stages run in order; each stage's output feeds the next.
    The prioritizer never mutates the input list and never mutates
    ``AlertState`` — the controller calls ``state.record(...)`` after.
    """

    def __init__(self, config: PrioritizerConfig | None = None) -> None:
        self.config = config or default_config()

    def prioritize(
        self,
        fresh: list[Alert],
        state: AlertState,
        *,
        now: datetime | None = None,
    ) -> list[Alert]:
        if not fresh:
            return []
        now = now or datetime.now(UTC)
        # Stage 1 — cooldown filter.
        kept = self._apply_cooldown(fresh, state, now)
        # Stage 2 — composite detection.
        kept = self._detect_composites(kept, now)
        # Stage 3 — severity sort.
        kept.sort(key=_sort_key)
        # Stage 4 — optional cap.
        if self.config.max_per_tick is not None:
            kept = kept[: self.config.max_per_tick]
        return kept

    # -- stages ----------------------------------------------------------

    def _apply_cooldown(
        self, fresh: list[Alert], state: AlertState, now: datetime
    ) -> list[Alert]:
        kept: list[Alert] = []
        cooldowns = self.config.cooldowns
        for alert in fresh:
            cooldown_s = cooldowns.get(alert.rule, 0)
            if cooldown_s <= 0:
                kept.append(alert)
                continue
            last = state.last_fired(alert.symbol, alert.rule)
            if last is None or now - last >= timedelta(seconds=cooldown_s):
                kept.append(alert)
            else:
                log.debug(
                    "alert.suppressed",
                    symbol=alert.symbol,
                    rule=alert.rule,
                    age_seconds=(now - last).total_seconds(),
                    cooldown_seconds=cooldown_s,
                )
        return kept

    def _detect_composites(
        self, kept: list[Alert], now: datetime
    ) -> list[Alert]:
        if not self.config.composites:
            return list(kept)

        by_symbol: dict[str, list[Alert]] = defaultdict(list)
        for alert in kept:
            by_symbol[alert.symbol].append(alert)

        out: list[Alert] = []
        for symbol, sym_alerts in by_symbol.items():
            fired_rules = {a.rule for a in sym_alerts}
            consumed: set[Alert] = set()
            for composite in self.config.composites:
                required = set(composite.requires)
                if required.issubset(fired_rules):
                    out.append(
                        Alert(
                            symbol=symbol,
                            rule=composite.name,
                            severity=composite.severity,
                            message=composite.message.format(symbol=symbol),
                            fired_at=now,
                        )
                    )
                    # Suppress the constituents that triggered this composite.
                    for a in sym_alerts:
                        if a.rule in required:
                            consumed.add(id(a))
                    # First-match wins so a constituent isn't double-consumed.
                    fired_rules -= required
            out.extend(a for a in sym_alerts if id(a) not in consumed)
        return out


def _sort_key(alert: Alert) -> tuple[int, str, str]:
    """Severity-first, then symbol, then rule_name for stable ordering."""
    return (_SEVERITY_RANK.get(alert.severity, 99), alert.symbol, alert.rule)


__all__ = [
    "AlertPrioritizer",
    "AlertState",
    "CompositeRule",
    "PrioritizerConfig",
    "default_config",
]
