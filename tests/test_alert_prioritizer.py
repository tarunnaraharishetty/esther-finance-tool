"""Tests for src.intelligence.alert_prioritizer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.intelligence.alert_prioritizer import (
    AlertPrioritizer,
    AlertState,
    CompositeRule,
    PrioritizerConfig,
    default_config,
)
from src.intelligence.alerts import Alert


def _ts(offset_seconds: int = 0) -> datetime:
    return datetime(2026, 5, 13, 14, 0, tzinfo=UTC) + timedelta(seconds=offset_seconds)


def _alert(
    symbol: str = "AAPL",
    rule: str = "action_changed",
    severity: str = "warn",
    message: str = "x",
    when: datetime | None = None,
) -> Alert:
    return Alert(
        symbol=symbol,
        rule=rule,
        severity=severity,
        message=message,
        fired_at=when or _ts(),
    )


# ---------------------------------------------------------------------------
# AlertState
# ---------------------------------------------------------------------------


def test_state_records_and_reports_last_fired() -> None:
    state = AlertState()
    a1 = _alert(when=_ts(0))
    a2 = _alert(symbol="MSFT", rule="confidence_threshold", when=_ts(60))
    state.record([a1, a2])
    assert state.last_fired("AAPL", "action_changed") == _ts(0)
    assert state.last_fired("MSFT", "confidence_threshold") == _ts(60)
    assert state.last_fired("NVDA", "action_changed") is None


def test_state_recent_is_newest_first_and_bounded() -> None:
    state = AlertState(max_history=3)
    for i in range(5):
        state.record([_alert(symbol=f"S{i}", when=_ts(i))])
    # Only the last 3 retained.
    recent = state.recent()
    assert [a.symbol for a in recent] == ["S4", "S3", "S2"]


def test_state_recent_caps_at_n() -> None:
    state = AlertState()
    for i in range(10):
        state.record([_alert(symbol=f"S{i}", when=_ts(i))])
    assert len(state.recent(n=3)) == 3


# ---------------------------------------------------------------------------
# Cooldown filter
# ---------------------------------------------------------------------------


def test_cooldown_drops_repeats_within_window() -> None:
    state = AlertState()
    state.record([_alert(rule="confidence_threshold", when=_ts(0))])
    prio = AlertPrioritizer(
        PrioritizerConfig(cooldowns={"confidence_threshold": 300})
    )
    fresh = [_alert(rule="confidence_threshold", when=_ts(100))]
    out = prio.prioritize(fresh, state, now=_ts(100))
    assert out == []


def test_cooldown_allows_after_window() -> None:
    state = AlertState()
    state.record([_alert(rule="confidence_threshold", when=_ts(0))])
    prio = AlertPrioritizer(
        PrioritizerConfig(cooldowns={"confidence_threshold": 300})
    )
    fresh = [_alert(rule="confidence_threshold", when=_ts(400))]
    out = prio.prioritize(fresh, state, now=_ts(400))
    assert len(out) == 1


def test_cooldown_is_per_symbol_rule() -> None:
    """Cooldown on (AAPL, confidence_threshold) must NOT suppress
    (MSFT, confidence_threshold)."""
    state = AlertState()
    state.record([_alert(symbol="AAPL", rule="confidence_threshold", when=_ts(0))])
    prio = AlertPrioritizer(
        PrioritizerConfig(cooldowns={"confidence_threshold": 300})
    )
    fresh = [_alert(symbol="MSFT", rule="confidence_threshold", when=_ts(100))]
    out = prio.prioritize(fresh, state, now=_ts(100))
    assert len(out) == 1


def test_zero_cooldown_never_suppresses() -> None:
    state = AlertState()
    state.record([_alert(rule="action_changed", when=_ts(0))])
    prio = AlertPrioritizer(PrioritizerConfig(cooldowns={"action_changed": 0}))
    fresh = [_alert(rule="action_changed", when=_ts(1))]
    assert len(prio.prioritize(fresh, state, now=_ts(1))) == 1


def test_unknown_rule_has_no_cooldown() -> None:
    """A rule not in the cooldowns map defaults to 0 (no cooldown)."""
    state = AlertState()
    state.record([_alert(rule="custom_rule", when=_ts(0))])
    prio = AlertPrioritizer(PrioritizerConfig(cooldowns={}))
    fresh = [_alert(rule="custom_rule", when=_ts(1))]
    assert len(prio.prioritize(fresh, state, now=_ts(1))) == 1


# ---------------------------------------------------------------------------
# Composite detection
# ---------------------------------------------------------------------------


def test_composite_emits_single_alert_when_all_requirements_present() -> None:
    config = PrioritizerConfig(
        composites=(
            CompositeRule(
                name="high_conviction_reversal",
                requires=("action_changed", "sentiment_shift"),
                severity="critical",
                message="action flip + sentiment swing",
            ),
        ),
    )
    prio = AlertPrioritizer(config)
    fresh = [
        _alert(rule="action_changed", severity="warn"),
        _alert(rule="sentiment_shift", severity="info"),
    ]
    out = prio.prioritize(fresh, AlertState())
    assert len(out) == 1
    assert out[0].rule == "high_conviction_reversal"
    assert out[0].severity == "critical"


def test_composite_does_not_fire_when_only_partial_match() -> None:
    config = PrioritizerConfig(
        composites=(
            CompositeRule(
                name="high_conviction_reversal",
                requires=("action_changed", "sentiment_shift"),
                severity="critical",
                message="x",
            ),
        ),
    )
    prio = AlertPrioritizer(config)
    fresh = [_alert(rule="action_changed", severity="warn")]
    out = prio.prioritize(fresh, AlertState())
    assert len(out) == 1
    assert out[0].rule == "action_changed"  # original preserved


def test_composite_is_per_symbol() -> None:
    """action_changed on AAPL + sentiment_shift on MSFT must NOT compose."""
    config = PrioritizerConfig(
        composites=(
            CompositeRule(
                name="high_conviction_reversal",
                requires=("action_changed", "sentiment_shift"),
                severity="critical",
                message="x",
            ),
        ),
    )
    prio = AlertPrioritizer(config)
    fresh = [
        _alert(symbol="AAPL", rule="action_changed"),
        _alert(symbol="MSFT", rule="sentiment_shift"),
    ]
    out = prio.prioritize(fresh, AlertState())
    # Both originals preserved; no composite.
    assert {a.rule for a in out} == {"action_changed", "sentiment_shift"}


def test_composite_message_template_substitutes_symbol() -> None:
    config = PrioritizerConfig(
        composites=(
            CompositeRule(
                name="combo",
                requires=("action_changed", "sentiment_shift"),
                severity="critical",
                message="{symbol}: combo fired",
            ),
        ),
    )
    prio = AlertPrioritizer(config)
    fresh = [
        _alert(symbol="NVDA", rule="action_changed"),
        _alert(symbol="NVDA", rule="sentiment_shift"),
    ]
    out = prio.prioritize(fresh, AlertState())
    assert out[0].message == "NVDA: combo fired"


def test_composite_unrelated_alerts_pass_through() -> None:
    """A composite over [action_changed, sentiment_shift] should not
    suppress an unrelated confidence_threshold alert for the same symbol."""
    config = PrioritizerConfig(
        composites=(
            CompositeRule(
                name="combo",
                requires=("action_changed", "sentiment_shift"),
                severity="critical",
                message="x",
            ),
        ),
    )
    prio = AlertPrioritizer(config)
    fresh = [
        _alert(rule="action_changed"),
        _alert(rule="sentiment_shift"),
        _alert(rule="confidence_threshold", severity="info"),
    ]
    out = prio.prioritize(fresh, AlertState())
    rules = sorted(a.rule for a in out)
    assert rules == ["combo", "confidence_threshold"]


# ---------------------------------------------------------------------------
# Severity sort
# ---------------------------------------------------------------------------


def test_severity_sort_orders_critical_first() -> None:
    prio = AlertPrioritizer(PrioritizerConfig())
    fresh = [
        _alert(severity="info", rule="r1"),
        _alert(severity="critical", rule="r2"),
        _alert(severity="warn", rule="r3"),
    ]
    out = prio.prioritize(fresh, AlertState())
    assert [a.severity for a in out] == ["critical", "warn", "info"]


def test_severity_sort_is_stable_by_symbol() -> None:
    prio = AlertPrioritizer(PrioritizerConfig())
    fresh = [
        _alert(symbol="NVDA", severity="warn", rule="r1"),
        _alert(symbol="AAPL", severity="warn", rule="r2"),
        _alert(symbol="MSFT", severity="warn", rule="r3"),
    ]
    out = prio.prioritize(fresh, AlertState())
    assert [a.symbol for a in out] == ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# Cap
# ---------------------------------------------------------------------------


def test_max_per_tick_cap_applies_after_sort() -> None:
    prio = AlertPrioritizer(PrioritizerConfig(max_per_tick=2))
    fresh = [
        _alert(symbol="AAPL", severity="info"),
        _alert(symbol="MSFT", severity="critical"),
        _alert(symbol="NVDA", severity="warn"),
    ]
    out = prio.prioritize(fresh, AlertState())
    assert len(out) == 2
    assert out[0].severity == "critical"
    assert out[1].severity == "warn"


# ---------------------------------------------------------------------------
# End-to-end: cooldown + composite + sort interacting
# ---------------------------------------------------------------------------


def test_full_pipeline_composite_then_cooldown_then_sort() -> None:
    """Verify the stages compose: composite alerts are NOT subject to
    cooldown (they fire fresh each tick they qualify), and severity
    sort orders the final list."""
    state = AlertState()
    # Pre-seed: confidence_threshold for AAPL fired 10s ago — should
    # still be in cooldown (default 300s).
    state.record([_alert(rule="confidence_threshold", when=_ts(0))])

    config = PrioritizerConfig(
        cooldowns={"confidence_threshold": 300},
        composites=(
            CompositeRule(
                name="combo",
                requires=("action_changed", "sentiment_shift"),
                severity="critical",
                message="x",
            ),
        ),
    )
    prio = AlertPrioritizer(config)

    fresh = [
        # Should be cooled-down → dropped.
        _alert(rule="confidence_threshold", severity="info", when=_ts(10)),
        # These two compose for AAPL.
        _alert(rule="action_changed", severity="warn", when=_ts(10)),
        _alert(rule="sentiment_shift", severity="info", when=_ts(10)),
    ]
    out = prio.prioritize(fresh, state, now=_ts(10))
    # One composite, nothing else.
    assert len(out) == 1
    assert out[0].rule == "combo"
    assert out[0].severity == "critical"


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------


def test_default_config_debounces_confidence_threshold() -> None:
    cfg = default_config()
    assert cfg.cooldowns["confidence_threshold"] == 300
    assert cfg.cooldowns["sentiment_shift"] == 600
    # action_changed must NOT be debounced — real flips are information.
    assert cfg.cooldowns["action_changed"] == 0


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_prioritize_empty_input_returns_empty() -> None:
    prio = AlertPrioritizer()
    assert prio.prioritize([], AlertState()) == []
