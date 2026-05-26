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
# OPP-related composites
# ---------------------------------------------------------------------------


def test_opp_entry_with_action_flip_composes_into_fresh_opp_from_flip() -> None:
    """The canonical (opportunity_entry, action_changed) pairing collapses
    into one critical alert that suppresses both constituents — the
    "fresh OPP from a flip" event the example YAML defines."""
    config = PrioritizerConfig(
        composites=(
            CompositeRule(
                name="fresh_opp_from_flip",
                requires=("opportunity_entry", "action_changed"),
                severity="critical",
                message="{symbol}: fresh top-N opportunity from an action flip",
            ),
        ),
    )
    prio = AlertPrioritizer(config)
    fresh = [
        _alert(symbol="NVDA", rule="opportunity_entry", severity="info"),
        _alert(symbol="NVDA", rule="action_changed", severity="warn"),
    ]
    out = prio.prioritize(fresh, AlertState())
    assert len(out) == 1
    assert out[0].rule == "fresh_opp_from_flip"
    assert out[0].severity == "critical"
    assert out[0].message == "NVDA: fresh top-N opportunity from an action flip"


def test_opp_entry_with_sentiment_swing_composes_into_warn_alert() -> None:
    """OPP entry + sentiment shift is the news-led composite — a step
    below the action-flip composite (warn rather than critical)."""
    config = PrioritizerConfig(
        composites=(
            CompositeRule(
                name="fresh_opp_from_sentiment_swing",
                requires=("opportunity_entry", "sentiment_shift"),
                severity="warn",
                message="{symbol}: fresh top-N opportunity on a sentiment swing",
            ),
        ),
    )
    prio = AlertPrioritizer(config)
    fresh = [
        _alert(symbol="AAPL", rule="opportunity_entry", severity="info"),
        _alert(symbol="AAPL", rule="sentiment_shift", severity="info"),
    ]
    out = prio.prioritize(fresh, AlertState())
    assert len(out) == 1
    assert out[0].rule == "fresh_opp_from_sentiment_swing"
    assert out[0].severity == "warn"


def test_triple_opp_composite_wins_over_pair_composite_when_ordered_first() -> None:
    """When the triple composite is listed before the pair subset, the
    triple wins — the prioritizer's first-match-wins rule. This is the
    ordering the shipped example YAML relies on so users get the
    highest-conviction alert when all three constituents fire."""
    config = PrioritizerConfig(
        composites=(
            # Triple FIRST so it gets first dibs on the constituents.
            CompositeRule(
                name="opp_flip_with_sentiment",
                requires=("opportunity_entry", "action_changed", "sentiment_shift"),
                severity="critical",
                message="{symbol}: triple confirm",
            ),
            CompositeRule(
                name="fresh_opp_from_flip",
                requires=("opportunity_entry", "action_changed"),
                severity="critical",
                message="{symbol}: pair",
            ),
        ),
    )
    prio = AlertPrioritizer(config)
    fresh = [
        _alert(symbol="NVDA", rule="opportunity_entry"),
        _alert(symbol="NVDA", rule="action_changed"),
        _alert(symbol="NVDA", rule="sentiment_shift"),
    ]
    out = prio.prioritize(fresh, AlertState())
    rules = {a.rule for a in out}
    assert "opp_flip_with_sentiment" in rules
    # The pair composite must NOT also fire — its constituents were
    # consumed by the triple.
    assert "fresh_opp_from_flip" not in rules


def test_pair_composite_wins_when_triple_listed_after() -> None:
    """Inverse: with the pair listed first, it consumes its constituents
    and the triple never fires. This documents WHY the example YAML
    orders triple-before-pair — to prevent this exact subset-shadowing."""
    config = PrioritizerConfig(
        composites=(
            # Pair FIRST — eats opportunity_entry + action_changed.
            CompositeRule(
                name="fresh_opp_from_flip",
                requires=("opportunity_entry", "action_changed"),
                severity="critical",
                message="{symbol}: pair",
            ),
            CompositeRule(
                name="opp_flip_with_sentiment",
                requires=("opportunity_entry", "action_changed", "sentiment_shift"),
                severity="critical",
                message="{symbol}: triple",
            ),
        ),
    )
    prio = AlertPrioritizer(config)
    fresh = [
        _alert(symbol="NVDA", rule="opportunity_entry"),
        _alert(symbol="NVDA", rule="action_changed"),
        _alert(symbol="NVDA", rule="sentiment_shift"),
    ]
    out = prio.prioritize(fresh, AlertState())
    rules = {a.rule for a in out}
    # Pair fired; triple did not. sentiment_shift survives as an
    # unrelated alert because it wasn't consumed.
    assert "fresh_opp_from_flip" in rules
    assert "opp_flip_with_sentiment" not in rules
    assert "sentiment_shift" in rules


def test_opp_composite_is_per_symbol() -> None:
    """opportunity_entry on NVDA and action_changed on AAPL must NOT
    compose — composites only collapse same-symbol firings."""
    config = PrioritizerConfig(
        composites=(
            CompositeRule(
                name="fresh_opp_from_flip",
                requires=("opportunity_entry", "action_changed"),
                severity="critical",
                message="x",
            ),
        ),
    )
    prio = AlertPrioritizer(config)
    fresh = [
        _alert(symbol="NVDA", rule="opportunity_entry"),
        _alert(symbol="AAPL", rule="action_changed"),
    ]
    out = prio.prioritize(fresh, AlertState())
    # Both originals preserved; no composite.
    assert {a.rule for a in out} == {"opportunity_entry", "action_changed"}


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
    # action_changed gets a short 60s debounce: enough to squash a
    # same-minute BUY → HOLD → BUY flap, short enough that a real new
    # flip still lands quickly. See BUGS.md B-22.
    assert cfg.cooldowns["action_changed"] == 60


def test_default_config_debounces_tier_and_opportunity_rules() -> None:
    """Tier and opportunity-entry rules can oscillate around their
    promotion thresholds; the BUGS.md B-22 contract requires every
    rule cited there to ship with a non-zero default cooldown."""
    cfg = default_config()
    assert cfg.cooldowns["tier_changed"] == 300
    assert cfg.cooldowns["opportunity_entry"] == 600


def test_every_default_rule_has_a_nonzero_default_cooldown() -> None:
    """Invariant pin: any rule returned by the default factories must
    have a non-zero default cooldown. Adding a new default rule without
    one trips this test rather than slipping a noisy alert into prod
    (B-22)."""
    from src.intelligence.alerts import _default_rules, _default_snapshot_rules

    cfg = default_config()
    every_rule = [
        *(r.name for r in _default_rules()),
        *(r.name for r in _default_snapshot_rules()),
    ]
    missing = [name for name in every_rule if cfg.cooldowns.get(name, 0) <= 0]
    assert missing == [], (
        f"default rules without a non-zero cooldown: {missing} — every "
        f"default rule must ship with a debounce window per BUGS.md B-22"
    )


def test_tier_changed_with_default_cooldown_suppresses_oscillation() -> None:
    """Behavioural pin: a tier oscillating within the cooldown window
    only emits one alert, not one per oscillation. Tier transitions are
    coarse-grained so the rule itself has no hysteresis — the cooldown
    is the only defence against a quality driver hovering near a
    promotion gate."""
    prio = AlertPrioritizer()  # uses default_config
    state = AlertState()
    # Three tier_changed firings inside the 300s default window.
    first = _alert(rule="tier_changed", when=_ts(0))
    second = _alert(rule="tier_changed", when=_ts(60))
    third = _alert(rule="tier_changed", when=_ts(120))
    kept_first = prio.prioritize([first], state)
    state.record(kept_first)
    kept_second = prio.prioritize([second], state, now=_ts(60))
    state.record(kept_second)
    kept_third = prio.prioritize([third], state, now=_ts(120))
    state.record(kept_third)
    # Only the first firing survived; the cooldown debounced the rest.
    assert [a.fired_at for a in kept_first] == [_ts(0)]
    assert kept_second == []
    assert kept_third == []


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_prioritize_empty_input_returns_empty() -> None:
    prio = AlertPrioritizer()
    assert prio.prioritize([], AlertState()) == []
