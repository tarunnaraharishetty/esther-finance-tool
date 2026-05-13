"""Tests for src.intelligence.alerts."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.dashboard.state import RecommendationRow
from src.intelligence.alerts import (
    ActionChangedRule,
    AlertEngine,
    ConfidenceThresholdRule,
    SentimentShiftRule,
    TierChangedRule,
    load_prioritizer_config_from_yaml,
    load_rules_from_yaml,
)
from src.strategy.base import RecommendationTier, SignalAction


def _row(
    *,
    symbol: str = "AAPL",
    action: SignalAction = SignalAction.HOLD,
    confidence: float = 0.3,
    sentiment: float = 0.0,
    tier: RecommendationTier | None = None,
    error: str | None = None,
) -> RecommendationRow:
    return RecommendationRow(
        symbol=symbol,
        action=action,
        confidence=confidence,
        combined_score=confidence if action == SignalAction.BUY else -confidence,
        technical_score=0.0,
        sentiment_score=sentiment,
        rsi=math.nan,
        macd=math.nan,
        bollinger=math.nan,
        last_price=100.0,
        num_news_articles=1,
        reasoning="",
        timestamp=datetime.now(UTC),
        tier=tier if tier is not None else RecommendationTier.from_action(action),
        error=error,
    )


def test_action_changed_rule_fires_on_flip() -> None:
    rule = ActionChangedRule()
    prev = _row(action=SignalAction.HOLD)
    curr = _row(action=SignalAction.BUY, confidence=0.5)
    alert = rule.evaluate(curr, prev)
    assert alert is not None
    assert "HOLD" in alert.message and "BUY" in alert.message


def test_action_changed_rule_silent_when_unchanged() -> None:
    rule = ActionChangedRule()
    prev = _row(action=SignalAction.BUY, confidence=0.5)
    curr = _row(action=SignalAction.BUY, confidence=0.55)
    assert rule.evaluate(curr, prev) is None


def test_action_changed_rule_silent_on_first_sight() -> None:
    rule = ActionChangedRule()
    assert rule.evaluate(_row(), None) is None


def test_confidence_threshold_fires_on_upward_cross() -> None:
    rule = ConfidenceThresholdRule(threshold=0.6)
    prev = _row(confidence=0.5)
    curr = _row(confidence=0.7)
    alert = rule.evaluate(curr, prev)
    assert alert is not None
    assert "0.60" in alert.message


def test_confidence_threshold_fires_on_downward_cross() -> None:
    rule = ConfidenceThresholdRule(threshold=0.6)
    prev = _row(confidence=0.7)
    curr = _row(confidence=0.4)
    alert = rule.evaluate(curr, prev)
    assert alert is not None


def test_confidence_threshold_silent_no_cross() -> None:
    rule = ConfidenceThresholdRule(threshold=0.6)
    assert rule.evaluate(_row(confidence=0.5), _row(confidence=0.4)) is None
    assert rule.evaluate(_row(confidence=0.8), _row(confidence=0.7)) is None


def test_sentiment_shift_fires_on_large_swing() -> None:
    rule = SentimentShiftRule(delta=0.4)
    prev = _row(sentiment=0.1)
    curr = _row(sentiment=-0.5)
    alert = rule.evaluate(curr, prev)
    assert alert is not None
    assert "-0.60" in alert.message  # +0.1 → -0.5 = -0.60


def test_sentiment_shift_silent_below_delta() -> None:
    rule = SentimentShiftRule(delta=0.4)
    assert rule.evaluate(_row(sentiment=0.1), _row(sentiment=0.3)) is None


# ---------------------------------------------------------------------------
# TierChangedRule
# ---------------------------------------------------------------------------


def test_tier_changed_fires_on_buy_to_strong_buy_promotion() -> None:
    rule = TierChangedRule()
    prev = _row(action=SignalAction.BUY, tier=RecommendationTier.BUY, confidence=0.55)
    curr = _row(
        action=SignalAction.BUY, tier=RecommendationTier.STRONG_BUY, confidence=0.78
    )
    alert = rule.evaluate(curr, prev)
    assert alert is not None
    assert "BUY" in alert.message and "STRONG BUY" in alert.message


def test_tier_changed_fires_on_strong_sell_to_sell_demotion() -> None:
    rule = TierChangedRule()
    prev = _row(
        action=SignalAction.SELL, tier=RecommendationTier.STRONG_SELL, confidence=0.72
    )
    curr = _row(action=SignalAction.SELL, tier=RecommendationTier.SELL, confidence=0.55)
    alert = rule.evaluate(curr, prev)
    assert alert is not None
    assert "STRONG SELL" in alert.message
    assert "-> SELL" in alert.message


def test_tier_changed_silent_when_action_flips() -> None:
    """ActionChangedRule already covers cross-action transitions. The
    tier rule should stay silent to avoid double-alerting."""
    rule = TierChangedRule()
    prev = _row(
        action=SignalAction.BUY, tier=RecommendationTier.STRONG_BUY, confidence=0.70
    )
    curr = _row(
        action=SignalAction.SELL, tier=RecommendationTier.STRONG_SELL, confidence=0.70
    )
    assert rule.evaluate(curr, prev) is None


def test_tier_changed_silent_when_tier_unchanged() -> None:
    rule = TierChangedRule()
    prev = _row(action=SignalAction.BUY, tier=RecommendationTier.BUY, confidence=0.55)
    curr = _row(action=SignalAction.BUY, tier=RecommendationTier.BUY, confidence=0.62)
    assert rule.evaluate(curr, prev) is None


def test_tier_changed_silent_on_first_sight() -> None:
    rule = TierChangedRule()
    assert rule.evaluate(_row(), None) is None


def test_tier_changed_silent_on_error_rows() -> None:
    rule = TierChangedRule()
    prev = _row(action=SignalAction.BUY, tier=RecommendationTier.BUY)
    err_curr = _row(action=SignalAction.BUY, tier=RecommendationTier.STRONG_BUY, error="boom")
    assert rule.evaluate(err_curr, prev) is None
    err_prev = _row(action=SignalAction.BUY, tier=RecommendationTier.BUY, error="boom")
    curr_ok = _row(action=SignalAction.BUY, tier=RecommendationTier.STRONG_BUY)
    assert rule.evaluate(curr_ok, err_prev) is None


def test_tier_changed_default_severity_is_info() -> None:
    """Same-direction tier shifts are notable but not urgent — info,
    not warn or critical."""
    assert TierChangedRule().severity == "info"


def test_tier_changed_rule_in_default_rule_set() -> None:
    """A fresh AlertEngine includes TierChangedRule out of the box."""
    engine = AlertEngine()
    rule_names = {r.name for r in engine.rules}
    assert "tier_changed" in rule_names


# ---------------------------------------------------------------------------
# AlertEngine + Rule interaction
# ---------------------------------------------------------------------------


def test_engine_collects_alerts_from_all_rules() -> None:
    engine = AlertEngine()
    first = _row(action=SignalAction.HOLD, confidence=0.2, sentiment=0.0)
    engine.evaluate([first])  # establish baseline
    second = _row(action=SignalAction.BUY, confidence=0.7, sentiment=0.5)
    alerts = engine.evaluate([second])
    rule_names = {a.rule for a in alerts}
    assert "action_changed" in rule_names
    assert "confidence_threshold" in rule_names
    assert "sentiment_shift" in rule_names


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "alerts.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_rules_returns_default_severities(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
rules:
  - type: action_changed
  - type: confidence_threshold
  - type: sentiment_shift
""",
    )
    rules = load_rules_from_yaml(path)
    assert [type(r).__name__ for r in rules] == [
        "ActionChangedRule",
        "ConfidenceThresholdRule",
        "SentimentShiftRule",
    ]


def test_load_rules_passes_kwargs_through(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
rules:
  - type: confidence_threshold
    threshold: 0.85
    severity: critical
  - type: sentiment_shift
    delta: 0.6
""",
    )
    rules = load_rules_from_yaml(path)
    assert isinstance(rules[0], ConfidenceThresholdRule)
    assert rules[0].threshold == 0.85
    assert rules[0].severity == "critical"
    assert isinstance(rules[1], SentimentShiftRule)
    assert rules[1].delta == 0.6


def test_load_rules_empty_file_yields_empty_list(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "")
    assert load_rules_from_yaml(path) == []


def test_load_rules_unknown_type_raises(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
rules:
  - type: not_a_real_rule
""",
    )
    with pytest.raises(ValueError, match="unknown rule type"):
        load_rules_from_yaml(path)


def test_load_rules_invalid_severity_raises(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
rules:
  - type: action_changed
    severity: HUGE
""",
    )
    with pytest.raises(ValueError, match="severity must be one of"):
        load_rules_from_yaml(path)


def test_load_rules_missing_type_raises(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
rules:
  - threshold: 0.5
""",
    )
    with pytest.raises(ValueError, match="missing 'type'"):
        load_rules_from_yaml(path)


def test_load_rules_unknown_kwarg_raises(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
rules:
  - type: action_changed
    nonsense_field: yes
""",
    )
    with pytest.raises(ValueError, match="action_changed"):
        load_rules_from_yaml(path)


def test_example_yaml_in_repo_parses() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "alerts.example.yaml"
    if not path.exists():
        pytest.skip(f"{path} not present")
    rules = load_rules_from_yaml(path)
    assert len(rules) > 0
    # Should round-trip to the right concrete classes
    type_names = {type(r).__name__ for r in rules}
    assert type_names <= {
        "ActionChangedRule",
        "ConfidenceThresholdRule",
        "SentimentShiftRule",
        "TierChangedRule",
    }


def test_yaml_loader_parses_tier_changed_rule(tmp_path: Path) -> None:
    """The YAML registry should recognize tier_changed and instantiate
    the right concrete rule class."""
    path = _write_yaml(
        tmp_path,
        """
rules:
  - type: tier_changed
    severity: warn
""",
    )
    rules = load_rules_from_yaml(path)
    assert len(rules) == 1
    assert isinstance(rules[0], TierChangedRule)
    assert rules[0].severity == "warn"


# ---------------------------------------------------------------------------
# Prioritizer YAML loader
# ---------------------------------------------------------------------------


def test_prio_loader_returns_default_when_block_absent(tmp_path: Path) -> None:
    """No `prioritizer:` block means the loader returns the default
    config — so old alerts.yaml files keep working."""
    from src.intelligence.alert_prioritizer import default_config

    path = _write_yaml(
        tmp_path,
        """
rules:
  - type: action_changed
""",
    )
    config = load_prioritizer_config_from_yaml(path)
    assert config == default_config()


def test_prio_loader_reads_cooldowns_and_composites(tmp_path: Path) -> None:
    from src.intelligence.alert_prioritizer import CompositeRule, PrioritizerConfig

    path = _write_yaml(
        tmp_path,
        """
rules: []
prioritizer:
  cooldowns:
    confidence_threshold: 120
    sentiment_shift: 240
  composites:
    - name: combo
      requires: [action_changed, sentiment_shift]
      severity: critical
      message: "{symbol}: combo"
  max_per_tick: 5
""",
    )
    config = load_prioritizer_config_from_yaml(path)
    assert isinstance(config, PrioritizerConfig)
    assert config.cooldowns == {"confidence_threshold": 120, "sentiment_shift": 240}
    assert config.composites == (
        CompositeRule(
            name="combo",
            requires=("action_changed", "sentiment_shift"),
            severity="critical",
            message="{symbol}: combo",
        ),
    )
    assert config.max_per_tick == 5


def test_prio_loader_rejects_negative_cooldown(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
prioritizer:
  cooldowns:
    confidence_threshold: -1
""",
    )
    with pytest.raises(ValueError, match="non-negative"):
        load_prioritizer_config_from_yaml(path)


def test_prio_loader_rejects_bad_composite_severity(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
prioritizer:
  composites:
    - name: combo
      requires: [action_changed, sentiment_shift]
      severity: APOCALYPSE
      message: "x"
""",
    )
    with pytest.raises(ValueError, match="severity must be one of"):
        load_prioritizer_config_from_yaml(path)


def test_prio_loader_rejects_empty_requires(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
prioritizer:
  composites:
    - name: combo
      requires: []
      severity: critical
      message: "x"
""",
    )
    with pytest.raises(ValueError, match="non-empty list"):
        load_prioritizer_config_from_yaml(path)


def test_prio_loader_example_yaml_in_repo_parses() -> None:
    """The shipped example must always parse cleanly."""
    from src.intelligence.alert_prioritizer import PrioritizerConfig

    path = Path(__file__).resolve().parents[1] / "config" / "alerts.example.yaml"
    if not path.exists():
        pytest.skip(f"{path} not present")
    config = load_prioritizer_config_from_yaml(path)
    assert isinstance(config, PrioritizerConfig)
    # The example ships at least one composite + the two documented cooldowns.
    assert len(config.composites) >= 1
    assert "confidence_threshold" in config.cooldowns


# ---------------------------------------------------------------------------
# Engine error-row handling
# ---------------------------------------------------------------------------


def test_engine_ignores_error_rows_for_baseline() -> None:
    engine = AlertEngine()
    good = _row(action=SignalAction.BUY, confidence=0.5)
    engine.evaluate([good])
    err = _row(action=SignalAction.HOLD, confidence=0.0, error="fetch failed")
    # Error row shouldn't fire alerts and shouldn't overwrite baseline.
    alerts = engine.evaluate([err])
    assert alerts == []
    # Now a fresh good row should still compare against the original baseline.
    next_good = _row(action=SignalAction.SELL, confidence=0.5)
    alerts = engine.evaluate([next_good])
    assert any(a.rule == "action_changed" for a in alerts)
