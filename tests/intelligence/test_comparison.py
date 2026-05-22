"""Tests for the pure side-by-side comparison module.

Exercises per-metric verdicts, direction handling, section grouping,
overall tally, headline strip formatting, and edge cases like missing
sides and identical reports.

The comparison module takes the analyzer wire-dict shape; these tests
construct minimal dicts that mirror that contract so they don't need
the full fundamentals stack to run.
"""

from __future__ import annotations

from typing import Any

from src.intelligence.comparison import (
    ComparisonView,
    MetricComparison,
    compare_reports,
)


def _report(
    *,
    symbol: str = "AAA",
    trust_score: float = 90.0,
    trust_grade: str = "A",
    last_price: float = 100.0,
    overall_analyzer_score: float = 70.0,
    provider_confidence: float = 0.85,
    valuation_confidence: float = 80.0,
    base_case: float | None = 120.0,
    overbought: float = 40.0,
    oversold: float = 60.0,
    pullback_risk: float = 30.0,
    rebound_potential: float = 70.0,
    warnings: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a minimal analyzer-report-shaped dict for comparison tests.

    Only the fields the comparison module reads are populated; the
    full analyzer report carries many more, but the comparison
    contract is narrow by design.
    """
    return {
        "symbol": symbol,
        "last_price": last_price,
        "overall_analyzer_score": overall_analyzer_score,
        "trust_score": {"score": trust_score, "grade": trust_grade, "components": []},
        "fundamentals_freshness": {"provider_confidence": provider_confidence},
        "valuation": {
            "confidence_score": valuation_confidence,
            "base_case": base_case,
        },
        "technicals": {
            "overbought_score": overbought,
            "oversold_score": oversold,
            "pullback_risk": pullback_risk,
            "rebound_potential": rebound_potential,
        },
        "warnings": list(warnings),
    }


def _by_metric(view: ComparisonView) -> dict[str, MetricComparison]:
    out: dict[str, MetricComparison] = {}
    for section in view.sections:
        for metric in section.metrics:
            out[metric.metric] = metric
    return out


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------


def test_compare_returns_well_shaped_view() -> None:
    view = compare_reports(_report(symbol="AAA"), _report(symbol="BBB"))
    assert view.left_symbol == "AAA"
    assert view.right_symbol == "BBB"
    # Four sections in declared order: Trust, Valuation, Technical, Risk.
    titles = [s.title for s in view.sections]
    assert titles == ["Trust", "Valuation", "Technical", "Risk"]
    # Headline has three rows: trust grade, last price, overall.
    assert len(view.headline) == 3


def test_to_dict_carries_every_metric() -> None:
    view = compare_reports(_report(), _report(symbol="BBB"))
    wire = view.to_dict()
    metric_names = [
        m["metric"] for s in wire["sections"] for m in s["metrics"]
    ]
    assert set(metric_names) == {
        "trust_score",
        "provider_confidence",
        "valuation_confidence",
        "upside_to_base_case",
        "overbought_score",
        "oversold_score",
        "pullback_risk",
        "rebound_potential",
    }


# ---------------------------------------------------------------------------
# Per-metric verdicts
# ---------------------------------------------------------------------------


def test_higher_trust_score_wins_left() -> None:
    left = _report(trust_score=95.0)
    right = _report(trust_score=70.0)
    view = compare_reports(left, right)
    assert _by_metric(view)["trust_score"].winner == "left"


def test_lower_overbought_wins_because_direction_is_inverted() -> None:
    """Overbought is bad for entries → lower value wins."""
    left = _report(overbought=80.0)  # heavily overbought, bad
    right = _report(overbought=30.0)  # not overbought, good
    view = compare_reports(left, right)
    metric = _by_metric(view)["overbought_score"]
    assert metric.winner == "right"
    assert metric.direction == "lower_better"


def test_values_within_tolerance_are_a_tie() -> None:
    """``tie_within`` rounds near-identical values to a tie verdict."""
    left = _report(trust_score=90.4)
    right = _report(trust_score=89.7)  # diff = 0.7 < tie_within=1.0
    view = compare_reports(left, right)
    assert _by_metric(view)["trust_score"].winner == "tie"


def test_missing_side_loses_by_default() -> None:
    """If one side has a value and the other doesn't, the populated
    side wins."""
    left = _report()
    right = _report(trust_score=90.0)
    # Strip the trust score from the right side to simulate missing data.
    right["trust_score"] = {"score": None, "grade": None, "components": []}
    view = compare_reports(left, right)
    assert _by_metric(view)["trust_score"].winner == "left"
    assert _by_metric(view)["trust_score"].right_value is None


def test_both_sides_missing_renders_na() -> None:
    left = _report()
    right = _report()
    left["valuation"] = {"confidence_score": None, "base_case": None}
    right["valuation"] = {"confidence_score": None, "base_case": None}
    view = compare_reports(left, right)
    metric = _by_metric(view)["valuation_confidence"]
    assert metric.winner == "n/a"
    assert metric.left_value is None
    assert metric.right_value is None


def test_upside_to_base_case_derives_percent() -> None:
    """``base_case - last_price`` / last_price expressed as %."""
    left = _report(last_price=100.0, base_case=130.0)   # +30% upside
    right = _report(last_price=100.0, base_case=110.0)  # +10% upside
    view = compare_reports(left, right)
    metric = _by_metric(view)["upside_to_base_case"]
    assert metric.left_value is not None
    assert metric.right_value is not None
    assert metric.left_value == 30.0
    assert metric.right_value == 10.0
    assert metric.winner == "left"
    assert metric.percent is True


def test_upside_returns_none_when_last_price_is_zero() -> None:
    """Avoid divide-by-zero on bad input."""
    left = _report(last_price=0.0, base_case=100.0)
    right = _report()
    view = compare_reports(left, right)
    assert _by_metric(view)["upside_to_base_case"].left_value is None


# ---------------------------------------------------------------------------
# Overall tally
# ---------------------------------------------------------------------------


def test_majority_metrics_decide_overall_winner() -> None:
    left = _report(
        trust_score=95.0,
        provider_confidence=0.95,
        overbought=20.0,  # left wins (lower-better)
    )
    right = _report(
        trust_score=70.0,
        provider_confidence=0.5,
        overbought=80.0,
    )
    view = compare_reports(left, right)
    assert view.overall_winner == "left"


def test_balanced_metrics_collapse_to_tie() -> None:
    """A symmetric scenario where each side wins the same number of metrics."""
    left = _report(
        trust_score=95.0,             # left
        provider_confidence=0.95,     # left
        valuation_confidence=85.0,    # left (within tolerance? 5 > 2 so left)
        base_case=130.0,              # left (+30%)
        overbought=80.0,              # right wins (lower-better)
        oversold=20.0,                # right wins (higher-better)
        pullback_risk=80.0,           # right wins (lower-better)
        rebound_potential=20.0,       # right wins (higher-better)
    )
    right = _report(
        trust_score=70.0,
        provider_confidence=0.5,
        valuation_confidence=70.0,
        base_case=110.0,
        overbought=30.0,
        oversold=80.0,
        pullback_risk=30.0,
        rebound_potential=80.0,
    )
    view = compare_reports(left, right)
    # 4 left wins + 4 right wins → tie.
    assert view.overall_winner == "tie"


def test_no_data_anywhere_overall_is_na() -> None:
    """When literally no metric resolved, overall is ``"n/a"``."""
    left = _report()
    right = _report()
    # Wipe every comparable field on both sides.
    for r in (left, right):
        r["trust_score"] = {"score": None, "grade": None, "components": []}
        r["fundamentals_freshness"] = {"provider_confidence": None}
        r["valuation"] = {"confidence_score": None, "base_case": None}
        r["technicals"] = {
            "overbought_score": None,
            "oversold_score": None,
            "pullback_risk": None,
            "rebound_potential": None,
        }
        r["last_price"] = None
    view = compare_reports(left, right)
    assert view.overall_winner == "n/a"


# ---------------------------------------------------------------------------
# Headline strip
# ---------------------------------------------------------------------------


def test_headline_includes_trust_grade_and_price() -> None:
    left = _report(trust_grade="A+", last_price=180.45)
    right = _report(trust_grade="C", last_price=42.10)
    view = compare_reports(left, right)
    labels = [h.label for h in view.headline]
    assert labels == ["Trust grade", "Last price", "Overall analyzer"]
    grade_row = view.headline[0]
    assert grade_row.left_display == "A+"
    assert grade_row.right_display == "C"
    assert grade_row.winner == "left"  # A+ > C
    price_row = view.headline[1]
    assert price_row.left_display == "$180.45"
    assert price_row.winner == "n/a"  # price is informational only


def test_headline_grade_winner_uses_explicit_order_not_alphabetic() -> None:
    """B > C even though alphabetically C > B; A+ > A by suffix."""
    left = _report(trust_grade="B")
    right = _report(trust_grade="C")
    view = compare_reports(left, right)
    assert view.headline[0].winner == "left"

    left2 = _report(trust_grade="A+")
    right2 = _report(trust_grade="A")
    view2 = compare_reports(left2, right2)
    assert view2.headline[0].winner == "left"


# ---------------------------------------------------------------------------
# Degraded inputs
# ---------------------------------------------------------------------------


def test_both_sides_none_produces_empty_view_with_warning() -> None:
    view = compare_reports(None, None)
    assert view.left_symbol == ""
    assert view.right_symbol == ""
    assert view.sections == ()
    assert view.headline == ()
    assert view.overall_winner == "n/a"
    assert len(view.warnings) == 1


def test_one_side_none_marks_other_side_winner_per_metric() -> None:
    """A missing right report → left wins everywhere it has data."""
    view = compare_reports(_report(symbol="AAA"), None)
    assert view.left_symbol == "AAA"
    assert view.right_symbol == "—"
    # Every metric has a left value but no right value → left wins.
    metrics = _by_metric(view)
    for m in metrics.values():
        assert m.right_value is None
        # Either the left side wins (it has data) or n/a (nothing on left
        # either, but for the test fixture every left field is populated).
        assert m.winner in {"left", "n/a"}
    # Comparison surfaces a warning about the degraded right side.
    assert any("only one report available" in w for w in view.warnings)
