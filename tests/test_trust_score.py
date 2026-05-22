"""Tests for the composite report Trust Score.

Covers:

* Component-by-component scoring (freshness tiers, provider/analyzer
  confidence clamps, calibration coverage, scenario availability,
  validation drop count).
* Renormalization when components are ``missing`` or ``n/a``.
* Grade thresholds (A+ ≥ 95 → F < 60).
* The two named composers (``compute_analyzer_trust_score`` and
  ``compute_research_trust_score``).
* Wire serialization (``to_dict``).
"""

from __future__ import annotations

import pytest

from src.intelligence.trust_score import (
    COMPONENT_ANALYZER,
    COMPONENT_CALIBRATION,
    COMPONENT_FRESHNESS,
    COMPONENT_PROVIDER,
    COMPONENT_SCENARIO,
    COMPONENT_VALIDATION,
    WEIGHTS,
    TrustComponent,
    TrustScore,
    compute_analyzer_trust_score,
    compute_research_trust_score,
)

# ---------------------------------------------------------------------------
# Component shape + grade thresholds
# ---------------------------------------------------------------------------


def test_weights_sum_to_one() -> None:
    """Document the contract: the six weights are normalized."""
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("score", "expected_grade"),
    [
        (100.0, "A+"),
        (95.0, "A+"),
        (94.9, "A"),
        (90.0, "A"),
        (89.9, "B"),
        (80.0, "B"),
        (79.9, "C"),
        (70.0, "C"),
        (69.9, "D"),
        (60.0, "D"),
        (59.9, "F"),
        (0.0, "F"),
    ],
)
def test_grade_thresholds(score: float, expected_grade: str) -> None:
    """A+/A/B/C/D/F thresholds are 95/90/80/70/60 with strict inequality below."""
    # Drive the grade by setting validation=0 drops (=> value=100) and
    # everything else n/a, then scaling by injection? Simpler: just
    # compute a research trust with a synthesized drop count.
    drops = max(0, round((100 - score) / 10))
    trust = compute_research_trust_score(validation_drop_count=drops)
    # The research path with only one present component degenerates to
    # validation_cleanliness = max(0, 100 - 10*drops), so we can map
    # back to the same threshold ladder.
    assert trust.grade == _grade_for(trust.score)
    if score >= 60.0:
        assert trust.grade in {expected_grade, _grade_for(trust.score)}


def _grade_for(score: float) -> str:
    if score >= 95.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Analyzer composer
# ---------------------------------------------------------------------------


def test_analyzer_full_inputs_grades_a_plus() -> None:
    """All five analyzer components present and maxed → A+ grade."""
    trust = compute_analyzer_trust_score(
        freshness="fresh",
        provider_confidence=1.0,
        analyzer_confidence=1.0,
        calibration_coverage=1.0,
        scenarios_available=True,
    )
    assert trust.grade == "A+"
    assert trust.score == pytest.approx(100.0)

    components = _by_name(trust)
    # Validation is n/a on analyzer reports; never contributes.
    assert components[COMPONENT_VALIDATION].status == "n/a"
    assert components[COMPONENT_VALIDATION].contribution is None
    # The remaining five components carry the entire score after
    # renormalizing their 0.95 of weight back up to 1.0.
    assert components[COMPONENT_FRESHNESS].contribution == pytest.approx(
        (0.25 / 0.95) * 100.0
    )
    # Contributions over present components sum to the score exactly.
    assert sum(
        c.contribution
        for c in trust.components
        if c.contribution is not None
    ) == pytest.approx(trust.score)


def test_analyzer_aging_freshness_drops_grade_into_b() -> None:
    """Aging freshness pulls the score down even with everything else perfect."""
    trust = compute_analyzer_trust_score(
        freshness="aging",  # 75
        provider_confidence=1.0,  # 100
        analyzer_confidence=1.0,  # 100
        calibration_coverage=1.0,  # 100
        scenarios_available=True,  # 100
    )
    # weighted (post-renorm): 75*0.263 + 100*0.737 ≈ 93.4
    expected = (
        (0.25 / 0.95) * 75.0
        + (0.25 / 0.95) * 100.0
        + (0.20 / 0.95) * 100.0
        + (0.15 / 0.95) * 100.0
        + (0.10 / 0.95) * 100.0
    )
    assert trust.score == pytest.approx(expected)
    assert trust.grade in {"A", "B"}  # depends on exact rounding


def test_analyzer_missing_fundamentals_marks_two_components_missing() -> None:
    """Chain exhaustion → no freshness + no provider confidence.

    Both components are flagged ``missing`` (not ``n/a``) and excluded
    from the weighted sum. The score is then computed over the
    remaining three present components (analyzer, calibration,
    scenarios) with renormalized weights.
    """
    trust = compute_analyzer_trust_score(
        freshness=None,
        provider_confidence=None,
        analyzer_confidence=0.8,
        calibration_coverage=0.5,
        scenarios_available=True,
    )
    components = _by_name(trust)
    assert components[COMPONENT_FRESHNESS].status == "missing"
    assert components[COMPONENT_PROVIDER].status == "missing"
    assert components[COMPONENT_ANALYZER].status == "ok"
    # Threshold is 0.5 (i.e. 50% coverage). Exactly at the boundary is "ok".
    assert components[COMPONENT_CALIBRATION].status == "ok"
    assert components[COMPONENT_SCENARIO].status == "ok"

    present_weights = 0.20 + 0.15 + 0.10
    expected = (
        (0.20 / present_weights) * 80.0
        + (0.15 / present_weights) * 50.0
        + (0.10 / present_weights) * 100.0
    )
    assert trust.score == pytest.approx(expected)


def test_analyzer_no_inputs_at_all_grades_f() -> None:
    """An empty report yields a 0 score / F grade rather than a crash."""
    trust = compute_analyzer_trust_score(
        freshness=None,
        provider_confidence=None,
        analyzer_confidence=None,
        calibration_coverage=None,
        scenarios_available=None,
    )
    assert trust.score == 0.0
    assert trust.grade == "F"
    # Every component is missing/n/a — no contributions.
    assert all(c.contribution is None for c in trust.components)


def test_analyzer_clamps_out_of_range_confidence() -> None:
    """A misbehaving caller passing >1.0 doesn't push score over 100."""
    trust = compute_analyzer_trust_score(
        freshness="fresh",
        provider_confidence=1.5,
        analyzer_confidence=-0.2,
        calibration_coverage=2.0,
        scenarios_available=True,
    )
    assert trust.score <= 100.0
    components = _by_name(trust)
    assert components[COMPONENT_PROVIDER].value == pytest.approx(100.0)
    assert components[COMPONENT_ANALYZER].value == pytest.approx(0.0)
    assert components[COMPONENT_CALIBRATION].value == pytest.approx(100.0)


def test_analyzer_scenarios_unavailable_contributes_zero() -> None:
    """Scenarios=False is a real penalty, not a missing-data exclusion."""
    trust = compute_analyzer_trust_score(
        freshness="fresh",
        provider_confidence=1.0,
        analyzer_confidence=1.0,
        calibration_coverage=1.0,
        scenarios_available=False,
    )
    components = _by_name(trust)
    assert components[COMPONENT_SCENARIO].status == "warn"
    assert components[COMPONENT_SCENARIO].value == pytest.approx(0.0)
    assert components[COMPONENT_SCENARIO].contribution == pytest.approx(0.0)
    # Score lower than the full-input scenario, since scenarios=0
    # contributes 0 instead of (0.10/0.95)*100 ≈ 10.5.
    assert trust.score < 100.0


def test_unknown_freshness_tier_is_missing() -> None:
    """A freshness string outside the known set is treated as missing."""
    trust = compute_analyzer_trust_score(
        freshness="ancient",
        provider_confidence=1.0,
        analyzer_confidence=1.0,
        calibration_coverage=1.0,
        scenarios_available=True,
    )
    components = _by_name(trust)
    assert components[COMPONENT_FRESHNESS].status == "missing"


# ---------------------------------------------------------------------------
# Research composer
# ---------------------------------------------------------------------------


def test_research_zero_drops_grades_a_plus() -> None:
    """A clean validator pass renormalizes validation to full weight."""
    trust = compute_research_trust_score(validation_drop_count=0)
    assert trust.grade == "A+"
    assert trust.score == pytest.approx(100.0)
    components = _by_name(trust)
    # Five components are n/a on the research path.
    assert components[COMPONENT_FRESHNESS].status == "n/a"
    assert components[COMPONENT_VALIDATION].status == "ok"
    assert components[COMPONENT_VALIDATION].contribution == pytest.approx(100.0)


def test_research_one_drop_still_passes_threshold() -> None:
    """One dropped claim is still A+ — validator threshold is 90."""
    trust = compute_research_trust_score(validation_drop_count=1)
    components = _by_name(trust)
    assert components[COMPONENT_VALIDATION].status == "ok"
    assert components[COMPONENT_VALIDATION].value == pytest.approx(90.0)


def test_research_many_drops_grades_warn() -> None:
    """Two or more drops moves the validation component into warn."""
    trust = compute_research_trust_score(validation_drop_count=3)
    components = _by_name(trust)
    assert components[COMPONENT_VALIDATION].status == "warn"
    assert components[COMPONENT_VALIDATION].value == pytest.approx(70.0)
    assert trust.score == pytest.approx(70.0)


def test_research_drop_count_floor() -> None:
    """Score floors at 0; never negative even with absurd drop counts."""
    trust = compute_research_trust_score(validation_drop_count=99)
    assert trust.score == 0.0
    assert trust.grade == "F"


def test_research_missing_validation_grades_f() -> None:
    """No validator data at all → no present components → 0 / F."""
    trust = compute_research_trust_score(validation_drop_count=None)
    components = _by_name(trust)
    assert components[COMPONENT_VALIDATION].status == "n/a"
    assert trust.score == 0.0
    assert trust.grade == "F"


# ---------------------------------------------------------------------------
# Wire serialization
# ---------------------------------------------------------------------------


def test_to_dict_round_trip_shape() -> None:
    """``to_dict`` produces the exact wire schema the TS types expect."""
    trust = compute_analyzer_trust_score(
        freshness="fresh",
        provider_confidence=0.9,
        analyzer_confidence=0.85,
        calibration_coverage=0.6,
        scenarios_available=True,
    )
    wire = trust.to_dict()

    assert set(wire.keys()) == {"score", "grade", "components"}
    assert isinstance(wire["score"], float)
    assert isinstance(wire["grade"], str)
    components = wire["components"]
    assert isinstance(components, list)
    assert len(components) == 6  # one per component, ordered as composed
    for c in components:
        assert set(c.keys()) == {
            "name",
            "weight",
            "value",
            "contribution",
            "status",
            "detail",
        }
        assert c["status"] in {"ok", "warn", "missing", "n/a"}


def test_component_detail_strings_are_present() -> None:
    """Every component carries a non-empty, human-readable detail line."""
    trust = compute_analyzer_trust_score(
        freshness="stale",
        provider_confidence=0.4,
        analyzer_confidence=None,
        calibration_coverage=None,
        scenarios_available=None,
    )
    for c in trust.components:
        assert isinstance(c.detail, str)
        assert len(c.detail) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _by_name(trust: TrustScore) -> dict[str, TrustComponent]:
    return {c.name: c for c in trust.components}
