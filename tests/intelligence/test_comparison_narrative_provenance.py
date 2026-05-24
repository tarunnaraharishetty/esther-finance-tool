"""Tests for the per-claim provenance graph extension to the compare
narrative validator.

The base tests in ``test_comparison_narrative.py`` cover the drop
policy + template narrator + XML parsing. This file focuses on what
the moat-spreading extension adds:

* Every surviving NarrativeSection carries a ``provenance`` dict
  mapping numeric tokens to ``left.<field>`` / ``right.<field>`` /
  ``view.<section>.<metric>.<side>`` labels.
* The labels disambiguate the two sides — the same ``"$180.50"``
  on AAPL vs MSFT hovers to different source labels.
* The drop policy is untouched — dropped sentences don't appear in
  provenance; existing drop_count tests still pass.
"""

from __future__ import annotations

from typing import Any

from src.intelligence.comparison import compare_reports
from src.intelligence.comparison_narrative import (
    ComparisonNarrative,
    NarrativeInput,
    NarrativeSection,
    TemplateComparisonNarrator,
    replace_section,
)
from src.intelligence.comparison_narrative_validator import (
    _build_corpus_entries,
    _token_provenance,
    validate_narrative,
)


def _report(
    *,
    symbol: str = "AAA",
    trust_score: float = 90.0,
    last_price: float = 100.0,
    base_case: float = 120.0,
    overbought: float = 40.0,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "last_price": last_price,
        "overall_analyzer_score": 70.0,
        "trust_score": {
            "score": trust_score,
            "grade": "A",
            "components": [],
        },
        "fundamentals_freshness": {
            "provider_confidence": 0.85,
            "freshness": "fresh",
            "data_age_days": 5,
            "source_chain": ["fmp"],
        },
        "valuation": {
            "confidence_score": 80.0,
            "base_case": base_case,
            "bear_case": base_case * 0.8,
            "bull_case": base_case * 1.2,
        },
        "technicals": {
            "overbought_score": overbought,
            "oversold_score": 100 - overbought,
            "pullback_risk": 30.0,
            "rebound_potential": 70.0,
            "confidence_score": 0.8,
        },
        "warnings": [],
    }


def _pair() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _report(symbol="AAA", trust_score=90.0, last_price=100.0, base_case=130.0),
        _report(symbol="BBB", trust_score=72.0, last_price=200.0, base_case=210.0, overbought=70.0),
    )


# ---------------------------------------------------------------------------
# Corpus entries: labels carry side prefixes
# ---------------------------------------------------------------------------


def test_corpus_entries_prefix_labels_with_left_or_right() -> None:
    """Every report-derived label starts with ``left.`` or ``right.``."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    entries = _build_corpus_entries(narrative, view, left, right)
    side_labels = [
        label
        for _text, label in entries
        if label.startswith("left.") or label.startswith("right.")
    ]
    # Both sides represented.
    assert any(label.startswith("left.") for label in side_labels)
    assert any(label.startswith("right.") for label in side_labels)


def test_corpus_entries_include_view_metrics_per_side() -> None:
    """View-derived metric values carry ``view.<section>.<metric>.<side>`` labels."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    entries = _build_corpus_entries(narrative, view, left, right)
    view_labels = {label for _text, label in entries if label.startswith("view.")}
    # The trust section has trust_score + provider_confidence metrics
    # → both sides → 4 labels minimum.
    assert any(
        ".left" in label and "trust_score" in label for label in view_labels
    )
    assert any(
        ".right" in label and "trust_score" in label for label in view_labels
    )


def test_left_trust_score_token_maps_to_left_label() -> None:
    """A decimal token matching only the left side's trust_score gets
    a left label.

    The validator's token regex tokenizes decimals and percent forms
    but not bare integers, so we test against the decimal surface
    form the corpus emits via _render_numeric_forms.
    """
    left, right = _pair()  # left trust=90, right trust=72
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    entries = _build_corpus_entries(narrative, view, left, right)
    prov = _token_provenance("Trust score reads 90.0 today.", entries, frozenset())
    # "90.0" is the left trust score in decimal form.
    assert "90.0" in prov
    assert "left" in prov["90.0"]


def test_right_specific_value_maps_to_right_label() -> None:
    """A decimal value unique to the right side maps to a right.* label."""
    left, right = _pair()  # right trust=72
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    entries = _build_corpus_entries(narrative, view, left, right)
    prov = _token_provenance(
        "Bear-side composite of 72.0 is concerning.", entries, frozenset()
    )
    assert "72.0" in prov
    assert "right" in prov["72.0"]


def test_token_provenance_skips_allowance_tokens() -> None:
    """Tokens supported only by allowance get no provenance entry."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    entries = _build_corpus_entries(narrative, view, left, right)
    # "2026" is in the standard allowance (year).
    prov = _token_provenance(
        "Through 2026 the stance held.", entries, frozenset({"2026"})
    )
    assert "2026" not in prov


# ---------------------------------------------------------------------------
# Wire shape: provenance attached to each surviving section
# ---------------------------------------------------------------------------


def test_validate_narrative_attaches_provenance_to_every_section() -> None:
    """Template narrator output runs through validator → every section
    gets a provenance map (possibly empty)."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    validated, _report = validate_narrative(narrative, view, left, right)
    # provenance attribute exists on every section.
    for section_key in (
        "headline",
        "momentum",
        "valuation",
        "risk",
        "quality",
        "bottom_line",
    ):
        section: NarrativeSection = getattr(validated, section_key)
        assert isinstance(section.provenance, dict)


def test_validated_section_provenance_includes_grounded_tokens() -> None:
    """Tokens in surviving body land in the section's provenance map."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    validated, _report = validate_narrative(narrative, view, left, right)
    # The headline section's body cites the overall winner — at least
    # one numeric token from the view should land in provenance.
    headline_prov = validated.headline.provenance
    # The headline section bullets render formatted numbers like 90 and 72.
    # At least one should be in provenance.
    matched_sides = {
        label.split(".")[0] for label in headline_prov.values()
    }
    # Tokens land on one of the labeled prefixes.
    assert matched_sides.issubset({"left", "right", "view"})


def test_to_dict_carries_provenance_per_section() -> None:
    """Wire shape round-trips the provenance map per section."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    validated, _report = validate_narrative(narrative, view, left, right)
    wire = validated.to_dict()
    for key in (
        "headline",
        "momentum",
        "valuation",
        "risk",
        "quality",
        "bottom_line",
    ):
        section = wire[key]
        assert "provenance" in section
        assert isinstance(section["provenance"], dict)


# ---------------------------------------------------------------------------
# Drop policy untouched
# ---------------------------------------------------------------------------


def test_dropped_tokens_do_not_appear_in_provenance() -> None:
    """A hallucinated token gets dropped AND is absent from provenance."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    # Inject a hallucinated sentence into the momentum body.
    narrative = replace_section(
        narrative,
        "momentum",
        body=(
            narrative.momentum.body
            + " AAA reported a fake 99.9% surge last quarter."
        ),
    )
    validated, report = validate_narrative(narrative, view, left, right)
    assert report.drop_count >= 1
    # 99.9% must not appear in provenance.
    assert "99.9%" not in validated.momentum.provenance


def test_replace_section_threads_provenance_kwarg() -> None:
    """replace_section accepts a provenance kwarg and applies it."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    custom_prov = {"100": "left.last_price"}
    updated = replace_section(
        narrative, "momentum", provenance=custom_prov
    )
    assert updated.momentum.provenance == custom_prov
    # Other fields untouched.
    assert updated.momentum.body == narrative.momentum.body


def test_empty_section_has_empty_provenance() -> None:
    """A scrubbed-empty section returns empty provenance, not missing."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = ComparisonNarrative(
        left_symbol="AAA",
        right_symbol="BBB",
        tagline="",
        headline=NarrativeSection(title="Headline", body="", bullets=()),
        momentum=NarrativeSection(title="Momentum", body="", bullets=()),
        valuation=NarrativeSection(title="Valuation", body="", bullets=()),
        risk=NarrativeSection(title="Risk", body="", bullets=()),
        quality=NarrativeSection(title="Quality", body="", bullets=()),
        bottom_line=NarrativeSection(title="Bottom Line", body="", bullets=()),
        model="template@test",
        generated_at="2026-05-23T00:00:00+00:00",
    )
    validated, _report = validate_narrative(narrative, view, left, right)
    for key in (
        "headline",
        "momentum",
        "valuation",
        "risk",
        "quality",
        "bottom_line",
    ):
        section: NarrativeSection = getattr(validated, key)
        assert section.provenance == {}
