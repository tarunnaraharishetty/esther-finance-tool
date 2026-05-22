"""Tests for the comparison-narrative module + validator.

Covers:

* TemplateComparisonNarrator deterministic output (every section
  populated, validator drops zero on the template path).
* LLMComparisonNarrator XML parsing (positive + degraded cases).
* Validator behavior (corpus build, sentence scrubbing, allowance,
  empty input edge cases, dropped-claim shape).
* Grounding rules constant carries the comparison-specific clauses.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.intelligence.comparison import compare_reports
from src.intelligence.comparison_narrative import (
    SECTION_KEYS,
    SECTION_TITLES,
    ComparisonNarrative,
    NarrativeInput,
    NarrativeSection,
    TemplateComparisonNarrator,
    _parse_narrative_xml,
    replace_section,
)
from src.intelligence.comparison_narrative_validator import (
    DroppedNarrativeClaim,
    narrative_validation_to_wire,
    validate_narrative,
)
from src.intelligence.grounding import (
    GROUNDING_RULES_COMPARISON,
    GROUNDING_RULES_PER_SYMBOL,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _report(
    *,
    symbol: str = "AAA",
    trust_score: float = 90.0,
    trust_grade: str = "A",
    last_price: float = 100.0,
    overall: float = 70.0,
    provider_confidence: float = 0.85,
    freshness: str = "fresh",
    valuation_confidence: float = 80.0,
    base_case: float | None = 120.0,
    overbought: float = 40.0,
    oversold: float = 60.0,
    pullback_risk: float = 30.0,
    rebound_potential: float = 70.0,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "last_price": last_price,
        "overall_analyzer_score": overall,
        "trust_score": {
            "score": trust_score,
            "grade": trust_grade,
            "components": [],
        },
        "fundamentals_freshness": {
            "provider_confidence": provider_confidence,
            "freshness": freshness,
            "data_age_days": 5,
            "source_chain": ["fmp"],
        },
        "valuation": {
            "confidence_score": valuation_confidence,
            "base_case": base_case,
            "bear_case": (base_case * 0.8) if base_case else None,
            "bull_case": (base_case * 1.2) if base_case else None,
        },
        "technicals": {
            "overbought_score": overbought,
            "oversold_score": oversold,
            "pullback_risk": pullback_risk,
            "rebound_potential": rebound_potential,
            "confidence_score": 0.8,
            "raw_rsi": 58.0,
            "raw_bollinger_z": 0.5,
            "raw_ma_distance_pct": 1.2,
            "raw_volume_z": 0.3,
            "raw_atr_ratio": 0.4,
        },
        "warnings": [],
    }


def _pair(*, left_revenue_seed: float = 1.0) -> tuple[dict[str, Any], dict[str, Any]]:
    left = _report(symbol="AAA")
    right = _report(
        symbol="BBB",
        trust_score=72.0,
        trust_grade="C",
        last_price=200.0,
        overall=55.0,
        provider_confidence=0.55,
        freshness="aging",
        valuation_confidence=60.0,
        base_case=210.0,
        overbought=70.0,
        oversold=25.0,
        pullback_risk=65.0,
        rebound_potential=35.0,
    )
    return left, right


# ---------------------------------------------------------------------------
# Grounding rules
# ---------------------------------------------------------------------------


def test_comparison_grounding_rules_inherit_core_clauses() -> None:
    """The comparison rules include the same core anti-hallucination text."""
    # Each of the core clauses is a verbatim substring in both flavors.
    for clause_prefix in [
        "Only summarize information present",
        "Quote headlines verbatim",
        "Describe the current state — do not predict",
    ]:
        assert clause_prefix in GROUNDING_RULES_PER_SYMBOL
        assert clause_prefix in GROUNDING_RULES_COMPARISON


def test_comparison_grounding_rules_add_two_symbol_scope() -> None:
    """The comparison-specific clauses must appear verbatim."""
    assert "two symbols" in GROUNDING_RULES_COMPARISON.lower()
    assert "pre-computed metric verdicts" in GROUNDING_RULES_COMPARISON
    assert "split" in GROUNDING_RULES_COMPARISON.lower()


# ---------------------------------------------------------------------------
# Template narrator
# ---------------------------------------------------------------------------


def test_template_narrator_populates_every_section() -> None:
    left, right = _pair()
    view = compare_reports(left, right)
    narrator = TemplateComparisonNarrator()
    narrative = narrator.generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    for key in SECTION_KEYS:
        section: NarrativeSection = getattr(narrative, key)
        assert section.title == SECTION_TITLES[key]
        assert section.body, f"section {key} body empty"


def test_template_narrator_includes_winning_symbol_in_tagline() -> None:
    left, right = _pair()
    view = compare_reports(left, right)
    narrator = TemplateComparisonNarrator()
    narrative = narrator.generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    assert "AAA" in narrative.tagline
    assert "BBB" in narrative.tagline


def test_template_narrator_validator_drops_zero_claims() -> None:
    """Template output is grounded by construction → drop count is 0."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    _validated, report = validate_narrative(narrative, view, left, right)
    assert report.drop_count == 0, (
        f"template narrator should ground everything, but dropped: "
        f"{report.dropped_claims}"
    )


def test_template_narrator_handles_tie_overall() -> None:
    """When the overall verdict is a tie, tagline reflects that."""
    # Build symmetric reports so each metric ties.
    left = _report(symbol="AAA")
    right = _report(symbol="BBB")
    view = compare_reports(left, right)
    narrator = TemplateComparisonNarrator()
    narrative = narrator.generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    # When inputs are identical every metric ties; the overall_winner
    # path renders the tie or n/a branch; either way tagline mentions both.
    assert "AAA" in narrative.tagline and "BBB" in narrative.tagline


def test_template_model_id_is_versioned() -> None:
    """``model_id`` carries the prompt version so cache keys can pin it."""
    assert "template" in TemplateComparisonNarrator().model_id
    assert "@" in TemplateComparisonNarrator().model_id


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def test_parse_xml_extracts_every_section() -> None:
    left, right = _pair()
    view = compare_reports(left, right)
    xml = """
    <narrative>
      <tagline>AAA edges BBB across the trust dimension.</tagline>
      <headline>
        <body>AAA leads the headline read with trust 90 vs 72.</body>
        <bullets>
          <item>Trust grade A vs C is the headline divergence.</item>
        </bullets>
      </headline>
      <momentum>
        <body>AAA shows oversold 60 vs BBB 25.</body>
        <bullets></bullets>
      </momentum>
      <valuation>
        <body>AAA confidence 80 vs BBB 60.</body>
        <bullets></bullets>
      </valuation>
      <risk>
        <body>AAA pullback risk 30 vs BBB 65.</body>
        <bullets></bullets>
      </risk>
      <quality>
        <body>AAA trust 90 vs BBB 72.</body>
        <bullets></bullets>
      </quality>
      <bottom_line>
        <body>The split favors AAA across measured sections.</body>
      </bottom_line>
    </narrative>
    """
    parsed = _parse_narrative_xml(xml, view=view, model_id="test@v1")
    assert parsed.tagline.startswith("AAA edges")
    assert "AAA leads" in parsed.headline.body
    assert "Trust grade A vs C" in parsed.headline.bullets[0]
    assert parsed.bottom_line.body.startswith("The split")
    assert parsed.model == "test@v1"


def test_parse_xml_strips_markdown_fence() -> None:
    """Defense against models that wrap XML in ```xml fences."""
    left, right = _pair()
    view = compare_reports(left, right)
    fenced = (
        "```xml\n"
        "<narrative><tagline>x</tagline>"
        "<headline><body>b</body><bullets></bullets></headline>"
        "<momentum><body></body><bullets></bullets></momentum>"
        "<valuation><body></body><bullets></bullets></valuation>"
        "<risk><body></body><bullets></bullets></risk>"
        "<quality><body></body><bullets></bullets></quality>"
        "<bottom_line><body></body></bottom_line>"
        "</narrative>\n"
        "```"
    )
    parsed = _parse_narrative_xml(fenced, view=view, model_id="test@v1")
    assert parsed.tagline == "x"


def test_parse_xml_missing_sections_become_empty_not_raises() -> None:
    """Missing sections degrade to empty NarrativeSection rather than raising."""
    left, right = _pair()
    view = compare_reports(left, right)
    minimal = "<narrative><tagline>just a tag</tagline></narrative>"
    parsed = _parse_narrative_xml(minimal, view=view, model_id="test@v1")
    for key in SECTION_KEYS:
        section: NarrativeSection = getattr(parsed, key)
        assert section.body == ""
        assert section.bullets == ()


def test_parse_xml_malformed_raises_value_error() -> None:
    """Genuine parse errors propagate as ValueError for the endpoint to handle."""
    left, right = _pair()
    view = compare_reports(left, right)
    with pytest.raises(ValueError, match="malformed XML"):
        _parse_narrative_xml("<narrative><tag", view=view, model_id="test@v1")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_drops_unsupported_numbers() -> None:
    """A hallucinated number not in the corpus → claim dropped."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrator = TemplateComparisonNarrator()
    narrative = narrator.generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    # Inject a hallucinated sentence into the momentum body.
    narrative = replace_section(
        narrative,
        "momentum",
        body=narrative.momentum.body
        + " AAA reported a 42.7% surge in oversold readings last quarter.",
    )
    validated, report = validate_narrative(narrative, view, left, right)
    assert report.drop_count >= 1
    # The dropped claim carries the path and the offending token.
    dropped_section_paths = [c.section for c in report.dropped_claims]
    assert "momentum.body" in dropped_section_paths
    tokens = [t for c in report.dropped_claims for t in c.unsupported_tokens]
    assert "42.7%" in tokens
    # The validated narrative keeps the original sentences.
    assert "42.7%" not in validated.momentum.body


def test_validator_drops_bullet_entirely_when_unsupported() -> None:
    """A whole bullet with no grounded sentences is dropped from the list."""
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    bad_bullet = "Made-up margin: AAA 99.9% vs BBB 88.8%."
    narrative = replace_section(
        narrative,
        "valuation",
        bullets=(*narrative.valuation.bullets, bad_bullet),
    )
    validated, report = validate_narrative(narrative, view, left, right)
    assert bad_bullet not in validated.valuation.bullets
    assert report.drop_count >= 1


def test_validator_corpus_includes_both_symbols() -> None:
    """Both symbol tickers must be allowed regardless of corpus content."""
    left, right = _pair()
    view = compare_reports(left, right)
    # Construct a synthetic narrative that only references the symbols
    # plus small integers — should drop nothing.
    narrative = ComparisonNarrative(
        left_symbol="AAA",
        right_symbol="BBB",
        tagline="AAA and BBB are compared here.",
        headline=NarrativeSection(
            title="Headline", body="AAA and BBB both present.", bullets=()
        ),
        momentum=NarrativeSection(title="Momentum", body="", bullets=()),
        valuation=NarrativeSection(title="Valuation", body="", bullets=()),
        risk=NarrativeSection(title="Risk", body="", bullets=()),
        quality=NarrativeSection(title="Quality", body="", bullets=()),
        bottom_line=NarrativeSection(
            title="Bottom Line", body="Bottom line is AAA vs BBB.", bullets=()
        ),
        model="template@test",
        generated_at="2026-05-22T00:00:00+00:00",
    )
    _validated, report = validate_narrative(narrative, view, left, right)
    assert report.drop_count == 0


def test_validator_handles_empty_narrative_without_raising() -> None:
    """All sections empty → drop_count 0 (nothing to scrub)."""
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
        generated_at="2026-05-22T00:00:00+00:00",
    )
    _validated, report = validate_narrative(narrative, view, left, right)
    assert report.drop_count == 0


def test_validator_wire_shape_is_documented() -> None:
    """``narrative_validation_to_wire`` mirrors the research validator's shape."""
    report = type(
        "stub",
        (),
        {
            "drop_count": 2,
            "dropped_claims": (
                DroppedNarrativeClaim(
                    section="momentum.body",
                    sentence="A bad sentence.",
                    unsupported_tokens=("9999.9",),
                ),
                DroppedNarrativeClaim(
                    section="risk.bullets[0]",
                    sentence="Another bad one.",
                    unsupported_tokens=("42.7%",),
                ),
            ),
        },
    )()
    wire = narrative_validation_to_wire(report)  # type: ignore[arg-type]
    assert wire["drop_count"] == 2
    claims = wire["dropped_claims"]
    assert isinstance(claims, list)
    assert claims[0]["section"] == "momentum.body"
    assert claims[0]["sentence"] == "A bad sentence."
    assert claims[0]["unsupported_tokens"] == ["9999.9"]


# ---------------------------------------------------------------------------
# Section-replace helper
# ---------------------------------------------------------------------------


def test_replace_section_returns_new_narrative_with_updated_section() -> None:
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    new_body = "Updated momentum body."
    updated = replace_section(narrative, "momentum", body=new_body)
    assert updated.momentum.body == new_body
    # Other sections untouched.
    assert updated.valuation.body == narrative.valuation.body


def test_replace_section_rejects_unknown_key() -> None:
    left, right = _pair()
    view = compare_reports(left, right)
    narrative = TemplateComparisonNarrator().generate(
        NarrativeInput(view=view, left_report=left, right_report=right)
    )
    with pytest.raises(KeyError):
        replace_section(narrative, "bogus_section", body="x")
