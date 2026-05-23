"""Tests for the per-claim provenance graph extension to the research validator.

The base validator tests (``test_research_validator.py``) cover the
drop policy. This file focuses on the additive provenance:

* Every surviving section/argument/catalyst/outlook/metric carries a
  ``provenance`` dict mapping numeric tokens to their source-field
  labels (e.g. ``"row.last_price"``).
* The drop policy is untouched — dropped sentences don't appear in
  provenance, and existing tests still pass.
* First-match-wins ordering: a token appearing in multiple corpus
  entries lands on the first (most informative) entry's label.
* Empty / no-prose sections degrade to empty provenance maps.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.dashboard.state import RecommendationRow
from src.intelligence.research_thesis import (
    BullBearArgument,
    Catalyst,
    MetricEntry,
    OutlookEntry,
    ResearchInput,
    ResearchThesis,
    ThesisSection,
)
from src.intelligence.research_validator import (
    _build_corpus_entries,
    _token_provenance,
    validate_thesis,
)
from src.strategy.base import RecommendationTier, SignalAction


def _row(
    *,
    symbol: str = "AAPL",
    rsi: float = 58.0,
    last_price: float = 180.50,
    confidence: float = 0.73,
    reasoning: str = "Healthy uptrend with moderate volume support.",
) -> RecommendationRow:
    return RecommendationRow(
        symbol=symbol,
        action=SignalAction.BUY,
        confidence=confidence,
        combined_score=0.42,
        technical_score=0.55,
        sentiment_score=0.21,
        rsi=rsi,
        macd=0.12,
        bollinger=0.18,
        last_price=last_price,
        num_news_articles=3,
        reasoning=reasoning,
        timestamp=datetime(2026, 5, 23, 12, 0, tzinfo=UTC),
        headlines=("Apple announces new chip",),
        tier=RecommendationTier.BUY,
        signal_quality="high",
        stability="stable",
        quality_reasons=(),
        intraday=None,
        error=None,
    )


def _input(**row_overrides: object) -> ResearchInput:
    return ResearchInput(
        row=_row(**row_overrides),  # type: ignore[arg-type]
        headlines=("Apple announces new chip",),
    )


def _thesis(**overrides: object) -> ResearchThesis:
    base: dict[str, object] = {
        "symbol": "AAPL",
        "generated_at": "2026-05-23T12:00:00+00:00",
        "model": "test-model",
        "rating": "buy",
        "confidence": 0.73,
        "tagline": "Healthy stance.",
        "company_overview": ThesisSection(title="Company Overview"),
        "technical_analysis": ThesisSection(title="Technical Analysis"),
        "fundamental_analysis": ThesisSection(title="Fundamental Analysis"),
        "sentiment_news": ThesisSection(title="Sentiment & News"),
        "risk_assessment": ThesisSection(title="Risk Assessment"),
        "explainability": ThesisSection(title="Why this read"),
    }
    base.update(overrides)
    return ResearchThesis(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Corpus entries (labeled source-of-truth)
# ---------------------------------------------------------------------------


def test_corpus_entries_carry_source_labels_per_row_scalar() -> None:
    """Every row scalar produces at least one labeled entry."""
    entries = _build_corpus_entries(_thesis(), _input())
    labels = {label for _text, label in entries}
    expected = {
        "row.symbol",
        "row.last_price",
        "row.rsi",
        "row.macd",
        "row.bollinger",
        "row.confidence",
        "row.combined_score",
        "row.technical_score",
        "row.sentiment_score",
        "row.num_news_articles",
        "row.reasoning",
        "headlines[0]",
    }
    assert expected.issubset(labels)


def test_corpus_entries_dedupe_surface_forms_under_one_label() -> None:
    """``180.5`` / ``180.50`` / ``$180.50`` all map to row.last_price."""
    entries = _build_corpus_entries(_thesis(), _input(last_price=180.50))
    by_label: dict[str, list[str]] = {}
    for text, label in entries:
        by_label.setdefault(label, []).append(text)
    forms = by_label.get("row.last_price", [])
    assert "180.5" in forms
    assert "180.50" in forms
    assert "$180.50" in forms


def test_empty_corpus_entries_are_filtered() -> None:
    """An empty-string entry would substring-match everything; the
    builder filters them so grounding stays meaningful."""
    entries = _build_corpus_entries(_thesis(), _input(reasoning=""))
    for text, _label in entries:
        assert text, f"empty corpus entry slipped through: {text!r}"


# ---------------------------------------------------------------------------
# Token provenance
# ---------------------------------------------------------------------------


def test_token_provenance_maps_decimal_to_row_label() -> None:
    """A decimal in prose that appears in row.last_price gets that label."""
    entries = _build_corpus_entries(_thesis(), _input(last_price=180.50))
    allowance: frozenset[str] = frozenset()
    prov = _token_provenance(
        "Last close was $180.50 on the tape.", entries, allowance
    )
    assert "$180.50" in prov
    assert prov["$180.50"] == "row.last_price"


def test_token_provenance_skips_allowance_tokens() -> None:
    """A token only matched by the allowance set has no provenance entry."""
    entries = _build_corpus_entries(_thesis(), _input())
    # "2026" is a year in the standard allowance — should not appear.
    allowance: frozenset[str] = frozenset({"2026"})
    prov = _token_provenance("Through 2026 the stance held.", entries, allowance)
    assert "2026" not in prov


def test_token_provenance_first_token_occurrence_wins() -> None:
    """Repeated tokens in the same body collapse to one provenance entry."""
    entries = _build_corpus_entries(_thesis(), _input(last_price=180.50))
    allowance: frozenset[str] = frozenset()
    prov = _token_provenance(
        "Close was $180.50 today; $180.50 again tomorrow.", entries, allowance
    )
    # One key regardless of duplicate occurrences.
    assert len([k for k in prov if k == "$180.50"]) == 1


def test_token_provenance_returns_empty_for_blank_text() -> None:
    entries = _build_corpus_entries(_thesis(), _input())
    assert _token_provenance("", entries, frozenset()) == {}


# ---------------------------------------------------------------------------
# Wire shape: provenance attached to surviving sections
# ---------------------------------------------------------------------------


def test_section_provenance_includes_grounded_tokens() -> None:
    """A surviving section's body tokens land in its ``provenance`` map."""
    thesis = _thesis(
        company_overview=ThesisSection(
            title="Company Overview",
            body=(
                "AAPL trades at $180.50 with RSI at 58.0 — a healthy stance "
                "with moderate volume support."
            ),
        )
    )
    validated, _report = validate_thesis(thesis, _input(last_price=180.50))
    prov = validated.company_overview.provenance
    assert "$180.50" in prov
    assert prov["$180.50"] == "row.last_price"
    assert "58.0" in prov
    assert prov["58.0"] == "row.rsi"


def test_section_provenance_includes_bullets() -> None:
    """Tokens that appear only in bullets still surface in the section
    provenance map."""
    thesis = _thesis(
        company_overview=ThesisSection(
            title="Company Overview",
            body="Healthy stance.",
            bullets=("RSI at 58.0 supports the read.",),
        )
    )
    validated, _report = validate_thesis(thesis, _input())
    assert "58.0" in validated.company_overview.provenance


def test_bullbear_argument_provenance_attached_per_arg() -> None:
    thesis = _thesis(
        bull_thesis=(
            BullBearArgument(
                label="Strong technical",
                weight=0.6,
                detail="RSI at 58.0 reads constructively.",
            ),
            BullBearArgument(
                label="Tape support",
                weight=0.4,
                detail="Last $180.50 holds the trend.",
            ),
        )
    )
    validated, _report = validate_thesis(thesis, _input())
    assert validated.bull_thesis[0].provenance.get("58.0") == "row.rsi"
    assert validated.bull_thesis[1].provenance.get("$180.50") == "row.last_price"


def test_catalyst_provenance_attached() -> None:
    thesis = _thesis(
        catalysts=(
            Catalyst(
                label="Q1 print",
                when="Next 30 days",
                impact="bullish",
                detail="Beat or miss against EPS 0.73.",
            ),
        )
    )
    validated, _report = validate_thesis(thesis, _input(confidence=0.73))
    # 0.73 in row.confidence yields provenance; the validator allowance
    # has "73%" but the bare "0.73" decimal is a corpus match.
    assert validated.catalysts[0].provenance.get("0.73") == "row.confidence"


def test_outlook_provenance_attached() -> None:
    thesis = _thesis(
        outlook=(
            OutlookEntry(
                horizon="short",
                bias="bullish",
                confidence=0.7,
                detail="RSI at 58.0 supports a short-term constructive read.",
            ),
        )
    )
    validated, _report = validate_thesis(thesis, _input())
    assert validated.outlook[0].provenance.get("58.0") == "row.rsi"


def test_metric_provenance_attached_to_value() -> None:
    thesis = _thesis(
        metrics=(
            MetricEntry(label="Last", value="180.50", tone="bull"),
        )
    )
    validated, _report = validate_thesis(thesis, _input(last_price=180.50))
    assert validated.metrics[0].provenance.get("180.50") == "row.last_price"


# ---------------------------------------------------------------------------
# Drop policy untouched
# ---------------------------------------------------------------------------


def test_dropped_section_has_no_provenance_for_dropped_tokens() -> None:
    """Tokens in dropped sentences don't end up in the surviving
    section's provenance map."""
    thesis = _thesis(
        company_overview=ThesisSection(
            title="Company Overview",
            body=(
                "Close was $180.50 today. Made-up gross margin of 99.9%."
            ),
        )
    )
    validated, report = validate_thesis(thesis, _input(last_price=180.50))
    # The second sentence ("99.9%") drops; the validator removes it.
    assert report.drop_count >= 1
    # 99.9% must not appear in the surviving provenance.
    assert "99.9%" not in validated.company_overview.provenance
    # The grounded token still surfaces.
    assert "$180.50" in validated.company_overview.provenance


def test_empty_section_has_empty_provenance() -> None:
    """A section with no prose carries an empty provenance dict."""
    thesis = _thesis(
        company_overview=ThesisSection(title="Company Overview", body=""),
    )
    validated, _report = validate_thesis(thesis, _input())
    assert validated.company_overview.provenance == {}


def test_existing_drop_count_invariant_preserved() -> None:
    """Adding provenance must not change the drop-count semantics."""
    thesis = _thesis(
        company_overview=ThesisSection(
            title="Company Overview",
            body="Close $180.50. Fake 99.9% margin.",
        )
    )
    _validated, report = validate_thesis(thesis, _input(last_price=180.50))
    assert report.drop_count == 1
    dropped = report.dropped_claims[0]
    assert "99.9%" in dropped.unsupported_tokens
