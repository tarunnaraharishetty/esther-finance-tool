"""Unit tests for :func:`validate_thesis`.

The validator is a pure function — given a thesis + input, return a
scrubbed thesis + a drop report. Tests build small synthetic inputs
so each invariant is verifiable without hitting Anthropic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.dashboard.state import RecommendationRow
from src.intelligence.research_thesis import (
    BullBearArgument,
    MetricEntry,
    OutlookEntry,
    ResearchInput,
    ResearchThesis,
    ThesisSection,
)
from src.intelligence.research_validator import (
    _TOKEN_PATTERN,
    _build_allowance,
    _build_corpus,
    validate_thesis,
)
from src.strategy.base import RecommendationTier, SignalAction

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


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
        timestamp=datetime(2026, 5, 22, 12, 0, tzinfo=UTC),
        headlines=("Apple announces new chip", "Q1 results released"),
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
        headlines=(
            "Apple announces new chip",
            "Q1 results released",
        ),
    )


def _thesis(**overrides: object) -> ResearchThesis:
    base: dict[str, object] = {
        "symbol": "AAPL",
        "generated_at": "2026-05-22T12:00:00+00:00",
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


# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("RSI is at 87.4 today.", ["87.4"]),
        ("Revenue grew 12.5% YoY.", ["12.5%"]),
        ("Market cap is $1.2B.", ["$1.2B"]),
        ("Trading at 12.5x earnings.", ["12.5x"]),
        ("Look at Q1 2025 for the catalyst.", ["Q1 2025"]),
        ("In 2024 the company doubled.", ["2024"]),
        # Long un-grouped currency: the ``$`` alternation wins over the bare
        # 6+ digit pattern; only one match is produced.
        ("Revenue of $250000000 reported.", ["$250000000"]),
        ("No numbers here at all.", []),
    ],
)
def test_token_pattern_extracts_each_class(
    text: str, expected: list[str]
) -> None:
    """Tokenizer must catch every documented surface form."""
    found = [m.group(0) for m in _TOKEN_PATTERN.finditer(text)]
    # Verify each expected token appears; order may vary across classes.
    for token in expected:
        assert token in found, (
            f"missing token {token!r} in {found!r} for input {text!r}"
        )


# -----------------------------------------------------------------------------
# Corpus
# -----------------------------------------------------------------------------


def test_corpus_includes_row_scalars() -> None:
    payload = _input()
    thesis = _thesis()
    corpus = _build_corpus(thesis, payload)
    # RSI value, last_price, and reasoning text should be findable.
    assert "58" in corpus
    assert "180" in corpus
    assert "healthy uptrend" in corpus
    # Headlines should be in the corpus too.
    assert "apple announces new chip" in corpus
    assert "q1 results released" in corpus


# -----------------------------------------------------------------------------
# Allowance
# -----------------------------------------------------------------------------


def test_allowance_includes_confidence_as_percent() -> None:
    """``confidence == 0.73`` → "73%" is always allowed even if not in corpus."""
    payload = _input()
    thesis = _thesis()
    allow = _build_allowance(thesis, payload)
    assert "73%" in allow


def test_allowance_includes_generated_year() -> None:
    thesis = _thesis(generated_at="2026-05-22T12:00:00+00:00")
    payload = _input()
    allow = _build_allowance(thesis, payload)
    assert "2026" in allow


def test_allowance_includes_small_integers() -> None:
    """Counts like "2 of 4" pass through without anchoring."""
    payload = _input()
    thesis = _thesis()
    allow = _build_allowance(thesis, payload)
    for n in range(0, 6):
        assert str(n) in allow


# -----------------------------------------------------------------------------
# Validation — sections
# -----------------------------------------------------------------------------


def test_supported_sentence_is_kept() -> None:
    """RSI 58 is in the row → "RSI is at 58.0" must survive."""
    payload = _input(rsi=58.0)
    thesis = _thesis(
        technical_analysis=ThesisSection(
            title="Technical Analysis",
            body="RSI is at 58.0 — neutral footing.",
        )
    )
    validated, report = validate_thesis(thesis, payload)
    assert report.drop_count == 0
    assert "58.0" in validated.technical_analysis.body


def test_unsupported_sentence_is_dropped_with_recorded_reason() -> None:
    payload = _input(rsi=58.0)
    thesis = _thesis(
        technical_analysis=ThesisSection(
            title="Technical Analysis",
            body="RSI is at 87.4, in deep oversold territory.",
        )
    )
    validated, report = validate_thesis(thesis, payload)
    assert report.drop_count == 1
    dropped = report.dropped_claims[0]
    assert dropped.section == "technical_analysis.body"
    assert "87.4" in dropped.unsupported_tokens
    # The bad sentence is gone from the validated thesis.
    assert "87.4" not in validated.technical_analysis.body


def test_mixed_sentences_keep_only_the_supported_one() -> None:
    payload = _input(rsi=58.0)
    thesis = _thesis(
        technical_analysis=ThesisSection(
            title="Technical Analysis",
            body=(
                "RSI is at 58.0 — neutral footing. "
                "Revenue grew 99.9% last quarter."
            ),
        )
    )
    validated, report = validate_thesis(thesis, payload)
    assert report.drop_count == 1
    # First sentence preserved; second dropped.
    assert "58.0" in validated.technical_analysis.body
    assert "99.9%" not in validated.technical_analysis.body


def test_sentence_with_any_unsupported_token_is_dropped() -> None:
    """A sentence mixing a good number and a bad one is dropped entirely.

    Trade-off documented in the module: better to drop a partially-
    correct sentence than to publish a fabricated number with the
    correct number alongside it.
    """
    payload = _input(rsi=58.0)
    thesis = _thesis(
        technical_analysis=ThesisSection(
            title="Technical Analysis",
            body="RSI at 58.0 implies a 95.7% probability of a pullback.",
        )
    )
    validated, report = validate_thesis(thesis, payload)
    assert report.drop_count == 1
    assert validated.technical_analysis.body == ""


# -----------------------------------------------------------------------------
# Validation — tuple fields
# -----------------------------------------------------------------------------


def test_bull_thesis_drops_only_the_bad_argument() -> None:
    payload = _input()
    good = BullBearArgument(
        label="Strong RSI",
        weight=0.6,
        detail="RSI is at 58.0, supportive of further upside.",
    )
    bad = BullBearArgument(
        label="Fake metric",
        weight=0.4,
        detail="Earnings beat by 87.4% last quarter.",
    )
    thesis = _thesis(bull_thesis=(good, bad))
    validated, report = validate_thesis(thesis, payload)
    assert report.drop_count == 1
    # Only the good arg survives.
    assert len(validated.bull_thesis) == 1
    assert validated.bull_thesis[0].label == "Strong RSI"


def test_outlook_entry_with_bad_detail_is_removed() -> None:
    payload = _input()
    good = OutlookEntry(
        horizon="short",
        bias="bullish",
        confidence=0.6,
        detail="Trend remains intact.",
    )
    bad = OutlookEntry(
        horizon="long",
        bias="bullish",
        confidence=0.4,
        detail="EPS will hit $9.99 by 2030.",
    )
    thesis = _thesis(outlook=(good, bad))
    validated, _report = validate_thesis(thesis, payload)
    assert len(validated.outlook) == 1
    assert validated.outlook[0].horizon == "short"


def test_metrics_with_unsupported_value_dropped() -> None:
    """Metric cards are atomic — bad value → whole card vanishes."""
    payload = _input()
    good = MetricEntry(label="RSI", value="58.0")
    bad = MetricEntry(label="Made-up margin", value="42.7%")
    thesis = _thesis(metrics=(good, bad))
    validated, report = validate_thesis(thesis, payload)
    # The "made-up" 42.7% has no anchor in the input.
    assert report.drop_count == 1
    assert any(m.label == "RSI" for m in validated.metrics)
    assert not any(m.label == "Made-up margin" for m in validated.metrics)


# -----------------------------------------------------------------------------
# Allowance integration
# -----------------------------------------------------------------------------


def test_confidence_percent_in_prose_is_allowed_even_without_corpus_match() -> None:
    """Confidence of 0.73 should let "73%" pass even if "73%" isn't in the input bundle."""
    payload = _input(confidence=0.73)
    thesis = _thesis(
        confidence=0.73,
        explainability=ThesisSection(
            title="Why this read",
            body="Overall confidence in this read is 73%.",
        ),
    )
    validated, report = validate_thesis(thesis, payload)
    assert report.drop_count == 0
    assert "73%" in validated.explainability.body


def test_small_integer_counts_are_allowed() -> None:
    """A claim like "2 of 4 brokerages cover this" must pass through."""
    payload = _input()
    thesis = _thesis(
        sentiment_news=ThesisSection(
            title="Sentiment & News",
            body="2 of 4 brokerages cover this name actively.",
        )
    )
    validated, report = validate_thesis(thesis, payload)
    # Despite "4" not appearing in the row scalars, small ints are allowed.
    assert report.drop_count == 0
    assert "2 of 4" in validated.sentiment_news.body


def test_generated_year_is_allowed() -> None:
    payload = _input()
    thesis = _thesis(
        generated_at="2026-05-22T12:00:00+00:00",
        sentiment_news=ThesisSection(
            title="Sentiment & News",
            body="Through mid-2026 the story has held.",
        ),
    )
    _validated, report = validate_thesis(thesis, payload)
    assert report.drop_count == 0


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


def test_empty_thesis_passes_unchanged() -> None:
    """A thesis with all-empty section bodies produces zero drops."""
    payload = _input()
    thesis = _thesis()
    validated, report = validate_thesis(thesis, payload)
    assert report.drop_count == 0
    assert validated == thesis  # frozen-dataclass equality preserved


def test_tagline_is_validated() -> None:
    payload = _input(rsi=58.0)
    thesis = _thesis(tagline="Watch for the 99.9% beat on next earnings.")
    validated, report = validate_thesis(thesis, payload)
    assert report.drop_count == 1
    # Bad tagline scrubbed.
    assert "99.9%" not in validated.tagline


def test_report_drop_count_equals_dropped_claims_length() -> None:
    payload = _input(rsi=58.0)
    thesis = _thesis(
        technical_analysis=ThesisSection(
            title="Technical Analysis",
            body=(
                "First fake sentence with 91.4 mentioned. "
                "Second fake sentence claims 99.9% growth."
            ),
        )
    )
    _validated, report = validate_thesis(thesis, payload)
    assert report.drop_count == len(report.dropped_claims)
    assert report.drop_count == 2


def test_report_records_section_path_for_each_drop() -> None:
    payload = _input(rsi=58.0)
    thesis = _thesis(
        technical_analysis=ThesisSection(
            title="Technical Analysis", body="RSI at 87.4 today."
        ),
        risk_assessment=ThesisSection(
            title="Risk Assessment", body="Drawdown of 42.7% last cycle."
        ),
    )
    _, report = validate_thesis(thesis, payload)
    sections = {d.section for d in report.dropped_claims}
    assert "technical_analysis.body" in sections
    assert "risk_assessment.body" in sections


# Suppress unused-import warnings for fixtures that show how to plug
# Decimal-backed scalars; not used in these tests directly.
_ = Decimal
