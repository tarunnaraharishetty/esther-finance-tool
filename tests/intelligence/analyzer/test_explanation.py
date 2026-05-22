"""Grounded explanation layer tests.

Verifies:

1. The citation validator's contract: every claim needs a tag, every
   tag must be allowed by the inputs.
2. The deterministic template builder produces only grounded claims
   and respects missing inputs.
3. The LLM XML parser routes through the validator — ungrounded LLM
   output cannot reach the caller.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.intelligence.analyzer.explanation import (
    CITATION_TAGS,
    AnalyzerExplanation,
    AnalyzerInputs,
    CitationValidationError,
    build_grounded_explanation,
    extract_citations,
    validate_citations,
)
from src.intelligence.analyzer.llm_explanation import parse_explanation_xml
from src.intelligence.analyzer.prompts import build_user_message
from src.intelligence.analyzer.technical import score_technicals
from src.intelligence.analyzer.valuation import build_valuation
from src.intelligence.fundamentals.models import (
    CompanyProfile,
    IncomeStatement,
    KeyRatios,
    NormalizedFundamentals,
    ProviderName,
    ReportPeriod,
)

# ---- helpers ----


def _scored_inputs(
    *, with_fundamentals: bool = False, with_valuation: bool = False
) -> AnalyzerInputs:
    """Build a realistic AnalyzerInputs for end-to-end-style tests."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed=11)
    n = 80
    daily_returns = rng.normal(loc=0.015, scale=0.015, size=n)
    close = 100.0 * np.cumprod(1.0 + daily_returns)
    df = pd.DataFrame(
        {
            "open": np.concatenate([[100.0], close[:-1]]),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=int),
        }
    )
    tech = score_technicals(df)

    funds = None
    val = None
    if with_fundamentals or with_valuation:
        funds = _profitable_tech()
    if with_valuation and funds is not None:
        val = build_valuation(funds)

    return AnalyzerInputs(
        symbol="TCHX",
        last_price=float(close[-1]),
        technicals=tech,
        fundamentals=funds,
        valuation=val,
        sentiment_score=0.42,
        headline_count=5,
        top_headlines=("Earnings beat; data-center revenue strong",),
        volume_z=1.8,
        volume_spike_score=60.0,
    )


def _profitable_tech() -> NormalizedFundamentals:
    incomes = tuple(
        IncomeStatement(
            period=ReportPeriod.ANNUAL,
            fiscal_date=datetime(year, 12, 31, tzinfo=UTC),
            revenue=100_000_000 * (1.15 ** i),
            net_income=20_000_000 * (1.15 ** i),
            ebitda=30_000_000 * (1.15 ** i),
            eps_diluted=2.0 * (1.15 ** i),
            shares_diluted=10_000_000,
        )
        for i, year in enumerate(range(2020, 2025))
    )
    return NormalizedFundamentals(
        symbol="TCHX",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(
            symbol="TCHX",
            sector="Information Technology",
            market_cap=1.5e9,
            shares_outstanding=10_000_000,
        ),
        income_statements=incomes,
        key_ratios=KeyRatios(pe_ratio=32.0, peg_ratio=2.0, debt_to_equity=0.5),
    )


# ---- citation regex / extractor ----


def test_extract_citations_finds_single_tag() -> None:
    assert extract_citations("RSI is 64 [technicals].") == ("technicals",)


def test_extract_citations_finds_multi_tag() -> None:
    assert extract_citations(
        "Trend is intact [technicals, volume]."
    ) == ("technicals", "volume")


def test_extract_citations_returns_empty_when_no_tags() -> None:
    assert extract_citations("Something happened.") == ()


def test_extract_citations_is_case_normalizing() -> None:
    # The regex is anchored on lowercase, so uppercase tags don't match.
    # This is by design — the LLM prompt requires lowercase tags.
    assert extract_citations("Claim [Technicals].") == ()


# ---- validator ----


def test_validate_citations_passes_when_all_grounded() -> None:
    inputs = AnalyzerInputs(
        symbol="X",
        technicals=None,  # we'll set only one stream
        sentiment_score=0.3,
    )
    # Sentiment is the only available tag.
    citations = validate_citations(
        ("Sentiment positive [sentiment].",), inputs
    )
    assert citations == {"sentiment": ("Sentiment positive [sentiment].",)}


def test_validate_citations_rejects_claim_with_no_tag() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.3)
    with pytest.raises(CitationValidationError) as exc_info:
        validate_citations(("This claim has no tag.",), inputs)
    assert "no citation tag" in str(exc_info.value)


def test_validate_citations_rejects_unavailable_tag() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.3)
    with pytest.raises(CitationValidationError) as exc_info:
        validate_citations(("Fundamentals strong [fundamentals].",), inputs)
    assert "[fundamentals]" in str(exc_info.value)


def test_validate_citations_rejects_unknown_tag() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.3)
    with pytest.raises(CitationValidationError) as exc_info:
        validate_citations(("Vibes feel good [vibes].",), inputs)
    assert "unknown tag" in str(exc_info.value)


def test_validate_citations_handles_multi_tag_claims() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.3, volume_z=2.0)
    citations = validate_citations(
        ("Tape and tape agree [sentiment, volume].",), inputs
    )
    # The claim is filed under both tags so the UI can render multiple chips.
    assert "sentiment" in citations
    assert "volume" in citations


def test_validate_citations_fails_when_any_multi_tag_unavailable() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.3)  # no volume stream
    with pytest.raises(CitationValidationError):
        validate_citations(
            ("Tape and tape agree [sentiment, volume].",), inputs
        )


# ---- AnalyzerInputs.available_tags ----


def test_available_tags_reflects_only_present_streams() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.5)
    assert inputs.available_tags() == frozenset({"sentiment"})


def test_available_tags_treats_zero_headline_count_as_present() -> None:
    """Headline_count=0 still means 'we asked and got 0' — a real signal."""
    inputs = AnalyzerInputs(symbol="X", headline_count=0)
    assert "news" in inputs.available_tags()


def test_available_tags_includes_volume_on_either_field() -> None:
    only_z = AnalyzerInputs(symbol="X", volume_z=1.5)
    only_score = AnalyzerInputs(symbol="X", volume_spike_score=70.0)
    assert "volume" in only_z.available_tags()
    assert "volume" in only_score.available_tags()


def test_available_tags_is_subset_of_vocabulary() -> None:
    inputs = _scored_inputs(with_fundamentals=True, with_valuation=True)
    assert inputs.available_tags() <= CITATION_TAGS


# ---- deterministic template builder ----


def test_template_builder_output_passes_validator() -> None:
    inputs = _scored_inputs(with_fundamentals=True, with_valuation=True)
    explanation = build_grounded_explanation(inputs)
    # Every claim should pass the validator with no surprises.
    citations = validate_citations(explanation.claims, inputs)
    assert citations  # at least one tag mapped to claims
    # Symbol propagates through.
    assert explanation.symbol == "TCHX"
    # Summary is present (no citation requirement on the summary).
    assert explanation.summary
    # Model id reflects the deterministic template.
    assert "template" in explanation.model


def test_template_builder_skips_streams_when_inputs_missing() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.5)
    explanation = build_grounded_explanation(inputs)
    # Only sentiment citations should appear.
    for claim in explanation.claims:
        for tag in extract_citations(claim):
            assert tag == "sentiment", (
                f"template builder cited unavailable tag {tag!r} in {claim!r}"
            )


def test_template_builder_handles_empty_inputs() -> None:
    """No grounded streams → empty claims, but no crash."""
    inputs = AnalyzerInputs(symbol="X")
    explanation = build_grounded_explanation(inputs)
    assert explanation.claims == ()
    assert explanation.citations == {}
    # Summary still renders so the UI can show "insufficient inputs".
    assert "insufficient" in explanation.summary.lower()


def test_template_builder_to_dict_is_json_safe() -> None:
    import json

    inputs = _scored_inputs(with_fundamentals=True, with_valuation=True)
    explanation = build_grounded_explanation(inputs)
    serialized = json.dumps(explanation.to_dict())
    restored = json.loads(serialized)
    assert restored["symbol"] == "TCHX"
    assert "claims" in restored


# ---- prompt template ----


def test_user_message_only_lists_available_tags() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.3, volume_z=1.0)
    msg = build_user_message(inputs)
    assert "ALLOWED CITATIONS: [sentiment], [volume]" in msg
    # No mention of [fundamentals] / [valuation] / [technicals] etc.
    assert "[fundamentals]" not in msg
    assert "[valuation]" not in msg
    assert "[technicals]" not in msg
    assert "[news]" not in msg


def test_user_message_renders_data_block_verbatim() -> None:
    inputs = _scored_inputs()
    msg = build_user_message(inputs)
    assert "DATA:" in msg
    assert "TECHNICALS:" in msg
    # Last price is rendered with two decimals.
    assert f"${inputs.last_price:.2f}" in msg
    # Top headline appears with bracket-escaping.
    safe = inputs.top_headlines[0].replace("[", "(").replace("]", ")")
    assert safe in msg


def test_user_message_when_no_streams_available_says_so() -> None:
    inputs = AnalyzerInputs(symbol="X")
    msg = build_user_message(inputs)
    assert "do not generate any claims" in msg


# ---- LLM XML parser + validator integration ----


def _explanation_xml(claims: list[str], risks: list[str] | None = None) -> str:
    risks = risks or []
    claim_xml = "".join(f"<claim>{c}</claim>" for c in claims)
    risk_xml = "".join(f"<risk>{r}</risk>" for r in risks)
    return (
        "<explanation>"
        "<summary>Symbol reads constructively.</summary>"
        f"{claim_xml}{risk_xml}"
        "</explanation>"
    )


def test_parser_accepts_well_formed_grounded_xml() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.4)
    xml = _explanation_xml(["Sentiment is positive [sentiment]."])
    explanation = parse_explanation_xml(
        xml, inputs=inputs, model_id="test-model"
    )
    assert explanation.claims == ("Sentiment is positive [sentiment].",)
    assert explanation.citations == {
        "sentiment": ("Sentiment is positive [sentiment].",)
    }
    assert explanation.summary.startswith("Symbol")


def test_parser_rejects_claim_missing_citation() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.4)
    xml = _explanation_xml(["Sentiment is positive."])
    with pytest.raises(CitationValidationError):
        parse_explanation_xml(xml, inputs=inputs, model_id="test-model")


def test_parser_rejects_unavailable_citation() -> None:
    """Hallucinated fundamentals citation must be blocked."""
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.4)
    xml = _explanation_xml(["Revenue grew [fundamentals]."])
    with pytest.raises(CitationValidationError) as exc_info:
        parse_explanation_xml(xml, inputs=inputs, model_id="test-model")
    assert "[fundamentals]" in str(exc_info.value)


def test_parser_rejects_unknown_citation() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.4)
    xml = _explanation_xml(["Vibes are good [vibes]."])
    with pytest.raises(CitationValidationError) as exc_info:
        parse_explanation_xml(xml, inputs=inputs, model_id="test-model")
    assert "unknown" in str(exc_info.value)


def test_parser_raises_value_error_on_missing_xml() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.4)
    with pytest.raises(ValueError) as exc_info:
        parse_explanation_xml(
            "no xml here", inputs=inputs, model_id="test-model"
        )
    assert "did not contain" in str(exc_info.value)


def test_parser_raises_value_error_on_malformed_xml() -> None:
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.4)
    with pytest.raises(ValueError):
        parse_explanation_xml(
            "<explanation><summary>bad</explanation>",
            inputs=inputs,
            model_id="test-model",
        )


def test_parser_validates_risk_citations_too() -> None:
    """A risk that cites an unavailable tag should also be rejected."""
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.4)
    xml = _explanation_xml(
        claims=["Sentiment positive [sentiment]."],
        risks=["Fundamentals weak [fundamentals]."],
    )
    with pytest.raises(CitationValidationError):
        parse_explanation_xml(xml, inputs=inputs, model_id="test-model")


def test_parser_strips_surrounding_prose() -> None:
    """The model sometimes emits prose around its XML; we tolerate that."""
    inputs = AnalyzerInputs(symbol="X", sentiment_score=0.4)
    xml = (
        "Here is the explanation:\n"
        + _explanation_xml(["Sentiment positive [sentiment]."])
        + "\n\nEnd."
    )
    explanation = parse_explanation_xml(xml, inputs=inputs, model_id="test-model")
    assert len(explanation.claims) == 1


# ---- dataclass shape ----


def test_analyzer_explanation_is_frozen() -> None:
    exp = AnalyzerExplanation(
        symbol="X",
        generated_at=datetime.now(UTC).isoformat(),
        model="m",
        summary="s",
        claims=(),
        risk_warnings=(),
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        exp.symbol = "Y"  # type: ignore[misc]
