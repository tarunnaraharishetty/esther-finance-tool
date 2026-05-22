"""Post-hoc numeric-claim validator for :class:`ResearchThesis`.

The grounding rules in :mod:`src.intelligence.grounding` are
prompt-time *guidance* — they tell the LLM "don't invent numbers".
This module is the programmatic *enforcement*: after the LLM (or
template) produces a thesis, every prose field is scanned for numeric
tokens (percentages, currency, ratios, years, quarters), each token
is substring-checked against the corpus we built from the input
bundle, and any sentence containing an unsupported token is dropped.

Why numbers, not qualitative claims
-----------------------------------
Numbers are the failure mode that costs traders money — a hallucinated
"$1.2B revenue" leads to a bet on a phantom company. Qualitative
claims ("strong revenue growth") would need a paraphrase model to
verify; the prompt-time grounding rules handle them. This module
catches the high-stakes class with high-confidence regex.

Trust framing
-------------
A drop is a *positive* signal: the system refused to publish a claim
it couldn't anchor in the input data. The wire contract surfaces
``drop_count`` so the UI can render "AI declined to make 3 unsupported
claims" alongside the validated thesis. Hiding drops would silently
ship hallucinations; loud drops are the institutional contract.

What we deliberately don't drop
-------------------------------
* The thesis itself when every claim is unsupported. The trader sees
  a thesis with empty sections and a high drop count — that's an
  honest result, more useful than a 500.
* Sentences that mention numbers from the allowance list: confidence
  rendered as a percentage, the year from ``generated_at``, integers
  in the 0-5 range (lists / counts), and the ticker itself.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING

from src.intelligence.research_thesis import (
    BullBearArgument,
    Catalyst,
    MetricEntry,
    OutlookEntry,
    ResearchThesis,
    ThesisSection,
)
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.intelligence.research_thesis import ResearchInput

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Token regex set
# ---------------------------------------------------------------------------
#
# Each pattern extracts a numeric token *with* its delimiter so the
# substring check operates on the same surface form the corpus uses.
# A bare ``87.4`` in prose must match ``87.4`` in the corpus — the
# regex captures decimals, percents, currency suffixes verbatim.

# Numbers with one of: %, $, x suffix, decimal, year, Q[1-4]. Order
# matters for the alternation: percentages must beat bare decimals so
# "87.4%" lands as one token, not two.
_TOKEN_PATTERN = re.compile(
    r"""
    (?:
        # Currency: comma-grouped ($1,250,000.00) OR contiguous ($250000000).
        \$(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?[KMBT]?
      | -?\d{1,3}(?:\.\d+)?%                         # 73%, -2.5%, 100%
      | \d+(?:\.\d+)?[xX]                            # 12.5x, 3X
      | \bQ[1-4](?:\s*\d{2,4})?\b                    # Q1, Q3 2025
      | \b20\d{2}\b                                  # 2024, 2026
      | \b\d+\.\d+\b                                 # 87.4 (PE, scores)
      | \b\d{6,}\b                                   # large integers (revenue $)
    )
    """,
    re.VERBOSE,
)


# Sentences end with ``. `` / ``? `` / ``! `` / newline. We keep punctuation
# attached so reconstruction preserves the original tone — the validator
# isn't a rewriter.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DroppedClaim:
    """One sentence the validator removed.

    ``section`` is a dotted path identifying where the claim lived
    on the thesis so the operator can trace which generator field
    produced the bad output. Examples: ``"technical_analysis.body"``,
    ``"bull_thesis[0].detail"``.
    """

    section: str
    sentence: str
    unsupported_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate result of :func:`validate_thesis`."""

    drop_count: int
    dropped_claims: tuple[DroppedClaim, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_thesis(
    thesis: ResearchThesis, payload: ResearchInput
) -> tuple[ResearchThesis, ValidationReport]:
    """Scan every prose field; return a thesis with unsupported sentences dropped.

    Returns ``(validated_thesis, report)`` — both are always returned,
    never raised. A thesis where every claim got dropped becomes a
    thesis with empty section bodies + a high ``drop_count``. The
    caller surfaces both verbatim on the wire.
    """
    corpus = _build_corpus(thesis, payload)
    allowance = _build_allowance(thesis, payload)
    dropped: list[DroppedClaim] = []

    # ThesisSection fields are validated in place via ``replace``.
    # Tuple fields (bull/bear/catalysts/outlook) need per-entry handling
    # so dropped entries vanish from the tuple instead of leaving holes.

    new_company = _scrub_section(
        thesis.company_overview, "company_overview", corpus, allowance, dropped
    )
    new_technical = _scrub_section(
        thesis.technical_analysis,
        "technical_analysis",
        corpus,
        allowance,
        dropped,
    )
    new_fundamental = _scrub_section(
        thesis.fundamental_analysis,
        "fundamental_analysis",
        corpus,
        allowance,
        dropped,
    )
    new_sentiment = _scrub_section(
        thesis.sentiment_news, "sentiment_news", corpus, allowance, dropped
    )
    new_risk = _scrub_section(
        thesis.risk_assessment, "risk_assessment", corpus, allowance, dropped
    )
    new_explainability = _scrub_section(
        thesis.explainability, "explainability", corpus, allowance, dropped
    )

    new_tagline, tagline_dropped = _scrub_text(
        thesis.tagline, "tagline", corpus, allowance
    )
    dropped.extend(tagline_dropped)

    new_bull = _scrub_arguments(
        thesis.bull_thesis, "bull_thesis", corpus, allowance, dropped
    )
    new_bear = _scrub_arguments(
        thesis.bear_thesis, "bear_thesis", corpus, allowance, dropped
    )
    new_catalysts = _scrub_catalysts(
        thesis.catalysts, "catalysts", corpus, allowance, dropped
    )
    new_outlook = _scrub_outlook(
        thesis.outlook, "outlook", corpus, allowance, dropped
    )
    new_metrics = _scrub_metrics(
        thesis.metrics, "metrics", corpus, allowance, dropped
    )

    validated = replace(
        thesis,
        tagline=new_tagline,
        company_overview=new_company,
        bull_thesis=new_bull,
        bear_thesis=new_bear,
        technical_analysis=new_technical,
        fundamental_analysis=new_fundamental,
        metrics=new_metrics,
        sentiment_news=new_sentiment,
        catalysts=new_catalysts,
        risk_assessment=new_risk,
        outlook=new_outlook,
        explainability=new_explainability,
    )
    report = ValidationReport(
        drop_count=len(dropped), dropped_claims=tuple(dropped)
    )
    if dropped:
        log.info(
            "research_validator.dropped",
            symbol=thesis.symbol,
            drop_count=len(dropped),
        )
    return validated, report


# ---------------------------------------------------------------------------
# Corpus + allowance construction
# ---------------------------------------------------------------------------


def _build_corpus(thesis: ResearchThesis, payload: ResearchInput) -> str:
    """Concatenate every grounding-eligible scalar from the input bundle.

    The corpus is the substring-match space. A token in the thesis
    prose is "supported" iff it appears verbatim somewhere in this
    string. Lowercase + whitespace-normalized so case + spacing don't
    cause false negatives.

    Every field that could legitimately ground a numeric claim must
    be included here — missing a field would cause the validator to
    drop valid claims. New adapters need to be added when they ship.
    """
    parts: list[str] = []
    row = payload.row
    # Scalars on the snapshot row. The currency-prefixed form of
    # ``last_price`` is included alongside the bare float so a
    # template/LLM rendering as "$101.61" still anchors against the
    # row that carries 101.61. Same with other dollar-like quantities
    # downstream — the corpus is a *surface-form* match space, not a
    # numeric-equivalence engine.
    parts.extend(
        [
            row.symbol,
            f"{row.last_price}",
            f"{row.last_price:.2f}",
            f"${row.last_price:.2f}",
            f"{row.rsi}",
            f"{row.rsi:.1f}",
            f"{row.macd}",
            f"{row.macd:.2f}",
            f"{row.bollinger}",
            f"{row.bollinger:.2f}",
            f"{row.confidence}",
            f"{row.confidence:.2f}",
            f"{row.combined_score}",
            f"{row.combined_score:.2f}",
            f"{row.technical_score}",
            f"{row.technical_score:.2f}",
            f"{row.sentiment_score}",
            f"{row.sentiment_score:.2f}",
            f"{row.num_news_articles}",
            row.reasoning,
        ]
    )
    parts.extend(payload.headlines)

    # Adapters — each contributes the scalars it carries when populated.
    if payload.company_profile is not None:
        prof = payload.company_profile
        parts.extend([prof.name, prof.sector, prof.industry, prof.summary])
        parts.extend(prof.revenue_drivers)
        if prof.competitive_moat is not None:
            parts.append(prof.competitive_moat)

    if payload.fundamentals is not None:
        f = payload.fundamentals
        for v in (
            f.pe,
            f.forward_pe,
            f.peg,
            f.ev_ebitda,
            f.price_to_sales,
            f.gross_margin,
            f.operating_margin,
            f.fcf_yield,
            f.revenue_growth_yoy,
            f.eps_growth_yoy,
            f.debt_to_equity,
        ):
            if v is not None:
                parts.append(str(v))
                parts.append(f"{v:.2f}")
                parts.append(f"{v:.1f}")

    if payload.earnings is not None:
        e = payload.earnings
        for v in (
            e.last_eps_actual,
            e.last_eps_estimate,
            e.last_revenue_actual,
            e.last_revenue_estimate,
        ):
            if v is not None:
                parts.append(str(v))
                parts.append(f"{v:.2f}")
        parts.append(str(e.surprise_streak))
        if e.next_report is not None:
            parts.append(e.next_report.isoformat())
            parts.append(str(e.next_report.year))

    if payload.analyst_ratings is not None:
        a = payload.analyst_ratings
        if a.consensus is not None:
            parts.append(a.consensus)
        parts.extend([str(a.buy_count), str(a.hold_count), str(a.sell_count)])
        for v in (a.avg_target, a.high_target, a.low_target):
            if v is not None:
                parts.append(str(v))
                parts.append(f"{v:.2f}")

    if payload.insider_activity is not None:
        i = payload.insider_activity
        if i.net_30d_usd is not None:
            parts.append(str(i.net_30d_usd))
        parts.extend([str(i.buys_30d), str(i.sells_30d)])
        if i.last_event is not None:
            parts.append(i.last_event)

    if payload.macro_context is not None:
        m = payload.macro_context
        for macro_field in (m.regime, m.next_fed_event, m.next_cpi, m.sector_trend):
            if macro_field is not None:
                parts.append(macro_field)

    if payload.social_sentiment is not None:
        s = payload.social_sentiment
        if s.overall is not None:
            parts.append(str(s.overall))
        if s.volume_24h is not None:
            parts.append(str(s.volume_24h))
        parts.extend(s.notable_threads)

    # Single lowercased blob — the substring matcher is case-insensitive.
    return " ".join(str(p) for p in parts).lower()


def _build_allowance(
    thesis: ResearchThesis, payload: ResearchInput
) -> frozenset[str]:
    """Tokens the validator always treats as supported.

    Even when the input corpus doesn't literally contain these
    surface forms, the validator allows them through:

    * The ``confidence`` percent (LLM commonly renders ``0.73`` as
      ``73%`` — we materialize the percent form).
    * The thesis ``generated_at`` year.
    * Small integers 0..5 (counts inside the prose like "2 of 4").
    * The symbol's own ticker.
    """
    out: set[str] = set()
    # Confidence percent form. Different generators render confidence
    # differently: the template uses ``int()`` truncation, an LLM
    # might use ``round()``, prose copy might use either. We add both
    # to the allowance so the surface form doesn't shift the answer.
    raw = payload.row.confidence * 100
    # ``round()`` returns int here (no second arg) so no cast is needed;
    # ruff RUF046 flags ``int(round(...))`` as redundant.
    for pct in {int(raw), round(raw)}:
        out.add(f"{pct}%")
        out.add(f"{pct}.0%")
    # Also the thesis-level confidence, which may have been promoted
    # to a tier-adjusted value distinct from row.confidence.
    thesis_raw = thesis.confidence * 100
    for pct in {int(thesis_raw), round(thesis_raw)}:
        out.add(f"{pct}%")
        out.add(f"{pct}.0%")
    # Year from generated_at.
    try:
        year = datetime.fromisoformat(thesis.generated_at).year
        out.add(str(year))
    except ValueError:
        pass
    # Small integers.
    for n in range(0, 6):
        out.add(str(n))
    # Symbol ticker (numerals like "3M").
    out.add(thesis.symbol.lower())
    return frozenset(out)


# ---------------------------------------------------------------------------
# Per-field scrubbers
# ---------------------------------------------------------------------------


def _scrub_section(
    section: ThesisSection,
    path: str,
    corpus: str,
    allowance: frozenset[str],
    dropped: list[DroppedClaim],
) -> ThesisSection:
    """Validate ``section.body`` + each bullet; return a new ThesisSection."""
    new_body, body_dropped = _scrub_text(section.body, f"{path}.body", corpus, allowance)
    new_bullets: list[str] = []
    for i, bullet in enumerate(section.bullets):
        kept, dropped_for_bullet = _scrub_text(
            bullet, f"{path}.bullets[{i}]", corpus, allowance
        )
        # A bullet is binary — keep it if anything survived, drop entirely
        # otherwise. Bullets are short; a partial-keep would be jarring.
        if kept.strip():
            new_bullets.append(kept)
        body_dropped.extend(dropped_for_bullet)
    dropped.extend(body_dropped)
    return replace(section, body=new_body, bullets=tuple(new_bullets))


def _scrub_arguments(
    args: tuple[BullBearArgument, ...],
    path: str,
    corpus: str,
    allowance: frozenset[str],
    dropped: list[DroppedClaim],
) -> tuple[BullBearArgument, ...]:
    """Validate each argument's ``detail``; drop entries whose detail is empty after scrub."""
    out: list[BullBearArgument] = []
    for i, arg in enumerate(args):
        new_detail, arg_dropped = _scrub_text(
            arg.detail, f"{path}[{i}].detail", corpus, allowance
        )
        dropped.extend(arg_dropped)
        if new_detail.strip():
            out.append(replace(arg, detail=new_detail))
    return tuple(out)


def _scrub_catalysts(
    items: tuple[Catalyst, ...],
    path: str,
    corpus: str,
    allowance: frozenset[str],
    dropped: list[DroppedClaim],
) -> tuple[Catalyst, ...]:
    out: list[Catalyst] = []
    for i, item in enumerate(items):
        new_detail, item_dropped = _scrub_text(
            item.detail, f"{path}[{i}].detail", corpus, allowance
        )
        dropped.extend(item_dropped)
        if new_detail.strip():
            out.append(replace(item, detail=new_detail))
    return tuple(out)


def _scrub_outlook(
    items: tuple[OutlookEntry, ...],
    path: str,
    corpus: str,
    allowance: frozenset[str],
    dropped: list[DroppedClaim],
) -> tuple[OutlookEntry, ...]:
    out: list[OutlookEntry] = []
    for i, item in enumerate(items):
        new_detail, item_dropped = _scrub_text(
            item.detail, f"{path}[{i}].detail", corpus, allowance
        )
        dropped.extend(item_dropped)
        if new_detail.strip():
            out.append(replace(item, detail=new_detail))
    return tuple(out)


def _scrub_metrics(
    items: tuple[MetricEntry, ...],
    path: str,
    corpus: str,
    allowance: frozenset[str],
    dropped: list[DroppedClaim],
) -> tuple[MetricEntry, ...]:
    """Metric values are numeric strings. Drop the entry whole if value is unsupported.

    Unlike prose, a metric card with no value is useless — a bad
    value means the whole card goes.
    """
    out: list[MetricEntry] = []
    for i, item in enumerate(items):
        unsupported = _unsupported_tokens(item.value, corpus, allowance)
        if unsupported:
            dropped.append(
                DroppedClaim(
                    section=f"{path}[{i}].value",
                    sentence=f"{item.label}: {item.value}",
                    unsupported_tokens=unsupported,
                )
            )
            continue
        out.append(item)
    return tuple(out)


# ---------------------------------------------------------------------------
# Token-level matching
# ---------------------------------------------------------------------------


def _scrub_text(
    text: str,
    path: str,
    corpus: str,
    allowance: frozenset[str],
) -> tuple[str, list[DroppedClaim]]:
    """Drop sentences in ``text`` containing any unsupported numeric token.

    Returns ``(kept_text, dropped_claims_for_this_field)``. Empty
    input → ``("", [])``. The sentence splitter is conservative — it
    splits on terminal punctuation followed by a capital letter, so
    quoted decimals ("the press release said 'Q4 was'") aren't split
    mid-token.
    """
    if not text or not text.strip():
        return text, []
    sentences = _split_sentences(text)
    kept: list[str] = []
    dropped: list[DroppedClaim] = []
    for sentence in sentences:
        unsupported = _unsupported_tokens(sentence, corpus, allowance)
        if unsupported:
            dropped.append(
                DroppedClaim(
                    section=path,
                    sentence=sentence.strip(),
                    unsupported_tokens=unsupported,
                )
            )
            continue
        kept.append(sentence)
    return " ".join(s.strip() for s in kept).strip(), dropped


def _unsupported_tokens(
    text: str, corpus: str, allowance: frozenset[str]
) -> tuple[str, ...]:
    """Return the tokens in ``text`` that don't appear in the corpus.

    A token is supported when its lowercase form is either in
    ``allowance`` or substring-matches the lowercase corpus.
    """
    out: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        lowered = token.lower()
        if lowered in allowance:
            continue
        if lowered in corpus:
            continue
        out.append(token)
    return tuple(out)


def _split_sentences(text: str) -> list[str]:
    """Conservative sentence splitter; preserves terminal punctuation.

    Splits on `.`, `?`, `!` followed by whitespace and a capital
    letter or quote — catches "First. Second." but not "0.73 is the
    confidence" or "Q4 2024" mid-stream.
    """
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


def validation_to_wire(report: ValidationReport) -> dict[str, object]:
    """Render a :class:`ValidationReport` to the JSON shape the API emits."""
    return {
        "drop_count": report.drop_count,
        "dropped_claims": [asdict(c) for c in report.dropped_claims],
    }


__all__ = [
    "DroppedClaim",
    "ValidationReport",
    "validate_thesis",
    "validation_to_wire",
]
