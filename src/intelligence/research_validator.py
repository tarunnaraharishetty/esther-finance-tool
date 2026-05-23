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
    corpus_entries = _build_corpus_entries(thesis, payload)
    corpus = " ".join(text for text, _label in corpus_entries).lower()
    allowance = _build_allowance(thesis, payload)
    dropped: list[DroppedClaim] = []

    # ThesisSection fields are validated in place via ``replace``.
    # Tuple fields (bull/bear/catalysts/outlook) need per-entry handling
    # so dropped entries vanish from the tuple instead of leaving holes.

    new_company = _scrub_section(
        thesis.company_overview, "company_overview", corpus, corpus_entries, allowance, dropped
    )
    new_technical = _scrub_section(
        thesis.technical_analysis,
        "technical_analysis",
        corpus,
        corpus_entries,
        allowance,
        dropped,
    )
    new_fundamental = _scrub_section(
        thesis.fundamental_analysis,
        "fundamental_analysis",
        corpus,
        corpus_entries,
        allowance,
        dropped,
    )
    new_sentiment = _scrub_section(
        thesis.sentiment_news, "sentiment_news", corpus, corpus_entries, allowance, dropped
    )
    new_risk = _scrub_section(
        thesis.risk_assessment, "risk_assessment", corpus, corpus_entries, allowance, dropped
    )
    new_explainability = _scrub_section(
        thesis.explainability, "explainability", corpus, corpus_entries, allowance, dropped
    )

    new_tagline, tagline_dropped = _scrub_text(
        thesis.tagline, "tagline", corpus, allowance
    )
    dropped.extend(tagline_dropped)

    new_bull = _scrub_arguments(
        thesis.bull_thesis, "bull_thesis", corpus, corpus_entries, allowance, dropped
    )
    new_bear = _scrub_arguments(
        thesis.bear_thesis, "bear_thesis", corpus, corpus_entries, allowance, dropped
    )
    new_catalysts = _scrub_catalysts(
        thesis.catalysts, "catalysts", corpus, corpus_entries, allowance, dropped
    )
    new_outlook = _scrub_outlook(
        thesis.outlook, "outlook", corpus, corpus_entries, allowance, dropped
    )
    new_metrics = _scrub_metrics(
        thesis.metrics, "metrics", corpus, corpus_entries, allowance, dropped
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

    Delegates to :func:`_build_corpus_entries` and flattens the
    labeled entries into the original concatenated-blob shape the
    drop-policy substring matcher expects. The two builders share a
    source-of-truth so a new corpus field automatically participates
    in both the drop policy AND the provenance graph.
    """
    return " ".join(text for text, _label in _build_corpus_entries(thesis, payload)).lower()


def _build_corpus_entries(
    thesis: ResearchThesis, payload: ResearchInput
) -> list[tuple[str, str]]:
    """Build the labeled corpus — each entry is ``(text, source_label)``.

    Source labels are short dotted paths the provenance tooltip
    renders verbatim (e.g. ``"row.last_price"``, ``"fundamentals.pe"``,
    ``"headlines[0]"``). Multiple surface forms of the same value
    (``101.61`` / ``101.61`` / ``$101.61``) share one source label
    so the tooltip is stable regardless of how the LLM rendered the
    number.

    Adding a new corpus field here automatically wires it into both
    the drop policy and the provenance graph.
    """
    entries: list[tuple[str, str]] = []
    row = payload.row
    entries.append((row.symbol, "row.symbol"))
    entries.append((f"{row.last_price}", "row.last_price"))
    entries.append((f"{row.last_price:.2f}", "row.last_price"))
    entries.append((f"${row.last_price:.2f}", "row.last_price"))
    entries.append((f"{row.rsi}", "row.rsi"))
    entries.append((f"{row.rsi:.1f}", "row.rsi"))
    entries.append((f"{row.macd}", "row.macd"))
    entries.append((f"{row.macd:.2f}", "row.macd"))
    entries.append((f"{row.bollinger}", "row.bollinger"))
    entries.append((f"{row.bollinger:.2f}", "row.bollinger"))
    entries.append((f"{row.confidence}", "row.confidence"))
    entries.append((f"{row.confidence:.2f}", "row.confidence"))
    entries.append((f"{row.combined_score}", "row.combined_score"))
    entries.append((f"{row.combined_score:.2f}", "row.combined_score"))
    entries.append((f"{row.technical_score}", "row.technical_score"))
    entries.append((f"{row.technical_score:.2f}", "row.technical_score"))
    entries.append((f"{row.sentiment_score}", "row.sentiment_score"))
    entries.append((f"{row.sentiment_score:.2f}", "row.sentiment_score"))
    entries.append((f"{row.num_news_articles}", "row.num_news_articles"))
    entries.append((row.reasoning, "row.reasoning"))
    for idx, headline in enumerate(payload.headlines):
        entries.append((headline, f"headlines[{idx}]"))

    # Adapters — each contributes the scalars it carries when populated.
    if payload.company_profile is not None:
        prof = payload.company_profile
        entries.append((prof.name, "company_profile.name"))
        entries.append((prof.sector, "company_profile.sector"))
        entries.append((prof.industry, "company_profile.industry"))
        entries.append((prof.summary, "company_profile.summary"))
        for idx, driver in enumerate(prof.revenue_drivers):
            entries.append((driver, f"company_profile.revenue_drivers[{idx}]"))
        if prof.competitive_moat is not None:
            entries.append(
                (prof.competitive_moat, "company_profile.competitive_moat")
            )

    if payload.fundamentals is not None:
        f = payload.fundamentals
        for field_name, value in (
            ("pe", f.pe),
            ("forward_pe", f.forward_pe),
            ("peg", f.peg),
            ("ev_ebitda", f.ev_ebitda),
            ("price_to_sales", f.price_to_sales),
            ("gross_margin", f.gross_margin),
            ("operating_margin", f.operating_margin),
            ("fcf_yield", f.fcf_yield),
            ("revenue_growth_yoy", f.revenue_growth_yoy),
            ("eps_growth_yoy", f.eps_growth_yoy),
            ("debt_to_equity", f.debt_to_equity),
        ):
            if value is not None:
                label = f"fundamentals.{field_name}"
                entries.append((str(value), label))
                entries.append((f"{value:.2f}", label))
                entries.append((f"{value:.1f}", label))

    if payload.earnings is not None:
        e = payload.earnings
        for field_name, value in (
            ("last_eps_actual", e.last_eps_actual),
            ("last_eps_estimate", e.last_eps_estimate),
            ("last_revenue_actual", e.last_revenue_actual),
            ("last_revenue_estimate", e.last_revenue_estimate),
        ):
            if value is not None:
                label = f"earnings.{field_name}"
                entries.append((str(value), label))
                entries.append((f"{value:.2f}", label))
        entries.append((str(e.surprise_streak), "earnings.surprise_streak"))
        if e.next_report is not None:
            entries.append((e.next_report.isoformat(), "earnings.next_report"))
            entries.append((str(e.next_report.year), "earnings.next_report.year"))

    if payload.analyst_ratings is not None:
        a = payload.analyst_ratings
        if a.consensus is not None:
            entries.append((a.consensus, "analyst_ratings.consensus"))
        entries.append((str(a.buy_count), "analyst_ratings.buy_count"))
        entries.append((str(a.hold_count), "analyst_ratings.hold_count"))
        entries.append((str(a.sell_count), "analyst_ratings.sell_count"))
        for field_name, value in (
            ("avg_target", a.avg_target),
            ("high_target", a.high_target),
            ("low_target", a.low_target),
        ):
            if value is not None:
                label = f"analyst_ratings.{field_name}"
                entries.append((str(value), label))
                entries.append((f"{value:.2f}", label))

    if payload.insider_activity is not None:
        ins = payload.insider_activity
        if ins.net_30d_usd is not None:
            entries.append(
                (str(ins.net_30d_usd), "insider_activity.net_30d_usd")
            )
        entries.append((str(ins.buys_30d), "insider_activity.buys_30d"))
        entries.append((str(ins.sells_30d), "insider_activity.sells_30d"))
        if ins.last_event is not None:
            entries.append((ins.last_event, "insider_activity.last_event"))

    if payload.macro_context is not None:
        m = payload.macro_context
        # Iteration order intentional — explicit typing so mypy doesn't
        # widen the value union with the unrelated fundamentals loop above.
        macro_fields: tuple[tuple[str, str | None], ...] = (
            ("regime", m.regime),
            ("next_fed_event", m.next_fed_event),
            ("next_cpi", m.next_cpi),
            ("sector_trend", m.sector_trend),
        )
        for macro_name, macro_value in macro_fields:
            if macro_value is not None:
                entries.append((macro_value, f"macro_context.{macro_name}"))

    if payload.social_sentiment is not None:
        s = payload.social_sentiment
        if s.overall is not None:
            entries.append((str(s.overall), "social_sentiment.overall"))
        if s.volume_24h is not None:
            entries.append((str(s.volume_24h), "social_sentiment.volume_24h"))
        for idx, thread in enumerate(s.notable_threads):
            entries.append(
                (thread, f"social_sentiment.notable_threads[{idx}]")
            )

    # Defensive: filter out empty-string entries (a None upstream would
    # have been skipped, but an empty string would substring-match
    # everything which is a grounding hole).
    return [(text, label) for text, label in entries if text]


def _token_provenance(
    text: str,
    corpus_entries: list[tuple[str, str]],
    allowance: frozenset[str],
) -> dict[str, str]:
    """Record the source label for each supported numeric token in ``text``.

    Returns ``{original_token: source_label}`` for tokens that
    substring-match a corpus entry. Tokens that only clear the
    allowance (small integers, year, ticker) are intentionally
    omitted — they're trivially supported but have no meaningful
    source to point at, so the UI renders them as plain text.

    First-match-wins: when a token substring-matches multiple corpus
    entries (e.g. ``"100"`` appears in both row.combined_score and
    a headline), the first entry in the list wins. The corpus
    builder orders entries by relevance — row scalars before
    headlines — so the chosen label is the most informative one.
    """
    if not text:
        return {}
    out: dict[str, str] = {}
    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        if token in out:
            continue  # First occurrence in text wins for UI stability.
        lowered = token.lower()
        if lowered in allowance:
            continue
        for entry_text, source_label in corpus_entries:
            if lowered in entry_text.lower():
                out[token] = source_label
                break
    return out


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
    corpus_entries: list[tuple[str, str]],
    allowance: frozenset[str],
    dropped: list[DroppedClaim],
) -> ThesisSection:
    """Validate ``section.body`` + each bullet; return a new ThesisSection.

    Also attaches per-token provenance covering tokens in the body
    AND any surviving bullets. The provenance map is a single dict
    per section — duplicate tokens (same token appearing in body
    and bullets) collapse to one entry (body takes precedence since
    we merge body first).
    """
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
    provenance = _token_provenance(new_body, corpus_entries, allowance)
    for bullet in new_bullets:
        for token, label in _token_provenance(bullet, corpus_entries, allowance).items():
            provenance.setdefault(token, label)
    return replace(
        section,
        body=new_body,
        bullets=tuple(new_bullets),
        provenance=provenance,
    )


def _scrub_arguments(
    args: tuple[BullBearArgument, ...],
    path: str,
    corpus: str,
    corpus_entries: list[tuple[str, str]],
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
            provenance = _token_provenance(
                new_detail, corpus_entries, allowance
            )
            out.append(replace(arg, detail=new_detail, provenance=provenance))
    return tuple(out)


def _scrub_catalysts(
    items: tuple[Catalyst, ...],
    path: str,
    corpus: str,
    corpus_entries: list[tuple[str, str]],
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
            provenance = _token_provenance(
                new_detail, corpus_entries, allowance
            )
            out.append(replace(item, detail=new_detail, provenance=provenance))
    return tuple(out)


def _scrub_outlook(
    items: tuple[OutlookEntry, ...],
    path: str,
    corpus: str,
    corpus_entries: list[tuple[str, str]],
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
            provenance = _token_provenance(
                new_detail, corpus_entries, allowance
            )
            out.append(replace(item, detail=new_detail, provenance=provenance))
    return tuple(out)


def _scrub_metrics(
    items: tuple[MetricEntry, ...],
    path: str,
    corpus: str,
    corpus_entries: list[tuple[str, str]],
    allowance: frozenset[str],
    dropped: list[DroppedClaim],
) -> tuple[MetricEntry, ...]:
    """Metric values are numeric strings. Drop the entry whole if value is unsupported.

    Unlike prose, a metric card with no value is useless — a bad
    value means the whole card goes. Surviving metrics get a
    provenance dict over the value tokens so the UI can tooltip
    the value chip.
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
        provenance = _token_provenance(item.value, corpus_entries, allowance)
        out.append(replace(item, provenance=provenance))
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
