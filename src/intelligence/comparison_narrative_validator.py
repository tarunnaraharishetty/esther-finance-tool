"""Substring-grounded validator for the comparison narrative.

Mirrors the contract of :mod:`~src.intelligence.research_validator`
but with a corpus tuned to the two-symbol comparison surface:

* Both analyzer reports' grounding-eligible scalars (symbols,
  prices, scores, freshness tier, valuation cases, trust grade).
* The pre-computed :class:`ComparisonView`\\ s metric values + the
  headline displays + section titles + symbol names.

A claim survives iff every numeric token in it appears verbatim
(case-insensitive substring) somewhere in this corpus, OR is in a
short allowance set (small integers, year, percentage forms of
known fractions). Anything else gets dropped and counted as a
``DroppedClaim`` so the UI can render "AI declined N unsupported
claims" as a positive trust signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.intelligence.comparison import ComparisonView
from src.intelligence.comparison_narrative import (
    SECTION_KEYS,
    ComparisonNarrative,
    NarrativeSection,
    replace_section,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


# Same token regex as research_validator, intentionally duplicated
# here so the comparison surface owns its own surface-form rules
# without coupling to the thesis validator's internals.
_TOKEN_PATTERN = re.compile(
    r"""
    (?:
        \$(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?[KMBT]?
      | -?\d{1,3}(?:\.\d+)?%
      | \d+(?:\.\d+)?[xX]
      | \bQ[1-4](?:\s*\d{2,4})?\b
      | \b20\d{2}\b
      | \b\d+\.\d+\b
      | \b\d{6,}\b
    )
    """,
    re.VERBOSE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")


@dataclass(frozen=True)
class DroppedNarrativeClaim:
    """One sentence the validator removed from the narrative."""

    section: str  # e.g. "momentum.body" or "valuation.bullets[1]"
    sentence: str
    unsupported_tokens: tuple[str, ...]


@dataclass(frozen=True)
class NarrativeValidationReport:
    """Aggregate validator output for one narrative."""

    drop_count: int
    dropped_claims: tuple[DroppedNarrativeClaim, ...]


def validate_narrative(
    narrative: ComparisonNarrative,
    view: ComparisonView,
    left_report: dict[str, Any] | None,
    right_report: dict[str, Any] | None,
) -> tuple[ComparisonNarrative, NarrativeValidationReport]:
    """Scrub every prose field; return ``(validated, report)``.

    Sections whose body fully scrubs out keep an empty body — the
    section header still renders so the user sees "we tried but had
    nothing to ground a momentum claim." Bullets that scrub out are
    dropped entirely.
    """
    corpus = _build_corpus(narrative, view, left_report, right_report)
    allowance = _build_allowance(narrative, view)
    dropped: list[DroppedNarrativeClaim] = []

    validated = narrative
    for key in SECTION_KEYS:
        section: NarrativeSection = getattr(validated, key)
        new_body, body_drops = _scrub_text(
            section.body, f"{key}.body", corpus, allowance
        )
        dropped.extend(body_drops)
        new_bullets: list[str] = []
        for i, bullet in enumerate(section.bullets):
            kept, bullet_drops = _scrub_text(
                bullet, f"{key}.bullets[{i}]", corpus, allowance
            )
            dropped.extend(bullet_drops)
            if kept.strip():
                new_bullets.append(kept)
        validated = replace_section(
            validated, key, body=new_body, bullets=tuple(new_bullets)
        )

    # Tagline is short prose — same scrub rules.
    new_tagline, tagline_drops = _scrub_text(
        narrative.tagline, "tagline", corpus, allowance
    )
    dropped.extend(tagline_drops)
    if new_tagline != narrative.tagline:
        from dataclasses import replace as dataclass_replace

        validated = dataclass_replace(validated, tagline=new_tagline)

    report = NarrativeValidationReport(
        drop_count=len(dropped),
        dropped_claims=tuple(dropped),
    )
    if dropped:
        log.info(
            "compare_narrative.validator.dropped",
            left=narrative.left_symbol,
            right=narrative.right_symbol,
            drop_count=len(dropped),
        )
    return validated, report


# ---------------------------------------------------------------------------
# Corpus / allowance construction
# ---------------------------------------------------------------------------


def _build_corpus(
    narrative: ComparisonNarrative,
    view: ComparisonView,
    left_report: dict[str, Any] | None,
    right_report: dict[str, Any] | None,
) -> str:
    """Concatenate every grounding-eligible scalar into one lowercase blob."""
    parts: list[str] = [
        narrative.left_symbol,
        narrative.right_symbol,
    ]
    # Pre-computed view metrics — these are the canonical numbers.
    for section in view.sections:
        for m in section.metrics:
            for v in (m.left_value, m.right_value):
                if v is not None:
                    parts.extend(_render_numeric_forms(v))
    # Headline display strings — already formatted for display, include verbatim.
    for h in view.headline:
        parts.append(h.left_display)
        parts.append(h.right_display)
    # Raw reports — extract every grounding-eligible scalar.
    for report in (left_report, right_report):
        if report is not None:
            parts.extend(_extract_report_scalars(report))
    return " ".join(str(p) for p in parts if p).lower()


def _extract_report_scalars(report: dict[str, Any]) -> list[str]:
    """Pull every value from the analyzer report worth grounding against."""
    parts: list[str] = []
    symbol = report.get("symbol")
    if symbol:
        parts.append(str(symbol))
    last_price = report.get("last_price")
    if last_price is not None:
        parts.extend(_render_numeric_forms(last_price))
    for key in (
        "overall_analyzer_score",
        "technical_score",
        "fundamental_score",
        "valuation_score",
    ):
        v = report.get(key)
        if v is not None:
            parts.extend(_render_numeric_forms(v))
    technicals = report.get("technicals") or {}
    for k in (
        "overbought_score",
        "oversold_score",
        "pullback_risk",
        "rebound_potential",
        "confidence_score",
        "raw_rsi",
        "raw_bollinger_z",
        "raw_ma_distance_pct",
        "raw_volume_z",
        "raw_atr_ratio",
    ):
        v = technicals.get(k)
        if v is not None:
            parts.extend(_render_numeric_forms(v))
    valuation = report.get("valuation") or {}
    for k in (
        "bear_case",
        "base_case",
        "bull_case",
        "weighted_ai_fair_value",
        "confidence_score",
    ):
        v = valuation.get(k)
        if v is not None:
            parts.extend(_render_numeric_forms(v))
    fresh = report.get("fundamentals_freshness") or {}
    for k in ("freshness", "data_age_days", "provider_confidence"):
        v = fresh.get(k)
        if v is not None:
            parts.append(str(v))
            if isinstance(v, (int, float)):
                parts.extend(_render_numeric_forms(v))
    chain = fresh.get("source_chain") if fresh else None
    if isinstance(chain, list):
        parts.extend(str(p) for p in chain)
    trust = report.get("trust_score") or {}
    score = trust.get("score")
    if score is not None:
        parts.extend(_render_numeric_forms(score))
    grade = trust.get("grade")
    if grade:
        parts.append(str(grade))
    return parts


def _render_numeric_forms(value: float | int) -> list[str]:
    """Render one number in every surface form the LLM might emit.

    Avoids false negatives when the model writes "92" but the corpus
    only carries "92.0", or writes "0.85" but the corpus only carries
    "85%". The set is small — keep it conservative; bigger surface
    forms = weaker grounding contract.
    """
    out: list[str] = [str(value)]
    try:
        f = float(value)
    except (TypeError, ValueError):
        return out
    out.append(f"{f:.0f}")
    out.append(f"{f:.1f}")
    out.append(f"{f:.2f}")
    out.append(f"{f:.3f}")
    out.append(f"${f:,.2f}")
    out.append(f"${f:.0f}")
    # Percent form when the value looks like a fraction (|v| <= 1).
    if -1.0 <= f <= 1.0:
        pct = f * 100.0
        out.append(f"{pct:.0f}%")
        out.append(f"{pct:.1f}%")
        out.append(f"{pct:.2f}%")
    # Percent form when the value already looks like a percentage
    # (so the LLM can write it back as a percent).
    if abs(f) >= 1.0:
        out.append(f"{f:.0f}%")
        out.append(f"{f:.1f}%")
    return out


def _build_allowance(
    narrative: ComparisonNarrative, view: ComparisonView
) -> frozenset[str]:
    """Tokens the validator always treats as supported.

    Small integers (counts inside prose like "wins 3 of 4"),
    current year, and the two symbol tickers themselves.
    """
    out: set[str] = set()
    # Small integers used for counts ("5 of 8 metrics", "0 ties").
    for n in range(0, 16):
        out.add(str(n))
        out.add(f"{n}.0")
    # Current year — common in body text.
    try:
        from datetime import datetime as _dt

        year = _dt.fromisoformat(narrative.generated_at).year
        out.add(str(year))
    except (ValueError, TypeError):
        pass
    # Tickers themselves (lowercase).
    out.add(narrative.left_symbol.lower())
    out.add(narrative.right_symbol.lower())
    return frozenset(out)


# ---------------------------------------------------------------------------
# Text scrubbing
# ---------------------------------------------------------------------------


def _scrub_text(
    text: str,
    path: str,
    corpus: str,
    allowance: frozenset[str],
) -> tuple[str, list[DroppedNarrativeClaim]]:
    """Drop sentences with unsupported numeric tokens; return (kept, drops)."""
    if not text or not text.strip():
        return text, []
    sentences = _split_sentences(text)
    kept: list[str] = []
    dropped: list[DroppedNarrativeClaim] = []
    for sentence in sentences:
        unsupported = _unsupported_tokens(sentence, corpus, allowance)
        if unsupported:
            dropped.append(
                DroppedNarrativeClaim(
                    section=path,
                    sentence=sentence.strip(),
                    unsupported_tokens=unsupported,
                )
            )
            continue
        kept.append(sentence)
    return " ".join(s.strip() for s in kept).strip(), dropped


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p for p in parts if p.strip()]


def _unsupported_tokens(
    text: str, corpus: str, allowance: frozenset[str]
) -> tuple[str, ...]:
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


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


def narrative_validation_to_wire(
    report: NarrativeValidationReport,
) -> dict[str, Any]:
    """Render the validator output to the JSON shape the API emits.

    Mirrors :func:`src.intelligence.research_validator.validation_to_wire`
    so the same ValidationBadge component on the frontend can consume
    either source.
    """
    return {
        "drop_count": report.drop_count,
        "dropped_claims": [
            {
                "section": c.section,
                "sentence": c.sentence,
                "unsupported_tokens": list(c.unsupported_tokens),
            }
            for c in report.dropped_claims
        ],
    }


__all__ = [
    "DroppedNarrativeClaim",
    "NarrativeValidationReport",
    "narrative_validation_to_wire",
    "validate_narrative",
]
