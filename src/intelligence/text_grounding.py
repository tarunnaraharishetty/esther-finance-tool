"""Reusable substring-grounded prose validator.

The thesis (``research_validator``) and compare-narrative
(``comparison_narrative_validator``) modules each ship a similar
"drop any sentence whose numeric tokens aren't in the corpus" routine.
This module exposes a single canonical implementation so other LLM
surfaces (today: :class:`~src.intelligence.llm_summary.LLMSummarizer`)
can adopt the same contract without re-implementing the regex and
the sentence splitter.

It is **not** a refactor of the two existing validators — they have
field-level logic and provenance maps that are surface-specific. The
shared helper is the *base case*: scrub a free-form blob of text
against a corpus and report what was dropped. Future cleanup can lift
the existing validators onto this helper; for now they stay
authoritative for their own surfaces.

Contract
--------
Given:

* ``text``: model-generated prose.
* ``corpus``: a single lowercase string containing every
  numeric scalar the model was allowed to mention (built by the
  caller from its own structured input).
* ``allowance``: a small ``frozenset[str]`` of tokens that are
  always supported (lowercase) — e.g. small integers, the year of
  ``generated_at``, the ticker.

A token is *supported* iff its lowercase form is in ``allowance`` OR
substring-matches the corpus. A sentence is dropped if it contains
any unsupported token. The function returns the cleaned text plus a
list of :class:`DroppedClaim` records the caller can log or surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical single source of truth for numeric-token surface forms
# across every grounded-AI validator (closes BUGS.md B-19). The two
# field-aware validators (research, comparison_narrative) import
# ``TOKEN_PATTERN`` + ``SENTENCE_SPLIT`` from this module instead of
# carrying private copies. The frontend ``ProvenanceProse.tsx`` mirrors
# the same alternation in a JS literal; a regression test reads that
# file and asserts behavioural parity on a full-coverage fixture so
# drift between Python and TS fails CI loudly.
TOKEN_PATTERN = re.compile(
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

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")

# Backward-compat aliases for any external reader that grabbed the
# underscore-prefixed names before B-19 promoted them. Both module-
# level names compile to the same object so behaviour is unchanged.
_TOKEN_PATTERN = TOKEN_PATTERN
_SENTENCE_SPLIT = SENTENCE_SPLIT


@dataclass(frozen=True)
class DroppedClaim:
    """One sentence the validator removed.

    ``where`` is a free-form label naming the surface that produced
    the text (e.g. ``"llm_summary"``) so callers logging multiple
    sites can disambiguate.
    """

    where: str
    sentence: str
    unsupported_tokens: tuple[str, ...]


def build_default_allowance(
    *, symbol: str, extra: tuple[str, ...] = ()
) -> frozenset[str]:
    """Build the always-supported token set.

    Includes:
    * ``symbol`` (lowercased) — the ticker is allowed to appear in
      prose as-is even when it lands inside a number regex match.
    * Single-digit integers and the small list / count integers a
      grounded brief uses to count contributors / headlines.
    * Common percentage forms of trivial fractions
      (``0%``, ``50%``, ``100%``).
    * Caller-provided ``extra`` tokens (already lowercased).
    """
    base: list[str] = [symbol.lower()]
    base.extend(str(i) for i in range(0, 11))
    base.extend(f"{i}%" for i in (0, 25, 50, 75, 100))
    base.extend(extra)
    return frozenset(base)


def build_corpus(*entries: str) -> str:
    """Concatenate the entries into a single lowercase substring corpus."""
    return " ".join(e for e in entries if e).lower()


def unsupported_tokens(
    text: str, corpus: str, allowance: frozenset[str]
) -> tuple[str, ...]:
    """Return the tokens in ``text`` that aren't grounded.

    Mirrors the contract of the surface-specific validators:
    case-insensitive substring match against the corpus, with the
    allowance set as a short-circuit override.
    """
    out: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        lowered = token.lower()
        if lowered in allowance:
            continue
        if lowered in corpus:
            continue
        out.append(token)
    return tuple(out)


def validate_prose(
    text: str,
    corpus: str,
    allowance: frozenset[str],
    *,
    where: str,
) -> tuple[str, tuple[DroppedClaim, ...]]:
    """Drop sentences in ``text`` whose numeric tokens aren't in ``corpus``.

    Returns ``(cleaned_text, dropped_claims)``. Empty or all-whitespace
    input is passed through unchanged with an empty dropped tuple.

    The sentence splitter is conservative — it splits on terminal
    punctuation followed by a capital letter or opening quote, so
    decimals inside a sentence ("revenue rose to 12.4% in Q4") are
    not split mid-token.
    """
    if not text or not text.strip():
        return text, ()
    parts = [p for p in SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    kept: list[str] = []
    dropped: list[DroppedClaim] = []
    for sentence in parts:
        unsupp = unsupported_tokens(sentence, corpus, allowance)
        if unsupp:
            dropped.append(
                DroppedClaim(
                    where=where,
                    sentence=sentence.strip(),
                    unsupported_tokens=unsupp,
                )
            )
            continue
        kept.append(sentence.strip())
    return " ".join(kept).strip(), tuple(dropped)


__all__ = [
    "SENTENCE_SPLIT",
    "TOKEN_PATTERN",
    "DroppedClaim",
    "build_corpus",
    "build_default_allowance",
    "unsupported_tokens",
    "validate_prose",
]
