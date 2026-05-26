"""Canonical anti-hallucination rules for every LLM-driven feature.

Both :mod:`~src.intelligence.llm_summary` and
:mod:`~src.intelligence.recap` embed the same boilerplate text into
their system prompts. That text lives here so:

* the contract is a single artifact reviewers can audit,
* a future LLM module starts from the same baseline,
* regression-guard tests on both modules continue to verify the
  rules appear in the assembled prompt verbatim — they just source
  from one canonical string.

Three flavors are exposed:

* :data:`GROUNDING_RULES_PER_SYMBOL` — used by ``llm_summary.py``;
  scoped to a single symbol's analysis.
* :data:`GROUNDING_RULES_WATCHLIST` — used by ``recap.py``; adds the
  recap-specific "do not reference symbols outside the watchlist" clause.
* :data:`GROUNDING_RULES_COMPARISON` — used by
  ``comparison_narrative.py``; adds the two-symbol-scope clause.

All three share the common core prefix; comparison + watchlist append
scope-specific clauses. If you change any of them, run the regression
guard tests in ``tests/test_llm_summary.py``, ``tests/test_recap.py``,
and ``tests/intelligence/test_comparison_narrative.py``.
"""

from __future__ import annotations

import re
from xml.sax.saxutils import escape as _xml_escape

from src.utils.logging import get_logger

log = get_logger(__name__)

# Cap on any single user-supplied string interpolated into a prompt
# (symbol, headline, free-text field). Defends against drowning out the
# system prompt with an attacker-supplied 100 KB headline and bounds
# token spend on pathological inputs.
_MAX_PROMPT_VALUE_CHARS = 600

# Sentinel returned by :func:`sanitize_symbol` when the input doesn't
# match the ticker shape. Logged so an operator triaging "why did the
# LLM see UNKNOWN?" can grep for the upstream offender.
_INVALID_SYMBOL_SENTINEL = "UNKNOWN"

# Canonical US-equity ticker shape: starts with an uppercase letter,
# 1-10 chars total, additional chars limited to A-Z 0-9 ``.`` ``-``.
# Covers BRK.B (period), BF-B (dash), and standard tickers up to 10
# chars. Per BUGS.md B-9: "reject symbols that aren't
# ^[A-Z][A-Z0-9.\\-]{0,9}$" — exactly this regex.
_SYMBOL_REGEX = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def sanitize_prompt_value(value: str, *, max_chars: int = _MAX_PROMPT_VALUE_CHARS) -> str:
    """Make an external string safe to interpolate into an LLM prompt.

    - Strips ASCII control characters (except space) so a crafted
      headline cannot inject newlines or NULs that confuse the
      model or the downstream XML response parser.
    - Caps length so attacker-supplied content cannot dominate the
      prompt budget.
    - XML-escapes ``& < > " '`` so any echoed text in the LLM's XML
      response parses cleanly.

    Mitigates (but does not eliminate) prompt injection: a determined
    attacker can still write "Ignore previous instructions" inside the
    cap. Pair this with the existing validators that drop unsupported
    sentences from the final output.

    For ticker symbols specifically, prefer :func:`sanitize_symbol` —
    that enforces the strict ``^[A-Z][A-Z0-9.\\-]{0,9}$`` shape per
    BUGS.md B-9 and returns a sentinel for malformed input.
    """
    if not isinstance(value, str):
        value = str(value)
    cleaned = "".join(ch for ch in value if ch == " " or (ch.isprintable() and ord(ch) >= 0x20))
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "…"
    return _xml_escape(cleaned, {'"': "&quot;", "'": "&apos;"})


def sanitize_symbol(value: str) -> str:
    """Validate ``value`` as a ticker symbol; return a sentinel if it isn't.

    Closes BUGS.md B-9's "reject symbols that aren't ``^[A-Z][A-Z0-9.\\-]{0,9}$``"
    requirement. Accepts the canonical US-equity ticker shape:

    * 1-10 characters total,
    * starts with an uppercase letter,
    * remaining chars limited to ``A-Z`` ``0-9`` ``.`` ``-``.

    Covers ``AAPL``, ``BRK.B`` (period), ``BF-B`` (dash), and standard
    tickers up to 10 chars. Lower-case input is normalized to upper
    before the regex check so a frontend that sends ``aapl`` still
    works.

    On a mismatch we log a warning and return
    :data:`_INVALID_SYMBOL_SENTINEL` (``"UNKNOWN"``) — the LLM sees a
    valid request and the operator gets a grep-able log line pointing
    at the upstream offender. Raising would let one malformed input
    bring down a whole batch of LLM calls; the sentinel preserves
    availability.
    """
    if not isinstance(value, str):
        # Defensive: a None / numeric / object input shouldn't be
        # coerced via ``str()`` because ``str(None)`` becomes ``"None"``
        # which matches the ticker regex post-uppercasing. The sentinel
        # is the safe contract here — the caller knows they passed a
        # non-string and the upstream offender shows up in the log.
        log.warning(
            "grounding.sanitize_symbol.rejected",
            value_type=type(value).__name__,
            reason="non-string input",
        )
        return _INVALID_SYMBOL_SENTINEL
    normalized = value.strip().upper()
    if _SYMBOL_REGEX.fullmatch(normalized) is None:
        log.warning(
            "grounding.sanitize_symbol.rejected",
            value=value[:64],  # cap the log payload too
            reason="did not match ^[A-Z][A-Z0-9.\\-]{0,9}$",
        )
        return _INVALID_SYMBOL_SENTINEL
    # The regex guarantees XML-safe output (only A-Z 0-9 . -). No
    # further escaping required, but pass through the helper for
    # consistency with sanitize_prompt_value's contract.
    return normalized


# Shared prefix: the four core grounding clauses applied to ANY LLM
# output Esther produces. Verbatim substrings of this text are asserted
# by the regression guards.
_CORE_RULES = """- Only summarize information present in the user message. Do not invent specific numbers, \
prices, percentages, or events not in the data provided.
- If a fact would require external knowledge (recent earnings beat, analyst price target, \
company news beyond what's quoted) and that fact is not in the user message, do not \
include it. Stay with what the data shows.
- Quote headlines verbatim or do not reference them at all. Do not paraphrase a headline \
into a stronger or weaker claim than its literal text. Do not extrapolate causation from a \
headline.
- Describe the current state — do not predict where the symbol will go. No forecasts, no \
"likely to" language, no probability claims."""


# Per-symbol grounding text (used by LLMSummarizer).
GROUNDING_RULES_PER_SYMBOL = (
    "GROUNDING RULES — these are non-negotiable and override any instinct to be helpful:"
    "\n\n"
    + _CORE_RULES
)


# Watchlist-wide grounding text (used by LLMRecapGenerator). Same four
# core rules + the recap-specific "stay within the watchlist" clause.
GROUNDING_RULES_WATCHLIST = (
    "GROUNDING RULES — non-negotiable:\n\n"
    + _CORE_RULES
    + "\n- The recap covers ONLY the symbols in the data. Do not bring up symbols, sectors, "
    "indices, or instruments outside the watchlist."
)


# Comparison-narrative grounding text (used by LLMComparisonNarrator).
# Adds the two-symbol-scope clause: the narrative must only reference
# the two symbols passed in. The pre-computed metric verdicts in the
# ComparisonView are the source of truth — the LLM is paraphrasing
# structured data into prose, not generating new claims.
GROUNDING_RULES_COMPARISON = (
    "GROUNDING RULES — non-negotiable:\n\n"
    + _CORE_RULES
    + "\n- This is a comparison between exactly two symbols. Reference ONLY those two "
    "symbols by name. Do not bring up indices, peers, sectors, or any other instrument."
    "\n- The pre-computed metric verdicts in the comparison view are the ground truth. "
    "Do not contradict them — if the view says the left side has a higher trust score, "
    "your narrative must reflect that."
    "\n- Express the trade-off honestly. If one side wins on momentum and the other on "
    "valuation, say so. Do not artificially declare a single winner when the metrics "
    "are split."
)


__all__ = [
    "GROUNDING_RULES_COMPARISON",
    "GROUNDING_RULES_PER_SYMBOL",
    "GROUNDING_RULES_WATCHLIST",
    "sanitize_prompt_value",
    "sanitize_symbol",
]
