"""Canonical anti-hallucination rules for every LLM-driven feature.

Both :mod:`~src.intelligence.llm_summary` and
:mod:`~src.intelligence.recap` embed the same boilerplate text into
their system prompts. That text lives here so:

* the contract is a single artifact reviewers can audit,
* a future LLM module starts from the same baseline,
* regression-guard tests on both modules continue to verify the
  rules appear in the assembled prompt verbatim — they just source
  from one canonical string.

Two flavors are exposed:

* :data:`GROUNDING_RULES_PER_SYMBOL` — used by ``llm_summary.py``;
  scoped to a single symbol's analysis.
* :data:`GROUNDING_RULES_WATCHLIST` — used by ``recap.py``; adds the
  recap-specific "do not reference symbols outside the watchlist" clause.

Both clauses share a common prefix; the watchlist version appends the
extra clause. If you change either, run the regression guard tests in
``tests/test_llm_summary.py`` and ``tests/test_recap.py``.
"""

from __future__ import annotations


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


__all__ = ["GROUNDING_RULES_PER_SYMBOL", "GROUNDING_RULES_WATCHLIST"]
