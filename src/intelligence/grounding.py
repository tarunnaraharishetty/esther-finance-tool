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
]
