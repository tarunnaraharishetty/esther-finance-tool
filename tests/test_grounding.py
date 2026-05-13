"""Tests for the centralized grounding-rules contract.

The two LLM modules (LLMSummarizer, LLMRecapGenerator) both embed the
canonical text from src.intelligence.grounding into their system
prompts. This test makes sure they continue to do so — if either ever
silently diverges, the canonical text loses its single-source-of-truth
property.
"""

from __future__ import annotations

from src.intelligence.grounding import (
    GROUNDING_RULES_PER_SYMBOL,
    GROUNDING_RULES_WATCHLIST,
)


def test_per_symbol_rules_contain_core_clauses() -> None:
    """The four core grounding clauses must all appear in the per-symbol text."""
    text = GROUNDING_RULES_PER_SYMBOL
    assert "Only summarize information present in the user message" in text
    assert "Do not invent" in text
    assert "Quote headlines verbatim" in text
    assert "do not predict" in text.lower()


def test_watchlist_rules_extend_per_symbol_with_scope_clause() -> None:
    """Watchlist version = core clauses + the recap-specific scope clause."""
    text = GROUNDING_RULES_WATCHLIST
    # Core clauses still present.
    assert "Only summarize information present in the user message" in text
    assert "Do not invent" in text
    assert "Quote headlines verbatim" in text
    assert "do not predict" in text.lower()
    # Recap-specific scope clause.
    assert "ONLY the symbols in the data" in text


def test_llm_summary_system_prompt_embeds_canonical_text() -> None:
    """Single source of truth: llm_summary.py's prompt contains the
    canonical per-symbol grounding text verbatim. If this fails, the
    LLM module has forked the rules — fix by re-embedding the constant."""
    from src.intelligence import llm_summary

    assert GROUNDING_RULES_PER_SYMBOL in llm_summary._SYSTEM_PROMPT


def test_recap_system_prompt_embeds_canonical_text() -> None:
    """Same contract for the recap generator."""
    from src.intelligence import recap

    assert GROUNDING_RULES_WATCHLIST in recap._SYSTEM_PROMPT
