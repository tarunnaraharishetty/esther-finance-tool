"""Prompt templates for the LLM-backed analyzer explanation.

Kept separate from the generator so the prompt body can be revised
without touching the SDK plumbing, and so tests can render the
``DATA`` block against a fixture and assert it contains exactly the
numbers it was meant to.

The prompt has three hard rules baked in:

1. The model may ONLY cite tags from the ``ALLOWED CITATIONS`` block
   the user message hands it. That block is derived at runtime from
   :meth:`AnalyzerInputs.available_tags`, so a missing fundamentals
   stream literally removes ``[fundamentals]`` from the citation
   vocabulary the model sees.
2. Every claim must end in at least one citation tag. Claims without
   tags are rejected by :func:`validate_citations` after the call.
3. The model may not reference any number that does not appear in the
   ``DATA`` block — the prompt makes this explicit and the validator
   enforces grounding via the citation requirement.
"""

from __future__ import annotations

from src.intelligence.analyzer.explanation import (
    CITATION_TAGS,
    AnalyzerInputs,
)
from src.intelligence.grounding import sanitize_prompt_value, sanitize_symbol

PROMPT_VERSION = "analyzer-explain-v1"


SYSTEM_PROMPT = """You are Esther's Financial Analyzer reasoning layer. Your single \
job is to explain *why* the analyzer's scoring landed where it did, using only the \
inputs handed to you in the DATA block.

GROUNDING CONTRACT (non-negotiable):

1. Every claim you emit must end with at least one citation tag in square brackets. \
The vocabulary is fixed: [technicals], [fundamentals], [news], [sentiment], \
[valuation], [volume]. You may emit multiple tags as [tag1, tag2] when a claim \
draws on multiple streams.

2. You may ONLY cite tags listed in the ALLOWED CITATIONS section of the user \
message. Cite anything else and the response is rejected and discarded.

3. Every number, ratio, headline, or score you mention MUST appear verbatim in the \
DATA block. Do not synthesize, project, or recall anything from training data — if \
it isn't in DATA, you cannot say it.

4. Empty input streams: if an input is missing, you cannot mention or imply it. \
Saying "the company has no fundamentals data" requires [fundamentals] to be on the \
allowed list — and it won't be, by construction. The correct behavior is silence \
on that stream.

5. No price targets. No buy/sell language. No imperative phrasing. You are \
explaining the analyzer's read, not telling the user what to do.

OUTPUT FORMAT — return exactly one XML document, no markdown, no prose outside:

<explanation>
  <summary>One sentence framing the analyzer's read. May omit citations because it \
is framed as the analyzer's view, not raw data.</summary>
  <claim>One supporting claim grounded in DATA, ending in [tag] or [tag, tag].</claim>
  <claim>...</claim>
  <!-- 3 to 7 claims total. -->
  <risk>One risk warning, ending in [tag].</risk>
  <!-- 0 to 4 risks. -->
</explanation>
"""


def build_user_message(inputs: AnalyzerInputs) -> str:
    """Render the structured DATA block plus the allowed-citation list."""
    allowed = sorted(inputs.available_tags())
    if not allowed:
        # No grounded streams — the LLM literally has nothing to say.
        # The caller should not invoke the LLM in this case; we still
        # render a coherent message in defense.
        allowed_block = "(none — do not generate any claims)"
    else:
        allowed_block = ", ".join(f"[{t}]" for t in allowed)

    sections: list[str] = [
        # Strict ticker-shape validation (BUGS.md B-9). Malformed
        # symbols become "UNKNOWN" + a logged warning rather than
        # flowing verbatim into the LLM system prompt.
        f"SYMBOL: {sanitize_symbol(inputs.symbol)}",
        f"ALLOWED CITATIONS: {allowed_block}",
        "",
        "DATA:",
    ]

    if inputs.last_price is not None:
        sections.append(f"- last_price: ${inputs.last_price:.2f}")

    if inputs.technicals is not None:
        sections.append(_render_technicals(inputs))

    if inputs.fundamentals is not None:
        sections.append(_render_fundamentals(inputs))

    if inputs.valuation is not None:
        sections.append(_render_valuation(inputs))

    if inputs.sentiment_score is not None:
        sections.append(f"- sentiment_composite: {inputs.sentiment_score:+.2f}")

    if inputs.headline_count is not None or inputs.top_headlines:
        sections.append(_render_news(inputs))

    if inputs.volume_z is not None or inputs.volume_spike_score is not None:
        sections.append(_render_volume(inputs))

    sections.append("")
    sections.append(
        "Produce the <explanation> XML now. Cite only the tags listed above; "
        "every claim must carry at least one tag."
    )
    return "\n".join(sections)


def _render_technicals(inputs: AnalyzerInputs) -> str:
    tech = inputs.technicals
    assert tech is not None  # caller-checked
    lines: list[str] = ["TECHNICALS:"]
    if tech.raw_rsi is not None:
        lines.append(f"- rsi_14: {tech.raw_rsi:.2f}")
    if tech.overbought_score is not None:
        lines.append(f"- overbought_score: {tech.overbought_score:.1f}/100")
    if tech.oversold_score is not None:
        lines.append(f"- oversold_score: {tech.oversold_score:.1f}/100")
    if tech.pullback_risk is not None:
        lines.append(f"- pullback_risk: {tech.pullback_risk:.1f}/100")
    if tech.rebound_potential is not None:
        lines.append(f"- rebound_potential: {tech.rebound_potential:.1f}/100")
    if tech.raw_ma_distance_pct is not None:
        lines.append(f"- distance_from_50bar_sma_pct: {tech.raw_ma_distance_pct:+.2f}")
    if tech.raw_atr_ratio is not None:
        lines.append(f"- atr_ratio_vs_baseline: {tech.raw_atr_ratio:.2f}")
    lines.append(f"- technical_confidence: {tech.confidence_score:.1f}/100")
    return "\n".join(lines)


def _render_fundamentals(inputs: AnalyzerInputs) -> str:
    funds = inputs.fundamentals
    assert funds is not None
    lines: list[str] = ["FUNDAMENTALS:"]
    p = funds.profile
    if p.sector:
        lines.append(f"- sector: {p.sector}")
    if p.market_cap:
        lines.append(f"- market_cap_usd: {p.market_cap:.0f}")
    income = funds.latest_annual_income
    if income is not None and income.revenue is not None:
        lines.append(f"- latest_annual_revenue_usd: {income.revenue:.0f}")
    if income is not None and income.net_income is not None:
        lines.append(f"- latest_annual_net_income_usd: {income.net_income:.0f}")
    r = funds.key_ratios
    if r.pe_ratio is not None:
        lines.append(f"- pe_ratio: {r.pe_ratio:.2f}")
    if r.peg_ratio is not None:
        lines.append(f"- peg_ratio: {r.peg_ratio:.2f}")
    if r.debt_to_equity is not None:
        lines.append(f"- debt_to_equity: {r.debt_to_equity:.2f}")
    if r.gross_margin is not None:
        lines.append(f"- gross_margin: {r.gross_margin:.3f}")
    return "\n".join(lines)


def _render_valuation(inputs: AnalyzerInputs) -> str:
    v = inputs.valuation
    assert v is not None
    lines: list[str] = ["VALUATION:"]
    if v.weighted_ai_fair_value is not None:
        lines.append(f"- weighted_ai_fair_value: ${v.weighted_ai_fair_value:.2f}")
    if v.bear_case is not None:
        lines.append(f"- bear_case: ${v.bear_case:.2f}")
    if v.base_case is not None:
        lines.append(f"- base_case: ${v.base_case:.2f}")
    if v.bull_case is not None:
        lines.append(f"- bull_case: ${v.bull_case:.2f}")
    lines.append(f"- valuation_confidence: {v.confidence_score:.1f}/100")
    if v.estimates:
        lines.append("- method_estimates:")
        for est in v.estimates:
            lines.append(
                f"    * {est.method}: ${est.fair_value:.2f} "
                f"(confidence {est.confidence:.2f}; inputs: {est.inputs_used})"
            )
    return "\n".join(lines)


def _render_news(inputs: AnalyzerInputs) -> str:
    lines: list[str] = ["NEWS:"]
    if inputs.headline_count is not None:
        lines.append(f"- headline_count: {inputs.headline_count}")
    if inputs.top_headlines:
        lines.append("- top_headlines:")
        for h in inputs.top_headlines[:5]:
            # Two layers (B-9):
            # * sanitize_prompt_value strips control chars, caps length,
            #   and XML-escapes — defeats injection via embedded newlines
            #   or markup.
            # * the bracket replacement keeps a headline from
            #   accidentally looking like a citation tag inside the DATA
            #   block (analyzer-specific concern).
            sanitized = sanitize_prompt_value(h)
            safe = sanitized.replace("[", "(").replace("]", ")")
            lines.append(f"    * {safe!r}")
    return "\n".join(lines)


def _render_volume(inputs: AnalyzerInputs) -> str:
    lines: list[str] = ["VOLUME:"]
    if inputs.volume_z is not None:
        lines.append(f"- volume_z_20: {inputs.volume_z:+.2f}")
    if inputs.volume_spike_score is not None:
        lines.append(f"- volume_spike_score: {inputs.volume_spike_score:.1f}/100")
    return "\n".join(lines)


__all__ = [
    "CITATION_TAGS",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "build_user_message",
]
