"""Anthropic-backed grounded-explanation generator.

Parallels the existing :class:`~src.intelligence.llm_research.LLMThesisGenerator`
but produces a much smaller artifact: the Financial Analyzer's
explanation card. The contract is the same — XML output, deterministic
parser, citation validator — but the prompt is strict about grounding
and the validator hard-rejects ungrounded claims.

Failure modes
-------------
* No API key configured → :class:`RuntimeError` at construction. The
  caller is expected to detect this and fall back to
  :func:`~src.intelligence.analyzer.explanation.build_grounded_explanation`.
* Anthropic transient errors → propagated unchanged so the route
  layer can decide whether to retry or fall back.
* Malformed XML → :class:`ValueError` with the prefix of the raw
  response (so logs are debuggable without leaking the full prompt).
* Validation failure (missing citation, citing an unavailable stream)
  → :class:`~src.intelligence.analyzer.explanation.CitationValidationError`.
  The caller may catch and fall back; not handling it is also fine —
  it surfaces an obvious bug rather than letting a hallucinated
  explanation reach the user.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from xml.etree import ElementTree as ET

import anthropic

from src.config import Settings, get_settings
from src.intelligence.analyzer.explanation import (
    AnalyzerExplanation,
    AnalyzerInputs,
    validate_citations,
)
from src.intelligence.analyzer.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_message,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


# Output cap: the analyzer explanation is short (1 summary + ≤7 claims
# + ≤4 risks). 1500 tokens leaves comfortable headroom for verbose
# bullet phrasing without paying for runaway prose.
_MAX_TOKENS = 1500

_XML_BLOCK_RE = re.compile(r"<explanation>.*?</explanation>", re.DOTALL)


class LLMExplanationGenerator:
    """Anthropic-backed grounded explanation builder.

    Construct once at startup; ``generate`` is synchronous. FastAPI
    handles the threadpool dispatch transparently for sync route
    handlers, mirroring the research-thesis generator.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.llm_model
        if client is not None:
            self.client = client
        else:
            key = self.settings.anthropic_api_key
            if key is None or not key.get_secret_value().strip():
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not configured. Set it in .env or "
                    "pass an explicit `client` argument."
                )
            self.client = anthropic.Anthropic(api_key=key.get_secret_value())

    @property
    def model_id(self) -> str:
        return f"{self.model}@{PROMPT_VERSION}"

    def generate(self, inputs: AnalyzerInputs) -> AnalyzerExplanation:
        """Run one Claude call and parse + validate the response."""
        user_message = build_user_message(inputs)
        log.info(
            "analyzer.llm.start",
            symbol=inputs.symbol,
            model=self.model,
            available_tags=sorted(inputs.available_tags()),
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        log.info(
            "analyzer.llm.done",
            symbol=inputs.symbol,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return parse_explanation_xml(
            text, inputs=inputs, model_id=self.model_id
        )


def parse_explanation_xml(
    text: str, *, inputs: AnalyzerInputs, model_id: str
) -> AnalyzerExplanation:
    """Parse the LLM XML output and run the citation validator.

    Public so the test suite can exercise the parser + validator
    without needing a live Anthropic client.
    """
    match = _XML_BLOCK_RE.search(text)
    if match is None:
        raise ValueError(
            f"LLM response did not contain an <explanation> block for "
            f"{inputs.symbol}. Raw head: {text[:200]!r}"
        )
    try:
        root = ET.fromstring(match.group(0))
    except ET.ParseError as exc:
        raise ValueError(
            f"LLM response was not parseable XML for {inputs.symbol}: {exc}. "
            f"Raw head: {text[:200]!r}"
        ) from exc

    summary_node = root.find("summary")
    summary = (summary_node.text or "").strip() if summary_node is not None else ""

    claim_nodes = root.findall("claim")
    claims = tuple(
        (node.text or "").strip()
        for node in claim_nodes
        if node.text and node.text.strip()
    )
    risk_nodes = root.findall("risk")
    risks = tuple(
        (node.text or "").strip()
        for node in risk_nodes
        if node.text and node.text.strip()
    )

    # Hard guard: every claim must carry a valid citation tag. The
    # validator raises CitationValidationError on the first offender.
    citations = validate_citations(claims, inputs) if claims else {}
    # Risks are not required to carry citations by spec (they're the
    # analyzer's warnings, not raw data points). When they DO carry
    # tags we still validate them; an offending risk falls back the
    # same way an offending claim would.
    if risks:
        validate_citations(risks, inputs)

    return AnalyzerExplanation(
        symbol=inputs.symbol,
        generated_at=datetime.now(UTC).isoformat(),
        model=model_id,
        summary=summary,
        claims=claims,
        risk_warnings=risks,
        citations=citations,
    )


__all__ = ["LLMExplanationGenerator", "parse_explanation_xml"]
