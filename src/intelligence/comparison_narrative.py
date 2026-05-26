"""Grounded narrative for the two-stock comparison view.

Takes a :class:`~src.intelligence.comparison.ComparisonView` plus both
underlying analyzer reports and produces a sectioned prose narrative
("Which has stronger momentum? Which is more attractively valued?").
Two backends share one protocol:

* :class:`TemplateComparisonNarrator` — deterministic, no LLM, always
  available. Phrases the structured verdict in plain prose. The
  template output is always grounded by construction.
* :class:`LLMComparisonNarrator` — Anthropic-backed. Reads the same
  inputs, paraphrases into more natural prose, subject to the
  validator's drop policy.

The endpoint chooses between them via ``?mode={template,llm,auto}``.

Why sections, not free-form
---------------------------
Six fixed sections (``headline``, ``momentum``, ``valuation``,
``risk``, ``quality``, ``bottom_line``) give the frontend a stable
shape for rendering. A free-form narrative would force the UI to
guess layout per response and would slip the grounding contract —
the LLM's job here is to write *one paragraph plus three bullets per
fixed section*, not to invent organization.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol
from xml.etree import ElementTree as ET

from src.config import Settings, get_settings
from src.intelligence.comparison import ComparisonView, MetricComparison
from src.intelligence.grounding import (
    GROUNDING_RULES_COMPARISON,
    sanitize_symbol,
)
from src.utils.logging import get_logger

if TYPE_CHECKING:
    import anthropic


log = get_logger(__name__)

_PROMPT_VERSION = "compare-narrative-v1"
_MAX_TOKENS = 3000


# Section names that every narrator must populate. Ordered as they
# render in the UI. Adding a new section here requires a UI update
# and a test for both narrators.
SECTION_KEYS: tuple[str, ...] = (
    "headline",
    "momentum",
    "valuation",
    "risk",
    "quality",
    "bottom_line",
)

SECTION_TITLES: dict[str, str] = {
    "headline": "Headline",
    "momentum": "Momentum & Technical",
    "valuation": "Valuation",
    "risk": "Risk",
    "quality": "Data Quality & Trust",
    "bottom_line": "Bottom Line",
}


@dataclass(frozen=True)
class NarrativeSection:
    """One labeled block of prose + bullets.

    ``body`` is a paragraph that paraphrases the structured verdicts.
    ``bullets`` are the supporting numeric reads — typically 2-4
    short data points. After validation either field may be empty
    if every claim got dropped.

    ``provenance`` maps surviving numeric tokens to their source-field
    labels (e.g. ``"left.trust_score.score"`` /
    ``"view.trust.trust_score.right"``). Empty when the validator
    hasn't run or no tokens grounded. The frontend renders provenance
    entries as hover tooltips so any numeric in the narrative is
    traceable to its source side + field with one mouseover.
    """

    title: str
    body: str
    bullets: tuple[str, ...] = ()
    provenance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "bullets": list(self.bullets),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ComparisonNarrative:
    """Full AI-written comparison between two symbols."""

    left_symbol: str
    right_symbol: str
    tagline: str
    headline: NarrativeSection
    momentum: NarrativeSection
    valuation: NarrativeSection
    risk: NarrativeSection
    quality: NarrativeSection
    bottom_line: NarrativeSection
    model: str
    generated_at: str  # ISO-8601 UTC
    # Free-form notes (e.g. "fundamentals missing on right side").
    warnings: tuple[str, ...] = ()

    def sections(self) -> tuple[tuple[str, NarrativeSection], ...]:
        """Iterate over (key, section) pairs in canonical order."""
        return tuple(
            (k, getattr(self, k))
            for k in SECTION_KEYS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_symbol": self.left_symbol,
            "right_symbol": self.right_symbol,
            "tagline": self.tagline,
            "headline": self.headline.to_dict(),
            "momentum": self.momentum.to_dict(),
            "valuation": self.valuation.to_dict(),
            "risk": self.risk.to_dict(),
            "quality": self.quality.to_dict(),
            "bottom_line": self.bottom_line.to_dict(),
            "model": self.model,
            "generated_at": self.generated_at,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class NarrativeInput:
    """Bundle of inputs every narrator consumes.

    ``view`` is the canonical pre-computed verdicts — both narrators
    must align with it (the LLM grounding rules forbid contradiction).
    ``left_report`` / ``right_report`` are the assembled analyzer
    payloads (wire-dict shape) — needed for richer prose like
    "trust grade B vs A" or "freshness aging vs fresh".
    """

    view: ComparisonView
    left_report: dict[str, Any] | None
    right_report: dict[str, Any] | None


class ComparisonNarrator(Protocol):
    """Common interface both backends implement."""

    @property
    def model_id(self) -> str: ...

    def generate(self, payload: NarrativeInput) -> ComparisonNarrative: ...


# ---------------------------------------------------------------------------
# Deterministic template narrator
# ---------------------------------------------------------------------------


class TemplateComparisonNarrator:
    """Deterministic narrator built from the structured view.

    Used as the fallback when the LLM path is disabled or fails, and
    as the always-available baseline. Every sentence references a
    metric from the comparison view, which guarantees grounding by
    construction — the validator's drop count on template output is
    expected to be zero in normal operation.
    """

    model_id: str = f"template@{_PROMPT_VERSION}"

    def generate(self, payload: NarrativeInput) -> ComparisonNarrative:
        view = payload.view
        sections: dict[str, NarrativeSection] = {}
        sections["headline"] = _template_headline(view)
        sections["momentum"] = _template_momentum(view, payload)
        sections["valuation"] = _template_valuation(view, payload)
        sections["risk"] = _template_risk(view, payload)
        sections["quality"] = _template_quality(view, payload)
        sections["bottom_line"] = _template_bottom_line(view)
        tagline = _template_tagline(view)
        return ComparisonNarrative(
            left_symbol=view.left_symbol,
            right_symbol=view.right_symbol,
            tagline=tagline,
            headline=sections["headline"],
            momentum=sections["momentum"],
            valuation=sections["valuation"],
            risk=sections["risk"],
            quality=sections["quality"],
            bottom_line=sections["bottom_line"],
            model=self.model_id,
            generated_at=_now_iso(),
            warnings=view.warnings,
        )


def _template_tagline(view: ComparisonView) -> str:
    if view.overall_winner == "left":
        return (
            f"{view.left_symbol} leads {view.right_symbol} across the "
            "majority of measured metrics."
        )
    if view.overall_winner == "right":
        return (
            f"{view.right_symbol} leads {view.left_symbol} across the "
            "majority of measured metrics."
        )
    if view.overall_winner == "tie":
        return (
            f"{view.left_symbol} and {view.right_symbol} are evenly "
            "matched across the measured metrics."
        )
    return (
        f"Comparison of {view.left_symbol} vs {view.right_symbol} — "
        "metric coverage is too thin to publish a verdict."
    )


def _template_headline(view: ComparisonView) -> NarrativeSection:
    # Headline metrics carry pre-formatted display strings (e.g. "A+",
    # "$182.45"), not raw floats — render them with their own helper.
    bullets = tuple(
        f"{h.label}: {view.left_symbol} {h.left_display} vs "
        f"{view.right_symbol} {h.right_display}"
        + (
            f" — {view.left_symbol} leads."
            if h.winner == "left"
            else f" — {view.right_symbol} leads."
            if h.winner == "right"
            else " — within tie tolerance."
            if h.winner == "tie"
            else "."
        )
        for h in view.headline
    )
    body = _winner_summary(view, prefix="Across the four measured sections, ")
    return NarrativeSection(
        title=SECTION_TITLES["headline"], body=body, bullets=bullets
    )


def _template_momentum(
    view: ComparisonView, payload: NarrativeInput
) -> NarrativeSection:
    metrics = _section_metrics(view, "Technical")
    return _build_section(
        view,
        section_title=SECTION_TITLES["momentum"],
        metrics=metrics,
        intro=(
            "Technical posture compares overbought and oversold composites — "
            "lower overbought scores indicate a cleaner entry; higher oversold "
            "scores indicate more rebound setup."
        ),
    )


def _template_valuation(
    view: ComparisonView, payload: NarrativeInput
) -> NarrativeSection:
    metrics = _section_metrics(view, "Valuation")
    return _build_section(
        view,
        section_title=SECTION_TITLES["valuation"],
        metrics=metrics,
        intro=(
            "Valuation compares the ensemble confidence and the upside implied "
            "by the base-case fair value."
        ),
    )


def _template_risk(
    view: ComparisonView, payload: NarrativeInput
) -> NarrativeSection:
    metrics = _section_metrics(view, "Risk")
    return _build_section(
        view,
        section_title=SECTION_TITLES["risk"],
        metrics=metrics,
        intro=(
            "Risk compares pullback risk against rebound potential — lower "
            "pullback risk is a safer hold, higher rebound potential is a "
            "stronger bounce setup."
        ),
    )


def _template_quality(
    view: ComparisonView, payload: NarrativeInput
) -> NarrativeSection:
    metrics = _section_metrics(view, "Trust")
    return _build_section(
        view,
        section_title=SECTION_TITLES["quality"],
        metrics=metrics,
        intro=(
            "Trust composites the data freshness, provider confidence, and "
            "calibration coverage for each side. Higher is better."
        ),
    )


def _template_bottom_line(view: ComparisonView) -> NarrativeSection:
    left_wins = sum(
        1
        for s in view.sections
        for m in s.metrics
        if m.winner == "left"
    )
    right_wins = sum(
        1
        for s in view.sections
        for m in s.metrics
        if m.winner == "right"
    )
    ties = sum(
        1
        for s in view.sections
        for m in s.metrics
        if m.winner == "tie"
    )
    body = (
        f"{view.left_symbol} wins {left_wins} of the measured metrics; "
        f"{view.right_symbol} wins {right_wins}; {ties} are within tie "
        "tolerance. The trade-off is not a recommendation — read the per-"
        "metric directions to interpret for your own setup."
    )
    return NarrativeSection(
        title=SECTION_TITLES["bottom_line"], body=body, bullets=()
    )


def _section_metrics(view: ComparisonView, title: str) -> tuple[MetricComparison, ...]:
    for s in view.sections:
        if s.title == title:
            return s.metrics
    return ()


def _build_section(
    view: ComparisonView,
    *,
    section_title: str,
    metrics: tuple[MetricComparison, ...],
    intro: str,
) -> NarrativeSection:
    if not metrics:
        return NarrativeSection(
            title=section_title,
            body=f"{intro} No comparable metrics on this section.",
            bullets=(),
        )
    left_wins = sum(1 for m in metrics if m.winner == "left")
    right_wins = sum(1 for m in metrics if m.winner == "right")
    if left_wins > right_wins:
        verdict = f"{view.left_symbol} leads this section ({left_wins}-{right_wins})."
    elif right_wins > left_wins:
        verdict = f"{view.right_symbol} leads this section ({right_wins}-{left_wins})."
    else:
        verdict = f"This section is even between {view.left_symbol} and {view.right_symbol}."
    bullets = tuple(
        _format_metric_line(m, view) for m in metrics if m.winner != "n/a"
    )
    body = f"{intro} {verdict}"
    return NarrativeSection(title=section_title, body=body, bullets=bullets)


def _winner_summary(view: ComparisonView, *, prefix: str) -> str:
    """Render the four section verdicts in one sentence."""
    section_lines = []
    for section in view.sections:
        left_wins = sum(1 for m in section.metrics if m.winner == "left")
        right_wins = sum(1 for m in section.metrics if m.winner == "right")
        if left_wins > right_wins:
            section_lines.append(f"{view.left_symbol} on {section.title.lower()}")
        elif right_wins > left_wins:
            section_lines.append(f"{view.right_symbol} on {section.title.lower()}")
        else:
            section_lines.append(f"{section.title.lower()} is even")
    return prefix + "; ".join(section_lines) + "."


def _format_metric_line(metric: MetricComparison, view: ComparisonView) -> str:
    """Render one metric as a bullet string.

    Example: ``Trust score: AAPL 92 vs MSFT 78 — AAPL leads.``
    """
    left_str = _format_value(metric.left_value, metric.percent)
    right_str = _format_value(metric.right_value, metric.percent)
    base = (
        f"{metric.label}: {view.left_symbol} {left_str} vs "
        f"{view.right_symbol} {right_str}"
    )
    if metric.winner == "left":
        return f"{base} — {view.left_symbol} leads."
    if metric.winner == "right":
        return f"{base} — {view.right_symbol} leads."
    if metric.winner == "tie":
        return f"{base} — within tie tolerance."
    return f"{base} — no comparison available."


def _format_value(value: float | None, percent: bool) -> str:
    if value is None:
        return "—"
    if percent:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.1f}%"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# LLM narrator (Anthropic, XML output)
# ---------------------------------------------------------------------------


_OUTPUT_SCHEMA = """OUTPUT FORMAT — return EXACTLY ONE XML document, no markdown, \
no explanatory prose outside the XML. The root element is <narrative>. Schema:

<narrative>
  <tagline>One-sentence gist. ≤140 chars. Plain prose.</tagline>
  <headline>
    <body>2-3 sentences naming the overall winner and the headline reason.</body>
    <bullets>
      <item>One concrete data point that anchors the verdict</item>
    </bullets>
  </headline>
  <momentum>
    <body>2-3 sentences comparing the two on technical / momentum metrics.</body>
    <bullets>
      <item>One bullet per metric you cite</item>
    </bullets>
  </momentum>
  <valuation>
    <body>2-3 sentences on valuation confidence and base-case upside.</body>
    <bullets>
      <item>One bullet per metric you cite</item>
    </bullets>
  </valuation>
  <risk>
    <body>2-3 sentences on pullback risk + rebound potential.</body>
    <bullets>
      <item>One bullet per metric you cite</item>
    </bullets>
  </risk>
  <quality>
    <body>2-3 sentences on data freshness, provider confidence, trust scores.</body>
    <bullets>
      <item>One bullet per metric you cite</item>
    </bullets>
  </quality>
  <bottom_line>
    <body>2-3 sentences synthesizing the trade-off. Honest about splits.</body>
  </bottom_line>
</narrative>
"""


_SYSTEM_PROMPT = f"""You are an institutional research analyst producing a side-by-side \
comparison of two equities for a discretionary trader. Your job is to paraphrase the \
pre-computed structured verdicts into clear prose. You are NOT generating new claims — \
you are surfacing what the data shows.

{GROUNDING_RULES_COMPARISON}

WRITING RULES:
- Reference both symbols by name. The two symbols are given in the user message.
- Use the exact numbers from the input. Never round in a way that creates a new value \
not present in the input (e.g. don't render 92.0 as 92.4).
- One concrete numeric anchor per sentence in section bodies.
- Bullets are short data reads, not opinions. Each bullet should reference one metric.
- Stay grounded — every claim must trace to a value in the input.
- When metrics split (one side wins momentum, the other wins valuation), say so. Don't \
manufacture a single winner.
- Avoid imperative phrasing ("buy", "sell", "enter here"). Frame as observations.

{_OUTPUT_SCHEMA}
"""


class LLMComparisonNarrator:
    """Anthropic-backed comparison narrator.

    Reads ``NarrativeInput`` (view + both reports), produces the same
    :class:`ComparisonNarrative` dataclass the template narrator emits.
    Output is XML so the parser can't drift — bad sections fall through
    to empty strings, the validator drops anything that didn't ground.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        # Late import: keep anthropic import lazy so tests that never
        # construct this class don't need the SDK on the import path.
        import anthropic as _anthropic

        self.settings = settings or get_settings()
        self.model = model or self.settings.llm_model
        if client is not None:
            self.client = client
        else:
            key = self.settings.anthropic_api_key
            if key is None or not key.get_secret_value().strip():
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not configured. Set it in .env or pass an "
                    "explicit `client` argument to LLMComparisonNarrator."
                )
            self.client = _anthropic.Anthropic(api_key=key.get_secret_value())

    @property
    def model_id(self) -> str:
        return f"{self.model}@{_PROMPT_VERSION}"

    def generate(self, payload: NarrativeInput) -> ComparisonNarrative:
        user_message = _build_user_message(payload)
        log.info(
            "compare_narrative.llm.start",
            left=payload.view.left_symbol,
            right=payload.view.right_symbol,
            model=self.model,
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            temperature=self.settings.llm_temperature,
            timeout=self.settings.llm_timeout_seconds,
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        log.info(
            "compare_narrative.llm.done",
            left=payload.view.left_symbol,
            right=payload.view.right_symbol,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return _parse_narrative_xml(
            text,
            view=payload.view,
            model_id=self.model_id,
        )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _build_user_message(payload: NarrativeInput) -> str:
    view = payload.view
    safe_left = sanitize_symbol(view.left_symbol)
    safe_right = sanitize_symbol(view.right_symbol)
    lines: list[str] = [
        f"Comparing: LEFT={safe_left} vs RIGHT={safe_right}",
        f"Overall winner: {view.overall_winner}",
        "",
        "Headline metrics:",
    ]
    for h in view.headline:
        lines.append(f"- {h.label}: {h.left_display} vs {h.right_display} (winner: {h.winner})")
    lines.append("")
    lines.append("Per-section verdicts:")
    for section in view.sections:
        lines.append(f"  [{section.title}]")
        for m in section.metrics:
            lines.append(
                f"  - {m.label} ({m.direction}): "
                f"left={_format_value(m.left_value, m.percent)} "
                f"right={_format_value(m.right_value, m.percent)} "
                f"→ winner: {m.winner}"
            )
    if view.warnings:
        lines.append("")
        lines.append("Warnings from data assembly:")
        for w in view.warnings:
            lines.append(f"- {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def _parse_narrative_xml(
    text: str, *, view: ComparisonView, model_id: str
) -> ComparisonNarrative:
    """Parse the model's XML response into a ComparisonNarrative.

    Defensive: extract whatever sections parsed cleanly; missing
    sections become empty :class:`NarrativeSection`\\ s rather than
    raising. The validator will drop anything unsupported downstream.
    """
    text = text.strip()
    # Some models prefix XML with a markdown fence; strip it defensively.
    # Handle both ```xml...``` and bare ```...``` shapes; preserve the
    # XML payload that sits between the opening + closing fences.
    if text.startswith("```"):
        text = text[3:].lstrip()
        if text.lower().startswith("xml"):
            text = text[3:].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"LLM returned malformed XML: {exc}") from exc

    tagline = _xml_text(root, "tagline")
    return ComparisonNarrative(
        left_symbol=view.left_symbol,
        right_symbol=view.right_symbol,
        tagline=tagline,
        headline=_xml_section(root, "headline", SECTION_TITLES["headline"]),
        momentum=_xml_section(root, "momentum", SECTION_TITLES["momentum"]),
        valuation=_xml_section(root, "valuation", SECTION_TITLES["valuation"]),
        risk=_xml_section(root, "risk", SECTION_TITLES["risk"]),
        quality=_xml_section(root, "quality", SECTION_TITLES["quality"]),
        bottom_line=_xml_section(
            root, "bottom_line", SECTION_TITLES["bottom_line"]
        ),
        model=model_id,
        generated_at=_now_iso(),
        warnings=view.warnings,
    )


def _xml_text(parent: ET.Element, tag: str) -> str:
    el = parent.find(tag)
    if el is None or el.text is None:
        return ""
    return el.text.strip()


def _xml_section(
    parent: ET.Element, tag: str, title: str
) -> NarrativeSection:
    section_el = parent.find(tag)
    if section_el is None:
        return NarrativeSection(title=title, body="", bullets=())
    body = _xml_text(section_el, "body")
    bullets_el = section_el.find("bullets")
    bullets: list[str] = []
    if bullets_el is not None:
        for item in bullets_el.findall("item"):
            if item.text and item.text.strip():
                bullets.append(item.text.strip())
    return NarrativeSection(title=title, body=body, bullets=tuple(bullets))


# ---------------------------------------------------------------------------
# Convenience constructor + reducer
# ---------------------------------------------------------------------------


def replace_section(
    narrative: ComparisonNarrative,
    key: str,
    *,
    body: str | None = None,
    bullets: tuple[str, ...] | None = None,
    provenance: dict[str, str] | None = None,
) -> ComparisonNarrative:
    """Return a copy of ``narrative`` with one section replaced.

    Used by the validator after scrubbing — sections come in as
    frozen dataclasses so the validator needs a structural way to
    rebuild the narrative without touching internal fields.
    """
    if key not in SECTION_KEYS:
        raise KeyError(f"unknown section key: {key!r}")
    section: NarrativeSection = getattr(narrative, key)
    updates: dict[str, Any] = {}
    if body is not None:
        updates["body"] = body
    if bullets is not None:
        updates["bullets"] = bullets
    if provenance is not None:
        updates["provenance"] = provenance
    new_section = replace(section, **updates)
    # mypy can't statically prove that ``key`` is one of the dataclass
    # field names — assert membership via SECTION_KEYS (checked above)
    # and cast the kwargs to ``Any`` for the dynamic replace call.
    return replace(narrative, **{key: new_section})  # type: ignore[arg-type]


__all__ = [
    "SECTION_KEYS",
    "SECTION_TITLES",
    "ComparisonNarrative",
    "ComparisonNarrator",
    "LLMComparisonNarrator",
    "NarrativeInput",
    "NarrativeSection",
    "TemplateComparisonNarrator",
    "replace_section",
]
