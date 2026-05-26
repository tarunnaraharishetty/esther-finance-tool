"""Anthropic-backed research thesis generator.

Produces the same :class:`~src.intelligence.research_thesis.ResearchThesis`
dataclass as the template generator, but with Claude writing the prose
instead of a deterministic template. Output schema is XML (one tag per
section, attributes for typed values) so the parser can't drift —
Claude either fills the slot or omits it; nothing leaks into a wrong
field.

This module is *the* place we talk to Anthropic for the research
surface. The endpoint layer (``src/api/research.py``) chooses between
this generator and the template fallback based on whether
``ANTHROPIC_API_KEY`` is set, and traps SDK exceptions so a transient
auth/rate-limit failure produces a degraded-but-rendering report
instead of a 500.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import anthropic

from src.config import Settings, get_settings
from src.intelligence.grounding import (
    GROUNDING_RULES_PER_SYMBOL,
    sanitize_prompt_value,
    sanitize_symbol,
)
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


# Prompt versioning — included in the cache key by the endpoint so a
# prompt revision invalidates old cached reports without touching the
# model id.
_PROMPT_VERSION = "research-v1"

# Cap output at 6000 tokens — enough for a full 10-section institutional
# report with prose + bullets, well below the 200K-token model ceiling.
# A typical fresh report comes in around 2-4k output tokens.
_MAX_TOKENS = 6000


# XML schema description embedded into the system prompt. Keeping the
# spec next to the prompt body means changing one side flags the other
# at code review.
_OUTPUT_SCHEMA = """OUTPUT FORMAT — return EXACTLY ONE XML document, no markdown, no \
explanatory prose outside the XML. The root element is <thesis>. \
Schema:

<thesis>
  <rating>strong_buy | buy | hold | sell | strong_sell</rating>
  <confidence>0.0–1.0</confidence>
  <tagline>One-sentence gist. ≤140 chars. Plain prose.</tagline>

  <company_overview>
    <body>2–4 sentences on what the company does, revenue drivers, competitive position. \
If you do not have the data, say "Data unavailable — fundamentals provider not connected." \
and stop the body there.</body>
    <bullets>
      <item>optional short bullet</item>
    </bullets>
  </company_overview>

  <bull_thesis>
    <arg weight="0.0–1.0">
      <label>Short label (≤6 words)</label>
      <detail>1–2 sentence rationale grounded in the input data.</detail>
    </arg>
    <!-- 1 to 5 args; weights sum to ~1.0 across the bull side -->
  </bull_thesis>

  <bear_thesis>
    <!-- same shape as bull_thesis; 1 to 5 args; weights sum to ~1.0 -->
  </bear_thesis>

  <technical_analysis>
    <body>Prose using the actual RSI / MACD / Bollinger values from input.</body>
    <bullets>
      <item>≤6 bullets, each a single technical observation</item>
    </bullets>
  </technical_analysis>

  <fundamental_analysis>
    <body>Prose. If no fundamentals were provided, say "Data unavailable — fundamentals \
provider not connected." and stop the body there.</body>
    <bullets>
      <item>optional fundamentals bullets when data is present</item>
    </bullets>
  </fundamental_analysis>

  <metrics>
    <metric label="Last Price" value="$XYZ" tone="" />
    <metric label="RSI" value="62" tone="bull|bear|warn|" delta="" />
    <!-- 4 to 8 metric cards. tone is one of bull/bear/warn or empty. -->
  </metrics>

  <sentiment_news>
    <body>What the news flow + FinBERT score is saying.</body>
    <bullets>
      <item>Direct verbatim quotes of the most material headlines (1-5 items)</item>
    </bullets>
  </sentiment_news>

  <catalysts>
    <catalyst label="Earnings" when="Q1 2026" impact="bullish|bearish|uncertain">
      <detail>1 sentence on what to watch.</detail>
    </catalyst>
    <!-- 0 to 6 catalysts. If you have no catalyst data, emit zero <catalyst> children. -->
  </catalysts>

  <risk_assessment>
    <body>Plain-prose risk readout.</body>
    <bullets>
      <item>1 to 6 specific risk flags</item>
    </bullets>
  </risk_assessment>

  <outlook>
    <entry horizon="short" bias="bullish|bearish|neutral" confidence="0.0–1.0">
      <detail>1–2 sentences on the 1–5 session view.</detail>
    </entry>
    <entry horizon="medium" bias="..." confidence="0.0–1.0">
      <detail>1–2 sentences on the 1–4 week view.</detail>
    </entry>
    <entry horizon="long" bias="..." confidence="0.0–1.0">
      <detail>1–2 sentences on the 3–12 month view.</detail>
    </entry>
  </outlook>

  <explainability>
    <body>2–3 sentences explaining WHY the rating + confidence landed where it did, \
referencing the inputs that drove the read. No new facts.</body>
  </explainability>
</thesis>"""


_SYSTEM_PROMPT = f"""You are Esther, an institutional-quality equity research analyst \
embedded in a decision-support trading workstation. Your role is to write one rigorous, \
structured research thesis per request — the kind of artifact a sell-side analyst would \
draft for a portfolio manager.

You are NOT a trade-execution system. You write reports; the human reading them decides \
whether and how to act.

{GROUNDING_RULES_PER_SYMBOL}

Additional rules specific to research output:
- If a section needs data you weren't given (fundamentals, analyst targets, insider \
activity, social sentiment, earnings calendar), write "Data unavailable — [what would \
need to land]" in that section's body and emit zero bullets / zero catalysts there. Do \
NOT invent numbers or events to fill the slot.
- Technical and sentiment sections MUST use the actual numbers from input (RSI value, \
MACD reading, Bollinger posture, sentiment score, headline count). These are always \
present.
- The bull and bear thesis sides must each have at least one argument when the signal \
stack admits one; emit zero <arg> children on a side only when the data truly gives no \
support for that direction.
- Weights inside one side should sum to roughly 1.0 — they represent the share of that \
side's case carried by each argument.
- Rating must be consistent with the AI engine's directional read in the input. If the \
engine says BUY at high confidence, you should not output STRONG_SELL.
- Never quote a price target or a stop-loss level. Never use imperative phrasing \
("buy here", "sell into strength"). Frame outlooks as observations of the current \
posture, not predictions.

{_OUTPUT_SCHEMA}

ONE EXAMPLE — illustrates the structure, brevity, and tone. Match this format exactly.

INPUT:
Symbol: NVDA
AI engine read: BUY (confidence 0.74, composite +0.58)
Tier: STRONG_BUY
Last price: $520.45
RSI: 64.2
MACD: +0.46
Bollinger: +0.32
Sentiment score: +0.51 across 9 articles
Signal quality: high
Stability: stable

Recent headlines:
1. "NVIDIA beats Q3 estimates as data-center revenue accelerates"
2. "Analysts upgrade NVIDIA citing supply-chain visibility into 2026"
3. "Hyperscaler AI capex guidance reinforced ahead of CES"

Fundamentals: not available
Analyst ratings: not available
Insider activity: not available

OUTPUT:
<thesis>
  <rating>strong_buy</rating>
  <confidence>0.74</confidence>
  <tagline>NVDA reads constructively — technical momentum, supportive sentiment, and \
data-center catalysts align at high signal quality.</tagline>

  <company_overview>
    <body>Data unavailable — fundamentals provider not connected. The pure signal-engine \
view based on price action and news flow is captured in the sections below.</body>
    <bullets></bullets>
  </company_overview>

  <bull_thesis>
    <arg weight="0.40">
      <label>Technical alignment</label>
      <detail>RSI at 64 with MACD at +0.46 and Bollinger at +0.32 form a coherent \
trend-continuation setup — momentum is engaged without being stretched into overbought.</detail>
    </arg>
    <arg weight="0.35">
      <label>News sentiment confirmation</label>
      <detail>FinBERT scores news flow +0.51 across 9 articles, anchored by "NVIDIA \
beats Q3 estimates as data-center revenue accelerates" — the narrative tape is moving \
with the chart.</detail>
    </arg>
    <arg weight="0.25">
      <label>Signal quality</label>
      <detail>Engine reports a high-quality, stable signal — the read is not coming from \
a noisy or flipping series.</detail>
    </arg>
  </bull_thesis>

  <bear_thesis>
    <arg weight="1.00">
      <label>Momentum stretch risk</label>
      <detail>RSI at 64 is constructive but the upper Bollinger lean (+0.32) means a \
mean-reversion pullback would not break the trend — entries here carry near-term \
giveback risk.</detail>
    </arg>
  </bear_thesis>

  <technical_analysis>
    <body>Composite technical score sits at +0.58 on a [-1, +1] scale with a stable, \
high-quality signal profile. Each individual contributor leans in the same direction \
without any one of them carrying the load.</body>
    <bullets>
      <item>RSI 64.2 — momentum bullish, room before overbought (70).</item>
      <item>MACD +0.46 — above signal line, histogram supportive of continuation.</item>
      <item>Bollinger +0.32 — price riding into the upper band, trend-continuation regime.</item>
    </bullets>
  </technical_analysis>

  <fundamental_analysis>
    <body>Data unavailable — fundamentals provider not connected. Revenue growth, \
margin trajectory, and valuation multiples will populate here when a fundamentals \
adapter (Polygon, FMP, or AlphaVantage) is wired in.</body>
    <bullets></bullets>
  </fundamental_analysis>

  <metrics>
    <metric label="Last Price" value="$520.45" tone="" />
    <metric label="Confidence" value="74%" tone="bull" />
    <metric label="Composite" value="+0.58" tone="bull" />
    <metric label="RSI" value="64.2" tone="" />
    <metric label="MACD" value="+0.46" tone="bull" />
    <metric label="Sentiment" value="+0.51" delta="9 headlines" tone="bull" />
  </metrics>

  <sentiment_news>
    <body>News flow scores +0.51 across 9 articles, a coherent constructive signal that \
moves with the technical posture rather than against it.</body>
    <bullets>
      <item>"NVIDIA beats Q3 estimates as data-center revenue accelerates"</item>
      <item>"Analysts upgrade NVIDIA citing supply-chain visibility into 2026"</item>
      <item>"Hyperscaler AI capex guidance reinforced ahead of CES"</item>
    </bullets>
  </sentiment_news>

  <catalysts></catalysts>

  <risk_assessment>
    <body>Risk readout combines technical posture and signal quality. The current setup \
is constructive but not without near-term mean-reversion risk.</body>
    <bullets>
      <item>Upper-band ride can resolve into a sharp pullback even within an intact uptrend.</item>
      <item>Concentration in data-center narrative — sector rotation away from AI capex \
would weigh quickly.</item>
    </bullets>
  </risk_assessment>

  <outlook>
    <entry horizon="short" bias="bullish" confidence="0.70">
      <detail>Next 1–5 sessions, the engine and tape both lean constructive. Watch \
whether the upper-band lean holds or reverts toward the middle band.</detail>
    </entry>
    <entry horizon="medium" bias="bullish" confidence="0.55">
      <detail>Multi-week view depends on whether the news flow keeps confirming the \
chart. The setup remains constructive as long as the data-center narrative holds.</detail>
    </entry>
    <entry horizon="long" bias="neutral" confidence="0.30">
      <detail>Long-term read requires fundamental data (revenue and margin trajectory) \
which isn't wired in. Treat the long-term column as informational only.</detail>
    </entry>
  </outlook>

  <explainability>
    <body>The STRONG_BUY rating at 74% confidence comes directly from the engine: \
composite score +0.58 with all three technical contributors leaning bullish (RSI 64, \
MACD +0.46, Bollinger +0.32) and FinBERT scoring news +0.51 across 9 articles in the \
same direction. Stable, high-quality signal profile means none of those inputs is \
coming from a noisy series.</body>
  </explainability>
</thesis>"""


class LLMThesisGenerator:
    """Anthropic-backed institutional research thesis generator.

    Construct once at startup and reuse — the Anthropic client is
    cheap to keep around. ``generate`` is synchronous; the endpoint
    runs it inside FastAPI's threadpool by default (FastAPI handles
    sync route handlers transparently).
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Allow overriding the model per generator instance so research
        # can default to a cheaper model than the dashboard summarizer
        # uses. Falls back to the global llm_model setting.
        self.model = model or self.settings.llm_model
        if client is not None:
            self.client = client
        else:
            key = self.settings.anthropic_api_key
            if key is None or not key.get_secret_value().strip():
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not configured. Set it in .env or pass an "
                    "explicit `client` argument to LLMThesisGenerator."
                )
            self.client = anthropic.Anthropic(api_key=key.get_secret_value())

    @property
    def model_id(self) -> str:
        """Stable id used by the endpoint for cache-key partitioning."""
        return f"{self.model}@{_PROMPT_VERSION}"

    def generate(self, payload: ResearchInput) -> ResearchThesis:
        """Run one Claude call and parse the response into a thesis."""
        user_message = _build_user_message(payload)
        log.info(
            "research.llm.start",
            symbol=payload.row.symbol,
            model=self.model,
            headlines=len(payload.headlines),
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
            "research.llm.done",
            symbol=payload.row.symbol,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read=getattr(response.usage, "cache_read_input_tokens", 0),
            cache_write=getattr(response.usage, "cache_creation_input_tokens", 0),
        )
        return _parse_thesis_xml(text, symbol=payload.row.symbol, model_id=self.model_id)


# ---------------------------------------------------------------------------
# Prompt assembly.
# ---------------------------------------------------------------------------


def _build_user_message(payload: ResearchInput) -> str:
    """Render the structured input as plain text for the user turn.

    Kept stable so future-us can A/B alternative wordings without
    surprising the cache key downstream.
    """
    row = payload.row
    safe_symbol = sanitize_symbol(row.symbol)
    lines: list[str] = [
        f"Symbol: {safe_symbol}",
        (
            f"AI engine read: {row.action.value.upper()} "
            f"(confidence {row.confidence:.2f}, composite {row.combined_score:+.2f})"
        ),
        f"Tier: {row.tier.value.upper()}",
        f"Last price: ${row.last_price:,.2f}",
        f"RSI: {row.rsi:.1f}",
        f"MACD: {row.macd:+.2f}",
        f"Bollinger: {row.bollinger:+.2f}",
        f"Sentiment score: {row.sentiment_score:+.2f} across {row.num_news_articles} articles",
        f"Signal quality: {row.signal_quality}",
        f"Stability: {row.stability}",
    ]
    if row.quality_reasons:
        lines.append(f"Quality flags: {', '.join(row.quality_reasons)}")

    if payload.headlines:
        lines.append("")
        lines.append("Recent headlines:")
        for i, h in enumerate(payload.headlines[:8], start=1):
            lines.append(f'{i}. "{sanitize_prompt_value(h)}"')
    else:
        lines.append("")
        lines.append("Recent headlines: none in the current window.")

    # Explicit "unavailable" markers for the adapter fields. The
    # prompt instructs the model to write "Data unavailable" in any
    # section that depends on missing data — giving it the absence
    # signal here makes that contract unambiguous.
    lines.append("")
    lines.append(_adapter_status_block(payload))
    return "\n".join(lines)


def _adapter_status_block(payload: ResearchInput) -> str:
    rows: list[str] = []
    rows.append(
        f"Company profile: {_provided_or_none(payload.company_profile)}"
    )
    rows.append(f"Fundamentals: {_provided_or_none(payload.fundamentals)}")
    rows.append(f"Earnings: {_provided_or_none(payload.earnings)}")
    rows.append(f"Analyst ratings: {_provided_or_none(payload.analyst_ratings)}")
    rows.append(f"Insider activity: {_provided_or_none(payload.insider_activity)}")
    rows.append(f"Macro context: {_provided_or_none(payload.macro_context)}")
    rows.append(f"Social sentiment: {_provided_or_none(payload.social_sentiment)}")
    return "Adapter data:\n" + "\n".join(f"- {r}" for r in rows)


def _provided_or_none(field: object | None) -> str:
    return "provided" if field is not None else "not available"


# ---------------------------------------------------------------------------
# XML parser — turns Claude's <thesis>...</thesis> into the dataclass.
# ---------------------------------------------------------------------------


_XML_BLOCK_RE = re.compile(r"<thesis>.*?</thesis>", re.DOTALL)


def _parse_thesis_xml(
    text: str, *, symbol: str, model_id: str
) -> ResearchThesis:
    """Parse Claude's XML output into a :class:`ResearchThesis`.

    Defensive against:
    * surrounding markdown fences / prose (we extract the first
      ``<thesis>...</thesis>`` block).
    * missing optional children — every getter degrades to a sensible
      default rather than raising.
    * malformed numeric attributes — coerced via ``_safe_float``.
    """
    from datetime import UTC, datetime

    block_match = _XML_BLOCK_RE.search(text)
    if block_match is None:
        raise ValueError(
            f"LLM response did not contain a <thesis> block for {symbol}. "
            f"Raw response head: {text[:200]!r}"
        )
    try:
        root = ET.fromstring(block_match.group(0))
    except ET.ParseError as e:
        raise ValueError(
            f"LLM response was not parseable XML for {symbol}: {e}. "
            f"Raw response head: {text[:200]!r}"
        ) from e

    rating = _text(root, "rating", default="hold").strip().lower()
    if rating not in {"strong_buy", "buy", "hold", "sell", "strong_sell"}:
        rating = "hold"
    confidence = _safe_float(_text(root, "confidence", default="0.5"), 0.5)
    tagline = _text(root, "tagline", default=f"{symbol} thesis — automated read.")

    return ResearchThesis(
        symbol=symbol,
        generated_at=datetime.now(UTC).isoformat(),
        model=model_id,
        rating=rating,
        confidence=max(0.0, min(1.0, confidence)),
        tagline=tagline.strip(),
        company_overview=_section(root, "company_overview", title="Company Overview"),
        bull_thesis=_arguments(root, "bull_thesis"),
        bear_thesis=_arguments(root, "bear_thesis"),
        technical_analysis=_section(root, "technical_analysis", title="Technical Analysis"),
        fundamental_analysis=_section(
            root, "fundamental_analysis", title="Fundamental Analysis"
        ),
        metrics=_metrics(root),
        sentiment_news=_section(root, "sentiment_news", title="Sentiment & News"),
        catalysts=_catalysts(root),
        risk_assessment=_section(root, "risk_assessment", title="Risk Assessment"),
        outlook=_outlook(root),
        explainability=_section(root, "explainability", title="Why this read"),
        data_sources=(
            "Alpaca paper-feed OHLCV",
            "Esther signal engine (RSI/MACD/Bollinger)",
            "Alpaca news API + FinBERT sentiment",
            f"Anthropic {model_id}",
        ),
    )


def _text(parent: ET.Element, tag: str, default: str = "") -> str:
    node = parent.find(tag)
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _section(root: ET.Element, tag: str, *, title: str) -> ThesisSection:
    node = root.find(tag)
    if node is None:
        return ThesisSection(title=title)
    body_node = node.find("body")
    body = (body_node.text or "").strip() if body_node is not None else ""
    bullets_node = node.find("bullets")
    bullets: tuple[str, ...] = ()
    if bullets_node is not None:
        bullets = tuple(
            (item.text or "").strip()
            for item in bullets_node.findall("item")
            if item.text and item.text.strip()
        )
    return ThesisSection(title=title, body=body, bullets=bullets)


def _arguments(root: ET.Element, tag: str) -> tuple[BullBearArgument, ...]:
    node = root.find(tag)
    if node is None:
        return ()
    out: list[BullBearArgument] = []
    for arg in node.findall("arg"):
        weight = _safe_float(arg.attrib.get("weight", "0"), 0.0)
        label = _text(arg, "label", default="(unlabeled)")
        detail = _text(arg, "detail", default="")
        out.append(
            BullBearArgument(
                label=label, weight=max(0.0, min(1.0, weight)), detail=detail
            )
        )
    return tuple(out)


def _metrics(root: ET.Element) -> tuple[MetricEntry, ...]:
    node = root.find("metrics")
    if node is None:
        return ()
    out: list[MetricEntry] = []
    for m in node.findall("metric"):
        tone_raw = (m.attrib.get("tone") or "").strip().lower()
        tone: str | None = tone_raw if tone_raw in {"bull", "bear", "warn"} else None
        delta = m.attrib.get("delta") or None
        if delta is not None and not delta.strip():
            delta = None
        out.append(
            MetricEntry(
                label=m.attrib.get("label", "—"),
                value=m.attrib.get("value", "—"),
                delta=delta,
                tone=tone,
            )
        )
    return tuple(out)


def _catalysts(root: ET.Element) -> tuple[Catalyst, ...]:
    node = root.find("catalysts")
    if node is None:
        return ()
    out: list[Catalyst] = []
    for c in node.findall("catalyst"):
        impact = (c.attrib.get("impact") or "uncertain").strip().lower()
        if impact not in {"bullish", "bearish", "uncertain"}:
            impact = "uncertain"
        out.append(
            Catalyst(
                label=c.attrib.get("label", "Catalyst"),
                when=c.attrib.get("when", ""),
                impact=impact,
                detail=_text(c, "detail", default=""),
            )
        )
    return tuple(out)


def _outlook(root: ET.Element) -> tuple[OutlookEntry, ...]:
    node = root.find("outlook")
    if node is None:
        return ()
    out: list[OutlookEntry] = []
    for entry in node.findall("entry"):
        horizon = (entry.attrib.get("horizon") or "short").strip().lower()
        if horizon not in {"short", "medium", "long"}:
            horizon = "short"
        bias = (entry.attrib.get("bias") or "neutral").strip().lower()
        if bias not in {"bullish", "bearish", "neutral"}:
            bias = "neutral"
        confidence = _safe_float(entry.attrib.get("confidence", "0.5"), 0.5)
        out.append(
            OutlookEntry(
                horizon=horizon,
                bias=bias,
                confidence=max(0.0, min(1.0, confidence)),
                detail=_text(entry, "detail", default=""),
            )
        )
    return tuple(out)


def _safe_float(raw: str, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


__all__ = ["LLMThesisGenerator"]
