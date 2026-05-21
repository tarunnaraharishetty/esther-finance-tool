"""Grounded explanation layer.

Produces a structured, citation-tagged narrative describing why the
analyzer landed where it did. The contract is hard:

* Every claim in the output is suffixed with one or more citation
  tags drawn from a fixed vocabulary —
  ``[technicals]``, ``[fundamentals]``, ``[news]``, ``[sentiment]``,
  ``[valuation]``, ``[volume]``.
* A tag may only be cited if the corresponding input stream was
  actually present in :class:`AnalyzerInputs`. A claim citing
  ``[fundamentals]`` when ``fundamentals`` was ``None`` is rejected.
* Output that fails citation validation never reaches the caller.

Two builders share this contract:

* :func:`build_grounded_explanation` — deterministic template. Always
  available, always passes validation because every line is written
  with the citations baked in.
* :class:`~src.intelligence.analyzer.llm_explanation.LLMExplanationGenerator`
  — Anthropic-backed. Its output is run through :func:`validate_citations`
  and the LLM is *required* to emit the same citation format. Falls
  back to the deterministic builder on validation failure.

By design the LLM never sees raw market data — it only sees the
``DATA`` block rendered from :class:`AnalyzerInputs`. So the universe
of facts it can cite is bounded by what we explicitly handed it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from src.intelligence.analyzer.technical import TechnicalScores
from src.intelligence.analyzer.valuation import ValuationEnsemble
from src.intelligence.fundamentals import NormalizedFundamentals


# Citation vocabulary. Frozen — changes here must travel into the LLM
# prompt template and the frontend chip rendering at the same time.
CITATION_TAGS: frozenset[str] = frozenset(
    {"technicals", "fundamentals", "news", "sentiment", "valuation", "volume"}
)


# Tag-finding regex. Tolerates ``[tag]``, ``[tag1, tag2]``, optional
# trailing whitespace inside the brackets. We split on ``,`` after
# extraction so multi-tag chips are easy to assemble.
_CITATION_RE = re.compile(r"\[([a-z, ]+)\]")


@dataclass(frozen=True)
class AnalyzerInputs:
    """Snapshot of every data stream the explanation layer may cite.

    A field set to ``None`` means "this stream was not present this
    call" — citing the corresponding tag is forbidden by the
    validator. Empty containers are treated as present-but-empty,
    which is a fine grounding signal (we can validly say "no recent
    news [news]").

    Numeric summaries live here as well so the prompt template and
    the LLM see the same numbers — no separate rendering paths that
    could drift.
    """

    symbol: str
    last_price: float | None = None
    technicals: TechnicalScores | None = None
    fundamentals: NormalizedFundamentals | None = None
    valuation: ValuationEnsemble | None = None
    sentiment_score: float | None = None
    headline_count: int | None = None
    top_headlines: tuple[str, ...] = ()
    volume_z: float | None = None
    volume_spike_score: float | None = None

    def available_tags(self) -> frozenset[str]:
        """Tags the validator will allow this run to cite.

        ``[volume]`` is granted when *either* of the two volume fields
        is populated — both sit on different parts of the upstream
        pipeline (raw z-score vs. the normalized sub-score). Either
        being present is enough grounding for a "volume" citation.
        """
        tags: set[str] = set()
        if self.technicals is not None:
            tags.add("technicals")
        if self.fundamentals is not None:
            tags.add("fundamentals")
        if self.valuation is not None:
            tags.add("valuation")
        if self.sentiment_score is not None:
            tags.add("sentiment")
        if self.headline_count is not None or self.top_headlines:
            tags.add("news")
        if self.volume_z is not None or self.volume_spike_score is not None:
            tags.add("volume")
        return frozenset(tags)


@dataclass(frozen=True)
class AnalyzerExplanation:
    """Grounded narrative + structured citations.

    Carries the citation map alongside the prose so the frontend can
    render badge chips next to each claim without re-parsing.
    """

    symbol: str
    generated_at: str
    model: str
    summary: str
    claims: tuple[str, ...]
    risk_warnings: tuple[str, ...]
    citations: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CitationValidationError(ValueError):
    """A claim was missing a citation, or cited an unavailable stream."""

    def __init__(self, message: str, claim: str = "") -> None:
        super().__init__(message)
        self.claim = claim


def extract_citations(claim: str) -> tuple[str, ...]:
    """Pull every ``[tag]`` (or ``[tag1, tag2]``) off the end of a claim."""
    out: list[str] = []
    for match in _CITATION_RE.finditer(claim):
        for raw in match.group(1).split(","):
            tag = raw.strip().lower()
            if tag:
                out.append(tag)
    return tuple(out)


def validate_citations(
    claims: tuple[str, ...], inputs: AnalyzerInputs
) -> dict[str, tuple[str, ...]]:
    """Raise :class:`CitationValidationError` for any non-grounded claim.

    Rules:

    * Every claim must carry at least one ``[tag]``. A claim without
      any tag is a hallucination risk by definition (we can't trace
      back what produced it).
    * Every cited tag must be in :data:`CITATION_TAGS` *and* in
      ``inputs.available_tags()``.

    On success, returns a ``{citation_tag: tuple-of-claims}`` mapping
    so the UI can render chips grouped by citation.
    """
    allowed = inputs.available_tags()
    by_tag: dict[str, list[str]] = {}
    for claim in claims:
        cites = extract_citations(claim)
        if not cites:
            raise CitationValidationError(
                f"claim has no citation tag: {claim!r}", claim=claim
            )
        for tag in cites:
            if tag not in CITATION_TAGS:
                raise CitationValidationError(
                    f"claim cites unknown tag [{tag}]: {claim!r}", claim=claim
                )
            if tag not in allowed:
                raise CitationValidationError(
                    f"claim cites [{tag}] but that input was not provided: "
                    f"{claim!r}",
                    claim=claim,
                )
            by_tag.setdefault(tag, []).append(claim)
    return {tag: tuple(claims) for tag, claims in by_tag.items()}


# ---------------------------------------------------------------------------
# Deterministic template builder.
# ---------------------------------------------------------------------------


_TEMPLATE_MODEL_ID = "analyzer-template-v1"


def build_grounded_explanation(inputs: AnalyzerInputs) -> AnalyzerExplanation:
    """Deterministic explanation builder.

    Every claim is written with its citation tag inline, so the
    resulting :class:`AnalyzerExplanation` is guaranteed to pass
    :func:`validate_citations`. Used as:

    * The "no-LLM" fallback path on Anthropic auth/rate-limit errors.
    * The ground-truth template the LLM prompt is benchmarked against
      in tests — if the LLM can't add value over this baseline we
      don't switch the user to the paid path.
    """
    claims: list[str] = []
    risks: list[str] = []

    if inputs.technicals is not None:
        claims.extend(_technical_claims(inputs.technicals))
        risks.extend(_technical_risks(inputs.technicals))

    if inputs.valuation is not None:
        claims.extend(_valuation_claims(inputs.valuation))
        risks.extend(_valuation_risks(inputs.valuation))

    if inputs.fundamentals is not None:
        claims.extend(_fundamentals_claims(inputs.fundamentals))

    if inputs.sentiment_score is not None:
        claims.extend(_sentiment_claims(inputs.sentiment_score))

    if inputs.headline_count is not None or inputs.top_headlines:
        claims.extend(_news_claims(inputs.headline_count, inputs.top_headlines))

    if inputs.volume_z is not None or inputs.volume_spike_score is not None:
        claims.extend(_volume_claims(inputs.volume_z, inputs.volume_spike_score))
        risks.extend(_volume_risks(inputs.volume_z))

    summary = _build_summary(inputs)
    if not claims:
        # Nothing was provided — emit a single grounded "no data" line.
        # We still need to cite a tag, so we cite whatever's available;
        # if literally nothing is available, the explanation is empty.
        claims = ()
    claims_tuple = tuple(claims)
    risks_tuple = tuple(risks)
    citations = validate_citations(claims_tuple, inputs) if claims_tuple else {}

    return AnalyzerExplanation(
        symbol=inputs.symbol,
        generated_at=datetime.now(UTC).isoformat(),
        model=_TEMPLATE_MODEL_ID,
        summary=summary,
        claims=claims_tuple,
        risk_warnings=risks_tuple,
        citations=citations,
    )


# ---- per-stream claim builders ----


def _technical_claims(tech: TechnicalScores) -> list[str]:
    out: list[str] = []
    if tech.raw_rsi is not None:
        out.append(
            f"RSI is at {tech.raw_rsi:.1f} on a 14-bar lookback, anchoring the "
            f"momentum read [technicals]."
        )
    if tech.overbought_score is not None and tech.oversold_score is not None:
        bias = (
            "stretched to the upside"
            if tech.overbought_score > tech.oversold_score + 20
            else "stretched to the downside"
            if tech.oversold_score > tech.overbought_score + 20
            else "balanced"
        )
        out.append(
            f"Overbought composite scores {tech.overbought_score:.0f}/100 vs "
            f"oversold {tech.oversold_score:.0f}/100 — posture is {bias} "
            f"[technicals]."
        )
    if tech.raw_ma_distance_pct is not None:
        direction = "above" if tech.raw_ma_distance_pct > 0 else "below"
        out.append(
            f"Price sits {abs(tech.raw_ma_distance_pct):.1f}% {direction} its "
            f"50-bar SMA [technicals]."
        )
    if tech.raw_atr_ratio is not None and tech.raw_atr_ratio > 1.2:
        out.append(
            f"ATR is running {tech.raw_atr_ratio:.2f}x its 50-bar baseline — "
            f"volatility is elevated [technicals]."
        )
    return out


def _technical_risks(tech: TechnicalScores) -> list[str]:
    out: list[str] = []
    if tech.pullback_risk is not None and tech.pullback_risk >= 60:
        out.append(
            f"Pullback risk reads {tech.pullback_risk:.0f}/100 — overheated "
            f"and showing early signs of mean-reversion pressure [technicals]."
        )
    if tech.rebound_potential is not None and tech.rebound_potential >= 60:
        out.append(
            f"Rebound potential reads {tech.rebound_potential:.0f}/100 — "
            f"oversold to a degree historically associated with mean-reversion "
            f"bounces [technicals]."
        )
    if tech.confidence_score < 40:
        out.append(
            f"Technical confidence is only {tech.confidence_score:.0f}/100 — "
            f"signal coverage is thin or sub-scores disagree [technicals]."
        )
    return out


def _valuation_claims(ensemble: ValuationEnsemble) -> list[str]:
    if not ensemble.estimates:
        return []
    out: list[str] = []
    if ensemble.weighted_ai_fair_value is not None:
        out.append(
            f"Weighted multi-method fair value lands at "
            f"${ensemble.weighted_ai_fair_value:.2f}, "
            f"with a bear / base / bull span of "
            f"${ensemble.bear_case:.2f} / "  # type: ignore[arg-type]
            f"${ensemble.base_case:.2f} / "  # type: ignore[arg-type]
            f"${ensemble.bull_case:.2f} [valuation]."  # type: ignore[arg-type]
        )
    out.append(
        f"{len(ensemble.estimates)} of 7 valuation methods produced an "
        f"estimate this run [valuation]."
    )
    # Surface the top-confidence method as a primary anchor.
    top = max(ensemble.estimates, key=lambda e: e.confidence)
    out.append(
        f"Highest-confidence method is "
        f"{top.method.replace('_', ' ')} at ${top.fair_value:.2f} "
        f"(confidence {top.confidence:.2f}) [valuation]."
    )
    return out


def _valuation_risks(ensemble: ValuationEnsemble) -> list[str]:
    if not ensemble.estimates:
        return []
    if ensemble.confidence_score < 30:
        return [
            f"Valuation confidence is low at "
            f"{ensemble.confidence_score:.0f}/100 — methods disagree or "
            f"few inputs were available [valuation]."
        ]
    return []


def _fundamentals_claims(funds: NormalizedFundamentals) -> list[str]:
    out: list[str] = []
    profile = funds.profile
    if profile.sector:
        out.append(
            f"Company sits in {profile.sector} per the fundamentals "
            f"feed [fundamentals]."
        )
    income = funds.latest_annual_income
    if income is not None and income.revenue is not None:
        out.append(
            f"Most recent annual revenue is ${income.revenue / 1e9:.2f}B "
            f"[fundamentals]."
        )
    ratios = funds.key_ratios
    if ratios.pe_ratio is not None:
        out.append(
            f"Trailing P/E is {ratios.pe_ratio:.1f} [fundamentals]."
        )
    if ratios.debt_to_equity is not None and ratios.debt_to_equity > 1.5:
        out.append(
            f"Debt-to-equity is elevated at {ratios.debt_to_equity:.2f} "
            f"[fundamentals]."
        )
    return out


def _sentiment_claims(score: float) -> list[str]:
    direction = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
    return [
        f"FinBERT sentiment composite is {score:+.2f}, a {direction} read "
        f"[sentiment]."
    ]


def _news_claims(count: int | None, headlines: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    if count is not None:
        out.append(
            f"{count} headlines landed in the active window [news]."
        )
    if headlines:
        first = headlines[0].replace("[", "(").replace("]", ")")
        out.append(f'Top headline: "{first}" [news].')
    return out


def _volume_claims(z: float | None, score: float | None) -> list[str]:
    out: list[str] = []
    if z is not None:
        out.append(
            f"Volume z-score sits at {z:+.2f} vs the 20-bar log-mean "
            f"[volume]."
        )
    if score is not None and score >= 50:
        out.append(
            f"Volume spike sub-score reads {score:.0f}/100 — recent prints "
            f"are well above the local norm [volume]."
        )
    return out


def _volume_risks(z: float | None) -> list[str]:
    if z is None:
        return []
    if z >= 2.5:
        return [
            f"Volume z is {z:+.2f} — anomaly territory; entries on this kind "
            f"of print often get faded [volume]."
        ]
    return []


def _build_summary(inputs: AnalyzerInputs) -> str:
    """One-sentence top-of-card summary, citation-free.

    The summary is allowed to be tag-free because it's framed as
    "the analyzer says..." not "the data shows..." — readers know it's
    the model's read. Each underlying claim that backs it up is in
    ``claims`` and carries its citation tag.
    """
    if inputs.technicals is None and inputs.valuation is None:
        return (
            f"{inputs.symbol}: insufficient grounded inputs for a full read."
        )
    tech_bias = ""
    if inputs.technicals is not None:
        ob = inputs.technicals.overbought_score or 0
        os = inputs.technicals.oversold_score or 0
        if ob > os + 20:
            tech_bias = "technically stretched to the upside"
        elif os > ob + 20:
            tech_bias = "technically stretched to the downside"
        else:
            tech_bias = "technically balanced"
    val_phrase = ""
    if inputs.valuation is not None and inputs.valuation.weighted_ai_fair_value is not None and inputs.last_price:
        diff = inputs.valuation.weighted_ai_fair_value - inputs.last_price
        pct = diff / inputs.last_price * 100
        if pct >= 10:
            val_phrase = "and valuation models point higher"
        elif pct <= -10:
            val_phrase = "and valuation models point lower"
        else:
            val_phrase = "with valuation roughly fair"
    parts = [p for p in (tech_bias, val_phrase) if p]
    if not parts:
        return f"{inputs.symbol}: analyzer read assembled from grounded inputs."
    return f"{inputs.symbol} is " + ", ".join(parts) + "."


__all__ = [
    "AnalyzerExplanation",
    "AnalyzerInputs",
    "CITATION_TAGS",
    "CitationValidationError",
    "build_grounded_explanation",
    "extract_citations",
    "validate_citations",
]
