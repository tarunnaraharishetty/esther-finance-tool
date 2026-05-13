"""Expanded opportunity drilldown — DetailPanel-ready breakdown.

Builds a structured per-driver view from a :class:`RankedOpportunity`
so the DetailPanel can render the seven drivers, quality labels, and
rationale phrases without re-deriving any of them in the render layer.

Pure data. Everything in the drilldown comes from fields already on
:class:`~src.intelligence.opportunities.RankedOpportunity` (the seven
driver scores, the :class:`~src.intelligence.signal_profile.SignalProfile`
classification, and the deterministic rationale strings) plus an
optional :class:`~src.intelligence.opportunity_history.OpportunityHistory`
for the NEW / Nx badge.

Quality labels are strict gates over those numerics — no forecasts,
no invented catalysts. The same anti-hallucination contract the rest
of ``intelligence/`` lives under.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.strategy.base import SignalAction

if TYPE_CHECKING:
    from src.dashboard.state import RecommendationRow
    from src.intelligence.opportunities import RankedOpportunity
    from src.intelligence.opportunity_history import OpportunityHistory


_DESCRIPTOR_FLOOR = 0.60
"""Per-driver descriptor only fires when the driver score is at least
this. Below the floor the bar alone communicates magnitude — adding a
descriptor for a quiet driver risks overstating its contribution."""

_HIGH_CONVICTION_QUALITY_FLOOR = 1.0
_HIGH_CONVICTION_COMPOSITE_FLOOR = 0.65
_BUILDING_MOMENTUM_ACCEL_FLOOR = 0.65
_BUILDING_MOMENTUM_PERSISTENCE_FLOOR = 0.50
_REVERSAL_CANDIDATE_FLOOR = 0.60
_SENTIMENT_DRIVEN_SENT_FLOOR = 0.60
_SENTIMENT_DRIVEN_TECH_CEIL = 0.40

# State-phrase gates — combinations of driver/profile/row state that
# earn a single trader-readable bullet. Each phrase describes current
# measurable state; none imply future direction.
_MOMENTUM_STRENGTHENING_ACCEL_FLOOR = 0.60
_MOMENTUM_STRENGTHENING_PERSISTENCE_FLOOR = 0.50
_SENTIMENT_BREADTH_ALIGNMENT_FLOOR = 0.60
_SENTIMENT_BREADTH_MIN_ARTICLES = 3


@dataclass(frozen=True)
class DriverBreakdown:
    """One driver's display payload.

    ``score`` is bounded ``[0, 1]`` (the same range
    :class:`RankedOpportunity` produces). ``descriptor`` is an
    observational phrase or empty string when the driver is below the
    descriptor floor.
    """

    key: str
    label: str
    score: float
    descriptor: str


@dataclass(frozen=True)
class OpportunityDrilldown:
    """Full breakdown for one ranked opportunity, sorted for display.

    ``drivers`` is pre-sorted strongest first so the render layer can
    iterate it directly. ``quality_labels`` is in display-priority
    order (positives first, then "unstable / choppy" caveat last when
    applicable). ``rationale`` is taken verbatim from the source
    :class:`RankedOpportunity` — single source of truth across the
    OPP line in the header and the drilldown block in the detail panel.
    ``state_phrases`` extends the drilldown with higher-abstraction
    observational bullets derived from combinations of driver scores,
    profile axes, and row state; the header doesn't render these.
    """

    rank: int
    symbol: str
    composite_score: float
    history: OpportunityHistory | None
    drivers: tuple[DriverBreakdown, ...]
    quality_labels: tuple[str, ...]
    rationale: tuple[str, ...]
    state_phrases: tuple[str, ...] = ()


_DRIVER_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("signal_quality_score", "signal quality"),
    ("technical_alignment", "technical alignment"),
    ("sentiment_alignment", "sentiment alignment"),
    ("momentum_persistence", "momentum persistence"),
    ("confidence_acceleration", "confidence acceleration"),
    ("unusual_activity", "unusual activity"),
    ("reversal_strength", "reversal strength"),
)


def build_drilldown(
    opp: RankedOpportunity,
    rank: int,
    history: OpportunityHistory | None,
    *,
    row: RecommendationRow | None = None,
) -> OpportunityDrilldown:
    """Assemble the drilldown payload for one ranked opportunity.

    ``rank`` is 1-based — i.e. the top OPP is rank=1. ``history`` may
    be ``None`` when the symbol isn't tracked yet (e.g. first tick
    after a re-add); the renderer treats that as "no badge".

    ``row`` enables action-aware state phrases — when omitted, the
    phrases that need action direction or news count simply don't
    fire. Callers that already have the row (the dashboard does)
    should pass it.
    """
    drivers = _drivers_for(opp)
    quality_labels = _quality_labels(opp)
    state_phrases = _state_phrases(opp, row)
    return OpportunityDrilldown(
        rank=rank,
        symbol=opp.symbol,
        composite_score=opp.composite_score,
        history=history,
        drivers=drivers,
        quality_labels=quality_labels,
        rationale=opp.rationale,
        state_phrases=state_phrases,
    )


def _drivers_for(opp: RankedOpportunity) -> tuple[DriverBreakdown, ...]:
    """Driver entries sorted by score descending.

    Stable across ties — Python's ``sort`` is stable so the canonical
    definition order in :data:`_DRIVER_DEFINITIONS` resolves ties
    deterministically.
    """
    descriptors = _descriptors_for(opp)
    entries: list[DriverBreakdown] = []
    for key, label in _DRIVER_DEFINITIONS:
        entries.append(
            DriverBreakdown(
                key=key,
                label=label,
                score=float(getattr(opp, key)),
                descriptor=descriptors.get(key, ""),
            )
        )
    entries.sort(key=lambda d: d.score, reverse=True)
    return tuple(entries)


def _descriptors_for(opp: RankedOpportunity) -> dict[str, str]:
    """Map each driver key → its observational descriptor (or omitted).

    Each phrase describes the *state* the driver score reflects, not
    a forecast. Phrasing mirrors the language used in
    :func:`src.intelligence.opportunities._rationale_from_drivers` so
    the header rationale and the drilldown read consistently.
    """
    out: dict[str, str] = {}
    if opp.technical_alignment >= _DESCRIPTOR_FLOOR:
        out["technical_alignment"] = "indicators aligned with action"
    if opp.sentiment_alignment >= _DESCRIPTOR_FLOOR:
        out["sentiment_alignment"] = "news sentiment matches direction"
    if opp.confidence_acceleration >= _DESCRIPTOR_FLOOR:
        out["confidence_acceleration"] = "confidence rising in current run"
    if opp.momentum_persistence >= _DESCRIPTOR_FLOOR:
        out["momentum_persistence"] = "momentum holding across recent ticks"
    if opp.unusual_activity >= _DESCRIPTOR_FLOOR:
        out["unusual_activity"] = "unusual confidence swing"
    if opp.reversal_strength >= _DESCRIPTOR_FLOOR:
        out["reversal_strength"] = "reversal intensity elevated"
    if opp.signal_quality_score >= _HIGH_CONVICTION_QUALITY_FLOOR:
        out["signal_quality_score"] = "high signal quality"
    return out


def _quality_labels(opp: RankedOpportunity) -> tuple[str, ...]:
    """Deterministic quality labels derived strictly from numeric state.

    Multiple labels may co-fire (e.g. "high conviction" alongside
    "building momentum"). When both a positive label and the
    "unstable / choppy" caveat apply, the caveat sits last so the eye
    reads the positives first — useful info that the composite is
    high *but* the underlying signal is noisy.
    """
    labels: list[str] = []
    if (
        opp.signal_quality_score >= _HIGH_CONVICTION_QUALITY_FLOOR
        and opp.composite_score >= _HIGH_CONVICTION_COMPOSITE_FLOOR
    ):
        labels.append("high conviction")
    if (
        opp.confidence_acceleration >= _BUILDING_MOMENTUM_ACCEL_FLOOR
        and opp.momentum_persistence >= _BUILDING_MOMENTUM_PERSISTENCE_FLOOR
    ):
        labels.append("building momentum")
    if opp.reversal_strength >= _REVERSAL_CANDIDATE_FLOOR:
        labels.append("reversal candidate")
    if (
        opp.sentiment_alignment >= _SENTIMENT_DRIVEN_SENT_FLOOR
        and opp.technical_alignment < _SENTIMENT_DRIVEN_TECH_CEIL
    ):
        labels.append("sentiment-driven")
    if opp.profile.stability == "noisy" or opp.profile.persistence == "flipping":
        labels.append("unstable / choppy")
    return tuple(labels)


def _state_phrases(
    opp: RankedOpportunity,
    row: RecommendationRow | None,
) -> tuple[str, ...]:
    """Higher-abstraction observational phrases for the drilldown.

    Each phrase fires from a numeric gate over a combination of
    existing fields (driver scores, profile axes, row action/news
    count). Strictly observational — phrases describe what the
    current state *is*, never what it might do next.

    The "reversal intensity elevated" language asked for in the spec
    is already produced as a per-driver descriptor at the
    ``reversal_strength`` line, so it isn't repeated here.
    """
    phrases: list[str] = []

    # Momentum strengthening — confidence rising within a run that's
    # had time to build (distinct from "momentum holding", which only
    # requires persistence).
    if (
        opp.confidence_acceleration >= _MOMENTUM_STRENGTHENING_ACCEL_FLOOR
        and opp.momentum_persistence >= _MOMENTUM_STRENGTHENING_PERSISTENCE_FLOOR
    ):
        phrases.append("momentum strengthening across recent ticks")

    # Sentiment breadth — alignment AND multi-article support. The
    # polarity word reflects the action direction (BUY = positive,
    # SELL = negative); only directional rows reach this code path
    # because rank_opportunities filters HOLD out upstream.
    if (
        row is not None
        and opp.sentiment_alignment >= _SENTIMENT_BREADTH_ALIGNMENT_FLOOR
        and row.num_news_articles >= _SENTIMENT_BREADTH_MIN_ARTICLES
    ):
        polarity = "positive" if row.action == SignalAction.BUY else "negative"
        phrases.append(f"{polarity} sentiment breadth")

    # Stable persistence — a settled run, with the action word inlined
    # so the trader doesn't have to cross-reference the row.
    if (
        row is not None
        and opp.profile.stability == "stable"
        and opp.profile.persistence == "persistent"
        and row.action in (SignalAction.BUY, SignalAction.SELL)
    ):
        phrases.append(f"stable {row.action.value.upper()} persistence")

    return tuple(phrases)


__all__ = ["DriverBreakdown", "OpportunityDrilldown", "build_drilldown"]
