"""Opportunity detection — ranked actionable setups across the watchlist.

Two concepts, both pure-data:

* :func:`detect_opportunities` — kind-based patterns. Returns 0-3
  named-pattern opportunities (convergence, reversal, high_conviction).
  Useful when you want "did a specific pattern fire?" semantics.

* :func:`rank_opportunities` — composite-score ranking. Scores every
  directional row across seven drivers and returns the top N. Useful
  for the dashboard's at-a-glance "what should I look at first?"
  question. Each ranked entry also carries a :class:`SignalProfile`
  classifying the symbol's signal as stable/noisy ×
  strengthening/weakening/flat × persistent/flipping.

Both stay grounded — every driver is a signed comparison of existing
scores against existing action direction. Rationale strings are
deterministic templates. No predictions, no invented catalysts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.intelligence.signal_profile import (
    SignalProfile,
    compute_signal_profile,
)
from src.strategy.base import RecommendationTier, SignalAction

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot, RecommendationRow
    from src.intelligence.history import SignalHistorySummary

# Tuning thresholds — kept inline so calibration is visible at the
# definition site rather than buried in a config.
_CONFIDENCE_HIGH = 0.55
_DIRECTIONAL_NONTRIVIAL = 0.20
_CONFIDENCE_RISE_DELTA = 0.05

# Ranking-engine tunables (separate from the kind-detector above).
_ALIGN_MIN_MAGNITUDE = 0.10
_SENTIMENT_DIRECTIONAL = 0.20
_DRIVER_WEIGHTS: dict[str, float] = {
    "technical_alignment": 0.20,
    "sentiment_alignment": 0.15,
    "confidence_acceleration": 0.10,
    "momentum_persistence": 0.15,
    "unusual_activity": 0.10,
    "reversal_strength": 0.10,
    "signal_quality_score": 0.20,
}
# When a driver score crosses this floor, contribute its rationale
# string to the human-readable bullets.
_RATIONALE_FLOOR = 0.60


@dataclass(frozen=True)
class Opportunity:
    """One ranked opportunity. Stable schema for UI + LLM inputs."""

    symbol: str
    kind: str  # "convergence" | "reversal" | "high_conviction"
    rationale: str  # plain-English; observational, no predictions
    score: float  # composite ranking score; non-negative


def detect_opportunities(
    snapshot: DashboardSnapshot,
    *,
    n: int = 3,
) -> list[Opportunity]:
    """Return up to ``n`` opportunities, highest score first.

    Pure function of the snapshot — call freely, no caching needed.
    Error rows are excluded everywhere.
    """
    healthy = [r for r in snapshot.rows if not r.error]
    candidates: list[Opportunity] = []
    for row in healthy:
        for opp in _candidates_for_row(row, snapshot):
            candidates.append(opp)

    candidates.sort(key=lambda o: o.score, reverse=True)

    # Dedup by (symbol, kind) — same symbol may surface multiple
    # candidates of the same kind from edge-case scoring; keep the
    # highest. Cross-kind duplicates on the same symbol are allowed
    # (a convergence AND a reversal on the same name is meaningful).
    seen: set[tuple[str, str]] = set()
    out: list[Opportunity] = []
    for opp in candidates:
        key = (opp.symbol, opp.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(opp)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Per-row candidate generators
# ---------------------------------------------------------------------------


def _candidates_for_row(row: RecommendationRow, snapshot: DashboardSnapshot) -> list[Opportunity]:
    out: list[Opportunity] = []
    conv = _check_convergence(row)
    if conv is not None:
        out.append(conv)
    rev = _check_reversal(row, snapshot)
    if rev is not None:
        out.append(rev)
    high = _check_high_conviction(row)
    if high is not None:
        out.append(high)
    return out


def _check_convergence(row: RecommendationRow) -> Opportunity | None:
    """All directional inputs lean the same way + the recommendation
    is itself directional (not HOLD)."""
    if row.action == SignalAction.HOLD:
        return None
    direction = 1 if row.action == SignalAction.BUY else -1

    def aligned(score: float) -> bool:
        # NaN-safe; treats NaN as not-aligned.
        if score != score:
            return False
        return (direction > 0 and score > 0) or (direction < 0 and score < 0)

    # All three indicators + sentiment need to agree. Sentiment only
    # counts if there's actual news; without it, convergence isn't real.
    has_news = row.num_news_articles > 0
    indicators_aligned = aligned(row.rsi) and aligned(row.macd) and aligned(row.bollinger)
    sentiment_aligned = has_news and aligned(row.sentiment_score)
    if not (indicators_aligned and sentiment_aligned):
        return None

    # Score: average of the four magnitudes — rewards strong alignment.
    magnitudes = [
        abs(row.rsi),
        abs(row.macd),
        abs(row.bollinger),
        abs(row.sentiment_score),
    ]
    score = sum(magnitudes) / len(magnitudes)
    direction_word = "bullish" if direction > 0 else "bearish"
    return Opportunity(
        symbol=row.symbol,
        kind="convergence",
        rationale=(
            f"RSI, MACD, Bollinger, and news sentiment all "
            f"{direction_word} (confidence {row.confidence:.2f})"
        ),
        score=score,
    )


def _check_reversal(row: RecommendationRow, snapshot: DashboardSnapshot) -> Opportunity | None:
    """Action flipped from a non-HOLD prior episode AND confidence is
    rising within the new episode."""
    if row.action == SignalAction.HOLD:
        return None
    summary = snapshot.signal_history.get(row.symbol)
    if summary is None or not summary.recent:
        return None
    prior = summary.recent[0]
    if prior.action == row.action:
        return None  # same direction = not a reversal
    if prior.action == SignalAction.HOLD:
        return None  # exiting HOLD is a trigger, not a reversal
    # Confidence must be rising within the new (current) episode.
    delta = summary.current.confidence_last - summary.current.confidence_first
    if delta < _CONFIDENCE_RISE_DELTA:
        return None

    # Score: confidence-rise magnitude × prior episode duration.
    # A long-held prior episode that just flipped is more notable than
    # a one-tick blip flipping.
    score = abs(delta) * max(prior.tick_count, 1)
    return Opportunity(
        symbol=row.symbol,
        kind="reversal",
        rationale=(
            f"flipped {prior.action.value.upper()} -> "
            f"{row.action.value.upper()} after {prior.tick_count} "
            f"tick{'s' if prior.tick_count != 1 else ''}; conf rising "
            f"{summary.current.confidence_first:.2f} -> "
            f"{summary.current.confidence_last:.2f}"
        ),
        score=score,
    )


def _check_high_conviction(row: RecommendationRow) -> Opportunity | None:
    """Top-tier confidence with non-trivial technical AND sentiment
    contribution in the same direction."""
    if row.action == SignalAction.HOLD:
        return None
    if row.confidence < _CONFIDENCE_HIGH:
        return None
    if abs(row.technical_score) < _DIRECTIONAL_NONTRIVIAL:
        return None
    if abs(row.sentiment_score) < _DIRECTIONAL_NONTRIVIAL:
        return None
    if row.num_news_articles == 0:
        return None  # sentiment without news isn't grounded
    # Both contributors must agree in direction with the action.
    direction = 1 if row.action == SignalAction.BUY else -1
    if direction * row.technical_score <= 0:
        return None
    if direction * row.sentiment_score <= 0:
        return None
    score = row.confidence
    return Opportunity(
        symbol=row.symbol,
        kind="high_conviction",
        rationale=(
            f"confidence {row.confidence:.2f}, technical "
            f"{row.technical_score:+.2f}, sentiment "
            f"{row.sentiment_score:+.2f} over {row.num_news_articles} "
            f"article{'s' if row.num_news_articles != 1 else ''}"
        ),
        score=score,
    )


# ---------------------------------------------------------------------------
# Composite ranking — Top Opportunities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankedOpportunity:
    """One entry in the ranked Top Opportunities list.

    Carries the composite score that drives ranking + transparency
    fields (the seven driver scores) + a :class:`SignalProfile`
    classifying the signal's stability / trend / persistence.

    Stable schema for both UI rendering and downstream LLM grounding.
    """

    symbol: str
    tier: RecommendationTier
    composite_score: float  # [0, 1]
    profile: SignalProfile
    rationale: tuple[str, ...]  # human-readable bullet points
    # Per-driver scores in [0, 1] for transparency.
    technical_alignment: float
    sentiment_alignment: float
    confidence_acceleration: float
    momentum_persistence: float
    unusual_activity: float
    reversal_strength: float
    signal_quality_score: float


def rank_opportunities_intraday(
    snapshot: DashboardSnapshot,
    *,
    n: int = 5,
) -> list[RankedOpportunity]:
    """Rank watchlist symbols by composite score on the **intraday**
    timeframe.

    Mirrors :func:`rank_opportunities` but reads from
    ``row.intraday`` (action, confidence, technical_score) and the
    parallel ``snapshot.intraday_signal_history``. Rows without an
    ``IntradayRead`` are skipped — no intraday data, no intraday
    ranking. Sentiment is timeframe-agnostic (news weighting is the
    same regardless of bar timeframe) and the signal-quality tier
    isn't computed separately for intraday yet (Phase 2c), so both
    of those drivers still read from the daily row.

    Returns the same :class:`RankedOpportunity` shape; ``tier`` is
    mapped directly from ``intraday.action`` (no STRONG promotion).
    """
    candidates: list[RankedOpportunity] = []
    for row in snapshot.rows:
        if row.error or row.intraday is None:
            continue
        if row.intraday.action not in (SignalAction.BUY, SignalAction.SELL):
            continue
        history = snapshot.intraday_signal_history.get(row.symbol)
        scores = _intraday_driver_scores(row, history)
        composite = _weighted_composite(scores)
        if composite <= 0:
            continue
        # Profile reuses the existing classifier — stability still
        # reads from row.stability (daily), but trend / persistence
        # come from the intraday history that gets passed in.
        profile = compute_signal_profile(row, history)
        rationale = _rationale_from_drivers(scores, profile, history)
        candidates.append(
            RankedOpportunity(
                symbol=row.symbol,
                tier=RecommendationTier.from_action(row.intraday.action),
                composite_score=composite,
                profile=profile,
                rationale=rationale,
                **scores,
            )
        )
    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return candidates[:n]


def rank_opportunities(
    snapshot: DashboardSnapshot,
    *,
    n: int = 5,
) -> list[RankedOpportunity]:
    """Rank watchlist symbols by a 7-driver composite score.

    Only directional rows (BUY / SELL) participate — HOLD has no
    "opportunity" to rank. Returns up to ``n`` entries, highest
    composite first. Empty list when no directional rows exist.

    Pure function — call freely, no caching needed. Same data flow
    pattern as :func:`detect_opportunities`; the two can co-exist.
    """
    candidates: list[RankedOpportunity] = []
    for row in snapshot.rows:
        if row.error:
            continue
        if row.action not in (SignalAction.BUY, SignalAction.SELL):
            continue
        history = snapshot.signal_history.get(row.symbol)
        scores = _driver_scores(row, history)
        composite = _weighted_composite(scores)
        if composite <= 0:
            continue
        profile = compute_signal_profile(row, history)
        rationale = _rationale_from_drivers(scores, profile, history)
        candidates.append(
            RankedOpportunity(
                symbol=row.symbol,
                tier=row.tier,
                composite_score=composite,
                profile=profile,
                rationale=rationale,
                **scores,
            )
        )
    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return candidates[:n]


# ---------------------------------------------------------------------------
# Driver scoring functions
# ---------------------------------------------------------------------------


def _driver_scores(
    row: RecommendationRow,
    history: SignalHistorySummary | None,
) -> dict[str, float]:
    direction = _direction(row.action)
    return {
        "technical_alignment": _score_technical_alignment(row, direction),
        "sentiment_alignment": _score_sentiment_alignment(row, direction),
        "confidence_acceleration": _score_confidence_acceleration(history),
        "momentum_persistence": _score_momentum_persistence(history),
        "unusual_activity": _score_unusual_activity(history),
        "reversal_strength": _score_reversal_strength(row, history, direction),
        "signal_quality_score": _score_signal_quality(row),
    }


def _intraday_driver_scores(
    row: RecommendationRow,
    history: SignalHistorySummary | None,
) -> dict[str, float]:
    """Driver scores for the intraday ranker.

    Differs from the daily ``_driver_scores`` in three places:

    * ``technical_alignment`` keys off the aggregate
      ``row.intraday.technical_score`` because IntradayRead doesn't
      carry per-indicator scores (RSI / MACD / Bollinger separately).
    * ``reversal_strength`` uses the intraday confidence, not the
      daily one.
    * Sentiment + signal-quality drivers pass through unchanged —
      news weighting is timeframe-agnostic and the tier-based
      quality grade isn't separately computed for intraday yet
      (Phase 2c if needed).

    Caller is responsible for ensuring ``row.intraday is not None``
    — this function assumes it.
    """
    assert row.intraday is not None
    direction = _direction(row.intraday.action)
    return {
        "technical_alignment": _score_technical_alignment_intraday(row, direction),
        "sentiment_alignment": _score_sentiment_alignment(row, direction),
        "confidence_acceleration": _score_confidence_acceleration(history),
        "momentum_persistence": _score_momentum_persistence(history),
        "unusual_activity": _score_unusual_activity(history),
        "reversal_strength": _score_reversal_strength_intraday(row, history, direction),
        "signal_quality_score": _score_signal_quality(row),
    }


def _score_technical_alignment_intraday(row: RecommendationRow, direction: int) -> float:
    """Intraday technical alignment from the aggregate technical
    score. Same magnitude scaling as the sentiment driver — score
    in the action direction gets mapped to ``[0, 1]``."""
    if direction == 0 or row.intraday is None:
        return 0.0
    directional = direction * row.intraday.technical_score
    if directional <= _ALIGN_MIN_MAGNITUDE:
        return 0.0
    return min(1.0, abs(row.intraday.technical_score) * 2.0)


def _score_reversal_strength_intraday(
    row: RecommendationRow,
    history: SignalHistorySummary | None,
    direction: int,
) -> float:
    """Same shape as ``_score_reversal_strength`` but uses the
    intraday confidence — a fresh flip on the intraday timeframe
    should reflect intraday conviction, not daily."""
    if direction == 0 or history is None or not history.recent or row.intraday is None:
        return 0.0
    prior = history.recent[0]
    prior_dir = _direction(prior.action)
    if prior_dir == 0 or prior_dir == direction:
        return 0.0
    score = row.intraday.confidence * prior.tick_count / 5.0
    return min(1.0, score)


def _direction(action: SignalAction) -> int:
    if action == SignalAction.BUY:
        return 1
    if action == SignalAction.SELL:
        return -1
    return 0


def _is_finite(x: float) -> bool:
    return x == x  # NaN != NaN


def _score_technical_alignment(row: RecommendationRow, direction: int) -> float:
    """Fraction of indicators (RSI, MACD, Bollinger) pointing the same
    way as the action. ``[0, 1]``; 1.0 when all three are aligned."""
    if direction == 0:
        return 0.0
    indicators = (row.rsi, row.macd, row.bollinger)
    finite = [s for s in indicators if _is_finite(s)]
    if not finite:
        return 0.0
    aligned = sum(1 for s in finite if direction * s > _ALIGN_MIN_MAGNITUDE)
    return aligned / len(finite)


def _score_sentiment_alignment(row: RecommendationRow, direction: int) -> float:
    """News-bearing sentiment in the action direction. ``[0, 1]``.

    Zero when there's no news (sentiment without articles is not
    grounded). Otherwise scales with the magnitude of the sentiment.
    """
    if direction == 0 or row.num_news_articles == 0:
        return 0.0
    directional = direction * row.sentiment_score
    if directional <= _ALIGN_MIN_MAGNITUDE:
        return 0.0
    # Magnitude up to ~0.5 = strong sentiment alignment.
    return min(1.0, abs(row.sentiment_score) * 2.0)


def _score_confidence_acceleration(history: SignalHistorySummary | None) -> float:
    """Rate of confidence change within the current episode. ``[0, 1]``.

    Normalized so 0.5 = flat, >0.5 = rising, <0.5 = falling. Delta of
    +/-0.4 across the episode saturates to 1.0 / 0.0. Short episodes
    (under 2 ticks) return the neutral 0.5 — no signal yet.
    """
    if history is None or history.current.tick_count < 2:
        return 0.5
    delta = history.current.confidence_last - history.current.confidence_first
    clipped = max(-0.4, min(0.4, delta))
    return 0.5 + clipped * 1.25  # ±0.4 → ±0.5 → [0, 1]


def _score_momentum_persistence(history: SignalHistorySummary | None) -> float:
    """Longer current episode = more persistent momentum. ``[0, 1]``.

    Linear: 1 tick → 0.2, 8 ticks → 1.0 (saturates). Without history
    we return 0.3 — slight penalty for the unknown.
    """
    if history is None:
        return 0.3
    ticks = history.current.tick_count
    return min(1.0, 0.2 + ticks * 0.1)


def _score_unusual_activity(history: SignalHistorySummary | None) -> float:
    """Big confidence swing inside a held episode. ``[0, 1]``.

    Rewards both stickiness and movement: |conf_delta| × tick_count,
    normalized. A 1-tick episode scores 0 (no delta possible).
    """
    if history is None or history.current.tick_count < 2:
        return 0.0
    delta = abs(history.current.confidence_last - history.current.confidence_first)
    score = delta * history.current.tick_count / 4.0
    return min(1.0, score)


def _score_reversal_strength(
    row: RecommendationRow,
    history: SignalHistorySummary | None,
    direction: int,
) -> float:
    """Strong when the action just flipped from a long-held opposite
    direction. ``[0, 1]``.

    Zero unless the most-recent prior episode had the opposite
    direction. Score = current confidence × prior tick count,
    normalized — a flip after 10 ticks of BUY is much stronger than
    a flip after 1 tick.
    """
    if direction == 0 or history is None or not history.recent:
        return 0.0
    prior = history.recent[0]
    prior_dir = _direction(prior.action)
    if prior_dir == 0 or prior_dir == direction:
        return 0.0
    score = row.confidence * prior.tick_count / 5.0
    return min(1.0, score)


def _score_signal_quality(row: RecommendationRow) -> float:
    """Map the tier-system quality label to a numeric score."""
    return {"high": 1.0, "moderate": 0.5, "low": 0.0}.get(row.signal_quality, 0.0)


def _weighted_composite(scores: dict[str, float]) -> float:
    """Weighted sum across all seven drivers. Always in ``[0, 1]``
    because weights sum to 1.0 and every driver is bounded ``[0, 1]``."""
    return sum(_DRIVER_WEIGHTS[k] * scores[k] for k in _DRIVER_WEIGHTS)


# ---------------------------------------------------------------------------
# Rationale builder — deterministic templates, no LLM
# ---------------------------------------------------------------------------


def _rationale_from_drivers(
    scores: dict[str, float],
    profile: SignalProfile,
    history: SignalHistorySummary | None,
) -> tuple[str, ...]:
    """Build human-readable bullets from the strongest drivers.

    Only mentions drivers above the rationale floor — quiet drivers
    don't contribute. Output is observational, not predictive — same
    contract as the rest of intelligence/.
    """
    out: list[str] = []
    if scores["technical_alignment"] >= _RATIONALE_FLOOR:
        out.append("indicators aligned with action")
    if scores["sentiment_alignment"] >= _RATIONALE_FLOOR:
        out.append("news sentiment matches direction")
    if scores["confidence_acceleration"] >= _RATIONALE_FLOOR:
        out.append("confidence rising in current run")
    if scores["momentum_persistence"] >= _RATIONALE_FLOOR and history is not None:
        out.append(
            f"momentum held {history.current.tick_count} "
            f"tick{'s' if history.current.tick_count != 1 else ''}"
        )
    if scores["unusual_activity"] >= _RATIONALE_FLOOR:
        out.append("unusual confidence swing")
    if scores["reversal_strength"] >= _RATIONALE_FLOOR and history is not None:
        prior = history.recent[0]
        out.append(f"fresh reversal from {prior.tick_count}-tick {prior.action.value.upper()}")
    if scores["signal_quality_score"] >= 1.0:
        out.append("high signal quality")
    return tuple(out)


__all__ = [
    "Opportunity",
    "RankedOpportunity",
    "detect_opportunities",
    "rank_opportunities",
    "rank_opportunities_intraday",
]
