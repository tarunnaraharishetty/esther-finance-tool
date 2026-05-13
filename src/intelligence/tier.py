"""Tier promotion — turns a base 3-tier recommendation into a 5-tier one.

The :class:`~src.strategy.recommendation.RecommendationEngine` knows
nothing about session history; it can only emit BUY / HOLD / SELL from
the current bar window. This module sits one layer above and decides
whether a directional recommendation qualifies for the STRONG tier
based on multi-system alignment, confidence stability, and reversal
frequency.

Inputs:
- :class:`TradingRecommendation` from the engine (with default
  ``tier = action``).
- Optional :class:`SignalHistorySummary` for the same symbol.

Output: a new (frozen) ``TradingRecommendation`` with ``tier``,
``signal_quality``, ``stability``, and ``quality_reasons`` populated.
Strong tier is only emitted when ALL FOUR gating conditions are met:

1. Action is directional (BUY or SELL).
2. ``signal_quality == "high"``.
3. ``stability == "stable"``.
4. ``confidence >= STRONG_CONFIDENCE_FLOOR``.

Anti-flip safeguard is intrinsic — recent reversals lower the stability
score, which by gate (3) prevents promotion to STRONG. No separate
"recently flipped" check needed.

Quality reasons are deterministic template strings. The recommendation
text remains observational; this module never invents catalysts or
makes predictions.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.strategy.base import RecommendationTier, SignalAction
from src.strategy.recommendation import TradingRecommendation

if TYPE_CHECKING:
    from src.intelligence.history import SignalHistorySummary


# ---------------------------------------------------------------------------
# Tunables — inlined so calibration is visible at the definition site
# ---------------------------------------------------------------------------

# Signal-quality thresholds
_QUALITY_HIGH = 0.70
_QUALITY_MODERATE = 0.40

# Stability thresholds
_STABILITY_STABLE = 0.65
_STABILITY_MODERATE = 0.40

# STRONG gate
STRONG_CONFIDENCE_FLOOR = 0.65

# How much of the action direction an indicator must show before counting
# as "aligned" for technical-alignment scoring.
_ALIGN_MIN_MAGNITUDE = 0.10

# Sentiment magnitude that counts as "directional" (vs noise).
_SENTIMENT_DIRECTIONAL = 0.20

# Momentum (MACD score) magnitude that counts as "strong".
_MOMENTUM_STRONG = 0.40

# Episode-length floors for stability scoring.
_EPISODE_STABLE_MIN_TICKS = 5
_EPISODE_BUILDING_MIN_TICKS = 2


def promote_to_tier(
    recommendation: TradingRecommendation,
    history: SignalHistorySummary | None = None,
) -> TradingRecommendation:
    """Return a new recommendation with tier + scores populated."""
    quality_reasons: list[str] = []

    quality_score = _compute_signal_quality(recommendation, quality_reasons)
    stability_score = _compute_stability(history, quality_reasons)

    signal_quality = _tier_for_quality(quality_score)
    stability = _tier_for_stability(stability_score)

    # Decide tier.
    base_tier = RecommendationTier.from_action(recommendation.action)
    promoted = _maybe_promote(
        recommendation, signal_quality=signal_quality, stability=stability
    )
    final_tier = promoted if promoted is not None else base_tier

    return recommendation.model_copy(
        update={
            "tier": final_tier,
            "signal_quality": signal_quality,
            "stability": stability,
            "quality_reasons": tuple(quality_reasons),
        }
    )


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def _maybe_promote(
    rec: TradingRecommendation,
    *,
    signal_quality: str,
    stability: str,
) -> RecommendationTier | None:
    """Return STRONG_BUY / STRONG_SELL when ALL four gates pass; else None."""
    if rec.action not in (SignalAction.BUY, SignalAction.SELL):
        return None
    if signal_quality != "high":
        return None
    if stability != "stable":
        return None
    if rec.confidence < STRONG_CONFIDENCE_FLOOR:
        return None
    return (
        RecommendationTier.STRONG_BUY
        if rec.action == SignalAction.BUY
        else RecommendationTier.STRONG_SELL
    )


# ---------------------------------------------------------------------------
# Signal-quality scoring
# ---------------------------------------------------------------------------


def _compute_signal_quality(
    rec: TradingRecommendation, reasons: list[str]
) -> float:
    """Return a 0-1 score; higher = better-aligned inputs.

    Drivers:
      - Technical alignment: fraction of indicators that point the same
        way as the action.
      - Sentiment alignment: news direction matches action (only if
        there's actual news).
      - Momentum strength: MACD magnitude above threshold.

    Each driver contributes a 0-1 sub-score; the final score is their
    mean. A driver that doesn't apply (e.g. no news) is simply skipped.
    """
    direction = _direction(rec.action)
    if direction == 0:
        # HOLD doesn't qualify for any quality bonus — there's no
        # direction to be aligned with.
        return 0.0

    sub_scores: list[float] = []

    # ---- Technical alignment ----
    # Strict: any indicator opposing the action caps the alignment
    # sub-score so a single contrarian reading can't be drowned out
    # by a strong sentiment lift. STRONG requires clean alignment.
    tech_values = [s for s in rec.indicator_scores.values() if _is_finite(s)]
    if tech_values:
        aligned = sum(
            1 for s in tech_values if direction * s > _ALIGN_MIN_MAGNITUDE
        )
        opposed = sum(
            1 for s in tech_values if direction * s < -_ALIGN_MIN_MAGNITUDE
        )
        alignment_frac = aligned / len(tech_values)
        if opposed > 0:
            # Hard cap: opposition means alignment is at best "moderate".
            alignment_frac = min(alignment_frac, 0.40)
        sub_scores.append(alignment_frac)
        if aligned == len(tech_values) and opposed == 0:
            reasons.append("Strong technical alignment")

    # ---- Sentiment alignment ----
    if rec.num_news_articles > 0:
        directional = direction * rec.sentiment_score
        if directional > _SENTIMENT_DIRECTIONAL:
            sub_scores.append(0.9)
            reasons.append(
                "Bullish sentiment acceleration"
                if direction > 0
                else "Bearish sentiment acceleration"
            )
        elif abs(rec.sentiment_score) < _SENTIMENT_DIRECTIONAL / 2:
            # Neutral news — neither helps nor hurts.
            sub_scores.append(0.5)
        else:
            # News disagrees with action.
            sub_scores.append(0.1)

    # ---- Momentum strength (MACD score) ----
    macd = rec.indicator_scores.get("macd", 0.0)
    if _is_finite(macd) and abs(macd) >= _MOMENTUM_STRONG:
        # Only credit when momentum aligns with action.
        if direction * macd > 0:
            sub_scores.append(0.85)
            reasons.append(
                "Positive momentum" if direction > 0 else "Negative momentum"
            )

    if not sub_scores:
        return 0.0
    return sum(sub_scores) / len(sub_scores)


def _tier_for_quality(score: float) -> str:
    if score >= _QUALITY_HIGH:
        return "high"
    if score >= _QUALITY_MODERATE:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Stability scoring
# ---------------------------------------------------------------------------


def _compute_stability(
    history: SignalHistorySummary | None, reasons: list[str]
) -> float:
    """Return a 0-1 stability score.

    Drivers:
      - Confidence trend within the current episode.
      - Reversal frequency (fewer episodes = more stable).
      - Episode duration.

    Hard vetoes ensure trader intuition holds:
      - Episode count >= 4 → immediate "volatile" return. A symbol
        that's flipped three or more times this session doesn't get
        the "stable" badge even if the current run is momentarily calm.
      - Falling confidence in a held run → soft cap below "stable".
        A run losing steam shouldn't qualify as stable.

    With no history (e.g. first tick of the session), we return a
    middle-of-the-road score so we neither falsely promote nor falsely
    block.
    """
    if history is None:
        return 0.5

    current = history.current
    episode_count = 1 + len(history.recent)

    # ---- Hard veto: many flips → volatile, full stop. ----
    if episode_count >= 4:
        reasons.append("High reversal frequency")
        return 0.15

    sub_scores: list[float] = []
    # Soft cap from individual signals that should disqualify "stable".
    soft_caps: list[float] = []

    # ---- Confidence stability ----
    if current.tick_count >= 3:
        conf_delta = current.confidence_last - current.confidence_first
        if abs(conf_delta) < 0.05:
            sub_scores.append(0.9)
            reasons.append("Stable confidence")
        elif conf_delta >= 0.10:
            sub_scores.append(0.8)
            reasons.append("Rising confidence")
        elif conf_delta <= -0.15:
            # Falling confidence in a held run is destabilizing — never
            # let this case earn "stable" no matter how clean the other
            # drivers look.
            sub_scores.append(0.2)
            soft_caps.append(_STABILITY_STABLE - 0.10)
        else:
            sub_scores.append(0.55)
    else:
        # Episode too young to judge.
        sub_scores.append(0.5)

    # ---- Reversal frequency (we already short-circuited >= 4) ----
    if episode_count == 1:
        sub_scores.append(0.9)
        reasons.append("Low reversal frequency")
    elif episode_count == 2:
        sub_scores.append(0.6)
    else:  # exactly 3
        sub_scores.append(0.35)

    # ---- Current episode duration ----
    if current.tick_count >= _EPISODE_STABLE_MIN_TICKS:
        sub_scores.append(0.85)
    elif current.tick_count >= _EPISODE_BUILDING_MIN_TICKS:
        sub_scores.append(0.55)
    else:
        sub_scores.append(0.25)

    score = sum(sub_scores) / len(sub_scores)
    if soft_caps:
        score = min(score, *soft_caps)
    return score


def _tier_for_stability(score: float) -> str:
    if score >= _STABILITY_STABLE:
        return "stable"
    if score >= _STABILITY_MODERATE:
        return "moderate"
    return "volatile"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _direction(action: SignalAction) -> int:
    if action == SignalAction.BUY:
        return 1
    if action == SignalAction.SELL:
        return -1
    return 0


def _is_finite(x: float) -> bool:
    """NaN-safe finite check without importing math at call sites."""
    return not math.isnan(x) and math.isfinite(x)


__all__ = ["STRONG_CONFIDENCE_FLOOR", "promote_to_tier"]
