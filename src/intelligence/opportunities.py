"""Opportunity detection — ranked actionable setups across the watchlist.

Pure data, no LLM, no state. Composites the existing snapshot fields
(rows, signal_history, rankings) into a small ranked list of named
opportunities the trader might want to look at first.

The "kinds" shipped in v1 are deliberately narrow — each one is a
specific, defensible pattern grounded in observable data:

- ``convergence``     RSI + MACD + Bollinger + sentiment all lean the
                      same direction (no internal disagreement).
- ``reversal``        The symbol just flipped direction AND confidence
                      is rising within the new episode.
- ``high_conviction`` Confidence is in the top tier AND both the
                      technical and sentiment scores are non-trivially
                      directional in the same sign.

An ``Opportunity`` has a stable schema. The dashboard renders the top N
in a small OPP section in the watchlist header; the LLM recap can use
the same list as structured grounding input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.strategy.base import SignalAction

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

# Tuning thresholds — kept inline so calibration is visible at the
# definition site rather than buried in a config.
_CONFIDENCE_HIGH = 0.55
_DIRECTIONAL_NONTRIVIAL = 0.20
_CONFIDENCE_RISE_DELTA = 0.05


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


def _candidates_for_row(
    row: RecommendationRow, snapshot: DashboardSnapshot
) -> list[Opportunity]:
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
    indicators_aligned = (
        aligned(row.rsi) and aligned(row.macd) and aligned(row.bollinger)
    )
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


def _check_reversal(
    row: RecommendationRow, snapshot: DashboardSnapshot
) -> Opportunity | None:
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


__all__ = ["Opportunity", "detect_opportunities"]
