"""Per-article quality weighting for news aggregation.

Every sentiment-driven feature in the system (alerts, ranking, pulse,
briefs) reads ``RecommendationRow.sentiment_score``, which is the
aggregate of per-article FinBERT scores. Without weighting, a stale
blog post weighs as much as a fresh wire-service headline — that
flattens the signal whenever Alpaca returns a long news tail.

This module supplies a per-article quality weight derived from two
observable properties:

* **Recency** — exponential decay in hours since publication, with a
  configurable half-life. Floor keeps very-old articles contributing
  a small amount rather than zeroing them out.
* **Source reputation** — a curated tier map; wire services and
  major financial press rank highest, blogs lowest, unknown sources
  fall back to a configurable default.

Pure functions, no I/O. Deterministic when ``now`` is passed
explicitly. Source matching is case-insensitive substring against
the curated keys — Alpaca's ``article.source`` strings come in many
casings ("The Motley Fool", "Benzinga Pro", etc.).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from src.config import Settings
    from src.data.models import NewsArticle


_DEFAULT_SOURCE_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        # Tier 1 — wire services + institutional financial press.
        "reuters": 1.0,
        "associated press": 1.0,
        "ap news": 1.0,
        "bloomberg": 1.0,
        # Tier 2 — major financial outlets.
        "wall street journal": 0.8,
        "wsj": 0.8,
        "financial times": 0.8,
        "ft.com": 0.8,
        "cnbc": 0.8,
        "marketwatch": 0.8,
        "barron": 0.8,
        # Tier 3 — Alpaca's typical feed.
        "benzinga": 0.6,
        "seeking alpha": 0.6,
        "motley fool": 0.6,
        "the fly": 0.6,
        "zacks": 0.6,
        # Tier 4 — general business / aggregator.
        "yahoo": 0.4,
        "business insider": 0.4,
        "forbes": 0.4,
        "investorplace": 0.4,
    }
)
"""Curated source-tier defaults. The mapping is read-only — wrap any
override in ``MappingProxyType`` (or pass a fresh dict) when
constructing a custom :class:`NewsQualityWeights`."""


_DEFAULT_UNKNOWN_SOURCE_WEIGHT = 0.30
_DEFAULT_HALF_LIFE_HOURS = 24.0
_DEFAULT_FLOOR_WEIGHT = 0.05


@dataclass(frozen=True)
class NewsQualityWeights:
    """Tunables governing per-article quality weighting.

    All weights are bounded ``[0, 1]``. ``floor_weight`` keeps stale
    articles contributing a small amount — zeroing them out would
    silently drop a real signal when the only news on a symbol is
    a day old.
    """

    half_life_hours: float = _DEFAULT_HALF_LIFE_HOURS
    floor_weight: float = _DEFAULT_FLOOR_WEIGHT
    source_weights: Mapping[str, float] = field(default_factory=lambda: _DEFAULT_SOURCE_WEIGHTS)
    unknown_source_weight: float = _DEFAULT_UNKNOWN_SOURCE_WEIGHT

    @classmethod
    def from_settings(cls, settings: Settings) -> NewsQualityWeights:
        """Pull the timing tunables from :class:`Settings`.

        The source map stays inline — env-loaded dicts are a UX wart
        and the curated defaults already cover Alpaca's common feed.
        Callers needing source overrides should construct directly.
        """
        return cls(
            half_life_hours=settings.news_recency_half_life_hours,
            floor_weight=settings.news_recency_floor_weight,
        )


def quality_weight(
    article: NewsArticle,
    now: datetime,
    weights: NewsQualityWeights,
) -> float:
    """Return the quality weight of one article in ``[floor, 1.0]``.

    Computed as ``recency × source_reputation``. Both factors are
    in ``[0, 1]``; the product is bounded above by 1.0 and below by
    ``weights.floor_weight`` (because recency is floored, source
    reputation is at least ``unknown_source_weight``, and their
    product is at least the recency floor times that — which we
    clamp to ``weights.floor_weight`` directly so callers don't have
    to reason about it).

    ``now`` is a required parameter — never read implicitly — so
    tests stay deterministic.
    """
    recency = _recency_weight(article, now, weights)
    source = _source_weight(article, weights)
    combined = recency * source
    return max(weights.floor_weight, combined)


def weighted_sentiment_mean(
    scores_and_weights: Iterable[tuple[float, float]],
) -> float:
    """Weighted arithmetic mean of signed sentiment scores.

    Each input is ``(signed_score, weight)``. Falls back to ``0.0``
    when the weight sum is zero — defensive; quality_weight floors
    above zero so this branch should be unreachable in practice.
    """
    total_weight = 0.0
    total = 0.0
    for score, weight in scores_and_weights:
        total_weight += weight
        total += score * weight
    if total_weight <= 0:
        return 0.0
    return total / total_weight


def _recency_weight(
    article: NewsArticle,
    now: datetime,
    weights: NewsQualityWeights,
) -> float:
    """Exponential decay of the article's age in hours.

    ``0.5 ** (hours_old / half_life)`` so a fresh article weighs 1.0,
    one half-life later weighs 0.5, two half-lives 0.25, etc.
    Floored at ``weights.floor_weight`` to keep stale items
    contributing a non-zero share.
    """
    if weights.half_life_hours <= 0:
        return 1.0
    delta = now - article.published_at
    hours_old = max(0.0, delta.total_seconds() / 3600.0)
    # ``float ** float`` returns Any in the current stubs; bind it
    # locally as a float so the strict return type stays clean.
    decayed: float = pow(0.5, hours_old / weights.half_life_hours)
    return max(weights.floor_weight, decayed)


def _source_weight(article: NewsArticle, weights: NewsQualityWeights) -> float:
    """Curated tier weight for the article's source.

    Case-insensitive substring match against
    ``weights.source_weights`` keys — Alpaca returns source strings
    like ``"The Motley Fool"``, ``"Benzinga Pro"``, ``"Zacks
    Investment Research"`` which won't equality-match a tidy key
    but will substring-match.
    """
    source = (article.source or "").lower().strip()
    if not source:
        return weights.unknown_source_weight
    for key, weight in weights.source_weights.items():
        if key in source:
            return weight
    return weights.unknown_source_weight


__all__ = [
    "NewsQualityWeights",
    "quality_weight",
    "weighted_sentiment_mean",
]
