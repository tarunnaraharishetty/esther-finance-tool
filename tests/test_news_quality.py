"""Tests for src.sentiment.news_quality."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from src.data.models import NewsArticle
from src.sentiment.news_quality import (
    NewsQualityWeights,
    quality_weight,
    weighted_sentiment_mean,
)

_NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _article(
    *,
    source: str = "Test Source",
    hours_ago: float = 0.0,
    headline: str = "Stock moves",
) -> NewsArticle:
    return NewsArticle(
        id=f"id-{source}-{hours_ago}-{abs(hash(headline)) % 100000}",
        headline=headline,
        summary="",
        source=source,
        symbols=["AAPL"],
        published_at=_NOW - timedelta(hours=hours_ago),
    )


# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------


def test_freshly_published_article_full_recency() -> None:
    """Zero-hour-old article from a top-tier source weighs 1.0."""
    weights = NewsQualityWeights()
    w = quality_weight(_article(source="Reuters", hours_ago=0.0), _NOW, weights)
    assert w == 1.0


def test_one_half_life_old_halves_the_weight() -> None:
    """At exactly one half-life (default 24h), recency halves. With
    Reuters source weight 1.0, combined weight should be ~0.5."""
    weights = NewsQualityWeights()
    w = quality_weight(_article(source="Reuters", hours_ago=24.0), _NOW, weights)
    assert math.isclose(w, 0.5, rel_tol=1e-6)


def test_two_half_lives_old_quarters_the_weight() -> None:
    """After two half-lives recency = 0.25; Reuters → ~0.25 combined."""
    weights = NewsQualityWeights()
    w = quality_weight(_article(source="Reuters", hours_ago=48.0), _NOW, weights)
    assert math.isclose(w, 0.25, rel_tol=1e-6)


def test_extremely_old_article_floors_at_minimum() -> None:
    """Very old articles still contribute a floor weight rather than
    silently dropping to zero — defends against losing the only news
    on a quiet-news session."""
    weights = NewsQualityWeights(floor_weight=0.05)
    # 30 days old — way past where exponential decay rounds to zero.
    w = quality_weight(_article(source="Reuters", hours_ago=24.0 * 30), _NOW, weights)
    assert w == 0.05


def test_future_published_at_clamps_to_zero_age() -> None:
    """An article timestamped in the future (clock-skew defense) reads
    as fresh — not negative age, not exploded recency."""
    weights = NewsQualityWeights()
    future = NewsArticle(
        id="future",
        headline="x",
        source="Reuters",
        symbols=["AAPL"],
        published_at=_NOW + timedelta(hours=2),
    )
    w = quality_weight(future, _NOW, weights)
    assert w == 1.0


# ---------------------------------------------------------------------------
# Source reputation
# ---------------------------------------------------------------------------


def test_wire_service_sources_get_tier_one_weight() -> None:
    """Reuters, AP, and Bloomberg are the canonical tier-1 sources."""
    weights = NewsQualityWeights()
    for source in ("Reuters", "Associated Press", "Bloomberg News"):
        w = quality_weight(_article(source=source), _NOW, weights)
        assert w == 1.0, f"{source} should weigh 1.0"


def test_major_financial_outlets_get_tier_two_weight() -> None:
    """WSJ, FT, CNBC, MarketWatch, Barron's are tier 2 (0.8)."""
    weights = NewsQualityWeights()
    for source in ("Wall Street Journal", "Financial Times", "CNBC", "Barron's"):
        w = quality_weight(_article(source=source), _NOW, weights)
        assert math.isclose(w, 0.8, rel_tol=1e-6), f"{source} should weigh 0.8"


def test_alpaca_feed_sources_get_tier_three_weight() -> None:
    """Alpaca's typical feed — Benzinga, Seeking Alpha, Motley Fool — at 0.6."""
    weights = NewsQualityWeights()
    for source in ("Benzinga Pro", "Seeking Alpha", "The Motley Fool", "The Fly"):
        w = quality_weight(_article(source=source), _NOW, weights)
        assert math.isclose(w, 0.6, rel_tol=1e-6), f"{source} should weigh 0.6"


def test_unknown_source_uses_default_weight() -> None:
    """An unfamiliar source falls back to ``unknown_source_weight``."""
    weights = NewsQualityWeights(unknown_source_weight=0.25)
    w = quality_weight(_article(source="My Niche Blog"), _NOW, weights)
    assert math.isclose(w, 0.25, rel_tol=1e-6)


def test_source_match_is_case_insensitive() -> None:
    """``REUTERS`` and ``reuters`` and ``The Reuters Wire`` all match
    the same tier — Alpaca returns inconsistent casing."""
    weights = NewsQualityWeights()
    for source in ("reuters", "REUTERS", "Reuters", "The Reuters Wire"):
        w = quality_weight(_article(source=source), _NOW, weights)
        assert w == 1.0, f"{source!r} should match Reuters tier"


def test_empty_source_falls_back_to_unknown() -> None:
    weights = NewsQualityWeights(unknown_source_weight=0.30)
    w = quality_weight(_article(source=""), _NOW, weights)
    assert math.isclose(w, 0.30, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Combined weight semantics
# ---------------------------------------------------------------------------


def test_combined_weight_is_product_of_recency_and_source() -> None:
    """Half-life-old Benzinga article: 0.5 recency × 0.6 source = 0.30."""
    weights = NewsQualityWeights()
    w = quality_weight(_article(source="Benzinga", hours_ago=24.0), _NOW, weights)
    assert math.isclose(w, 0.30, rel_tol=1e-6)


def test_fresh_high_tier_beats_old_low_tier() -> None:
    """The whole point: a fresh wire-service article should weigh
    more than a day-old blog post."""
    weights = NewsQualityWeights()
    fresh_top = quality_weight(_article(source="Reuters", hours_ago=0.5), _NOW, weights)
    old_blog = quality_weight(_article(source="My Niche Blog", hours_ago=48.0), _NOW, weights)
    assert fresh_top > old_blog


# ---------------------------------------------------------------------------
# Weighted mean
# ---------------------------------------------------------------------------


def test_weighted_mean_collapses_when_all_weights_equal() -> None:
    """Equal weights → weighted mean equals the plain arithmetic mean.
    Backwards compatibility guard: existing unit-weight tests of the
    aggregator still produce the same result."""
    result = weighted_sentiment_mean([(0.8, 0.5), (-0.4, 0.5), (0.0, 0.5)])
    assert math.isclose(result, (0.8 - 0.4 + 0.0) / 3.0, rel_tol=1e-6)


def test_weighted_mean_pulls_toward_higher_weighted_score() -> None:
    """A high-weight bullish article should outweigh a low-weight
    bearish one — the mean tilts toward the bullish side."""
    result = weighted_sentiment_mean([(0.8, 0.9), (-0.8, 0.1)])
    # (0.8 * 0.9 + (-0.8) * 0.1) / (0.9 + 0.1) = (0.72 - 0.08) / 1.0 = 0.64
    assert math.isclose(result, 0.64, rel_tol=1e-6)


def test_weighted_mean_falls_back_to_zero_when_no_weight() -> None:
    """Defensive: empty-weight aggregate returns 0.0 rather than
    raising ZeroDivisionError. The quality_weight floor keeps this
    branch unreachable in practice."""
    assert weighted_sentiment_mean([]) == 0.0
    assert weighted_sentiment_mean([(0.5, 0.0), (-0.3, 0.0)]) == 0.0


def test_single_article_weighted_mean_is_the_score() -> None:
    """One article in, one weight, mean = score regardless of weight.
    Preserves single-article test cases that asserted ``score == confidence``."""
    assert weighted_sentiment_mean([(0.85, 0.5)]) == 0.85
    assert weighted_sentiment_mean([(0.85, 1.0)]) == 0.85
    assert weighted_sentiment_mean([(-0.9, 0.2)]) == -0.9


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------


def test_from_settings_picks_up_recency_tunables() -> None:
    """Half-life + floor come from Settings; source map stays inline.
    Build a tiny stand-in to avoid needing real Settings credentials."""

    class _FakeSettings:
        news_recency_half_life_hours = 6.0
        news_recency_floor_weight = 0.10

    weights = NewsQualityWeights.from_settings(_FakeSettings())  # type: ignore[arg-type]
    assert weights.half_life_hours == 6.0
    assert weights.floor_weight == 0.10
    # Source defaults unchanged.
    assert "reuters" in weights.source_weights
