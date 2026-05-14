"""Trading recommendation engine.

Combines technical indicator signals (RSI / MACD / Bollinger Bands) with
news sentiment (FinBERT) into a single BUY / HOLD / SELL recommendation
with a confidence score and a human-readable explanation.

Paper-trading only: this module computes recommendations; it does NOT
submit orders. Wiring into execution lives in :mod:`src.execution` and
must run through the risk layer first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from src.data.models import TimeFrame
from src.indicators.bollinger import BollingerBands
from src.indicators.macd import MACD
from src.indicators.rsi import RSI
from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.news_quality import (
    NewsQualityWeights,
    quality_weight,
    weighted_sentiment_mean,
)
from src.strategy.base import RecommendationTier, SignalAction
from src.strategy.multi_timeframe import IntradayRead

if TYPE_CHECKING:
    import pandas as pd

    from src.data.models import NewsArticle
    from src.sentiment.analyzer import SentimentScore


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TradingRecommendation(BaseModel):
    """A single recommendation for one symbol.

    ``action`` is the base 3-tier label (BUY / HOLD / SELL).
    ``tier`` is the 5-tier label surfaced to the trader; defaults to the
    same value as ``action`` and is upgraded to STRONG_BUY / STRONG_SELL
    only when :func:`src.intelligence.tier.promote_to_tier` confirms
    multi-system alignment, stability, and high confidence.

    ``confidence`` is the absolute value of the weighted combined score
    and sits in [0, 1]. ``combined_score`` is signed: positive =
    bullish, negative = bearish.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    action: SignalAction
    confidence: float = Field(ge=0.0, le=1.0)
    combined_score: float = Field(ge=-1.0, le=1.0)
    technical_score: float = Field(ge=-1.0, le=1.0)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    indicator_scores: dict[str, float] = Field(default_factory=dict)
    reasoning: str
    timestamp: datetime
    num_news_articles: int = 0
    # ---- 5-tier surface (filled by promote_to_tier; defaults stay sane
    # for callers that bypass the promoter) ----
    tier: RecommendationTier = RecommendationTier.HOLD
    signal_quality: str = "moderate"  # "high" | "moderate" | "low"
    stability: str = "stable"  # "stable" | "moderate" | "volatile"
    quality_reasons: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class IndicatorWeights:
    """How much each technical sub-signal counts inside the technical score."""

    rsi: float = 1.0
    macd: float = 1.0
    bollinger: float = 1.0


@dataclass
class RecommendationEngine:
    """Produce a :class:`TradingRecommendation` from OHLCV + news.

    Scoring conventions:

    * Every sub-score is normalised to [-1, 1], positive = bullish.
    * Indicator scores are weighted-averaged into ``technical_score``.
    * ``sentiment_score`` is the mean signed FinBERT score over the latest
      ``max_news_articles`` headlines.
    * ``combined_score = technical_weight * tech + sentiment_weight * sent``
      (the two weights need not sum to 1; the result is clipped).
    * Action thresholds: ``BUY`` if combined >= ``buy_threshold``,
      ``SELL`` if combined <= ``-sell_threshold``, else ``HOLD``.
    """

    technical_weight: float = 0.6
    sentiment_weight: float = 0.4
    buy_threshold: float = 0.2
    sell_threshold: float = 0.2
    indicator_weights: IndicatorWeights = field(default_factory=IndicatorWeights)
    max_news_articles: int = 20
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    sentiment_analyzer: SentimentAnalyzer | None = None
    # News quality weighting (recency × source reputation) applied when
    # aggregating per-article FinBERT scores into the per-symbol
    # sentiment_score. Defaults are deliberately conservative — see
    # src.sentiment.news_quality. ``None`` lazily builds defaults so
    # call sites that don't customize sentiment stay one-liner.
    news_quality_weights: NewsQualityWeights | None = None

    # -- top-level entry point ---------------------------------------------

    def recommend(
        self,
        symbol: str,
        df: pd.DataFrame,
        news: list[NewsArticle] | None = None,
        *,
        now: datetime | None = None,
    ) -> TradingRecommendation:
        """Return a recommendation for ``symbol`` given OHLCV + news."""
        ts = now or datetime.now(UTC)
        indicator_scores = self._score_indicators(df)
        technical_score = self._combine_indicator_scores(indicator_scores)

        scored_articles = self._score_news(news or [], now=ts)
        sentiment_score = weighted_sentiment_mean(scored_articles) if scored_articles else 0.0

        combined = _clip(
            self.technical_weight * technical_score + self.sentiment_weight * sentiment_score
        )
        action = self._decide_action(combined)
        confidence = _clip(abs(combined), 0.0, 1.0)
        reasoning = self._build_reasoning(
            symbol=symbol,
            action=action,
            indicator_scores=indicator_scores,
            technical_score=technical_score,
            sentiment_score=sentiment_score,
            combined=combined,
            num_news=len(scored_articles),
        )

        return TradingRecommendation(
            symbol=symbol,
            action=action,
            confidence=confidence,
            combined_score=combined,
            technical_score=_clip(technical_score),
            sentiment_score=_clip(sentiment_score),
            indicator_scores=indicator_scores,
            reasoning=reasoning,
            timestamp=ts,
            num_news_articles=len(scored_articles),
            # Default tier = direct map from action. The promoter in
            # src.intelligence.tier upgrades to STRONG when conditions
            # are met; the engine itself doesn't have history access.
            tier=RecommendationTier.from_action(action),
        )

    def recommend_intraday(
        self,
        symbol: str,
        df: pd.DataFrame,
        *,
        timeframe: TimeFrame,
    ) -> IntradayRead:
        """Technical-only intraday read for one symbol.

        Reuses the daily indicator-scoring path on a different bar
        dataframe — same RSI / MACD / Bollinger math, just on bars at
        ``timeframe``. Sentiment is skipped (news is timeframe-
        agnostic), so ``combined_score == technical_score`` and
        ``confidence == |combined_score|``. The result is meant as
        alignment context for the daily recommendation, not as a
        standalone signal — the dashboard surfaces it as a chip
        beside the daily action.

        Empty / too-thin dataframes return a HOLD with zero scores
        so the renderer can still surface the chip without branching
        on edge cases.
        """
        indicator_scores = self._score_indicators(df)
        technical_score = self._combine_indicator_scores(indicator_scores)
        combined = _clip(technical_score)
        action = self._decide_action(combined)
        confidence = _clip(abs(combined), 0.0, 1.0)
        return IntradayRead(
            timeframe=timeframe,
            action=action,
            confidence=confidence,
            combined_score=combined,
            technical_score=_clip(technical_score),
        )

    # -- technicals --------------------------------------------------------

    def _score_indicators(self, df: pd.DataFrame) -> dict[str, float]:
        if df.empty or "close" not in df.columns:
            return {"rsi": 0.0, "macd": 0.0, "bollinger": 0.0}
        return {
            "rsi": self._score_rsi(df),
            "macd": self._score_macd(df),
            "bollinger": self._score_bollinger(df),
        }

    def _score_rsi(self, df: pd.DataFrame) -> float:
        """Score in [-1, 1]. Oversold → bullish (+), overbought → bearish (-)."""
        series = RSI(period=self.rsi_period).compute(df).dropna()
        if series.empty:
            return 0.0
        latest = float(series.iloc[-1])
        # Linear ramp: at oversold = +1, at overbought = -1, at 50 = 0.
        midpoint = (self.rsi_oversold + self.rsi_overbought) / 2.0
        half_range = (self.rsi_overbought - self.rsi_oversold) / 2.0
        if half_range <= 0:
            return 0.0
        return _clip((midpoint - latest) / half_range)

    def _score_macd(self, df: pd.DataFrame) -> float:
        """Use the MACD line (fast EMA - slow EMA) normalised by price.

        The MACD *line* tracks trend direction; positive ⇒ short-term EMA
        above long-term ⇒ bullish. The histogram is intentionally avoided
        here because in sustained trends it captures *acceleration* and
        flips sign late-trend, which is misleading for a directional read.
        """
        full = MACD().compute_full(df).dropna()
        if full.empty:
            return 0.0
        macd_line = float(full["macd"].iloc[-1])
        price = float(df["close"].iloc[-1])
        if price <= 0:
            return 0.0
        # 2% of price worth of MACD line saturates the signal.
        return _clip(macd_line / (0.02 * price))

    def _score_bollinger(self, df: pd.DataFrame) -> float:
        """Mean-reversion read: below lower band → bullish, above upper → bearish."""
        bands = BollingerBands().compute_full(df).dropna()
        if bands.empty:
            return 0.0
        price = float(df["close"].iloc[-1])
        upper = float(bands["upper"].iloc[-1])
        middle = float(bands["middle"].iloc[-1])
        lower = float(bands["lower"].iloc[-1])
        half_width = (upper - lower) / 2.0
        if half_width <= 0:
            return 0.0
        # +1 at lower band, -1 at upper band, 0 at middle.
        return _clip((middle - price) / half_width)

    def _combine_indicator_scores(self, scores: dict[str, float]) -> float:
        w = self.indicator_weights
        weights = {"rsi": w.rsi, "macd": w.macd, "bollinger": w.bollinger}
        total = sum(weights.values())
        if total <= 0:
            return 0.0
        weighted = sum(scores.get(k, 0.0) * weights[k] for k in weights)
        return _clip(weighted / total)

    # -- sentiment ---------------------------------------------------------

    def _score_news(
        self,
        news: list[NewsArticle],
        *,
        now: datetime,
    ) -> list[tuple[float, float]]:
        """Score each article and pair it with its quality weight.

        Returns ``(signed_score, weight)`` for the latest
        ``max_news_articles`` items, sorted most-recent first. ``now``
        is threaded so the recency-decay component is deterministic
        in tests.
        """
        if not news:
            return []
        ordered = sorted(news, key=lambda a: a.published_at, reverse=True)
        ordered = ordered[: self.max_news_articles]
        analyzer = self.sentiment_analyzer or SentimentAnalyzer()
        weights = self.news_quality_weights or NewsQualityWeights()
        scored: list[tuple[float, float]] = []
        for article in ordered:
            score: SentimentScore = analyzer.score_article(article)
            weight = quality_weight(article, now, weights)
            scored.append((score.signed, weight))
        return scored

    # -- decision + reasoning ---------------------------------------------

    def _decide_action(self, combined: float) -> SignalAction:
        if combined >= self.buy_threshold:
            return SignalAction.BUY
        if combined <= -self.sell_threshold:
            return SignalAction.SELL
        return SignalAction.HOLD

    def _build_reasoning(
        self,
        *,
        symbol: str,
        action: SignalAction,
        indicator_scores: dict[str, float],
        technical_score: float,
        sentiment_score: float,
        combined: float,
        num_news: int,
    ) -> str:
        parts = [
            f"{symbol}: {action.value.upper()} (combined={combined:+.2f})",
            f"technical={technical_score:+.2f} "
            f"[rsi={indicator_scores.get('rsi', 0.0):+.2f}, "
            f"macd={indicator_scores.get('macd', 0.0):+.2f}, "
            f"bollinger={indicator_scores.get('bollinger', 0.0):+.2f}]",
            (
                f"sentiment={sentiment_score:+.2f} over {num_news} article(s)"
                if num_news
                else "sentiment=n/a (no news)"
            ),
        ]
        verdict = {
            SignalAction.BUY: "bullish enough to enter",
            SignalAction.SELL: "bearish enough to exit/short",
            SignalAction.HOLD: "signal below action threshold",
        }[action]
        parts.append(f"verdict: {verdict}.")
        return " | ".join(parts)
