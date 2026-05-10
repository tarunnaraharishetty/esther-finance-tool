"""News sentiment analyzer (FinBERT by default).

Lazy-loads the HuggingFace pipeline on first use to keep startup fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING

from src.config import Settings, get_settings
from src.data.models import NewsArticle
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import Pipeline

log = get_logger(__name__)


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class SentimentScore:
    label: SentimentLabel
    confidence: float

    @property
    def signed(self) -> float:
        """+confidence for positive, -confidence for negative, 0 for neutral."""
        if self.label == SentimentLabel.POSITIVE:
            return self.confidence
        if self.label == SentimentLabel.NEGATIVE:
            return -self.confidence
        return 0.0


class SentimentAnalyzer:
    """Wraps a HuggingFace text-classification pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @cached_property
    def _pipeline(self) -> Pipeline:
        from transformers import pipeline

        log.info(
            "sentiment.loading_model",
            model=self._settings.sentiment_model,
            device=self._settings.sentiment_device.value,
        )
        return pipeline(
            "text-classification",
            model=self._settings.sentiment_model,
            device=self._settings.sentiment_device.value,
        )

    def score_text(self, text: str) -> SentimentScore:
        if not text.strip():
            return SentimentScore(SentimentLabel.NEUTRAL, 0.0)
        result = self._pipeline(text)[0]
        label_raw = str(result["label"]).lower()
        label = SentimentLabel(label_raw) if label_raw in SentimentLabel else SentimentLabel.NEUTRAL
        return SentimentScore(label=label, confidence=float(result["score"]))

    def score_article(self, article: NewsArticle) -> SentimentScore:
        text = f"{article.headline}. {article.summary}".strip(". ")
        return self.score_text(text)
