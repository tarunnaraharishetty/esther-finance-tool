"""News sentiment analyzer (FinBERT by default).

Lazy-loads the HuggingFace pipeline on first use to keep startup fast.
Per-tick fan-out (500 symbols × 20 articles = 10k texts) goes through
:meth:`SentimentAnalyzer.score_texts`, which runs a single batched
forward pass per ``_MAX_BATCH_SIZE`` chunk instead of one inference per
text. The single-text :meth:`score_text` and per-article
:meth:`score_article` wrap the batched API so a subclass override of
``score_texts`` propagates everywhere — see B-15 in BUGS.md.
"""

from __future__ import annotations

from collections.abc import Sequence
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


# Articles per forward pass. FinBERT's encoder runs on ~512 token
# inputs; 32 inputs at once is the sweet spot between throughput and
# memory on CPU + a single mid-range GPU. Larger batches don't help on
# CPU (it serializes anyway) and risk OOM on GPU; smaller batches
# leave throughput on the table.
_MAX_BATCH_SIZE = 32


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

    def score_texts(self, texts: Sequence[str]) -> list[SentimentScore]:
        """Score every text in ``texts``; return one SentimentScore per input.

        Closes B-15: instead of N forward passes (one per text), this
        runs ``ceil(N / _MAX_BATCH_SIZE)`` batched passes. With 500
        symbols × 20 articles the savings are dominant — FinBERT on CPU
        spends most of its budget on tokenization + the model's first
        attention layer, both of which amortize across a batch.

        Empty inputs (whitespace-only strings) are returned as neutral
        sentinels without consulting the model so an article with no
        body doesn't waste a slot in the batch. The output list is
        always the same length and order as ``texts``.
        """
        if not texts:
            return []
        # Pre-fill output with the neutral sentinel; we'll overwrite
        # the non-empty slots with real model output below. This keeps
        # the response in input order regardless of how many empty
        # entries were skipped over.
        results: list[SentimentScore] = [
            SentimentScore(SentimentLabel.NEUTRAL, 0.0) for _ in texts
        ]
        nonempty_indices: list[int] = []
        nonempty_texts: list[str] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                nonempty_indices.append(i)
                nonempty_texts.append(text)
        if not nonempty_texts:
            return results
        # Run the pipeline batch-by-batch. The transformers pipeline
        # natively accepts ``list[str]`` and returns a parallel list,
        # which is what we want — one forward pass per chunk.
        for start in range(0, len(nonempty_texts), _MAX_BATCH_SIZE):
            chunk = nonempty_texts[start : start + _MAX_BATCH_SIZE]
            raw_batch = self._pipeline(chunk)
            for offset, raw in enumerate(raw_batch):
                output_index = nonempty_indices[start + offset]
                results[output_index] = _decode_pipeline_output(raw)
        return results

    def score_text(self, text: str) -> SentimentScore:
        """Score a single text — thin wrapper over the batched API.

        Kept for callers that genuinely have one text in hand (CLI
        helpers, tests). Hot per-tick paths should use
        :meth:`score_texts` directly so the batching kicks in.
        """
        return self.score_texts([text])[0]

    def score_articles(
        self, articles: Sequence[NewsArticle]
    ) -> list[SentimentScore]:
        """Batched per-article variant of :meth:`score_article`.

        Composes each article's headline + summary into one text and
        passes the whole sequence through :meth:`score_texts` for a
        single batched forward pass per chunk. The hot caller in
        :func:`~src.strategy.recommendation.RecommendationEngine._score_news`
        uses this.
        """
        texts = [
            f"{article.headline}. {article.summary}".strip(". ")
            for article in articles
        ]
        return self.score_texts(texts)

    def score_article(self, article: NewsArticle) -> SentimentScore:
        """Single-article wrapper over the batched API. Same back-compat
        contract as :meth:`score_text`."""
        return self.score_articles([article])[0]


def _decode_pipeline_output(raw: dict[str, object] | object) -> SentimentScore:
    """Coerce one pipeline result row into a :class:`SentimentScore`.

    The transformers pipeline returns ``[{"label": str, "score":
    float}]`` per input; chunking gives us ``list[dict]``. Pulled out
    so the batched path and the (potential) future single-text fast
    path share one decode shape.
    """
    if not isinstance(raw, dict):
        return SentimentScore(SentimentLabel.NEUTRAL, 0.0)
    label_raw = str(raw.get("label", "")).lower()
    label = (
        SentimentLabel(label_raw)
        if label_raw in SentimentLabel
        else SentimentLabel.NEUTRAL
    )
    raw_score = raw.get("score", 0.0)
    try:
        # ``raw`` is a dict[str, object] in our annotation; the pipeline
        # actually returns floats here but mypy doesn't know that, so
        # cast through str first which accepts any object.
        confidence = float(raw_score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        confidence = 0.0
    return SentimentScore(label=label, confidence=confidence)
