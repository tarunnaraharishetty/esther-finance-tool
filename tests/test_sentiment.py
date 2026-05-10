"""Sentiment analyzer tests.

The pipeline-loading test is marked ``slow`` because it downloads ~440MB of
FinBERT weights on first run. Skipped by default; run explicitly with::

    pytest -m slow tests/test_sentiment.py
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.data.models import NewsArticle
from src.sentiment.analyzer import SentimentAnalyzer, SentimentLabel, SentimentScore


def test_signed_score_directionality() -> None:
    pos = SentimentScore(SentimentLabel.POSITIVE, 0.8)
    neg = SentimentScore(SentimentLabel.NEGATIVE, 0.7)
    neu = SentimentScore(SentimentLabel.NEUTRAL, 0.9)
    assert pos.signed == 0.8
    assert neg.signed == -0.7
    assert neu.signed == 0.0


def test_score_empty_text_returns_neutral() -> None:
    """Must not invoke the heavy pipeline for empty input."""
    analyzer = SentimentAnalyzer()
    score = analyzer.score_text("   ")
    assert score.label == SentimentLabel.NEUTRAL
    assert score.confidence == 0.0


@pytest.mark.slow
def test_finbert_pipeline_loads_and_scores() -> None:
    """End-to-end: actually load FinBERT and score a finance headline.

    Requires ``transformers`` + ``torch`` installed and ~440MB model download
    on first run. Marked slow so CI/dev runs skip it by default.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    analyzer = SentimentAnalyzer()
    article = NewsArticle(
        id="t1",
        headline="Apple beats earnings expectations, raises guidance",
        summary="Strong iPhone sales drive record quarterly revenue.",
        source="test",
        symbols=["AAPL"],
        published_at=datetime.now(UTC),
    )
    score = analyzer.score_article(article)
    assert isinstance(score, SentimentScore)
    assert score.confidence > 0.0
    # Strong positive headline should not score as negative.
    assert score.label != SentimentLabel.NEGATIVE
