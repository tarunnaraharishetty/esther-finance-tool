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


# ---------------------------------------------------------------------------
# B-15: batched inference replaces the per-article serial loop
# ---------------------------------------------------------------------------


class _FakePipeline:
    """Stand-in for the transformers text-classification pipeline.

    Records every input batch it sees so tests can assert on the
    shape of the calls (one call per ``_MAX_BATCH_SIZE`` chunk vs.
    one call per article — that's the B-15 contract).
    """

    def __init__(
        self, outputs: dict[str, dict[str, str | float]] | None = None
    ) -> None:
        self.calls: list[list[str]] = []
        self._outputs = outputs or {}

    def __call__(self, inputs: list[str]) -> list[dict[str, str | float]]:
        # Real pipeline accepts list[str] and returns list[dict]; that's
        # the shape our batched code targets.
        self.calls.append(list(inputs))
        return [
            self._outputs.get(t, {"label": "neutral", "score": 0.5})
            for t in inputs
        ]


def _analyzer_with_fake(
    outputs: dict[str, dict[str, str | float]] | None = None,
) -> tuple[SentimentAnalyzer, _FakePipeline]:
    """Build an analyzer whose ``_pipeline`` descriptor is replaced by
    a fake. Setting the attribute on the instance bypasses the
    ``cached_property`` descriptor in the class."""
    analyzer = SentimentAnalyzer()
    fake = _FakePipeline(outputs)
    analyzer.__dict__["_pipeline"] = fake
    return analyzer, fake


def test_score_texts_empty_input_returns_empty() -> None:
    analyzer, fake = _analyzer_with_fake()
    assert analyzer.score_texts([]) == []
    assert fake.calls == []  # no model call for empty input


def test_score_texts_returns_one_per_input_in_order() -> None:
    """Output shape matches input shape exactly; order preserved."""
    outputs = {
        "good news for AAPL": {"label": "positive", "score": 0.91},
        "bad news for MSFT": {"label": "negative", "score": 0.77},
    }
    analyzer, _ = _analyzer_with_fake(outputs)
    out = analyzer.score_texts(["good news for AAPL", "bad news for MSFT"])
    assert len(out) == 2
    assert out[0].label == SentimentLabel.POSITIVE
    assert out[0].confidence == pytest.approx(0.91)
    assert out[1].label == SentimentLabel.NEGATIVE
    assert out[1].confidence == pytest.approx(0.77)


def test_score_texts_skips_empty_strings_without_calling_pipeline() -> None:
    """Whitespace-only inputs become neutral sentinels without consuming
    a slot in the batched pass — so a sparse article set doesn't waste
    GPU/CPU budget on empty bodies."""
    analyzer, fake = _analyzer_with_fake(
        {"real text": {"label": "positive", "score": 0.6}}
    )
    out = analyzer.score_texts(["   ", "real text", "", "\n\t"])
    # Output length preserved; non-empty slots resolved via the model.
    assert len(out) == 4
    assert out[0].label == SentimentLabel.NEUTRAL
    assert out[0].confidence == 0.0
    assert out[1].label == SentimentLabel.POSITIVE
    assert out[2].label == SentimentLabel.NEUTRAL
    assert out[3].label == SentimentLabel.NEUTRAL
    # Pipeline was called once with just the non-empty input.
    assert fake.calls == [["real text"]]


def test_score_texts_chunks_at_batch_cap() -> None:
    """B-15 contract: 70 inputs run in ceil(70 / 32) = 3 forward
    passes, NOT 70. Bounds the GPU/CPU memory footprint per pass."""
    from src.sentiment.analyzer import _MAX_BATCH_SIZE

    analyzer, fake = _analyzer_with_fake()
    inputs = [f"text {i}" for i in range(70)]
    analyzer.score_texts(inputs)
    # Exactly 3 batched calls (32 + 32 + 6).
    assert len(fake.calls) == 3
    assert len(fake.calls[0]) == _MAX_BATCH_SIZE
    assert len(fake.calls[1]) == _MAX_BATCH_SIZE
    assert len(fake.calls[2]) == 70 - 2 * _MAX_BATCH_SIZE
    # And the concatenated batches must equal the original input.
    seen = [t for batch in fake.calls for t in batch]
    assert seen == inputs


def test_score_text_delegates_to_batched_api() -> None:
    """Single-text wrapper goes through the batched pipeline with one
    element — uniform contract, no separate code path."""
    analyzer, fake = _analyzer_with_fake(
        {"hello": {"label": "neutral", "score": 0.5}}
    )
    analyzer.score_text("hello")
    assert fake.calls == [["hello"]]


def test_score_articles_runs_one_batched_pass_for_small_input() -> None:
    """The hot caller in RecommendationEngine._score_news passes a
    list of articles; under the cap, that's one forward pass."""
    analyzer, fake = _analyzer_with_fake()
    articles = [
        NewsArticle(
            id=f"a{i}",
            headline=f"headline {i}",
            summary=f"body {i}",
            source="test",
            symbols=["X"],
            published_at=datetime.now(UTC),
        )
        for i in range(5)
    ]
    out = analyzer.score_articles(articles)
    assert len(out) == 5
    assert len(fake.calls) == 1  # one batched call, not five
    assert len(fake.calls[0]) == 5


def test_score_article_single_wraps_batched_api() -> None:
    analyzer, fake = _analyzer_with_fake()
    article = NewsArticle(
        id="solo",
        headline="hello",
        summary="world",
        source="test",
        symbols=["X"],
        published_at=datetime.now(UTC),
    )
    analyzer.score_article(article)
    assert fake.calls == [["hello. world"]]


def test_score_texts_handles_unknown_label_as_neutral() -> None:
    """A model returning an unexpected label (model upgrade, OOV) is
    decoded as neutral rather than crashing — defensive but quiet."""
    analyzer, _ = _analyzer_with_fake(
        {"x": {"label": "ambivalent", "score": 0.42}}
    )
    out = analyzer.score_texts(["x"])
    assert out[0].label == SentimentLabel.NEUTRAL
    assert out[0].confidence == pytest.approx(0.42)
