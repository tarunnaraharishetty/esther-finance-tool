"""Tests for the SQLAlchemy storage layer.

Each test gets a fresh SQLite file under tmp_path. We override DATABASE_URL,
clear the cached engine + settings, and call init_db() to materialise the
schema on the new database.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from src.data.models import Bar, NewsArticle, TimeFrame
from src.data.repositories import (
    BarRepository,
    NewsRepository,
    SentimentRepository,
)
from src.data.storage import init_db, reset_engine, session_scope
from src.sentiment.analyzer import SentimentLabel, SentimentScore


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Per-test SQLite file. Resets engine + settings caches around the test."""
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")

    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    reset_engine()
    init_db()
    try:
        yield
    finally:
        reset_engine()
        settings_mod.get_settings.cache_clear()


def _bar(symbol: str, ts: datetime, close: float = 100.0) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=ts,
        open=Decimal(str(close - 1)),
        high=Decimal(str(close + 1)),
        low=Decimal(str(close - 2)),
        close=Decimal(str(close)),
        volume=1_000,
        timeframe=TimeFrame.DAY_1,
    )


# ---------------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------------


def test_bar_upsert_and_query(db: None) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [_bar("AAPL", base + timedelta(days=i), close=100 + i) for i in range(5)]

    with session_scope() as s:
        repo = BarRepository(s)
        inserted = repo.upsert(bars)
        assert inserted == 5

    with session_scope() as s:
        repo = BarRepository(s)
        out = repo.get("AAPL", TimeFrame.DAY_1, base, base + timedelta(days=4))
        assert len(out) == 5
        assert [b.close for b in out] == [Decimal(str(100 + i)) for i in range(5)]
        # Ordered ascending by timestamp.
        assert all(out[i].timestamp <= out[i + 1].timestamp for i in range(4))


def test_bar_upsert_duplicates_ignored(db: None) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [_bar("AAPL", base, close=100), _bar("AAPL", base, close=999)]

    with session_scope() as s:
        repo = BarRepository(s)
        repo.upsert(bars[:1])

    # Re-upsert with same PK + a different close: should be ignored, not updated.
    with session_scope() as s:
        repo = BarRepository(s)
        repo.upsert(bars)

    with session_scope() as s:
        repo = BarRepository(s)
        out = repo.get("AAPL", TimeFrame.DAY_1, base, base)
        assert len(out) == 1
        assert out[0].close == Decimal("100")


def test_bar_latest(db: None) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [_bar("MSFT", base + timedelta(days=i), close=400 + i) for i in range(3)]

    with session_scope() as s:
        BarRepository(s).upsert(bars)

    with session_scope() as s:
        latest = BarRepository(s).latest("MSFT", TimeFrame.DAY_1)
        assert latest is not None
        assert latest.close == Decimal("402")


def test_bar_latest_missing_returns_none(db: None) -> None:
    with session_scope() as s:
        assert BarRepository(s).latest("NOPE", TimeFrame.DAY_1) is None


def test_bar_get_range_excludes_outside_window(db: None) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [_bar("AAPL", base + timedelta(days=i), close=100 + i) for i in range(10)]

    with session_scope() as s:
        BarRepository(s).upsert(bars)

    with session_scope() as s:
        out = BarRepository(s).get(
            "AAPL", TimeFrame.DAY_1, base + timedelta(days=2), base + timedelta(days=5)
        )
        assert len(out) == 4
        assert out[0].timestamp == base + timedelta(days=2)
        assert out[-1].timestamp == base + timedelta(days=5)


def test_bar_upsert_empty_is_noop(db: None) -> None:
    with session_scope() as s:
        assert BarRepository(s).upsert([]) == 0


# ---------------------------------------------------------------------------
# news
# ---------------------------------------------------------------------------


def _article(id_: str, symbols: list[str], ts: datetime, headline: str = "h") -> NewsArticle:
    return NewsArticle(
        id=id_,
        headline=headline,
        summary="s",
        source="alpaca",
        symbols=symbols,
        published_at=ts,
    )


def test_news_upsert_and_query_by_symbol(db: None) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    articles = [
        _article("a1", ["AAPL", "MSFT"], base, "Apple beats"),
        _article("a2", ["AAPL"], base + timedelta(hours=1), "iPhone launch"),
        _article("a3", ["TSLA"], base + timedelta(hours=2), "Cybertruck news"),
    ]

    with session_scope() as s:
        NewsRepository(s).upsert(articles)

    with session_scope() as s:
        repo = NewsRepository(s)
        aapl = repo.get_for_symbol("AAPL", base - timedelta(days=1), base + timedelta(days=1))
        assert {a.id for a in aapl} == {"a1", "a2"}
        # Ordered desc by published_at.
        assert aapl[0].published_at > aapl[1].published_at

        tsla = repo.get_for_symbol("TSLA", base - timedelta(days=1), base + timedelta(days=1))
        assert [a.id for a in tsla] == ["a3"]


def test_news_upsert_overwrites_existing(db: None) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)

    with session_scope() as s:
        NewsRepository(s).upsert([_article("x1", ["AAPL"], base, "Old headline")])

    with session_scope() as s:
        NewsRepository(s).upsert(
            [_article("x1", ["AAPL", "MSFT"], base, "New headline")]
        )

    with session_scope() as s:
        repo = NewsRepository(s)
        article = repo.get_by_id("x1")
        assert article is not None
        assert article.headline == "New headline"
        # Symbol set was replaced, not appended.
        assert set(article.symbols) == {"AAPL", "MSFT"}


def test_news_get_by_id_missing(db: None) -> None:
    with session_scope() as s:
        assert NewsRepository(s).get_by_id("nope") is None


# ---------------------------------------------------------------------------
# sentiment
# ---------------------------------------------------------------------------


def test_sentiment_upsert_and_get(db: None) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)

    with session_scope() as s:
        NewsRepository(s).upsert([_article("n1", ["AAPL"], ts)])

    with session_scope() as s:
        SentimentRepository(s).upsert(
            "n1", "ProsusAI/finbert", SentimentScore(SentimentLabel.POSITIVE, 0.92)
        )

    with session_scope() as s:
        score = SentimentRepository(s).get("n1", "ProsusAI/finbert")
        assert score is not None
        assert score.label == SentimentLabel.POSITIVE
        assert score.confidence == pytest.approx(0.92)


def test_sentiment_rescore_overwrites_same_model(db: None) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    with session_scope() as s:
        NewsRepository(s).upsert([_article("n2", ["AAPL"], ts)])
        SentimentRepository(s).upsert(
            "n2", "finbert", SentimentScore(SentimentLabel.NEGATIVE, 0.6)
        )

    with session_scope() as s:
        SentimentRepository(s).upsert(
            "n2", "finbert", SentimentScore(SentimentLabel.POSITIVE, 0.8)
        )

    with session_scope() as s:
        score = SentimentRepository(s).get("n2", "finbert")
        assert score is not None
        assert score.label == SentimentLabel.POSITIVE
        assert score.confidence == pytest.approx(0.8)


def test_sentiment_different_models_coexist(db: None) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    with session_scope() as s:
        NewsRepository(s).upsert([_article("n3", ["AAPL"], ts)])
        repo = SentimentRepository(s)
        repo.upsert("n3", "finbert", SentimentScore(SentimentLabel.POSITIVE, 0.7))
        repo.upsert("n3", "roberta", SentimentScore(SentimentLabel.NEGATIVE, 0.6))

    with session_scope() as s:
        repo = SentimentRepository(s)
        a = repo.get("n3", "finbert")
        b = repo.get("n3", "roberta")
        assert a is not None and a.label == SentimentLabel.POSITIVE
        assert b is not None and b.label == SentimentLabel.NEGATIVE


def test_news_cascade_deletes_sentiment(db: None) -> None:
    """Deleting a news article should cascade to its sentiment + symbols."""
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    with session_scope() as s:
        NewsRepository(s).upsert([_article("n4", ["AAPL"], ts)])
        SentimentRepository(s).upsert(
            "n4", "finbert", SentimentScore(SentimentLabel.POSITIVE, 0.5)
        )

    with session_scope() as s:
        from src.data.orm import NewsArticleORM

        article = s.get(NewsArticleORM, "n4")
        assert article is not None
        s.delete(article)

    with session_scope() as s:
        assert NewsRepository(s).get_by_id("n4") is None
        assert SentimentRepository(s).get("n4", "finbert") is None
