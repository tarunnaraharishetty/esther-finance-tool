"""Repositories: typed query/upsert helpers over the ORM.

Each repository takes an open :class:`Session`. Callers manage the
transaction (typically via :func:`session_scope`).

Upsert semantics:
- :class:`BarRepository` does INSERT-OR-IGNORE — bars are immutable, so
  duplicates from re-fetching the same range are silently dropped.
- :class:`NewsRepository` does INSERT-OR-UPDATE — articles can be edited
  (corrected headlines, added summaries) by the upstream provider.
- :class:`SentimentRepository` is keyed on (news_id, model) so re-scoring
  with the same model overwrites; a new model adds a new row.

Bulk paths use dialect-specific INSERT ... ON CONFLICT for SQLite + Postgres
(the two dialects we plan to ship). Other dialects fall back to per-row
``session.merge()`` — slower but correct.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, delete, select
from sqlalchemy.orm import Session


def _utc(dt: datetime | None) -> datetime | None:
    """Attach UTC tzinfo if missing.

    SQLite stores datetimes as naive ISO strings (it has no native TZ type),
    so values round-tripped through SQLite come back tz-naive. We treat the
    storage layer as UTC-only and re-attach on read.
    """
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)

from src.data.models import Bar, NewsArticle, TimeFrame
from src.data.orm import BarORM, NewsArticleORM, NewsSymbolORM, SentimentScoreORM
from src.sentiment.analyzer import SentimentLabel, SentimentScore
from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------------


class BarRepository:
    """Persistence for OHLCV bars."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, bars: Sequence[Bar]) -> int:
        """Insert bars, ignoring duplicates on (symbol, timeframe, timestamp).

        Returns the number of rows inserted (excludes ignored duplicates).
        """
        if not bars:
            return 0

        rows = [_bar_to_row(b) for b in bars]
        dialect = self.session.bind.dialect.name  # type: ignore[union-attr]

        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            stmt = sqlite_insert(BarORM).values(rows).on_conflict_do_nothing(
                index_elements=["symbol", "timeframe", "timestamp"]
            )
            result = self.session.execute(stmt)
            return result.rowcount or 0

        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = pg_insert(BarORM).values(rows).on_conflict_do_nothing(
                index_elements=["symbol", "timeframe", "timestamp"]
            )
            result = self.session.execute(stmt)
            return result.rowcount or 0

        # Generic fallback: row-by-row merge.
        for row in rows:
            self.session.merge(BarORM(**row))
        return len(rows)

    def get(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        stmt = (
            select(BarORM)
            .where(
                and_(
                    BarORM.symbol == symbol,
                    BarORM.timeframe == timeframe.value,
                    BarORM.timestamp >= start,
                    BarORM.timestamp <= end,
                )
            )
            .order_by(BarORM.timestamp)
        )
        return [_row_to_bar(r) for r in self.session.scalars(stmt)]

    def latest(self, symbol: str, timeframe: TimeFrame) -> Bar | None:
        stmt = (
            select(BarORM)
            .where(BarORM.symbol == symbol, BarORM.timeframe == timeframe.value)
            .order_by(BarORM.timestamp.desc())
            .limit(1)
        )
        row = self.session.scalars(stmt).first()
        return _row_to_bar(row) if row else None


def _bar_to_row(bar: Bar) -> dict[str, Any]:
    return {
        "symbol": bar.symbol,
        "timeframe": bar.timeframe.value,
        "timestamp": bar.timestamp,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }


def _row_to_bar(row: BarORM) -> Bar:
    return Bar(
        symbol=row.symbol,
        timeframe=TimeFrame(row.timeframe),
        timestamp=_utc(row.timestamp),  # type: ignore[arg-type]
        open=Decimal(str(row.open)),
        high=Decimal(str(row.high)),
        low=Decimal(str(row.low)),
        close=Decimal(str(row.close)),
        volume=row.volume,
    )


# ---------------------------------------------------------------------------
# news
# ---------------------------------------------------------------------------


class NewsRepository:
    """Persistence for news articles + their symbol associations."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, articles: Iterable[NewsArticle]) -> int:
        """Insert or update articles. Replaces the symbol set on each.

        Returns the number of articles touched.
        """
        articles = list(articles)
        if not articles:
            return 0

        # Easiest portable upsert: SELECT existing -> overwrite fields, else
        # add. Article volume is low enough (hundreds/day) that this is fine.
        ids = [a.id for a in articles]
        existing = {
            row.id: row
            for row in self.session.scalars(
                select(NewsArticleORM).where(NewsArticleORM.id.in_(ids))
            )
        }

        # Wipe symbol associations for the IDs we're touching; we'll re-insert.
        if existing:
            self.session.execute(
                delete(NewsSymbolORM).where(NewsSymbolORM.news_id.in_(existing.keys()))
            )

        for art in articles:
            row = existing.get(art.id)
            if row is None:
                row = NewsArticleORM(id=art.id)
                self.session.add(row)
            row.headline = art.headline
            row.summary = art.summary or ""
            row.author = art.author
            row.url = art.url
            row.source = art.source
            row.published_at = art.published_at
            row.updated_at = art.updated_at
            for symbol in art.symbols:
                self.session.add(NewsSymbolORM(news_id=art.id, symbol=symbol))

        self.session.flush()
        return len(articles)

    def get_for_symbol(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int = 100,
    ) -> list[NewsArticle]:
        stmt = (
            select(NewsArticleORM)
            .join(NewsSymbolORM, NewsSymbolORM.news_id == NewsArticleORM.id)
            .where(
                NewsSymbolORM.symbol == symbol,
                NewsArticleORM.published_at >= start,
                NewsArticleORM.published_at <= end,
            )
            .order_by(NewsArticleORM.published_at.desc())
            .limit(limit)
        )
        return [_row_to_news(r) for r in self.session.scalars(stmt)]

    def get_by_id(self, news_id: str) -> NewsArticle | None:
        row = self.session.get(NewsArticleORM, news_id)
        return _row_to_news(row) if row else None


def _row_to_news(row: NewsArticleORM) -> NewsArticle:
    return NewsArticle(
        id=row.id,
        headline=row.headline,
        summary=row.summary,
        author=row.author,
        url=row.url,
        source=row.source,
        symbols=[s.symbol for s in row.symbols],
        published_at=_utc(row.published_at),  # type: ignore[arg-type]
        updated_at=_utc(row.updated_at),
    )


# ---------------------------------------------------------------------------
# sentiment
# ---------------------------------------------------------------------------


class SentimentRepository:
    """Persistence for FinBERT/RoBERTa sentiment scores keyed by (news, model)."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, news_id: str, model: str, score: SentimentScore) -> None:
        existing = self.session.scalar(
            select(SentimentScoreORM).where(
                SentimentScoreORM.news_id == news_id,
                SentimentScoreORM.model == model,
            )
        )
        if existing is None:
            self.session.add(
                SentimentScoreORM(
                    news_id=news_id,
                    model=model,
                    label=score.label.value,
                    confidence=score.confidence,
                    scored_at=datetime.now(UTC),
                )
            )
        else:
            existing.label = score.label.value
            existing.confidence = score.confidence
            existing.scored_at = datetime.now(UTC)
        self.session.flush()

    def get(self, news_id: str, model: str) -> SentimentScore | None:
        row = self.session.scalar(
            select(SentimentScoreORM).where(
                SentimentScoreORM.news_id == news_id,
                SentimentScoreORM.model == model,
            )
        )
        if row is None:
            return None
        return SentimentScore(label=SentimentLabel(row.label), confidence=row.confidence)
