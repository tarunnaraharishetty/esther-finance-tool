"""SQLAlchemy ORM models for persisted data.

Tables:
- ``bars``            OHLCV bars, composite PK (symbol, timeframe, timestamp).
- ``news_articles``   News articles, PK on provider id.
- ``news_symbols``    Many-to-many link between articles and symbols.
- ``sentiment_scores`` Per-(article, model) sentiment label + confidence.

Quote/Trade ticks are not persisted here — they are typically too high-volume
for a relational store and should be streamed or written to a TSDB instead.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.data.storage import Base

# Prices: 18 digits, 6 decimals — fits equities + most crypto sub-cent precision.
_PRICE = Numeric(18, 6)


class BarORM(Base):
    __tablename__ = "bars"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(8), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)

    open: Mapped[Decimal] = mapped_column(_PRICE, nullable=False)
    high: Mapped[Decimal] = mapped_column(_PRICE, nullable=False)
    low: Mapped[Decimal] = mapped_column(_PRICE, nullable=False)
    close: Mapped[Decimal] = mapped_column(_PRICE, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        # Common query pattern: "give me bars for AAPL, ordered by time".
        Index("ix_bars_symbol_ts", "symbol", "timestamp"),
    )


class NewsArticleORM(Base):
    __tablename__ = "news_articles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    author: Mapped[str | None] = mapped_column(String(255))
    url: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    symbols: Mapped[list["NewsSymbolORM"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    sentiment: Mapped[list["SentimentScoreORM"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_news_published_at", "published_at"),
        Index("ix_news_source", "source"),
    )


class NewsSymbolORM(Base):
    __tablename__ = "news_symbols"

    news_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("news_articles.id", ondelete="CASCADE"), primary_key=True
    )
    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)

    article: Mapped[NewsArticleORM] = relationship(back_populates="symbols")

    __table_args__ = (
        # Strategies query "all news for AAPL ordered by time" — index on
        # (symbol, news_id) supports the join from this side.
        Index("ix_news_symbols_symbol", "symbol"),
    )


class SentimentScoreORM(Base):
    __tablename__ = "sentiment_scores"

    # Integer (not BigInteger) so SQLite uses it as a rowid alias and
    # auto-increments. Postgres maps Integer to INTEGER which is fine for
    # this volume; bump to BigInteger if scores ever exceed 2^31.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    news_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("news_articles.id", ondelete="CASCADE"), nullable=False
    )
    # Model name like "ProsusAI/finbert" — keep alongside score so we can
    # re-score articles with a new model and compare without losing history.
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    label: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    article: Mapped[NewsArticleORM] = relationship(back_populates="sentiment")

    __table_args__ = (
        UniqueConstraint("news_id", "model", name="uq_sentiment_news_model"),
        Index("ix_sentiment_news", "news_id"),
    )
