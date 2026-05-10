"""News ingestion across providers (Alpaca News / NewsAPI / Benzinga)."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime

from src.config import NewsProvider, get_settings
from src.data.alpaca_client import AlpacaClient
from src.data.models import NewsArticle
from src.utils.logging import get_logger
from src.utils.rate_limiter import with_async_retry

log = get_logger(__name__)


class NewsSource(ABC):
    """Provider-agnostic news source."""

    @abstractmethod
    async def fetch(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        limit: int = 50,
    ) -> list[NewsArticle]: ...


class AlpacaNewsSource(NewsSource):
    def __init__(self, client: AlpacaClient | None = None) -> None:
        self.client = client or AlpacaClient()

    @with_async_retry(attempts=3, min_wait=1.0, max_wait=10.0)
    async def fetch(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        limit: int = 50,
    ) -> list[NewsArticle]:
        await self.client.acquire_quota()

        from alpaca.data.requests import NewsRequest

        # Alpaca expects a comma-separated symbols string.
        request = NewsRequest(
            symbols=",".join(symbols) if symbols else None,
            start=start,
            end=end,
            limit=limit,
            include_content=False,
            exclude_contentless=True,
        )
        result = await asyncio.to_thread(self.client.news.get_news, request)
        raw_news = getattr(result, "news", None) or getattr(result, "data", []) or []
        return [_article_from_alpaca(n) for n in raw_news]


def get_news_source() -> NewsSource:
    """Factory: returns the configured provider."""
    provider = get_settings().news_provider
    if provider == NewsProvider.ALPACA:
        return AlpacaNewsSource()
    raise NotImplementedError(f"news provider {provider} not yet implemented")


def _article_from_alpaca(raw: object) -> NewsArticle:
    return NewsArticle(
        id=str(getattr(raw, "id")),
        headline=str(getattr(raw, "headline", "")),
        summary=str(getattr(raw, "summary", "") or ""),
        author=getattr(raw, "author", None),
        url=getattr(raw, "url", None),
        source=str(getattr(raw, "source", "alpaca")),
        symbols=list(getattr(raw, "symbols", []) or []),
        published_at=getattr(raw, "created_at"),
        updated_at=getattr(raw, "updated_at", None),
    )
