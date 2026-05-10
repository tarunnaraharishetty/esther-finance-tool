"""News ingestion across providers (Alpaca News / NewsAPI / Benzinga)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.config import NewsProvider, get_settings
from src.data.alpaca_client import AlpacaClient
from src.data.models import NewsArticle
from src.utils.logging import get_logger

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

    async def fetch(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        limit: int = 50,
    ) -> list[NewsArticle]:
        await self.client.acquire_quota()
        raise NotImplementedError("wire up alpaca-py NewsClient")


def get_news_source() -> NewsSource:
    """Factory: returns the configured provider."""
    provider = get_settings().news_provider
    if provider == NewsProvider.ALPACA:
        return AlpacaNewsSource()
    raise NotImplementedError(f"news provider {provider} not yet implemented")
