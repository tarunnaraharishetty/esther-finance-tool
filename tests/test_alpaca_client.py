"""Tests for the Alpaca data layer.

Mocks alpaca-py at the cached-property boundary on AlpacaClient so we never
hit the network. Each test patches the relevant subclient via
monkeypatch.setattr on the AlpacaClient instance.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.data.alpaca_client import AlpacaClient
from src.data.market_data import MarketDataService
from src.data.models import TimeFrame
from src.data.news_ingestion import AlpacaNewsSource


def _stub_client(monkeypatch: pytest.MonkeyPatch, **subclients: object) -> AlpacaClient:
    """Build an AlpacaClient with subclients pre-stubbed to MagicMocks.

    Bypasses the cached_property network setup entirely by writing the value
    into the instance ``__dict__`` directly.
    """
    client = AlpacaClient()
    for name, value in subclients.items():
        client.__dict__[name] = value
    return client


# ---------------------------------------------------------------------------
# market data
# ---------------------------------------------------------------------------


def _fake_alpaca_bar(close: float = 100.0) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=datetime.now(UTC),
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1_000_000,
    )


@pytest.mark.asyncio
async def test_get_bars_returns_domain_models(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_bars = {"AAPL": [_fake_alpaca_bar(150.0), _fake_alpaca_bar(151.0)]}
    bar_set = SimpleNamespace(data=raw_bars)
    market_mock = MagicMock()
    market_mock.get_stock_bars.return_value = bar_set
    client = _stub_client(monkeypatch, market_data=market_mock)

    svc = MarketDataService(client=client)
    bars = await svc.get_bars(
        symbols=["AAPL"],
        timeframe=TimeFrame.DAY_1,
        start=datetime.now(UTC) - timedelta(days=2),
        end=datetime.now(UTC),
    )

    assert len(bars) == 2
    assert bars[0].symbol == "AAPL"
    assert bars[0].close == Decimal("150.0")
    assert bars[1].close == Decimal("151.0")
    assert bars[0].timeframe == TimeFrame.DAY_1
    assert market_mock.get_stock_bars.call_count == 1


@pytest.mark.asyncio
async def test_get_bars_empty_symbols_skips_call(monkeypatch: pytest.MonkeyPatch) -> None:
    market_mock = MagicMock()
    client = _stub_client(monkeypatch, market_data=market_mock)
    svc = MarketDataService(client=client)
    bars = await svc.get_bars(
        symbols=[],
        timeframe=TimeFrame.DAY_1,
        start=datetime.now(UTC),
        end=datetime.now(UTC),
    )
    assert bars == []
    market_mock.get_stock_bars.assert_not_called()


@pytest.mark.asyncio
async def test_get_latest_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    market_mock = MagicMock()
    market_mock.get_stock_latest_bar.return_value = {"MSFT": _fake_alpaca_bar(420.0)}
    client = _stub_client(monkeypatch, market_data=market_mock)

    svc = MarketDataService(client=client)
    bar = await svc.get_latest_bar("MSFT")
    assert bar.symbol == "MSFT"
    assert bar.close == Decimal("420.0")


def test_to_dataframe_handles_empty() -> None:
    svc = MarketDataService(client=AlpacaClient())
    df = svc.to_dataframe([])
    assert df.empty
    assert "close" in df.columns


# ---------------------------------------------------------------------------
# news
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_news_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = SimpleNamespace(
        id=42,
        headline="AAPL beats earnings",
        summary="Strong iPhone sales.",
        author="reporter",
        url="https://example.com/article",
        source="benzinga",
        symbols=["AAPL"],
        created_at=datetime.now(UTC),
        updated_at=None,
    )
    response = SimpleNamespace(news=[raw])
    news_mock = MagicMock()
    news_mock.get_news.return_value = response
    client = _stub_client(monkeypatch, news=news_mock)

    src = AlpacaNewsSource(client=client)
    articles = await src.fetch(
        symbols=["AAPL"],
        start=datetime.now(UTC) - timedelta(days=1),
        end=datetime.now(UTC),
        limit=10,
    )

    assert len(articles) == 1
    a = articles[0]
    assert a.id == "42"
    assert a.headline == "AAPL beats earnings"
    assert a.symbols == ["AAPL"]

    request = news_mock.get_news.call_args.args[0]
    assert request.symbols == "AAPL"
    assert request.limit == 10


@pytest.mark.asyncio
async def test_news_fetch_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    news_mock = MagicMock()
    news_mock.get_news.return_value = SimpleNamespace(news=[])
    client = _stub_client(monkeypatch, news=news_mock)

    src = AlpacaNewsSource(client=client)
    articles = await src.fetch(
        symbols=["AAPL"],
        start=datetime.now(UTC) - timedelta(days=1),
        end=datetime.now(UTC),
    )
    assert articles == []
