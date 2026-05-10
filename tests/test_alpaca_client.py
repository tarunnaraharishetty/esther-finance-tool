"""Tests for the Alpaca data + broker layer.

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
from src.execution.broker import (
    AlpacaBroker,
    OrderRequest,
    OrderSide,
    OrderType,
    TimeInForce,
)


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
# broker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_account_equity(monkeypatch: pytest.MonkeyPatch) -> None:
    trading_mock = MagicMock()
    trading_mock.get_account.return_value = SimpleNamespace(
        equity="125000.50", portfolio_value="125000.50"
    )
    client = _stub_client(monkeypatch, trading=trading_mock)

    broker = AlpacaBroker(client=client)
    equity = await broker.get_account_equity()
    assert equity == Decimal("125000.50")
    trading_mock.get_account.assert_called_once()


@pytest.mark.asyncio
async def test_submit_market_order_no_brackets(monkeypatch: pytest.MonkeyPatch) -> None:
    from alpaca.trading.requests import MarketOrderRequest

    trading_mock = MagicMock()
    trading_mock.submit_order.return_value = SimpleNamespace(
        id="order-1", status="accepted", filled_qty="0", filled_avg_price=None
    )
    client = _stub_client(monkeypatch, trading=trading_mock)

    broker = AlpacaBroker(client=client)
    result = await broker.submit(
        OrderRequest(
            symbol="AAPL",
            qty=Decimal("10"),
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
        )
    )

    assert result.id == "order-1"
    assert result.status == "accepted"
    submitted = trading_mock.submit_order.call_args.args[0]
    assert isinstance(submitted, MarketOrderRequest)
    assert submitted.symbol == "AAPL"
    assert submitted.qty == 10.0
    # No bracket params -> order_class left unset (alpaca defaults to simple).
    assert submitted.order_class in (None,) or submitted.order_class.value == "simple"
    assert submitted.take_profit is None
    assert submitted.stop_loss is None


@pytest.mark.asyncio
async def test_submit_market_order_with_bracket(monkeypatch: pytest.MonkeyPatch) -> None:
    from alpaca.trading.enums import OrderClass

    trading_mock = MagicMock()
    trading_mock.submit_order.return_value = SimpleNamespace(
        id="order-2", status="accepted", filled_qty="0", filled_avg_price=None
    )
    client = _stub_client(monkeypatch, trading=trading_mock)

    broker = AlpacaBroker(client=client)
    await broker.submit(
        OrderRequest(
            symbol="MSFT",
            qty=Decimal("5"),
            side=OrderSide.BUY,
            take_profit=Decimal("450.00"),
            stop_loss=Decimal("400.00"),
        )
    )
    submitted = trading_mock.submit_order.call_args.args[0]
    assert submitted.order_class == OrderClass.BRACKET
    assert submitted.take_profit.limit_price == 450.0
    assert submitted.stop_loss.stop_price == 400.0


@pytest.mark.asyncio
async def test_cancel_order(monkeypatch: pytest.MonkeyPatch) -> None:
    trading_mock = MagicMock()
    client = _stub_client(monkeypatch, trading=trading_mock)
    broker = AlpacaBroker(client=client)
    await broker.cancel("abc-123")
    trading_mock.cancel_order_by_id.assert_called_once_with("abc-123")


def test_limit_order_requires_limit_price() -> None:
    broker = AlpacaBroker(client=AlpacaClient())
    with pytest.raises(ValueError, match="limit_price"):
        broker._build_request(
            OrderRequest(
                symbol="AAPL",
                qty=Decimal("1"),
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
            )
        )


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
