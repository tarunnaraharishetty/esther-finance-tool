"""Tests for the streaming wrapper.

We mock alpaca-py's StockDataStream / NewsDataStream at the AlpacaClient
cached-property boundary so no socket is ever opened. The SDK's blocking
``run()`` is replaced with a sleep so the start/stop lifecycle can be
exercised within the test event loop.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.data.alpaca_client import AlpacaClient
from src.data.streaming import (
    MarketStream,
    _to_domain_bar,
    _to_domain_news,
    _to_domain_quote,
    _to_domain_trade,
)


def _build_stream_mock(blocking: bool = True) -> MagicMock:
    """A MagicMock that mimics StockDataStream.

    ``run()`` blocks until ``stop()`` is called so the start/stop lifecycle
    looks realistic to the wrapper.
    """
    mock = MagicMock()
    # We can't await from sync code; use a threading event instead.
    import threading

    stop_flag = threading.Event()

    def fake_run() -> None:
        if blocking:
            stop_flag.wait(timeout=5.0)

    def fake_stop() -> None:
        stop_flag.set()

    mock.run.side_effect = fake_run
    mock.stop.side_effect = fake_stop
    mock._stop_flag = stop_flag  # exposed for tests
    return mock


def _client_with_streams(market: object | None = None, news: object | None = None) -> AlpacaClient:
    client = AlpacaClient()
    if market is not None:
        client.__dict__["stream"] = market
    if news is not None:
        client.__dict__["news_stream"] = news
    return client


# ---------------------------------------------------------------------------
# domain conversion
# ---------------------------------------------------------------------------


def test_to_domain_bar_from_object() -> None:
    raw = SimpleNamespace(
        symbol="AAPL",
        timestamp=datetime.now(UTC),
        open=150.0,
        high=151.0,
        low=149.5,
        close=150.5,
        volume=1234,
    )
    bar = _to_domain_bar(raw)
    assert bar.symbol == "AAPL"
    assert bar.close == Decimal("150.5")
    assert bar.volume == 1234


def test_to_domain_bar_from_dict() -> None:
    raw = {
        "symbol": "MSFT",
        "timestamp": datetime.now(UTC),
        "open": 420.0,
        "high": 421.0,
        "low": 419.0,
        "close": 420.5,
        "volume": 99,
    }
    bar = _to_domain_bar(raw)
    assert bar.symbol == "MSFT"
    assert bar.close == Decimal("420.5")


def test_to_domain_quote() -> None:
    raw = SimpleNamespace(
        symbol="AAPL",
        timestamp=datetime.now(UTC),
        bid_price=150.0,
        bid_size=100,
        ask_price=150.05,
        ask_size=200,
    )
    q = _to_domain_quote(raw)
    assert q.bid_price == Decimal("150.0")
    assert q.ask_size == 200


def test_to_domain_trade() -> None:
    raw = SimpleNamespace(
        symbol="AAPL",
        timestamp=datetime.now(UTC),
        price=150.25,
        size=50,
    )
    t = _to_domain_trade(raw)
    assert t.price == Decimal("150.25")
    assert t.size == 50


def test_to_domain_news_uses_created_at() -> None:
    raw = SimpleNamespace(
        id=7,
        headline="hello",
        summary="world",
        author="r",
        url="u",
        source="s",
        symbols=["AAPL"],
        created_at=datetime.now(UTC),
        updated_at=None,
    )
    n = _to_domain_news(raw)
    assert n.id == "7"
    assert n.symbols == ["AAPL"]


# ---------------------------------------------------------------------------
# subscription wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_subscribes_handlers_for_each_event_type() -> None:
    market_mock = _build_stream_mock()
    client = _client_with_streams(market=market_mock)
    stream = MarketStream(client=client)

    async def bar_h(_b: object) -> None: ...
    async def quote_h(_q: object) -> None: ...
    async def trade_h(_t: object) -> None: ...

    stream.on_bar(bar_h)
    stream.on_quote(quote_h)
    stream.on_trade(trade_h)
    stream.subscribe(["AAPL", "MSFT"])

    await stream.start()
    try:
        # All three subscribe_* called once with our internal dispatcher
        market_mock.subscribe_bars.assert_called_once()
        market_mock.subscribe_quotes.assert_called_once()
        market_mock.subscribe_trades.assert_called_once()
        # symbols passed positionally after the handler
        bars_args = market_mock.subscribe_bars.call_args.args
        assert set(bars_args[1:]) == {"AAPL", "MSFT"}
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_start_skips_event_types_with_no_handler() -> None:
    market_mock = _build_stream_mock()
    client = _client_with_streams(market=market_mock)
    stream = MarketStream(client=client)

    async def bar_h(_b: object) -> None: ...

    stream.on_bar(bar_h)
    stream.subscribe(["AAPL"])

    await stream.start()
    try:
        market_mock.subscribe_bars.assert_called_once()
        market_mock.subscribe_quotes.assert_not_called()
        market_mock.subscribe_trades.assert_not_called()
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_start_requires_handlers_and_symbols() -> None:
    stream = MarketStream(client=_client_with_streams(market=_build_stream_mock()))
    with pytest.raises(RuntimeError, match="no handlers"):
        await stream.start()

    async def bar_h(_b: object) -> None: ...

    stream.on_bar(bar_h)
    with pytest.raises(RuntimeError, match="no symbols"):
        await stream.start()


@pytest.mark.asyncio
async def test_double_start_raises() -> None:
    market_mock = _build_stream_mock()
    client = _client_with_streams(market=market_mock)
    stream = MarketStream(client=client)

    async def bar_h(_b: object) -> None: ...

    stream.on_bar(bar_h)
    stream.subscribe(["AAPL"])
    await stream.start()
    try:
        with pytest.raises(RuntimeError, match="twice"):
            await stream.start()
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_stop_is_idempotent_when_not_started() -> None:
    stream = MarketStream(client=_client_with_streams(market=_build_stream_mock()))
    await stream.stop()  # should not raise


# ---------------------------------------------------------------------------
# dispatch / fan-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_fans_out_to_all_handlers() -> None:
    market_mock = _build_stream_mock()
    client = _client_with_streams(market=market_mock)
    stream = MarketStream(client=client)

    seen: list[str] = []

    async def h1(b: object) -> None:
        seen.append(f"h1:{b.symbol}")

    async def h2(b: object) -> None:
        seen.append(f"h2:{b.symbol}")

    stream.on_bar(h1)
    stream.on_bar(h2)
    stream.subscribe(["AAPL"])

    raw = SimpleNamespace(
        symbol="AAPL",
        timestamp=datetime.now(UTC),
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10,
    )
    await stream._dispatch_bar(raw)
    assert seen == ["h1:AAPL", "h2:AAPL"]


@pytest.mark.asyncio
async def test_dispatch_swallows_handler_errors() -> None:
    """One failing handler must not stop the others or kill the stream."""
    stream = MarketStream(client=_client_with_streams(market=_build_stream_mock()))
    seen: list[str] = []

    async def boom(_b: object) -> None:
        raise ValueError("boom")

    async def ok(b: object) -> None:
        seen.append(b.symbol)

    stream.on_bar(boom)
    stream.on_bar(ok)

    raw = SimpleNamespace(
        symbol="AAPL",
        timestamp=datetime.now(UTC),
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10,
    )
    # Should not raise
    await stream._dispatch_bar(raw)
    assert seen == ["AAPL"]


# ---------------------------------------------------------------------------
# news stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_news_stream_only_started_when_enabled() -> None:
    market_mock = _build_stream_mock()
    news_mock = _build_stream_mock()
    client = _client_with_streams(market=market_mock, news=news_mock)

    # include_news=False (default): news stream untouched even with handler
    stream = MarketStream(client=client)

    async def bar_h(_b: object) -> None: ...
    async def news_h(_n: object) -> None: ...

    stream.on_bar(bar_h)
    stream.on_news(news_h)
    stream.subscribe(["AAPL"])
    stream.subscribe_news(["AAPL"])

    await stream.start()
    try:
        news_mock.subscribe_news.assert_not_called()
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_news_stream_started_when_enabled() -> None:
    market_mock = _build_stream_mock()
    news_mock = _build_stream_mock()
    client = _client_with_streams(market=market_mock, news=news_mock)

    stream = MarketStream(client=client, include_news=True)

    async def bar_h(_b: object) -> None: ...
    async def news_h(_n: object) -> None: ...

    stream.on_bar(bar_h)
    stream.on_news(news_h)
    stream.subscribe(["AAPL"])
    stream.subscribe_news(["AAPL"])

    await stream.start()
    try:
        news_mock.subscribe_news.assert_called_once()
    finally:
        await stream.stop()
