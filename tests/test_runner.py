"""Tests for the TradingLoop orchestrator and signal conversion.

All dependencies are mocked — no network, no FinBERT, no Alpaca.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.models import NewsArticle, TimeFrame
from src.execution.broker import OrderResult
from src.runner import (
    TickResult,
    TradingLoop,
    TradingLoopConfig,
    recommendation_to_signal,
)
from src.strategy.base import SignalAction
from src.strategy.recommendation import TradingRecommendation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bars_df(n: int = 30, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rets = rng.normal(loc=0.001, scale=0.01, size=n)
    close = start * np.exp(np.cumsum(rets))
    idx = pd.date_range(end=datetime.now(UTC), periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=int),
        },
        index=idx,
    )


def _make_rec(
    symbol: str = "AAPL",
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.6,
    combined: float = 0.6,
) -> TradingRecommendation:
    return TradingRecommendation(
        symbol=symbol,
        action=action,
        confidence=confidence,
        combined_score=combined,
        technical_score=0.4,
        sentiment_score=0.8,
        indicator_scores={"rsi": 0.1, "macd": 0.5, "bollinger": 0.6},
        reasoning="test reasoning",
        timestamp=datetime.now(UTC),
        num_news_articles=2,
    )


def _build_loop(
    *,
    config: TradingLoopConfig | None = None,
    engine_rec: TradingRecommendation | None = None,
    broker_result: OrderResult | None = None,
    risk_allowed: bool = True,
    bars_df: pd.DataFrame | None = None,
) -> tuple[TradingLoop, dict[str, Any]]:
    """Build a TradingLoop with all deps mocked. Returns (loop, mocks_dict)."""
    config = config or TradingLoopConfig(watchlist=["AAPL"], execute=False)

    market = MagicMock()
    market.get_bars = AsyncMock(return_value=[])
    market.to_dataframe = MagicMock(return_value=bars_df if bars_df is not None else _bars_df())

    news_source = MagicMock()
    news_source.fetch = AsyncMock(return_value=[])

    engine = MagicMock()
    engine.recommend = MagicMock(
        return_value=engine_rec if engine_rec is not None else _make_rec()
    )

    broker = MagicMock()
    broker.get_account_equity = AsyncMock(return_value=Decimal(100_000))
    broker.submit = AsyncMock(
        return_value=broker_result
        if broker_result is not None
        else OrderResult(id="ord-1", status="accepted", filled_qty=Decimal(0), filled_avg_price=None)
    )

    risk_gate = MagicMock()
    from src.risk.exposure import GateResult

    risk_gate.check = MagicMock(
        return_value=GateResult(allowed=risk_allowed, reason="" if risk_allowed else "blocked")
    )

    from src.execution.order_manager import OrderManager

    order_manager = OrderManager(broker=broker, risk_gate=risk_gate)

    loop = TradingLoop(
        config=config,
        engine=engine,
        market=market,
        news_source=news_source,
        order_manager=order_manager,
    )
    return loop, {
        "market": market,
        "news_source": news_source,
        "engine": engine,
        "broker": broker,
        "risk_gate": risk_gate,
    }


# ---------------------------------------------------------------------------
# recommendation_to_signal
# ---------------------------------------------------------------------------


def test_signal_conversion_carries_fields() -> None:
    rec = _make_rec(symbol="TSLA", action=SignalAction.SELL, confidence=0.42, combined=-0.42)
    sig = recommendation_to_signal(rec)
    assert sig.symbol == "TSLA"
    assert sig.action == SignalAction.SELL
    assert sig.confidence == pytest.approx(0.42)
    assert sig.source == "recommendation_engine"
    assert "test reasoning" in sig.rationale
    assert sig.metadata["combined_score"] == pytest.approx(-0.42)
    assert sig.metadata["technical_score"] == pytest.approx(0.4)
    assert sig.metadata["sentiment_score"] == pytest.approx(0.8)
    assert sig.metadata["num_news_articles"] == 2


# ---------------------------------------------------------------------------
# Loop construction guards
# ---------------------------------------------------------------------------


def test_loop_refuses_non_paper_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")  # live URL
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="non-paper"):
        _build_loop()
    settings_mod.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Tick behavior
# ---------------------------------------------------------------------------


async def test_tick_returns_recommendation_per_symbol() -> None:
    cfg = TradingLoopConfig(watchlist=["AAPL", "MSFT"], execute=False)
    loop, mocks = _build_loop(config=cfg)
    results = await loop.tick()
    assert [r.symbol for r in results] == ["AAPL", "MSFT"]
    assert all(r.recommendation is not None for r in results)


async def test_dry_run_does_not_call_broker() -> None:
    cfg = TradingLoopConfig(watchlist=["AAPL"], execute=False)
    loop, mocks = _build_loop(config=cfg)
    results = await loop.tick()
    assert results[0].order is None
    assert results[0].note == "dry-run"
    mocks["broker"].submit.assert_not_called()


async def test_execute_mode_submits_buy_order() -> None:
    cfg = TradingLoopConfig(watchlist=["AAPL"], execute=True, min_confidence=0.0)
    rec = _make_rec(action=SignalAction.BUY, confidence=0.7, combined=0.7)
    loop, mocks = _build_loop(config=cfg, engine_rec=rec)
    results = await loop.tick()
    assert results[0].order is not None
    assert results[0].order.id == "ord-1"
    mocks["broker"].submit.assert_called_once()


async def test_hold_does_not_submit() -> None:
    cfg = TradingLoopConfig(watchlist=["AAPL"], execute=True, min_confidence=0.0)
    rec = _make_rec(action=SignalAction.HOLD, confidence=0.0, combined=0.0)
    loop, mocks = _build_loop(config=cfg, engine_rec=rec)
    results = await loop.tick()
    assert results[0].order is None
    mocks["broker"].submit.assert_not_called()


async def test_low_confidence_is_skipped_before_risk_gate() -> None:
    cfg = TradingLoopConfig(watchlist=["AAPL"], execute=True, min_confidence=0.5)
    rec = _make_rec(action=SignalAction.BUY, confidence=0.3, combined=0.3)
    loop, mocks = _build_loop(config=cfg, engine_rec=rec)
    results = await loop.tick()
    assert results[0].order is None
    assert "confidence" in results[0].note
    mocks["risk_gate"].check.assert_not_called()
    mocks["broker"].submit.assert_not_called()


async def test_risk_gate_block_prevents_order() -> None:
    cfg = TradingLoopConfig(watchlist=["AAPL"], execute=True, min_confidence=0.0)
    rec = _make_rec(action=SignalAction.BUY, confidence=0.7, combined=0.7)
    loop, mocks = _build_loop(config=cfg, engine_rec=rec, risk_allowed=False)
    results = await loop.tick()
    assert results[0].order is None
    mocks["risk_gate"].check.assert_called_once()
    mocks["broker"].submit.assert_not_called()


async def test_per_symbol_errors_do_not_break_loop() -> None:
    cfg = TradingLoopConfig(watchlist=["AAPL", "BAD", "MSFT"], execute=False)
    loop, mocks = _build_loop(config=cfg)
    # Second symbol blows up; first and third should still produce results.
    mocks["market"].get_bars.side_effect = [
        [],  # AAPL bars (returns empty list → df from to_dataframe stays mocked)
        RuntimeError("boom"),  # BAD
        [],  # MSFT
    ]
    results = await loop.tick()
    assert [r.symbol for r in results] == ["AAPL", "BAD", "MSFT"]
    assert results[0].error is None and results[0].recommendation is not None
    assert results[1].error is not None and "boom" in (results[1].error or "")
    assert results[2].error is None and results[2].recommendation is not None


async def test_empty_df_returns_note_no_bars() -> None:
    cfg = TradingLoopConfig(watchlist=["AAPL"], execute=True, min_confidence=0.0)
    loop, mocks = _build_loop(config=cfg, bars_df=pd.DataFrame())
    results = await loop.tick()
    assert results[0].recommendation is None
    assert results[0].order is None
    assert results[0].note == "no bars"
    mocks["engine"].recommend.assert_not_called()


# ---------------------------------------------------------------------------
# Loop driver
# ---------------------------------------------------------------------------


async def test_loop_run_stops_at_max_iterations() -> None:
    cfg = TradingLoopConfig(
        watchlist=["AAPL"], execute=False, max_iterations=3, interval_seconds=0.0
    )
    loop, mocks = _build_loop(config=cfg)
    await loop.run()
    # tick is called once per iteration, so 3 calls into engine.recommend
    assert mocks["engine"].recommend.call_count == 3


async def test_loop_run_passes_news_to_engine() -> None:
    cfg = TradingLoopConfig(
        watchlist=["AAPL"], execute=False, max_iterations=1, interval_seconds=0.0
    )
    loop, mocks = _build_loop(config=cfg)
    article = NewsArticle(
        id="x1", headline="ok", source="t", symbols=["AAPL"], published_at=datetime.now(UTC)
    )
    mocks["news_source"].fetch.return_value = [article]
    await loop.run()
    # Engine was called with our news list
    call_kwargs = mocks["engine"].recommend.call_args.kwargs
    assert call_kwargs["news"] == [article]


async def test_loop_uses_configured_timeframe() -> None:
    cfg = TradingLoopConfig(
        watchlist=["AAPL"],
        execute=False,
        max_iterations=1,
        interval_seconds=0.0,
        timeframe=TimeFrame.HOUR_1,
    )
    loop, mocks = _build_loop(config=cfg)
    await loop.run()
    call_args = mocks["market"].get_bars.call_args
    # positional args: [symbols], timeframe, start, end
    assert call_args.args[1] == TimeFrame.HOUR_1
