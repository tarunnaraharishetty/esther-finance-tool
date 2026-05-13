"""Tests for the dashboard controllers and a headless smoke test of the app."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.dashboard.controller import (
    DashboardController,
    MockDashboardController,
)
from src.dashboard.state import (
    DashboardSnapshot,
    EventBuffer,
)
from src.data.models import NewsArticle
from src.intelligence.alerts import AlertEngine
from src.sentiment.analyzer import SentimentAnalyzer, SentimentLabel, SentimentScore
from src.strategy.base import SignalAction
from src.strategy.recommendation import RecommendationEngine

# ---------------------------------------------------------------------------
# State plumbing
# ---------------------------------------------------------------------------


def test_event_buffer_is_bounded() -> None:
    buf = EventBuffer(capacity=3)
    for i in range(5):
        buf.info(f"msg-{i}")
    snap = buf.snapshot()
    assert len(snap) == 3
    assert [e.message for e in snap] == ["msg-2", "msg-3", "msg-4"]


def test_event_buffer_levels() -> None:
    buf = EventBuffer()
    buf.info("a")
    buf.warn("b")
    buf.error("c")
    levels = [e.level for e in buf.snapshot()]
    assert levels == ["info", "warn", "error"]


# ---------------------------------------------------------------------------
# MockDashboardController
# ---------------------------------------------------------------------------


def test_mock_controller_returns_full_snapshot() -> None:
    ctrl = MockDashboardController(watchlist=["AAPL", "MSFT", "NVDA"], seed=7)
    snap = asyncio.run(ctrl.fetch_snapshot())
    assert isinstance(snap, DashboardSnapshot)
    assert snap.tick == 1
    assert [r.symbol for r in snap.rows] == ["AAPL", "MSFT", "NVDA"]
    assert all(0.0 <= r.confidence <= 1.0 for r in snap.rows)
    assert all(-1.0 <= r.combined_score <= 1.0 for r in snap.rows)


def test_mock_controller_advances_tick_and_emits_events() -> None:
    ctrl = MockDashboardController(watchlist=["AAPL"], seed=3)
    s1 = asyncio.run(ctrl.fetch_snapshot())
    s2 = asyncio.run(ctrl.fetch_snapshot())
    assert s1.tick == 1 and s2.tick == 2
    # Each tick should add at least one "tick start" event + per-symbol events.
    assert len(s2.events) > len(s1.events)
    levels = {e.level for e in s2.events}
    assert "info" in levels


def test_mock_controller_action_is_a_valid_signal() -> None:
    ctrl = MockDashboardController(watchlist=["AAPL", "MSFT"], seed=42)
    snap = asyncio.run(ctrl.fetch_snapshot())
    valid = {SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD}
    for r in snap.rows:
        assert r.action in valid


def test_mock_controller_surfaces_alerts_across_ticks() -> None:
    """First tick has no baseline so produces no alerts; second tick has
    the chance to fire if any rule's condition is met."""
    ctrl = MockDashboardController(watchlist=["AAPL", "MSFT"], seed=42)
    s1 = asyncio.run(ctrl.fetch_snapshot())
    # First tick: no previous row → no alert can fire.
    assert s1.alerts == []
    s2 = asyncio.run(ctrl.fetch_snapshot())
    # Alerts are a list — may be empty, but the field must exist and the
    # second snapshot must have run the engine (baseline now populated).
    assert isinstance(s2.alerts, list)


def test_controller_uses_injected_alert_engine() -> None:
    """The controller should hand rows to whatever AlertEngine was injected."""
    engine = AlertEngine(rules=[])  # zero rules → never any alerts
    ctrl = MockDashboardController(
        watchlist=["AAPL", "MSFT"], seed=7, alert_engine=engine
    )
    s1 = asyncio.run(ctrl.fetch_snapshot())
    s2 = asyncio.run(ctrl.fetch_snapshot())
    assert s1.alerts == []
    assert s2.alerts == []


# ---------------------------------------------------------------------------
# Live DashboardController — fully mocked deps so no Alpaca / network
# ---------------------------------------------------------------------------


class _Neut(SentimentAnalyzer):
    def __init__(self) -> None:
        pass

    def score_text(self, _t: str) -> SentimentScore:  # type: ignore[override]
        return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

    def score_article(self, _a: NewsArticle) -> SentimentScore:  # type: ignore[override]
        return self.score_text("")


def _df(n: int = 60, drift: float = 0.005) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rets = rng.normal(loc=drift, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
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


def _build_live_controller(
    *, df: pd.DataFrame | None = None
) -> DashboardController:
    market = MagicMock()
    market.get_bars = AsyncMock(return_value=["bar"])  # sentinel
    market.to_dataframe = MagicMock(return_value=df if df is not None else _df())

    news_source = MagicMock()
    news_source.fetch = AsyncMock(return_value=[])

    engine = RecommendationEngine(sentiment_analyzer=_Neut())
    return DashboardController(
        watchlist=["AAPL"],
        engine=engine,
        market=market,
        news_source=news_source,
    )


def test_live_controller_returns_recommendation() -> None:
    ctrl = _build_live_controller()
    snap = asyncio.run(ctrl.fetch_snapshot())
    assert len(snap.rows) == 1
    row = snap.rows[0]
    assert row.symbol == "AAPL"
    assert row.error is None


def test_live_controller_handles_empty_bars() -> None:
    ctrl = _build_live_controller(df=pd.DataFrame())
    snap = asyncio.run(ctrl.fetch_snapshot())
    assert snap.rows[0].error == "no bars"
    assert snap.rows[0].action == SignalAction.HOLD


def test_live_controller_handles_market_error() -> None:
    ctrl = _build_live_controller()
    ctrl.market.get_bars = AsyncMock(side_effect=RuntimeError("alpaca down"))  # type: ignore[method-assign]
    snap = asyncio.run(ctrl.fetch_snapshot())
    assert snap.rows[0].error is not None
    assert "alpaca down" in snap.rows[0].error
    # error-level event should be in the buffer
    assert any(e.level == "error" for e in snap.events)


def test_live_controller_merges_cached_bars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """If data/cache/ has bars for the symbol, the controller's df should
    include them — verified by checking the row count exceeds what the
    market mock returned alone."""
    from pathlib import Path

    from src.config import settings as settings_mod
    from src.data import cache
    from src.data.models import TimeFrame

    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()
    monkeypatch.setattr(s, "data_dir", Path(str(tmp_path)), raising=False)

    # Seed the cache with 50 daily bars dated well in the past.
    cached = pd.DataFrame(
        {
            "open": np.linspace(100, 150, 50),
            "high": np.linspace(101, 151, 50),
            "low": np.linspace(99, 149, 50),
            "close": np.linspace(100, 150, 50),
            "volume": np.full(50, 1_000_000, dtype=int),
        },
        index=pd.date_range(
            start=datetime(2026, 1, 1, tzinfo=UTC), periods=50, freq="D", tz="UTC"
        ),
    )
    cache.write_bars("AAPL", TimeFrame.DAY_1, cached)

    # Live mock returns just 5 fresh bars; engine should see 55 total.
    live = pd.DataFrame(
        {
            "open": np.linspace(155, 160, 5),
            "high": np.linspace(156, 161, 5),
            "low": np.linspace(154, 159, 5),
            "close": np.linspace(155, 160, 5),
            "volume": np.full(5, 1_000_000, dtype=int),
        },
        index=pd.date_range(
            start=datetime(2026, 3, 1, tzinfo=UTC), periods=5, freq="D", tz="UTC"
        ),
    )
    captured: dict[str, int] = {}

    def _capture_recommend(self: object, *args: object, **kwargs: object) -> object:
        df = kwargs.get("df") if "df" in kwargs else (args[1] if len(args) > 1 else None)
        if isinstance(df, pd.DataFrame):
            captured["rows_seen_by_engine"] = len(df)
        # Delegate to a minimal stub recommendation so the controller succeeds.
        from datetime import datetime as _dt

        from src.strategy.base import SignalAction
        from src.strategy.recommendation import TradingRecommendation

        return TradingRecommendation(
            symbol="AAPL",
            action=SignalAction.HOLD,
            confidence=0.1,
            combined_score=0.0,
            technical_score=0.0,
            sentiment_score=0.0,
            indicator_scores={"rsi": 0.0, "macd": 0.0, "bollinger": 0.0},
            reasoning="stub",
            timestamp=_dt.now(UTC),
            num_news_articles=0,
        )

    ctrl = _build_live_controller(df=live)
    monkeypatch.setattr(
        RecommendationEngine, "recommend", _capture_recommend, raising=True
    )
    snap = asyncio.run(ctrl.fetch_snapshot())
    assert snap.rows[0].error is None
    # Engine should have seen the merged dataframe: 50 cached + 5 live.
    assert captured["rows_seen_by_engine"] == 55


def test_live_controller_refuses_non_paper_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="non-paper"):
            DashboardController(
                watchlist=["AAPL"],
                engine=RecommendationEngine(sentiment_analyzer=_Neut()),
                market=MagicMock(),
                news_source=MagicMock(),
            )
    finally:
        settings_mod.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Headless app smoke test — verifies the Textual layout + keybindings boot.
# ---------------------------------------------------------------------------


async def test_app_renders_and_responds_to_keys() -> None:
    from textual.widgets import DataTable

    from src.dashboard.app import DashboardApp

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL", "MSFT", "NVDA"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        table = app.query_one(DataTable)
        assert table.row_count == 3
        assert len(table.columns) == 11  # see compose()

        # Pause / resume key toggles state
        await pilot.press("p")
        assert app.paused is True
        await pilot.press("p")
        assert app.paused is False

        # Manual refresh leaves the table populated
        await pilot.press("r")
        await pilot.pause(0.2)
        assert table.row_count == 3


async def test_app_watchlist_header_renders() -> None:
    """The new header widget should mount and render at least the action mix."""
    from src.dashboard.app import DashboardApp, WatchlistHeader

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL", "MSFT"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        header = app.query_one(WatchlistHeader)
        text = header.render()
        assert "Action mix" in text
        # First frame: no prior snapshot to diff.
        assert "first frame" in text or "no changes" in text


async def test_app_s_keypress_with_no_summarizer_surfaces_error() -> None:
    """`s` with summarizer=None should put DetailPanel into an error state
    pointing at ANTHROPIC_API_KEY — no crash, no network call."""
    from src.dashboard.app import DashboardApp, DetailPanel

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL", "MSFT"]),
        refresh_seconds=999.0,
        summarizer=None,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        await pilot.press("s")
        await pilot.pause(0.1)
        detail = app.query_one(DetailPanel)
        assert detail.brief_state == "error"
        assert "ANTHROPIC_API_KEY" in detail.brief_text


async def test_app_s_keypress_with_mock_summarizer_loads_and_caches() -> None:
    """`s` with a real Summarizer: spawn worker, brief lands, cache hit on re-press."""
    from unittest.mock import MagicMock

    from src.dashboard.app import DashboardApp, DetailPanel

    summarizer = MagicMock()
    summarizer.summarize = MagicMock(
        return_value="AAPL leans bullish with high confidence."
    )

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL", "MSFT"]),
        refresh_seconds=999.0,
        summarizer=summarizer,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        await pilot.press("s")
        # Let the worker complete.
        await pilot.pause(0.3)
        detail = app.query_one(DetailPanel)
        assert detail.brief_state == "ready"
        assert detail.brief_text == "AAPL leans bullish with high confidence."
        assert summarizer.summarize.call_count == 1

        # Re-press: same row, should hit cache, no second SDK call.
        await pilot.press("s")
        await pilot.pause(0.1)
        assert summarizer.summarize.call_count == 1
