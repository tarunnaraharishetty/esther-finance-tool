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


def test_event_buffer_dedups_consecutive_identical_pushes() -> None:
    """Same (level, message) in a row should collapse into one entry
    with the count bumped, not three separate entries."""
    buf = EventBuffer()
    buf.info("tick 5 start")
    buf.info("tick 5 start")
    buf.info("tick 5 start")
    snap = buf.snapshot()
    assert len(snap) == 1
    assert snap[0].count == 3
    assert snap[0].message == "tick 5 start"


def test_event_buffer_dedup_updates_timestamp() -> None:
    """When a duplicate is collapsed, the entry's timestamp advances to
    the latest occurrence."""
    import time

    buf = EventBuffer()
    first = buf.info("foo")
    time.sleep(0.01)
    second = buf.info("foo")
    assert second.timestamp > first.timestamp
    assert second.count == 2


def test_event_buffer_different_message_breaks_dedup() -> None:
    """A different message ends the dedup run; the next identical push
    starts a fresh entry, not a bump of an older one."""
    buf = EventBuffer()
    buf.info("a")
    buf.info("a")  # bumps to count=2
    buf.info("b")  # new entry
    buf.info("a")  # NEW entry, count=1 (not 3 — the old "a" is no longer the tail)
    snap = buf.snapshot()
    assert [(e.message, e.count) for e in snap] == [
        ("a", 2),
        ("b", 1),
        ("a", 1),
    ]


def test_event_buffer_different_level_breaks_dedup() -> None:
    """Same message at a different level is not a duplicate."""
    buf = EventBuffer()
    buf.info("AAPL slow tick")
    buf.warn("AAPL slow tick")
    snap = buf.snapshot()
    assert len(snap) == 2
    assert snap[0].level == "info"
    assert snap[1].level == "warn"
    assert all(e.count == 1 for e in snap)


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


def test_mock_controller_signal_history_grows_across_ticks() -> None:
    """The signal_history field on the snapshot should populate after the
    first tick and accumulate episode metadata over subsequent ticks."""
    ctrl = MockDashboardController(watchlist=["AAPL", "MSFT"], seed=7)
    s1 = asyncio.run(ctrl.fetch_snapshot())
    assert set(s1.signal_history.keys()) == {"AAPL", "MSFT"}
    # First tick: each symbol has one episode, tick_count == 1.
    for sym in ("AAPL", "MSFT"):
        summary = s1.signal_history[sym]
        assert summary.current.tick_count == 1
        assert summary.recent == ()

    s2 = asyncio.run(ctrl.fetch_snapshot())
    # Second tick: tick_count grew (if action stayed) OR a new episode
    # started (if action flipped). Either way, the engine accumulated.
    for sym in ("AAPL", "MSFT"):
        summary = s2.signal_history[sym]
        # Either tick_count >= 2 in current episode, or a prior episode appeared.
        assert summary.current.tick_count >= 1
        # Total observation count across episodes should be exactly 2.
        eps = ctrl.signal_history.episodes_for(sym)
        total_ticks = sum(ep.tick_count for ep in eps)
        assert total_ticks == 2


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
    # With zero rules, recent_alerts stays empty across ticks too.
    assert s1.recent_alerts == ()
    assert s2.recent_alerts == ()


def test_controller_runs_alerts_through_prioritizer() -> None:
    """The controller's pipeline should be:
    AlertEngine → AlertPrioritizer → AlertState.record. We verify by
    seeding a prioritizer that always drops everything and confirming
    snapshot.alerts is empty even when raw rules would fire."""
    from src.intelligence.alert_prioritizer import (
        AlertPrioritizer,
        PrioritizerConfig,
    )

    drop_all = AlertPrioritizer(PrioritizerConfig(max_per_tick=0))
    ctrl = MockDashboardController(
        watchlist=["AAPL", "MSFT"], seed=7, alert_prioritizer=drop_all
    )
    # Run a few ticks to let raw rules generate firings.
    asyncio.run(ctrl.fetch_snapshot())
    s2 = asyncio.run(ctrl.fetch_snapshot())
    s3 = asyncio.run(ctrl.fetch_snapshot())
    # Prioritizer dropped everything → both fields empty.
    assert s2.alerts == []
    assert s3.alerts == []
    assert s2.recent_alerts == ()
    assert s3.recent_alerts == ()


def test_controller_recent_alerts_accumulates_across_ticks() -> None:
    """With default prioritizer + the mock controller's synthetic
    movement, action_changed alerts naturally fire. recent_alerts
    should be empty on first tick (no baseline) and may grow after."""
    ctrl = MockDashboardController(watchlist=["AAPL", "MSFT"], seed=42)
    s1 = asyncio.run(ctrl.fetch_snapshot())
    s2 = asyncio.run(ctrl.fetch_snapshot())
    s3 = asyncio.run(ctrl.fetch_snapshot())
    # First tick: no prior row → no alerts → recent empty.
    assert s1.recent_alerts == ()
    # Across three ticks the controller has recorded zero or more alerts
    # into AlertState — verify recent_alerts is at least as long as the
    # total alerts ever fired this session.
    total_fresh = sum(len(s.alerts) for s in (s2, s3))
    assert len(s3.recent_alerts) <= len(ctrl.alert_state)
    assert len(ctrl.alert_state) >= total_fresh or total_fresh == 0


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


async def test_burst_fires_when_alerts_present() -> None:
    """A snapshot with any alert should trigger burst mode so the
    dashboard refreshes faster than the base interval."""
    from src.dashboard.app import DashboardApp
    from src.dashboard.state import DashboardSnapshot
    from src.intelligence.alerts import Alert

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    fired = datetime.now(UTC)
    snap_with_alert = DashboardSnapshot(
        tick=1,
        rows=[],
        alerts=[
            Alert(symbol="AAPL", rule="action_changed", severity="warn",
                  message="x", fired_at=fired),
        ],
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert app._should_burst(snap_with_alert) is True


async def test_burst_fires_when_action_flips_vs_prev() -> None:
    """A new snapshot whose row action differs from the prior snapshot's
    recorded action should burst — that's the 'something just moved' case."""
    from src.dashboard.app import DashboardApp
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )

    def _row(action: SignalAction) -> RecommendationRow:
        return RecommendationRow(
            symbol="AAPL",
            action=action,
            confidence=0.5,
            combined_score=0.5,
            technical_score=0.5,
            sentiment_score=0.0,
            rsi=0.5, macd=0.5, bollinger=0.5,
            last_price=150.0,
            num_news_articles=0,
            reasoning="",
            timestamp=datetime.now(UTC),
        )

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        app._previous_actions = {}  # clear mock-controller's seeded baseline
        # First call seeds the baseline; can't flip yet.
        app._should_burst(DashboardSnapshot(tick=1, rows=[_row(SignalAction.HOLD)]))
        # Second call has a different action → burst.
        result = app._should_burst(
            DashboardSnapshot(tick=2, rows=[_row(SignalAction.BUY)])
        )
        assert result is True


async def test_burst_quiet_when_no_alerts_and_actions_stable() -> None:
    """Stable watchlist + empty alerts → no burst."""
    from src.dashboard.app import DashboardApp
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )

    def _row() -> RecommendationRow:
        return RecommendationRow(
            symbol="AAPL",
            action=SignalAction.HOLD,
            confidence=0.3,
            combined_score=0.0,
            technical_score=0.0,
            sentiment_score=0.0,
            rsi=0.0, macd=0.0, bollinger=0.0,
            last_price=150.0,
            num_news_articles=0,
            reasoning="",
            timestamp=datetime.now(UTC),
        )

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        # The mock controller's on_mount refresh seeded _previous_actions
        # with whatever AAPL's initial mock action was; clear it so the
        # test starts from a known no-prior state.
        app._previous_actions = {}
        snap1 = DashboardSnapshot(tick=1, rows=[_row()])
        snap2 = DashboardSnapshot(tick=2, rows=[_row()])
        # First call: no prior, no alerts → no burst.
        assert app._should_burst(snap1) is False
        # Second call: same action, no alerts → no burst.
        assert app._should_burst(snap2) is False


async def test_burst_ignores_error_rows_for_flip_detection() -> None:
    """An error row replacing a healthy row shouldn't count as a flip —
    a transient fetch failure isn't a market move."""
    from src.dashboard.app import DashboardApp
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )

    def _row(action: SignalAction, error: str | None = None) -> RecommendationRow:
        return RecommendationRow(
            symbol="AAPL",
            action=action,
            confidence=0.5,
            combined_score=0.5,
            technical_score=0.5,
            sentiment_score=0.0,
            rsi=0.5, macd=0.5, bollinger=0.5,
            last_price=150.0,
            num_news_articles=0,
            reasoning="",
            timestamp=datetime.now(UTC),
            error=error,
        )

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        app._previous_actions = {}  # clear mock-controller's seeded baseline
        app._should_burst(DashboardSnapshot(tick=1, rows=[_row(SignalAction.BUY)]))
        # Error row this tick — baseline preserved, no burst.
        result = app._should_burst(
            DashboardSnapshot(tick=2, rows=[_row(SignalAction.HOLD, error="boom")])
        )
        assert result is False


async def test_pulse_line_surfaces_breadth_when_directional() -> None:
    """When the watchlist has BUY rows with aligned MACD, momentum-breadth
    should appear as a percentage chunk in the PULSE line."""
    from src.dashboard.app import DashboardApp, WatchlistHeader
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    fired = datetime.now(UTC)

    def _aligned_buy(symbol: str) -> RecommendationRow:
        return RecommendationRow(
            symbol=symbol,
            action=SignalAction.BUY,
            confidence=0.5,
            combined_score=0.5,
            technical_score=0.5,
            sentiment_score=0.5,
            rsi=0.4, macd=0.5, bollinger=0.4,  # MACD aligned with BUY
            last_price=100.0,
            num_news_articles=3,  # news-bearing → sentiment_breadth counts
            reasoning="",
            timestamp=fired,
        )

    snap = DashboardSnapshot(
        tick=1,
        rows=[_aligned_buy("AAPL"), _aligned_buy("MSFT")],
    )

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        header = app.query_one(WatchlistHeader)
        header.snapshot = snap
        text = header.render()
        # Both BUY rows have aligned MACD → momentum breadth 100%.
        assert "mom" in text
        assert "100%" in text
        # Sentiment breadth — news-bearing rows aligned → also 100%.
        assert "sent" in text


async def test_pulse_line_omits_breadth_when_no_directional_rows() -> None:
    """All-HOLD watchlist → no directional rows → breadth chunks omitted."""
    from src.dashboard.app import DashboardApp, WatchlistHeader
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    fired = datetime.now(UTC)
    hold_row = RecommendationRow(
        symbol="AAPL",
        action=SignalAction.HOLD,
        confidence=0.2,
        combined_score=0.0,
        technical_score=0.0,
        sentiment_score=0.0,
        rsi=0.0, macd=0.0, bollinger=0.0,
        last_price=100.0,
        num_news_articles=0,
        reasoning="",
        timestamp=fired,
    )
    snap = DashboardSnapshot(tick=1, rows=[hold_row])

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        header = app.query_one(WatchlistHeader)
        header.snapshot = snap
        text = header.render()
        # No directional rows → momentum/sentiment breadth chunks absent.
        # The "mom" / "sent" labels would appear only inside breadth chunks.
        assert "mom" not in text
        assert "sent" not in text


async def test_pulse_line_surfaces_strong_tier_symbols() -> None:
    """A STRONG_BUY row should produce a 'STRONG' chunk in the PULSE
    line carrying the symbol + tier label."""
    from src.dashboard.app import DashboardApp, WatchlistHeader
    from src.dashboard.state import DashboardSnapshot, RecommendationRow
    from src.strategy.base import RecommendationTier

    fired = datetime.now(UTC)
    strong_row = RecommendationRow(
        symbol="NVDA",
        action=SignalAction.BUY,
        confidence=0.85,
        combined_score=0.8,
        technical_score=0.6,
        sentiment_score=0.5,
        rsi=0.5, macd=0.6, bollinger=0.4,
        last_price=520.0,
        num_news_articles=5,
        reasoning="",
        timestamp=fired,
        tier=RecommendationTier.STRONG_BUY,
        signal_quality="high",
        stability="stable",
    )
    snap = DashboardSnapshot(tick=1, rows=[strong_row])

    app = DashboardApp(
        MockDashboardController(watchlist=["NVDA"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        header = app.query_one(WatchlistHeader)
        header.snapshot = snap
        text = header.render()
        assert "STRONG" in text
        assert "NVDA" in text
        assert "STRONG BUY" in text


async def test_pulse_line_omits_alerts_chunk_when_zero() -> None:
    """Quiet markets stay tight — no alerts chunk when alert_intensity is 0."""
    from src.dashboard.app import DashboardApp, WatchlistHeader
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    fired = datetime.now(UTC)
    row = RecommendationRow(
        symbol="AAPL",
        action=SignalAction.HOLD,
        confidence=0.3,
        combined_score=0.0,
        technical_score=0.0,
        sentiment_score=0.0,
        rsi=0.0, macd=0.0, bollinger=0.0,
        last_price=100.0,
        num_news_articles=0,
        reasoning="",
        timestamp=fired,
    )
    snap = DashboardSnapshot(tick=1, rows=[row])  # no recent_alerts

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        header = app.query_one(WatchlistHeader)
        header.snapshot = snap
        text = header.render()
        assert "alerts" not in text


async def test_status_line_idle_when_no_rows() -> None:
    """Empty snapshot → status reads 'idle'."""
    from src.dashboard.app import DashboardApp, StatusLine

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        status = app.query_one(StatusLine)
        status.snapshot = None
        assert "idle" in status.render().lower()


async def test_status_line_shows_sentiment_and_tick() -> None:
    """Healthy snapshot → status includes a sentiment chip + tick #."""
    from src.dashboard.app import DashboardApp, StatusLine

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL", "MSFT", "NVDA"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        status = app.query_one(StatusLine)
        text = status.render()
        # One of the four sentiment tiers must appear.
        assert any(s in text for s in ("bullish", "bearish", "mixed", "neutral"))
        assert "tick" in text


async def test_status_line_surfaces_strong_tier_symbol() -> None:
    """A row promoted to STRONG_BUY/STRONG_SELL gets surfaced by symbol
    in the status line — high-conviction signals deserve always-visible
    real estate."""
    from src.dashboard.app import DashboardApp, StatusLine
    from src.dashboard.state import DashboardSnapshot, RecommendationRow
    from src.strategy.base import RecommendationTier

    fired = datetime.now(UTC)
    row = RecommendationRow(
        symbol="NVDA",
        action=SignalAction.BUY,
        confidence=0.78,
        combined_score=0.7,
        technical_score=0.6,
        sentiment_score=0.5,
        rsi=0.5, macd=0.6, bollinger=0.4,
        last_price=520.0,
        num_news_articles=8,
        reasoning="",
        timestamp=fired,
        tier=RecommendationTier.STRONG_BUY,
        signal_quality="high",
        stability="stable",
    )
    snap = DashboardSnapshot(tick=42, rows=[row])

    app = DashboardApp(
        MockDashboardController(watchlist=["NVDA"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        status = app.query_one(StatusLine)
        status.snapshot = snap
        text = status.render()
        assert "STRONG BUY" in text
        assert "NVDA" in text


async def test_status_line_shows_critical_alert_count() -> None:
    """Critical alerts in recent_alerts surface as a dedicated chunk."""
    from src.dashboard.app import DashboardApp, StatusLine
    from src.dashboard.state import DashboardSnapshot, RecommendationRow
    from src.intelligence.alerts import Alert

    fired = datetime.now(UTC)
    row = RecommendationRow(
        symbol="AAPL",
        action=SignalAction.HOLD,
        confidence=0.3,
        combined_score=0.1,
        technical_score=0.0,
        sentiment_score=0.0,
        rsi=0.0, macd=0.0, bollinger=0.0,
        last_price=150.0,
        num_news_articles=0,
        reasoning="",
        timestamp=fired,
    )
    alerts = (
        Alert(symbol="AAPL", rule="tier_changed", severity="critical",
              message="x", fired_at=fired),
        Alert(symbol="MSFT", rule="action_changed", severity="critical",
              message="x", fired_at=fired),
        Alert(symbol="NVDA", rule="confidence_threshold", severity="info",
              message="x", fired_at=fired),
    )
    snap = DashboardSnapshot(tick=1, rows=[row], recent_alerts=alerts)

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        status = app.query_one(StatusLine)
        status.snapshot = snap
        text = status.render()
        assert "2 critical" in text


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
        assert len(table.columns) == 12  # see compose() — SYM..NEWS + MTF

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
        # New layout uses fixed-width section labels.
        assert "MIX" in text
        assert "BUY" in text  # action chip is always present
        # First frame: no prior snapshot to diff.
        assert "first frame" in text or "no changes" in text


def test_header_signature_is_deterministic_for_identical_inputs() -> None:
    """Pure-function helper: same inputs → same signature, so the
    watch_snapshot guard can short-circuit identical renders."""
    from src.dashboard.app import _header_signature
    from src.dashboard.state import DashboardSnapshot

    snap = DashboardSnapshot(tick=5, rows=[])
    prev = DashboardSnapshot(tick=4, rows=[])
    assert _header_signature(snap, prev) == _header_signature(snap, prev)


def test_header_signature_changes_with_action_mix() -> None:
    """A different row action under cursor must produce a new
    signature — the watch guard otherwise wouldn't refresh."""
    from datetime import datetime

    from src.dashboard.app import _header_signature
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    def _row(action: SignalAction) -> RecommendationRow:
        return RecommendationRow(
            symbol="AAPL",
            action=action,
            confidence=0.5,
            combined_score=0.5,
            technical_score=0.5,
            sentiment_score=0.0,
            rsi=0.5,
            macd=0.5,
            bollinger=0.5,
            last_price=150.0,
            num_news_articles=0,
            reasoning="",
            timestamp=datetime.now(UTC),
        )

    snap_buy = DashboardSnapshot(tick=1, rows=[_row(SignalAction.BUY)])
    snap_sell = DashboardSnapshot(tick=1, rows=[_row(SignalAction.SELL)])
    assert _header_signature(snap_buy, None) != _header_signature(snap_sell, None)


async def test_detail_panel_short_circuits_on_identical_inputs() -> None:
    """When the selected row's data + brief state are unchanged across
    renders, the second render must return the cached string by
    reference (no recomputation)."""
    from src.dashboard.app import DashboardApp, DetailPanel
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    fired = datetime.now(UTC)
    row = RecommendationRow(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.5,
        combined_score=0.5,
        technical_score=0.5,
        sentiment_score=0.0,
        rsi=0.5,
        macd=0.5,
        bollinger=0.5,
        last_price=150.0,
        num_news_articles=0,
        reasoning="",
        timestamp=fired,
    )
    snap = DashboardSnapshot(tick=1, rows=[row])

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        detail = app.query_one(DetailPanel)
        detail.snapshot = snap
        detail.row_index = 0
        first = detail.render()
        # Hand the same data again — should hit the cache.
        second = detail.render()
        assert first == second
        # The cache hit returns the exact same string object by identity.
        assert first is detail._last_rendered


async def test_detail_panel_invalidates_cache_when_row_changes() -> None:
    """Cache must invalidate when meaningful row data changes."""
    from src.dashboard.app import DashboardApp, DetailPanel
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    fired = datetime.now(UTC)

    def _make_row(action: SignalAction, conf: float) -> RecommendationRow:
        from src.strategy.base import RecommendationTier

        return RecommendationRow(
            symbol="AAPL",
            action=action,
            confidence=conf,
            combined_score=conf,
            technical_score=conf,
            sentiment_score=0.0,
            rsi=0.5,
            macd=0.5,
            bollinger=0.5,
            last_price=150.0,
            num_news_articles=0,
            reasoning="",
            timestamp=fired,
            tier=RecommendationTier.from_action(action),
        )

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        detail = app.query_one(DetailPanel)
        detail.snapshot = DashboardSnapshot(
            tick=1, rows=[_make_row(SignalAction.BUY, 0.5)]
        )
        detail.row_index = 0
        first = detail.render()
        sig_one = detail._last_signature

        # Same inputs again — cached.
        detail.render()
        assert detail._last_signature == sig_one

        # Now flip the action — cache must invalidate.
        detail.snapshot = DashboardSnapshot(
            tick=2, rows=[_make_row(SignalAction.SELL, 0.5)]
        )
        second = detail.render()
        assert detail._last_signature != sig_one
        assert "SELL" in second
        assert "BUY" not in second or first != second


async def test_watchlist_header_renders_opp_lines_when_qualifying_rows_exist() -> None:
    """A row that qualifies for a high_conviction opportunity should
    produce an OPP line in the header."""
    from src.dashboard.app import DashboardApp, WatchlistHeader
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    fired = datetime.now(UTC)
    qualifying = RecommendationRow(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.7,
        combined_score=0.7,
        technical_score=0.5,
        sentiment_score=0.5,
        rsi=0.5, macd=0.5, bollinger=0.5,
        last_price=150.0,
        num_news_articles=5,
        reasoning="",
        timestamp=fired,
    )
    snap = DashboardSnapshot(tick=1, rows=[qualifying])

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        header = app.query_one(WatchlistHeader)
        header.snapshot = snap
        text = header.render()
        # OPP section now renders the ranked-composite output: tier
        # display + profile chip axes. At least one of the profile
        # axis labels must appear.
        assert "OPP" in text
        assert any(
            axis in text
            for axis in (
                "stable", "noisy",
                "strengthening", "weakening", "flat",
                "persistent", "flipping",
            )
        )


async def test_opp_line_shows_composite_score_and_profile_chip() -> None:
    """A qualifying directional row should produce an OPP line that
    surfaces the composite score (formatted .NN) and a profile chip
    with all three axis labels."""
    from src.dashboard.app import DashboardApp, WatchlistHeader
    from src.dashboard.state import DashboardSnapshot, RecommendationRow
    from src.strategy.base import RecommendationTier

    fired = datetime.now(UTC)
    row = RecommendationRow(
        symbol="NVDA",
        action=SignalAction.BUY,
        confidence=0.85,
        combined_score=0.8,
        technical_score=0.7,
        sentiment_score=0.6,
        rsi=0.5, macd=0.7, bollinger=0.4,
        last_price=520.0,
        num_news_articles=8,
        reasoning="",
        timestamp=fired,
        tier=RecommendationTier.STRONG_BUY,
        signal_quality="high",
        stability="stable",
    )
    snap = DashboardSnapshot(tick=1, rows=[row])

    app = DashboardApp(
        MockDashboardController(watchlist=["NVDA"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        header = app.query_one(WatchlistHeader)
        header.snapshot = snap
        text = header.render()
        assert "OPP" in text
        # Composite score formatted to two decimals appears somewhere.
        import re
        assert re.search(r"\b0\.\d{2}\b", text), f"expected composite score in {text}"
        # Profile chip has all three axes.
        assert "stable" in text
        # NVDA is the symbol; STRONG BUY is the tier display.
        assert "NVDA" in text
        assert "STRONG BUY" in text


async def test_opp_section_omitted_when_no_directional_rows() -> None:
    """All-HOLD watchlist → no qualifying rows → OPP section absent."""
    from src.dashboard.app import DashboardApp, WatchlistHeader
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    fired = datetime.now(UTC)
    hold_row = RecommendationRow(
        symbol="AAPL",
        action=SignalAction.HOLD,
        confidence=0.2,
        combined_score=0.0,
        technical_score=0.0,
        sentiment_score=0.0,
        rsi=0.0, macd=0.0, bollinger=0.0,
        last_price=100.0,
        num_news_articles=0,
        reasoning="",
        timestamp=fired,
    )
    snap = DashboardSnapshot(tick=1, rows=[hold_row])

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        header = app.query_one(WatchlistHeader)
        header.snapshot = snap
        text = header.render()
        # OPP label only appears as a row-section header — absent when
        # no entries qualify.
        assert "OPP        " not in text  # the padded label


async def test_table_action_cell_renders_tier_display() -> None:
    """The table's ACTION column should render the 5-tier display
    (e.g. 'STRONG BUY'), not the base 3-tier action string."""
    from textual.widgets import DataTable

    from src.dashboard.app import DashboardApp
    from src.strategy.base import RecommendationTier

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        # The mock controller doesn't reliably produce STRONG (needs
        # specific scoring/history), so we just assert the cell content
        # is one of the five valid display strings.
        table = app.query_one(DataTable)
        # Walk the first row's ACTION cell text.
        if table.row_count:
            valid_displays = {t.display for t in RecommendationTier}
            cell_text = str(table.get_cell_at((0, 1)))
            # Strip Rich markup for the comparison.
            assert any(d in cell_text for d in valid_displays), (
                f"action cell should contain one of {valid_displays}, got: {cell_text}"
            )


async def test_detail_panel_renders_tier_quality_stability_when_reasons_present() -> None:
    """When a recommendation has quality_reasons, the detail panel
    shows the 'Why this tier' section."""
    from src.dashboard.app import DashboardApp, DetailPanel
    from src.dashboard.state import DashboardSnapshot, RecommendationRow
    from src.strategy.base import RecommendationTier

    fired = datetime.now(UTC)
    row = RecommendationRow(
        symbol="NVDA",
        action=SignalAction.BUY,
        confidence=0.78,
        combined_score=0.7,
        technical_score=0.5,
        sentiment_score=0.5,
        rsi=0.5,
        macd=0.6,
        bollinger=0.4,
        last_price=520.0,
        num_news_articles=8,
        reasoning="",
        timestamp=fired,
        tier=RecommendationTier.STRONG_BUY,
        signal_quality="high",
        stability="stable",
        quality_reasons=(
            "Strong technical alignment",
            "Bullish sentiment acceleration",
            "Low reversal frequency",
        ),
    )
    snap = DashboardSnapshot(tick=1, rows=[row])

    app = DashboardApp(
        MockDashboardController(watchlist=["NVDA"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        detail = app.query_one(DetailPanel)
        detail.snapshot = snap
        detail.row_index = 0
        text = detail.render()
        assert "STRONG BUY" in text
        assert "HIGH" in text  # quality tier
        assert "STABLE" in text  # stability tier
        assert "Why this tier" in text
        assert "Strong technical alignment" in text


async def test_detail_panel_omits_tier_section_when_no_reasons() -> None:
    """No quality_reasons → no 'Why this tier' section. Keeps the
    detail panel tight on data-thin rows."""
    from src.dashboard.app import DashboardApp, DetailPanel
    from src.dashboard.state import DashboardSnapshot, RecommendationRow

    fired = datetime.now(UTC)
    row = RecommendationRow(
        symbol="AAPL",
        action=SignalAction.HOLD,
        confidence=0.2,
        combined_score=0.1,
        technical_score=0.05,
        sentiment_score=0.0,
        rsi=0.0, macd=0.0, bollinger=0.0,
        last_price=150.0,
        num_news_articles=0,
        reasoning="",
        timestamp=fired,
        # quality_reasons defaults to empty tuple — no reasons surfaced
    )
    snap = DashboardSnapshot(tick=1, rows=[row])

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        detail = app.query_one(DetailPanel)
        detail.snapshot = snap
        detail.row_index = 0
        text = detail.render()
        assert "Why this tier" not in text


async def test_watchlist_header_renders_pulse_line_when_rows_present() -> None:
    """The pulse line is always at the top when any healthy rows
    exist. Tier values are stable schema (bullish/bearish/mixed/neutral
    × strong/moderate/weak × volatile/active/calm) so we can assert
    that at least one of each tier shows up."""
    from src.dashboard.app import DashboardApp, WatchlistHeader

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL", "MSFT", "NVDA"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        header = app.query_one(WatchlistHeader)
        text = header.render()
        assert "PULSE" in text
        # Conviction abbreviated to "conv" in the dense single-line layout.
        assert "conv" in text
        # One of the sentiment tiers must appear.
        assert any(s in text for s in ("bullish", "bearish", "mixed", "neutral"))


async def test_watchlist_header_renders_alerts_summary_when_present() -> None:
    """When recent_alerts is non-empty, the header should show an
    ALERTS line with severity counts."""
    from src.dashboard.app import DashboardApp, WatchlistHeader
    from src.dashboard.state import DashboardSnapshot
    from src.intelligence.alerts import Alert

    fired = datetime.now(UTC)
    snap = DashboardSnapshot(
        tick=1,
        rows=[],  # don't care for this test
        recent_alerts=(
            Alert(symbol="AAPL", rule="action_changed", severity="critical",
                  message="boom", fired_at=fired),
            Alert(symbol="MSFT", rule="confidence_threshold", severity="warn",
                  message="x", fired_at=fired),
            Alert(symbol="NVDA", rule="sentiment_shift", severity="info",
                  message="x", fired_at=fired),
        ),
    )
    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        header = app.query_one(WatchlistHeader)
        header.snapshot = snap
        text = header.render()
        assert "ALERTS" in text
        assert "1 critical" in text
        assert "1 warn" in text
        assert "1 info" in text
        assert "3 this session" in text


async def test_watchlist_header_skips_alerts_when_none() -> None:
    """No recent_alerts → no ALERTS line. Keeps the header tight on
    quiet sessions."""
    from src.dashboard.app import DashboardApp, WatchlistHeader

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        header = app.query_one(WatchlistHeader)
        # First tick of the mock controller: no alerts yet.
        text = header.render()
        assert "ALERTS" not in text


async def test_detail_panel_shows_per_symbol_alerts_filtered_from_recent() -> None:
    """The detail panel's new Alerts section should list only alerts
    for the symbol under cursor, not all session alerts."""
    from src.dashboard.app import DashboardApp, DetailPanel
    from src.dashboard.state import DashboardSnapshot, RecommendationRow
    from src.intelligence.alerts import Alert

    fired = datetime.now(UTC)
    row_aapl = RecommendationRow(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.5,
        combined_score=0.5,
        technical_score=0.5,
        sentiment_score=0.0,
        rsi=0.5,
        macd=0.5,
        bollinger=0.5,
        last_price=150.0,
        num_news_articles=0,
        reasoning="",
        timestamp=fired,
    )
    snap = DashboardSnapshot(
        tick=1,
        rows=[row_aapl],
        recent_alerts=(
            Alert(symbol="AAPL", rule="action_changed", severity="critical",
                  message="AAPL flipped HOLD->BUY", fired_at=fired),
            Alert(symbol="MSFT", rule="confidence_threshold", severity="info",
                  message="MSFT crossed 0.6", fired_at=fired),
        ),
    )
    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        detail = app.query_one(DetailPanel)
        detail.snapshot = snap
        detail.row_index = 0
        text = detail.render()
        # AAPL's alert appears, MSFT's does not.
        assert "AAPL flipped HOLD->BUY" in text
        assert "MSFT crossed 0.6" not in text
        assert "Alerts" in text  # the section header


async def test_detail_panel_omits_alerts_section_when_symbol_has_none() -> None:
    from src.dashboard.app import DashboardApp, DetailPanel
    from src.dashboard.state import DashboardSnapshot, RecommendationRow
    from src.intelligence.alerts import Alert

    fired = datetime.now(UTC)
    row_aapl = RecommendationRow(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.5,
        combined_score=0.5,
        technical_score=0.5,
        sentiment_score=0.0,
        rsi=0.5,
        macd=0.5,
        bollinger=0.5,
        last_price=150.0,
        num_news_articles=0,
        reasoning="",
        timestamp=fired,
    )
    # MSFT has alerts; AAPL doesn't. Looking at AAPL → no Alerts section.
    snap = DashboardSnapshot(
        tick=1,
        rows=[row_aapl],
        recent_alerts=(
            Alert(symbol="MSFT", rule="action_changed", severity="warn",
                  message="x", fired_at=fired),
        ),
    )
    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        detail = app.query_one(DetailPanel)
        detail.snapshot = snap
        detail.row_index = 0
        text = detail.render()
        # No Alerts section since AAPL has no alerts.
        # Use a tight check that doesn't false-match other text:
        assert "[bold cyan]Alerts[/]" not in text


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


# ---------------------------------------------------------------------------
# Multi-timeframe column + Timeframes detail-panel section
# ---------------------------------------------------------------------------


def _mtf_view(
    *,
    short: str = "bullish",
    medium: str = "bullish",
    long: str = "bullish",
    short_pct: float = 3.0,
    medium_pct: float = 8.0,
    long_pct: float = 15.0,
    alignment: str = "aligned_bullish",
    alignment_strength: float = 0.6,
) -> object:
    """Construct a MultiTimeframeView with sane defaults for rendering tests."""
    from src.intelligence.multi_timeframe import (
        MultiTimeframeView,
        TimeframeTrend,
    )

    def _t(horizon: str, direction: str, pct: float, bars: int) -> TimeframeTrend:
        return TimeframeTrend(
            horizon=horizon,
            direction=direction,
            strength=min(1.0, abs(pct) / 20.0),
            delta_pct=pct,
            bars_used=bars,
        )

    return MultiTimeframeView(
        symbol="AAPL",
        short=_t("short", short, short_pct, 5),
        medium=_t("medium", medium, medium_pct, 20),
        long=_t("long", long, long_pct, 60),
        alignment=alignment,
        alignment_strength=alignment_strength,
    )


def test_mtf_cell_renders_aligned_bullish_with_three_up_arrows() -> None:
    from src.dashboard.app import _mtf_cell

    cell = _mtf_cell(_mtf_view())  # type: ignore[arg-type]
    assert "↑↑↑" in cell
    assert "green" in cell


def test_mtf_cell_renders_aligned_bearish_in_red() -> None:
    from src.dashboard.app import _mtf_cell

    view = _mtf_view(
        short="bearish",
        medium="bearish",
        long="bearish",
        short_pct=-3.0,
        medium_pct=-8.0,
        long_pct=-15.0,
        alignment="aligned_bearish",
    )
    cell = _mtf_cell(view)  # type: ignore[arg-type]
    assert "↓↓↓" in cell
    assert "red" in cell


def test_mtf_cell_renders_short_reversal_with_mixed_glyphs() -> None:
    """Short bearish + long bullish — the headline 'pullback in uptrend'
    pattern. Cell shows actual per-horizon direction glyphs colored
    yellow."""
    from src.dashboard.app import _mtf_cell

    view = _mtf_view(
        short="bearish",
        medium="neutral",
        long="bullish",
        short_pct=-3.0,
        medium_pct=0.5,
        long_pct=12.0,
        alignment="short_reversal",
    )
    cell = _mtf_cell(view)  # type: ignore[arg-type]
    assert "↓·↑" in cell
    assert "yellow" in cell


def test_mtf_cell_renders_mixed_in_dim() -> None:
    from src.dashboard.app import _mtf_cell

    view = _mtf_view(
        short="bullish",
        medium="neutral",
        long="neutral",
        short_pct=2.0,
        medium_pct=0.5,
        long_pct=1.0,
        alignment="mixed",
    )
    cell = _mtf_cell(view)  # type: ignore[arg-type]
    assert "↑··" in cell
    assert "dim" in cell


def test_mtf_cell_renders_dim_dash_when_view_is_none() -> None:
    from src.dashboard.app import _mtf_cell

    cell = _mtf_cell(None)
    assert "—" in cell
    assert "dim" in cell


def test_mtf_strength_label_tiers() -> None:
    from src.dashboard.app import _mtf_strength_label

    assert _mtf_strength_label(0.1) == "weak"
    assert _mtf_strength_label(0.33) == "moderate"
    assert _mtf_strength_label(0.5) == "moderate"
    assert _mtf_strength_label(0.67) == "strong"
    assert _mtf_strength_label(1.0) == "strong"


def test_render_timeframes_block_contains_three_horizon_lines() -> None:
    from src.dashboard.app import _render_timeframes_block

    block = _render_timeframes_block(_mtf_view())  # type: ignore[arg-type]
    # Each horizon label appears on its own line.
    assert "short" in block
    assert "medium" in block
    assert "long" in block
    # Direction labels render.
    assert "bullish" in block
    # Signed pct change renders.
    assert "+3.0%" in block or "+3.00%" in block or "+3.0" in block
    # Alignment label and strength surface on the last line.
    assert "aligned_bullish" in block
    assert "strength" in block


def test_render_timeframes_block_shows_bar_counts() -> None:
    """The trader needs to know which window each horizon used — if
    we fall back to a short df, the value depends on len(df)."""
    from src.dashboard.app import _render_timeframes_block

    block = _render_timeframes_block(_mtf_view())  # type: ignore[arg-type]
    assert "5 bars" in block
    assert "20 bars" in block
    assert "60 bars" in block


async def test_app_table_renders_mtf_column() -> None:
    """End-to-end smoke: MockController produces rows with MTF; the
    table has an MTF column header."""
    from textual.widgets import DataTable

    from src.dashboard.app import DashboardApp

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL", "MSFT"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        table = app.query_one(DataTable)
        column_labels = [str(c.label) for c in table.columns.values()]
        assert "MTF" in column_labels


async def test_app_detail_panel_includes_timeframes_section() -> None:
    """Smoke: when the mock controller produces a row with MTF, the
    detail panel includes the Timeframes block."""
    from src.dashboard.app import DashboardApp, DetailPanel

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL", "MSFT"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        detail = app.query_one(DetailPanel)
        rendered = detail.render()
        # Render() returns a Rich-markup string; the section header is plain text.
        assert "Timeframes" in rendered


async def test_app_detail_panel_omits_timeframes_when_mtf_none() -> None:
    """If a row has mtf=None (df too short), the section is omitted —
    no empty 'Timeframes' header with no content."""
    from dataclasses import replace

    from src.dashboard.app import DashboardApp, DetailPanel

    app = DashboardApp(
        MockDashboardController(watchlist=["AAPL"]),
        refresh_seconds=999.0,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        # Strip the mtf field from the snapshot's row to simulate the
        # short-df fallback path.
        snap = app._snapshot
        assert snap is not None
        snap.rows[:] = [replace(snap.rows[0], mtf=None)]
        detail = app.query_one(DetailPanel)
        # Invalidate cached signature so render runs again.
        detail._last_signature = None
        detail.snapshot = snap
        rendered = detail.render()
        assert "Timeframes" not in rendered


def test_recommendation_row_carries_mtf_field() -> None:
    """The dashboard row schema includes the new mtf field with default
    None — guards against accidental removal during refactors."""
    from src.dashboard.state import RecommendationRow

    fields = RecommendationRow.__dataclass_fields__
    assert "mtf" in fields
    # Default is None so legacy callers / tests don't have to set it.
    assert fields["mtf"].default is None


def test_no_prediction_language_in_mtf_render_helpers() -> None:
    """Regression guard: the rendered strings carry no forecast / prediction
    language. Same anti-hallucination contract as pulse / opportunities."""
    from src.dashboard.app import _mtf_cell, _render_timeframes_block

    view = _mtf_view(
        short="bearish",
        medium="neutral",
        long="bullish",
        short_pct=-3.0,
        medium_pct=0.5,
        long_pct=12.0,
        alignment="short_reversal",
    )
    combined = _mtf_cell(view) + "\n" + _render_timeframes_block(view)  # type: ignore[arg-type]
    lowered = combined.lower()
    for word in ("likely", "will rise", "will fall", "expected to", "forecast", "predict"):
        assert word not in lowered, f"forbidden word {word!r} in render"
