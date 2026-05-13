"""Controllers that produce DashboardSnapshots for the UI to render.

Two flavours:

* :class:`DashboardController` — live: pulls bars + news from Alpaca and
  scores with :class:`RecommendationEngine`. Paper-feed only.
* :class:`MockDashboardController` — synthetic OHLCV + news, no Alpaca
  required. Useful for local demos and tests.
"""

from __future__ import annotations

import asyncio
import math
import random
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.dashboard.state import (
    DashboardSnapshot,
    EventBuffer,
    RecommendationRow,
)
from src.data import cache as bar_cache
from src.data.alpaca_client import AlpacaClient
from src.data.market_data import MarketDataService
from src.data.models import NewsArticle, TimeFrame
from src.data.news_ingestion import NewsSource, get_news_source
from src.intelligence.alert_prioritizer import AlertPrioritizer, AlertState
from src.intelligence.alerts import AlertEngine
from src.intelligence.history import SignalHistory, SignalHistorySummary
from src.intelligence.opportunities import rank_opportunities
from src.intelligence.opportunity_history import (
    OpportunityHistory,
    OpportunityMembershipTracker,
)
from src.intelligence.pulse import compute_pulse
from src.intelligence.pulse_history import PulseHistoryTracker
from src.intelligence.tier import promote_to_tier
from src.persistence.session_store import SessionSnapshot, SessionStore
from src.sentiment.analyzer import SentimentAnalyzer, SentimentLabel, SentimentScore
from src.strategy.base import SignalAction
from src.strategy.recommendation import RecommendationEngine

# ---------------------------------------------------------------------------
# Base controller
# ---------------------------------------------------------------------------


_OPP_TOP_N = 3
"""Canonical top-N for opportunity ranking. Matches the renderer's
:func:`rank_opportunities` call and OpportunityEntryRule's default.
Picking one source of truth here means the membership tracker, the
entry-alert rule, and the rendered OPP lines all agree on what
"top-N" means within a session."""


class BaseController(ABC):
    """Common state — tick counter, event buffer, watchlist, alert engine."""

    def __init__(
        self,
        *,
        watchlist: list[str],
        engine: RecommendationEngine,
        events: EventBuffer | None = None,
        alert_engine: AlertEngine | None = None,
        alert_prioritizer: AlertPrioritizer | None = None,
        alert_state: AlertState | None = None,
        signal_history: SignalHistory | None = None,
        opp_tracker: OpportunityMembershipTracker | None = None,
        pulse_tracker: PulseHistoryTracker | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.watchlist = list(watchlist)
        self.engine = engine
        self.events = events or EventBuffer()
        self.alert_engine = alert_engine or AlertEngine()
        self.alert_prioritizer = alert_prioritizer or AlertPrioritizer()
        self.alert_state = alert_state or AlertState()
        self.signal_history = signal_history or SignalHistory()
        self.opp_tracker = opp_tracker or OpportunityMembershipTracker()
        self.pulse_tracker = pulse_tracker or PulseHistoryTracker()
        self.session_store = session_store
        self._tick = 0
        # If a SessionStore is wired up, hydrate the trackers + tick
        # counter from the saved snapshot. Missing or corrupt file →
        # cold start; logged at the store layer.
        if self.session_store is not None:
            self._hydrate_from_store()

    @abstractmethod
    async def fetch_snapshot(self) -> DashboardSnapshot:
        """Produce one frame of dashboard state."""

    def add_symbol(self, symbol: str) -> bool:
        """Append ``symbol`` to the watchlist for the next tick onward.

        Returns ``True`` if added, ``False`` if it was already present
        (no duplicate insert). The symbol is uppercased and stripped
        before comparison; downstream fetches will surface a friendly
        error row for any symbol the data layer can't resolve.

        Cold-start indicators may be NaN on the first tick — the
        controller's per-symbol fetch path warms up naturally as bars
        accumulate, but pre-populating via ``esther backfill`` is the
        cleaner path.
        """
        sym = symbol.strip().upper()
        if not sym or sym in self.watchlist:
            return False
        self.watchlist.append(sym)
        return True

    def remove_symbol(self, symbol: str) -> bool:
        """Drop ``symbol`` from the watchlist for the next tick onward.

        Returns ``True`` if removed, ``False`` if it wasn't present.
        Per-symbol session state (signal_history, opp tracker membership,
        alert engine baselines) is intentionally left in place — if the
        symbol gets re-added later the historical context resumes; if
        not, the orphaned state costs nothing and the trackers
        self-prune over their windows.
        """
        sym = symbol.strip().upper()
        if sym not in self.watchlist:
            return False
        self.watchlist.remove(sym)
        return True

    def _record_pulse(self, snap: DashboardSnapshot) -> None:
        """Compute the pulse once + record it into the rolling-window
        tracker, then attach both ``snap.pulse`` and ``snap.pulse_history``.

        Computing in the controller (rather than re-computing in every
        renderer that needs it) means the WatchlistHeader, StatusLine,
        and the header signature cache all see the same value — and the
        rolling history reflects the *rendered* pulse, not a slightly
        different recomputation.
        """
        pulse = compute_pulse(snap)
        self.pulse_tracker.record(pulse)
        snap.pulse = pulse
        snap.pulse_history = self.pulse_tracker.summary()

    def _record_opp_history(self, snap: DashboardSnapshot) -> dict[str, OpportunityHistory]:
        """Update the membership tracker for this tick + return per-symbol
        summaries for the symbols currently in top-N.

        Symbols outside top-N are intentionally excluded from the result
        — the renderer only consults this for OPP lines it draws, and
        carrying summaries for absent symbols would just be dead weight
        on the snapshot.
        """
        ranked = rank_opportunities(snap, n=_OPP_TOP_N)
        top_symbols = [opp.symbol for opp in ranked]
        self.opp_tracker.record(top_symbols)
        out: dict[str, OpportunityHistory] = {}
        for symbol in top_symbols:
            summary = self.opp_tracker.summary_for(symbol)
            if summary is not None:
                out[symbol] = summary
        return out

    def _record_history(
        self, rows: list[RecommendationRow], now: datetime
    ) -> dict[str, SignalHistorySummary]:
        """Append healthy rows to history and return per-symbol summaries.

        Error rows are intentionally NOT recorded — a transient fetch
        failure shouldn't reset the episode counter for that symbol.
        """
        for row in rows:
            if row.error:
                continue
            self.signal_history.record(row.symbol, row.action, row.confidence, now)
        summaries: dict[str, SignalHistorySummary] = {}
        for row in rows:
            summary = self.signal_history.summary_for(row.symbol)
            if summary is not None:
                summaries[row.symbol] = summary
        return summaries

    def _hydrate_from_store(self) -> None:
        """Replace tracker state with the saved snapshot if one exists.

        Called once from ``__init__`` when a ``SessionStore`` is
        wired up. Missing / corrupt / schema-mismatched snapshots
        return ``None`` from the store and we leave the freshly-
        constructed trackers in place — i.e. a cold start.
        """
        assert self.session_store is not None
        snapshot = self.session_store.load()
        if snapshot is None:
            return
        self.signal_history.apply_snapshot(snapshot.signal_episodes)
        self.opp_tracker.apply_snapshot(snapshot.opp_membership)
        self.pulse_tracker.apply_snapshot(snapshot.pulse_records)
        self.alert_state.apply_snapshot(snapshot.alert_log, snapshot.alert_last_fired)
        self._tick = snapshot.tick
        self.events.info(
            f"session restored from {self.session_store.path.name} (tick {snapshot.tick})"
        )

    def _persist_if_enabled(self) -> None:
        """Write the current tracker state to the session store.

        No-op when no store is wired up. Called at the end of each
        ``fetch_snapshot`` in the concrete controllers.
        """
        if self.session_store is None:
            return
        alert_log, alert_last_fired = self.alert_state.to_snapshot()
        snapshot = SessionSnapshot(
            saved_at=datetime.now(UTC),
            tick=self._tick,
            signal_episodes=self.signal_history.to_snapshot(),
            opp_membership=self.opp_tracker.to_snapshot(),
            pulse_records=self.pulse_tracker.to_snapshot(),
            alert_log=alert_log,
            alert_last_fired=alert_last_fired,
        )
        self.session_store.save(snapshot)


# ---------------------------------------------------------------------------
# Live controller
# ---------------------------------------------------------------------------


class DashboardController(BaseController):
    """Live data: Alpaca bars + news + engine recommendations."""

    def __init__(
        self,
        *,
        watchlist: list[str],
        engine: RecommendationEngine,
        market: MarketDataService | None = None,
        news_source: NewsSource | None = None,
        settings: Settings | None = None,
        lookback_days: int = 60,
        news_hours: int = 48,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        events: EventBuffer | None = None,
        alert_engine: AlertEngine | None = None,
        alert_prioritizer: AlertPrioritizer | None = None,
        alert_state: AlertState | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        super().__init__(
            watchlist=watchlist,
            engine=engine,
            events=events,
            alert_engine=alert_engine,
            alert_prioritizer=alert_prioritizer,
            alert_state=alert_state,
            session_store=session_store,
        )
        self.settings = settings or get_settings()
        if not self.settings.is_paper_trading:
            raise RuntimeError(
                f"Dashboard refuses non-paper Alpaca URL (got {self.settings.alpaca_base_url!r})"
            )
        self.market = market or MarketDataService(client=AlpacaClient(self.settings))
        self.news_source = news_source or get_news_source()
        self.lookback_days = lookback_days
        self.news_hours = news_hours
        self.timeframe = timeframe

    async def fetch_snapshot(self) -> DashboardSnapshot:
        self._tick += 1
        now = datetime.now(UTC)
        self.events.info(f"tick {self._tick} start ({len(self.watchlist)} symbols)")
        rows = await asyncio.gather(
            *(self._row_for(sym, now) for sym in self.watchlist),
            return_exceptions=False,
        )
        rows_list = list(rows)
        history = self._record_history(rows_list, now)
        # Build the snapshot first so snapshot-level rules can see the
        # same view the dashboard will render. Alerts get filled in
        # after prioritization.
        snap = DashboardSnapshot(
            tick=self._tick,
            rows=rows_list,
            events=self.events.snapshot(),
            signal_history=history,
            timestamp=now,
        )
        self._record_pulse(snap)
        snap.opp_history = self._record_opp_history(snap)
        fresh_alerts = self.alert_engine.evaluate(rows_list)
        fresh_alerts.extend(self.alert_engine.evaluate_snapshot(snap))
        snap.alerts = self.alert_prioritizer.prioritize(fresh_alerts, self.alert_state, now=now)
        self.alert_state.record(snap.alerts)
        snap.recent_alerts = self.alert_state.recent(20)
        self._persist_if_enabled()
        return snap

    async def _row_for(self, symbol: str, now: datetime) -> RecommendationRow:
        bars_start = now - timedelta(days=self.lookback_days)
        news_start = now - timedelta(hours=self.news_hours)
        try:
            bars = await self.market.get_bars([symbol], self.timeframe, bars_start, now)
            df = self.market.to_dataframe(bars)
            if not df.empty:
                for c in ("open", "high", "low", "close"):
                    df[c] = df[c].astype(float)
                df["volume"] = df["volume"].astype(int)
            # Prepend any cached history so first-launch indicators have warmup.
            df = bar_cache.merge_with_cache(symbol, self.timeframe, df)
            if df.empty:
                self.events.warn(f"{symbol}: no bars returned and no cache")
                return _empty_row(symbol, now, error="no bars")
            news = await self.news_source.fetch([symbol], news_start, now, limit=20)
            rec = self.engine.recommend(symbol, df, news=news, now=now)
            # Promote using the session history we'll record below.
            history = self.signal_history.summary_for(symbol)
            rec = promote_to_tier(rec, history)
            self.events.info(
                f"{symbol} → {rec.tier.display} "
                f"(conf {rec.confidence:.2f}, combined {rec.combined_score:+.2f})"
            )
            top_headlines = tuple(
                a.headline for a in sorted(news, key=lambda a: a.published_at, reverse=True)[:5]
            )
            return _row_from_recommendation(
                rec,
                last_price=float(df["close"].iloc[-1]),
                headlines=top_headlines,
            )
        except Exception as e:
            self.events.error(f"{symbol}: {e}")
            return _empty_row(symbol, now, error=str(e))


# ---------------------------------------------------------------------------
# Mock controller — for local demo without Alpaca
# ---------------------------------------------------------------------------


class _NeutralAnalyzer(SentimentAnalyzer):
    """Skip FinBERT — emit a fixed neutral score."""

    def __init__(self) -> None:
        pass

    def score_text(self, _t: str) -> SentimentScore:
        return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

    def score_article(self, _a: NewsArticle) -> SentimentScore:
        return self.score_text("")


class _RandomSentiment(SentimentAnalyzer):
    """Random positive/negative — for the mock dashboard to show variation."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def score_text(self, _t: str) -> SentimentScore:
        roll = self._rng.random()
        if roll < 0.4:
            return SentimentScore(SentimentLabel.POSITIVE, 0.6 + self._rng.random() * 0.3)
        if roll < 0.7:
            return SentimentScore(SentimentLabel.NEGATIVE, 0.5 + self._rng.random() * 0.3)
        return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

    def score_article(self, article: NewsArticle) -> SentimentScore:
        return self.score_text(article.headline)


class MockDashboardController(BaseController):
    """Drives the UI with synthetic OHLCV + headlines. No Alpaca needed."""

    def __init__(
        self,
        *,
        watchlist: list[str] | None = None,
        seed: int = 7,
        use_sentiment: bool = True,
        events: EventBuffer | None = None,
        alert_engine: AlertEngine | None = None,
        alert_prioritizer: AlertPrioritizer | None = None,
        alert_state: AlertState | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        watchlist = watchlist or ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]
        rng = random.Random(seed)
        analyzer = _RandomSentiment(rng) if use_sentiment else _NeutralAnalyzer()
        engine = RecommendationEngine(sentiment_analyzer=analyzer)
        super().__init__(
            watchlist=watchlist,
            engine=engine,
            events=events,
            alert_engine=alert_engine,
            alert_prioritizer=alert_prioritizer,
            alert_state=alert_state,
            session_store=session_store,
        )
        self._rng = rng
        self._np_rng = np.random.default_rng(seed)
        # Per-symbol drift/noise — gives BUYs/SELLs different reasons across rows.
        self._sym_profiles = {
            s: {"drift": rng.uniform(-0.005, 0.015), "noise": rng.uniform(0.008, 0.02)}
            for s in watchlist
        }

    async def fetch_snapshot(self) -> DashboardSnapshot:
        self._tick += 1
        now = datetime.now(UTC)
        self.events.info(f"mock tick {self._tick} ({len(self.watchlist)} symbols)")
        rows = [self._mock_row(sym, now) for sym in self.watchlist]
        history = self._record_history(rows, now)
        snap = DashboardSnapshot(
            tick=self._tick,
            rows=rows,
            events=self.events.snapshot(),
            signal_history=history,
            timestamp=now,
        )
        self._record_pulse(snap)
        snap.opp_history = self._record_opp_history(snap)
        fresh_alerts = self.alert_engine.evaluate(rows)
        fresh_alerts.extend(self.alert_engine.evaluate_snapshot(snap))
        snap.alerts = self.alert_prioritizer.prioritize(fresh_alerts, self.alert_state, now=now)
        self.alert_state.record(snap.alerts)
        snap.recent_alerts = self.alert_state.recent(20)
        self._persist_if_enabled()
        return snap

    def _mock_row(self, symbol: str, now: datetime) -> RecommendationRow:
        prof = self._sym_profiles[symbol]
        # Build a fresh synthetic OHLCV window for each tick — adds slow
        # variation so the dashboard isn't a static screenshot.
        n = 80
        # Vary the seed each tick so rows refresh meaningfully.
        rng = np.random.default_rng(
            self._np_rng.integers(0, 2**31 - 1) + hash(symbol) % 1000 + self._tick
        )
        rets = rng.normal(loc=prof["drift"], scale=prof["noise"], size=n)
        close = 100.0 * np.exp(np.cumsum(rets))
        idx = pd.bdate_range(end=now, periods=n)
        # bdate_range may return n-1; align.
        if len(idx) < n:
            close = close[-len(idx) :]
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.003,
                "low": close * 0.997,
                "close": close,
                "volume": np.full(len(close), 1_000_000, dtype=int),
            },
            index=idx,
        )
        news = [
            NewsArticle(
                id=f"{symbol}-{self._tick}-{i}",
                headline=self._rng.choice(
                    [
                        f"{symbol} beats analyst expectations on revenue",
                        f"{symbol} announces leadership change",
                        f"{symbol} faces regulatory headwinds",
                        f"{symbol} unveils new product line",
                        f"{symbol} cuts guidance amid macro pressure",
                    ]
                ),
                source="mock",
                symbols=[symbol],
                published_at=now - timedelta(hours=self._rng.randint(1, 24)),
            )
            for i in range(self._rng.randint(0, 4))
        ]
        rec = self.engine.recommend(symbol, df, news=news, now=now)
        history = self.signal_history.summary_for(symbol)
        rec = promote_to_tier(rec, history)
        self.events.info(f"{symbol} → {rec.tier.display} (conf {rec.confidence:.2f})")
        top_headlines = tuple(
            a.headline for a in sorted(news, key=lambda a: a.published_at, reverse=True)[:5]
        )
        return _row_from_recommendation(
            rec,
            last_price=float(df["close"].iloc[-1]),
            headlines=top_headlines,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_from_recommendation(
    rec: object, last_price: float, headlines: tuple[str, ...] = ()
) -> RecommendationRow:
    """Convert a TradingRecommendation to a dashboard row."""
    indicators = getattr(rec, "indicator_scores", {}) or {}
    return RecommendationRow(
        symbol=rec.symbol,  # type: ignore[attr-defined]
        action=rec.action,  # type: ignore[attr-defined]
        confidence=float(rec.confidence),  # type: ignore[attr-defined]
        combined_score=float(rec.combined_score),  # type: ignore[attr-defined]
        technical_score=float(rec.technical_score),  # type: ignore[attr-defined]
        sentiment_score=float(rec.sentiment_score),  # type: ignore[attr-defined]
        rsi=float(indicators.get("rsi", math.nan)),
        macd=float(indicators.get("macd", math.nan)),
        bollinger=float(indicators.get("bollinger", math.nan)),
        last_price=last_price,
        num_news_articles=int(rec.num_news_articles),  # type: ignore[attr-defined]
        reasoning=str(rec.reasoning),  # type: ignore[attr-defined]
        timestamp=rec.timestamp,  # type: ignore[attr-defined]
        headlines=headlines,
        tier=rec.tier,  # type: ignore[attr-defined]
        signal_quality=str(rec.signal_quality),  # type: ignore[attr-defined]
        stability=str(rec.stability),  # type: ignore[attr-defined]
        quality_reasons=tuple(rec.quality_reasons),  # type: ignore[attr-defined]
    )


def _empty_row(symbol: str, now: datetime, *, error: str) -> RecommendationRow:
    return RecommendationRow(
        symbol=symbol,
        action=SignalAction.HOLD,
        confidence=0.0,
        combined_score=0.0,
        technical_score=0.0,
        sentiment_score=0.0,
        rsi=math.nan,
        macd=math.nan,
        bollinger=math.nan,
        last_price=math.nan,
        num_news_articles=0,
        reasoning="",
        timestamp=now,
        error=error,
    )
