"""Pure-data structures the dashboard UI renders.

No Textual / Rich imports here — keeping these decoupled lets the
controller layer be unit-tested without spinning up a TUI.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.strategy.base import RecommendationTier, SignalAction

if TYPE_CHECKING:
    from src.intelligence.alerts import Alert
    from src.intelligence.history import SignalHistorySummary
    from src.intelligence.opportunities import RankedOpportunity
    from src.intelligence.opportunity_history import OpportunityHistory
    from src.intelligence.pulse import MarketPulse
    from src.intelligence.pulse_evolution import PulseEvolution
    from src.intelligence.pulse_history import PulseHistory
    from src.persistence.session_store import SessionStoreStatus
    from src.strategy.multi_timeframe import IntradayRead


@dataclass(frozen=True)
class RecommendationRow:
    """One symbol's worth of dashboard data for the watchlist table."""

    symbol: str
    action: SignalAction
    confidence: float  # [0, 1]
    combined_score: float  # [-1, 1]
    technical_score: float
    sentiment_score: float
    rsi: float
    macd: float
    bollinger: float
    last_price: float
    num_news_articles: int
    reasoning: str
    timestamp: datetime
    # Most-recent headline strings (top ~5), kept on the row so the dashboard
    # AI-brief worker can pass them to the LLM without a second Alpaca fetch.
    headlines: tuple[str, ...] = ()
    # 5-tier recommendation surface (filled by promote_to_tier in the
    # controller after engine.recommend()). Defaults preserve the
    # invariant that tier always exists; STRONG only appears when the
    # promoter's four gates pass.
    tier: RecommendationTier = RecommendationTier.HOLD
    signal_quality: str = "moderate"
    stability: str = "stable"
    quality_reasons: tuple[str, ...] = ()
    # Optional secondary intraday read — populated only when the
    # controller's ``intraday_enabled`` Setting is on. ``None`` on
    # symbols where the intraday fetch failed or the setting is off.
    intraday: IntradayRead | None = None
    error: str | None = None  # populated if the per-symbol fetch failed


@dataclass(frozen=True)
class EventEntry:
    """A single line in the events / log panel.

    ``count`` tracks how many times this same (level, message) was
    pushed consecutively; the buffer collapses duplicates into one
    entry instead of accumulating them, and the dashboard renders an
    "× N" suffix when ``count > 1``.
    """

    timestamp: datetime
    level: str  # "info" | "warn" | "error"
    message: str
    count: int = 1


@dataclass
class DashboardSnapshot:
    """One frame of dashboard state, produced by the controller each tick."""

    tick: int
    rows: list[RecommendationRow]
    events: list[EventEntry] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    recent_alerts: tuple[Alert, ...] = ()
    signal_history: dict[str, SignalHistorySummary] = field(default_factory=dict)
    # Per-symbol top-N opportunity-membership history, populated by the
    # controller for symbols currently in top-N. Symbols outside top-N
    # are intentionally absent (the renderer only consults this for OPP
    # lines it's already drawing).
    opp_history: dict[str, OpportunityHistory] = field(default_factory=dict)
    # Pre-computed top-N ranked opportunities for the current tick.
    # Carried on the snapshot so HTTP / web clients render the same
    # composite-score ordering as the TUI without having to re-run
    # ``rank_opportunities`` themselves. Empty tuple when no symbol
    # qualifies (default RankedOpportunity threshold).
    ranked_opportunities: tuple[RankedOpportunity, ...] = ()
    # Pulse: pre-computed once by the controller so the renderer,
    # signature cache, status line, and snapshot rules all see the
    # same value. ``None`` means the snapshot was constructed outside
    # the controller (unit tests, fixtures) — the renderer falls back
    # to recomputing in that case.
    pulse: MarketPulse | None = None
    # Rolling-window history of recent pulses. ``None`` until the
    # controller's tracker has at least one entry — same fallback
    # semantics as ``pulse``.
    pulse_history: PulseHistory | None = None
    # Trajectory-level synthesis (regime + patterns) over the pulse
    # history. ``None`` when history is too thin for a meaningful
    # classification — the renderer treats that as "skip the chip".
    pulse_evolution: PulseEvolution | None = None
    # Parallel intraday signal history — populated only when
    # ``settings.intraday_enabled`` is on and the row carries an
    # IntradayRead. MT2 phase 2a foundation; downstream phases will
    # surface this alongside the daily ``signal_history`` field.
    intraday_signal_history: dict[str, SignalHistorySummary] = field(default_factory=dict)
    # Parallel intraday top-N opportunity history (MT2 phase 2b).
    # Same shape as the daily ``opp_history`` but populated from
    # ``rank_opportunities_intraday`` + the controller's parallel
    # intraday tracker.
    intraday_opp_history: dict[str, OpportunityHistory] = field(default_factory=dict)
    # Parallel intraday ranked opportunities. Mirrors
    # ``ranked_opportunities`` but sourced from
    # ``rank_opportunities_intraday`` so the active-view OPP panel can
    # flip between timeframes without recomputing client-side.
    intraday_ranked_opportunities: tuple[RankedOpportunity, ...] = ()
    # Parallel intraday pulse triad (MT2 phase 2c). All three fields
    # default to ``None`` so daily-only sessions render unchanged.
    intraday_pulse: MarketPulse | None = None
    intraday_pulse_history: PulseHistory | None = None
    intraday_pulse_evolution: PulseEvolution | None = None
    # SessionStore health snapshot (polish item #3). Populated by
    # the controller after each tick's persist when a session store
    # is wired. ``None`` when persistence is disabled — the status
    # chip then renders empty.
    session_store_status: SessionStoreStatus | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


def filter_snapshot_for_symbols(
    snap: DashboardSnapshot, allowed: frozenset[str] | None
) -> DashboardSnapshot:
    """Return ``snap`` filtered down to a user's allowed symbol set.

    ``allowed=None`` is the firehose path — used by the TUI and any
    operator endpoint that wants the full controller view. When a set
    is given, every per-symbol collection on the snapshot is filtered
    to only those symbols; watchlist-wide aggregates (pulse, regime,
    session-store health, events, tick, timestamp) pass through
    unchanged because they describe the operating environment, not
    any individual user's positions.

    Symbol matching is case-insensitive — the WatchlistStore upper-cases
    on write, but defence-in-depth: a stray lower-case key never leaks
    past the boundary.
    """
    if allowed is None:
        return snap
    norm = frozenset(s.upper() for s in allowed)

    def _keep_row(sym: str) -> bool:
        return sym.upper() in norm

    rows = [r for r in snap.rows if _keep_row(r.symbol)]
    alerts = [a for a in snap.alerts if _keep_row(a.symbol)]
    recent_alerts = tuple(a for a in snap.recent_alerts if _keep_row(a.symbol))
    signal_history = {s: v for s, v in snap.signal_history.items() if _keep_row(s)}
    opp_history = {s: v for s, v in snap.opp_history.items() if _keep_row(s)}
    ranked = tuple(o for o in snap.ranked_opportunities if _keep_row(o.symbol))
    intraday_history = {
        s: v for s, v in snap.intraday_signal_history.items() if _keep_row(s)
    }
    intraday_opp_history = {
        s: v for s, v in snap.intraday_opp_history.items() if _keep_row(s)
    }
    intraday_ranked = tuple(
        o for o in snap.intraday_ranked_opportunities if _keep_row(o.symbol)
    )

    return replace(
        snap,
        rows=rows,
        alerts=alerts,
        recent_alerts=recent_alerts,
        signal_history=signal_history,
        opp_history=opp_history,
        ranked_opportunities=ranked,
        intraday_signal_history=intraday_history,
        intraday_opp_history=intraday_opp_history,
        intraday_ranked_opportunities=intraday_ranked,
    )


class EventBuffer:
    """Bounded FIFO of EventEntry — used by the controller to accumulate log lines.

    Consecutive duplicate pushes (same level + message as the most-recent
    entry) are collapsed: the buffer's last entry has its ``count``
    bumped and ``timestamp`` advanced to the latest occurrence. The
    dashboard's events pane is then responsible for showing the updated
    count — see ``DashboardApp._render_events``.
    """

    def __init__(self, capacity: int = 200) -> None:
        self._buf: deque[EventEntry] = deque(maxlen=capacity)

    def push(self, level: str, message: str) -> EventEntry:
        now = datetime.now(UTC)
        if self._buf and self._buf[-1].level == level and self._buf[-1].message == message:
            # Collapse the duplicate: bump count, advance timestamp.
            last = self._buf[-1]
            bumped = replace(last, count=last.count + 1, timestamp=now)
            self._buf[-1] = bumped
            return bumped
        ev = EventEntry(timestamp=now, level=level, message=message)
        self._buf.append(ev)
        return ev

    def info(self, message: str) -> EventEntry:
        return self.push("info", message)

    def warn(self, message: str) -> EventEntry:
        return self.push("warn", message)

    def error(self, message: str) -> EventEntry:
        return self.push("error", message)

    def snapshot(self) -> list[EventEntry]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)
