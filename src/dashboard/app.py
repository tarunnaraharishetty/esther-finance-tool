"""Textual dashboard app.

Layout::

    ┌─ Header (env, paper, tick clock) ──────────────────────────────┐
    │ ┌─ Recommendations (DataTable) ──────────────────────────────┐ │
    │ │ SYM  ACT  CONF  TECH  SENT  RSI ...                        │ │
    │ │ ...                                                        │ │
    │ └────────────────────────────────────────────────────────────┘ │
    │ ┌─ Selected row detail ──────────────────────────────────────┐ │
    │ │ AAPL  reasoning: ...                                       │ │
    │ └────────────────────────────────────────────────────────────┘ │
    │ ┌─ Events (RichLog) ─────────────────────────────────────────┐ │
    │ │ 19:50:12 tick start ...                                    │ │
    │ └────────────────────────────────────────────────────────────┘ │
    └─ q quit · r refresh · p pause · ↑↓ select ────────────────────┘
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rich.markup import escape as rich_escape
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, RichLog, Static

from src.dashboard.state import DashboardSnapshot, RecommendationRow
from src.intelligence.explain import Explanation, explain
from src.intelligence.history import SignalEpisode, SignalHistorySummary
from src.intelligence.watchlist import (
    action_breakdown,
    diff_snapshots,
    top_movers,
)
from src.strategy.base import SignalAction

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController
    from src.intelligence.summary import Summarizer


_ACTION_STYLES = {
    SignalAction.BUY: "bold green",
    SignalAction.SELL: "bold red",
    SignalAction.HOLD: "yellow",
}


def _fmt_signed(x: float) -> str:
    """Signed score rendered with sign-based color markup.

    Positive = green, negative = red, near-zero = yellow. NaN renders as
    a plain dim dash so empty cells stay visually quiet.
    """
    if x != x:  # NaN
        return "[dim]  -  [/dim]"
    return f"[{_score_style(x)}]{x:+.2f}[/]"


def _fmt_price(x: float) -> str:
    if x != x:
        return "—"
    return f"${x:,.2f}"


def _action_text(action: SignalAction) -> str:
    style = _ACTION_STYLES.get(action, "")
    return f"[{style}]{action.value.upper():4}[/{style}]"


def _confidence_bar(conf: float, width: int = 10) -> str:
    """Mini ASCII bar — width chars filled proportional to conf in [0,1]."""
    filled = max(0, min(width, int(round(conf * width))))
    return "█" * filled + "·" * (width - filled)


class WatchlistHeader(Static):
    """Three-line header above the watchlist: movers / action mix / diff.

    Holds an internal reference to the previous snapshot so it can render
    "Since last refresh" without coupling to the controller's state.
    """

    snapshot: reactive[DashboardSnapshot | None] = reactive(None)

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._prev_snapshot: DashboardSnapshot | None = None

    def watch_snapshot(self, _old: object, new: object) -> None:
        """When a new snapshot arrives, render against the previous one,
        then promote the new one to prev for the next refresh."""
        if new is None:
            return
        self.refresh()  # trigger render
        # Promote *after* render so the next tick diffs against this frame.
        self._prev_snapshot = new  # type: ignore[assignment]

    def render(self) -> str:
        snap = self.snapshot
        if snap is None:
            return "[dim]watchlist intel: loading…[/dim]"

        # Line 1 — top movers (3 strongest abs(combined_score))
        movers = top_movers(snap, n=3)
        if movers:
            mover_chunks = [
                f"[bold]{m.symbol}[/bold] [{_score_style(m.combined_score)}]"
                f"{m.combined_score:+.2f}[/]"
                for m in movers
            ]
            line1 = "[dim]Top movers:[/dim]  " + "  |  ".join(mover_chunks)
        else:
            line1 = "[dim]Top movers: (none — all rows errored)[/dim]"

        # Line 2 — action breakdown
        counts = action_breakdown(snap)
        line2 = (
            "[dim]Action mix:[/dim]  "
            f"[bold green]{counts[SignalAction.BUY]} BUY[/]  "
            f"[bold yellow]{counts[SignalAction.HOLD]} HOLD[/]  "
            f"[bold red]{counts[SignalAction.SELL]} SELL[/]"
        )

        # Line 3 — diff against prev
        changes = diff_snapshots(snap, self._prev_snapshot)
        if changes:
            change_chunks = [
                f"[bold]{c.symbol}[/bold] [dim]({c.kind})[/dim]" for c in changes[:4]
            ]
            line3 = "[dim]Since last refresh:[/dim]  " + "  ·  ".join(change_chunks)
        elif self._prev_snapshot is None:
            line3 = "[dim]Since last refresh: (first frame)[/dim]"
        else:
            line3 = "[dim]Since last refresh: no changes[/dim]"

        return f"{line1}\n{line2}\n{line3}"


def _score_style(score: float) -> str:
    if score >= 0.3:
        return "bold green"
    if score <= -0.3:
        return "bold red"
    return "yellow"


class DetailPanel(Static):
    """Reasoning + indicator detail for the selected row."""

    row_index: reactive[int] = reactive(0)
    snapshot: reactive[DashboardSnapshot | None] = reactive(None)
    brief_state: reactive[str] = reactive("idle")  # idle | loading | ready | error
    brief_text: reactive[str] = reactive("")

    def render(self) -> str:
        snap = self.snapshot
        if snap is None or not snap.rows:
            return "[dim]select a row for details[/dim]"
        idx = max(0, min(self.row_index, len(snap.rows) - 1))
        r = snap.rows[idx]
        if r.error:
            return f"[bold cyan]{r.symbol}[/]  [red]error:[/]  {rich_escape(r.error)}"

        explanation = _explain_row(r)
        history = (snap.signal_history or {}).get(r.symbol)

        # Header line — symbol + action + confidence, tightly packed.
        header = (
            f"[bold cyan]{r.symbol}[/]  {_action_text(r.action)}  "
            f"[dim]conf[/] [bold]{r.confidence:.2f}[/] "
            f"[dim]({explanation.confidence_label}, combined "
            f"[/][{_score_style(r.combined_score)}]{r.combined_score:+.2f}[/][dim])[/]"
        )

        # Tagline — the engine's plain-English verdict.
        tagline = f"[italic dim]{rich_escape(explanation.headline)}[/italic dim]"

        # Numeric line — price first (most-asked datum), then indicators, then news.
        numbers = (
            f"[dim]price[/] [bold]{_fmt_price(r.last_price)}[/]  "
            f"[dim]·[/]  "
            f"[dim]rsi[/] {_fmt_signed(r.rsi)}  "
            f"[dim]macd[/] {_fmt_signed(r.macd)}  "
            f"[dim]bb[/] {_fmt_signed(r.bollinger)}  "
            f"[dim]·[/]  "
            f"[dim]news[/] [bold]{r.num_news_articles}[/]"
        )

        # Section: Signals (contributing factors).
        if explanation.contributors:
            signals_lines = [
                f"  • [bold]{c.name:<14}[/] {_fmt_signed(c.score)}  "
                f"[dim]{rich_escape(c.note)}[/dim]"
                for c in explanation.contributors
            ]
        else:
            signals_lines = ["  [dim]no contributing signals[/dim]"]
        signals_block = "[bold cyan]Signals[/]\n" + "\n".join(signals_lines)

        # Section: History (this session).
        history_block = (
            "[bold cyan]History[/]\n" + _render_history_block(history)
            if history is not None
            else ""
        )

        # Section: AI brief.
        brief_block = ""
        if self.brief_state == "loading":
            brief_block = (
                "[bold cyan]AI brief[/]\n  [dim italic]loading…[/dim italic]"
            )
        elif self.brief_state == "ready":
            brief_block = (
                "[bold cyan]AI brief[/]\n  "
                f"[italic]{rich_escape(self.brief_text)}[/italic]"
            )
        elif self.brief_state == "error":
            brief_block = (
                "[bold cyan]AI brief[/]\n  "
                f"[red]failed:[/] {rich_escape(self.brief_text)}"
            )

        sections = [header, tagline, numbers, signals_block]
        if history_block:
            sections.append(history_block)
        if brief_block:
            sections.append(brief_block)
        return "\n".join(sections)


def _render_history_block(h: SignalHistorySummary) -> str:
    """Render the 'Now / Was / Was' history lines for the detail panel."""
    lines: list[str] = []
    current = h.current
    lines.append(_render_episode_line(current, "Now ", show_trend=True))
    for ep in h.recent:
        lines.append(_render_episode_line(ep, "Was ", show_trend=False))
    return "\n".join(lines)


def _render_episode_line(ep: SignalEpisode, label: str, *, show_trend: bool) -> str:
    """One episode row in the History section."""
    action_cell = _action_text(ep.action)
    tick_word = "tick" if ep.tick_count == 1 else "ticks"
    base = f"  [dim]{label}[/] {action_cell}  [dim]for[/] {ep.tick_count} {tick_word}"
    if show_trend:
        trend = ep.confidence_trend
        trend_style = {
            "rising": "green",
            "falling": "red",
            "flat": "dim",
        }[trend]
        base += (
            f"  [dim]·[/]  [dim]conf[/] [{trend_style}]{trend}[/] "
            f"{ep.confidence_first:.2f} → {ep.confidence_last:.2f}"
        )
    return base


def _explain_row(r: RecommendationRow) -> Explanation:
    """Build an Explanation from a dashboard row.

    Reconstructs the indicator-score dict from the per-column NaN-safe
    fields on the row (NaN means the indicator wasn't computed and
    should be omitted from the contributor list).
    """
    indicator_scores: dict[str, float] = {}
    for key, value in (("rsi", r.rsi), ("macd", r.macd), ("bollinger", r.bollinger)):
        if value == value:  # filter NaN
            indicator_scores[key] = float(value)
    return explain(
        symbol=r.symbol,
        action=r.action,
        confidence=r.confidence,
        combined_score=r.combined_score,
        indicator_scores=indicator_scores,
        sentiment_score=r.sentiment_score,
        num_news_articles=r.num_news_articles,
    )


class DashboardApp(App[None]):
    """Esther terminal dashboard — observational, no order submission."""

    CSS = """
    Screen { layout: vertical; }
    #watchlist_header { height: 3; padding: 0 1; }
    #detail { height: auto; padding: 0 1 1 1; border-top: solid $primary 30%; }
    #alerts { height: 6; border-top: solid $warning 50%; }
    #events { height: 10; border-top: solid $primary 30%; }
    DataTable { height: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("r", "refresh_now", "refresh"),
        Binding("p", "toggle_pause", "pause/resume"),
        Binding("s", "summarize_selected", "AI brief"),
        Binding("up,down", "noop", "select", show=False),
    ]

    paused: reactive[bool] = reactive(False)

    def __init__(
        self,
        controller: "BaseController",
        refresh_seconds: float = 5.0,
        summarizer: "Summarizer | None" = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.refresh_seconds = refresh_seconds
        self.summarizer = summarizer
        self._snapshot: DashboardSnapshot | None = None
        self._tick_handle = None
        # Brief cache keyed by (symbol, action_str). Action change invalidates.
        self._brief_cache: dict[tuple[str, str], str] = {}

    # -- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield WatchlistHeader(id="watchlist_header")
            table = DataTable(zebra_stripes=True, cursor_type="row")
            table.add_columns(
                "SYM", "ACTION", "CONF", "BAR", "TECH", "SENT",
                "RSI", "MACD", "BBAND", "PRICE", "NEWS",
            )
            yield table
            yield DetailPanel(id="detail")
            yield RichLog(id="alerts", highlight=False, markup=True, wrap=False)
            yield RichLog(id="events", highlight=True, markup=True, wrap=False)
        yield Footer()

    # -- lifecycle --------------------------------------------------------

    async def on_mount(self) -> None:
        self.title = "Esther — quant dashboard"
        self.sub_title = f"watchlist: {', '.join(self.controller.watchlist)}"
        # First refresh immediately; then schedule recurring.
        await self._refresh_snapshot()
        self._tick_handle = self.set_interval(self.refresh_seconds, self._tick)

    # -- actions ----------------------------------------------------------

    def action_refresh_now(self) -> None:
        self.run_worker(self._refresh_snapshot, exclusive=True)

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        msg = "paused" if self.paused else "resumed"
        self.query_one("#events", RichLog).write(
            f"[bold yellow]{datetime.now(UTC).strftime('%H:%M:%S')}[/] {msg}"
        )

    def action_noop(self) -> None:  # bound for footer hint only
        pass

    def action_summarize_selected(self) -> None:
        """`s`: load the LLM brief for the currently-highlighted row."""
        row = self._selected_row()
        if row is None or row.error:
            return
        detail = self.query_one(DetailPanel)
        if self.summarizer is None:
            detail.brief_state = "error"
            detail.brief_text = (
                "ANTHROPIC_API_KEY not configured — see ALPACA_SETUP.md."
            )
            detail.refresh()
            return

        cache_key = (row.symbol, row.action.value)
        if cache_key in self._brief_cache:
            detail.brief_state = "ready"
            detail.brief_text = self._brief_cache[cache_key]
            detail.refresh()
            return

        detail.brief_state = "loading"
        detail.brief_text = ""
        detail.refresh()
        self.run_worker(
            self._fetch_brief(row, cache_key),
            exclusive=False,
            group="brief",
        )

    def _selected_row(self) -> RecommendationRow | None:
        snap = self._snapshot
        if snap is None or not snap.rows:
            return None
        table = self.query_one(DataTable)
        idx = table.cursor_row if table.row_count else 0
        idx = max(0, min(idx, len(snap.rows) - 1))
        return snap.rows[idx]

    async def _fetch_brief(
        self, row: RecommendationRow, cache_key: tuple[str, str]
    ) -> None:
        """Compute the AI brief off the UI thread, then push to DetailPanel."""
        import asyncio

        assert self.summarizer is not None  # checked by caller

        explanation = _explain_row(row)
        headlines = list(row.headlines) or None
        try:
            text = await asyncio.to_thread(
                self.summarizer.summarize, explanation, headlines=headlines
            )
        except Exception as e:  # noqa: BLE001 — surfaced to the UI as an error state
            detail = self.query_one(DetailPanel)
            detail.brief_state = "error"
            detail.brief_text = str(e)
            detail.refresh()
            return

        self._brief_cache[cache_key] = text
        # Only push if the user hasn't navigated away from this symbol.
        current = self._selected_row()
        detail = self.query_one(DetailPanel)
        if current is not None and (current.symbol, current.action.value) == cache_key:
            detail.brief_state = "ready"
            detail.brief_text = text
            detail.refresh()

    # -- refresh ----------------------------------------------------------

    async def _tick(self) -> None:
        if self.paused:
            return
        await self._refresh_snapshot()

    async def _refresh_snapshot(self) -> None:
        try:
            snap = await self.controller.fetch_snapshot()
        except Exception as e:  # noqa: BLE001
            self.query_one("#events", RichLog).write(
                f"[red]controller error:[/] {e}"
            )
            return
        self._snapshot = snap
        self._render_table(snap)
        self._render_panels(snap)
        self._render_alerts(snap)
        self._render_events(snap)

    def _render_table(self, snap: DashboardSnapshot) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=False)
        for r in snap.rows:
            action_cell = _action_text(r.action) if not r.error else "[red]ERR[/red]"
            bar = _confidence_bar(r.confidence)
            row_cells: tuple[object, ...] = (
                r.symbol,
                action_cell,
                f"{r.confidence:.2f}" if not r.error else "—",
                bar,
                _fmt_signed(r.technical_score) if not r.error else "—",
                _fmt_signed(r.sentiment_score) if not r.error else "—",
                _fmt_signed(r.rsi),
                _fmt_signed(r.macd),
                _fmt_signed(r.bollinger),
                _fmt_price(r.last_price),
                str(r.num_news_articles) if not r.error else r.error or "",
            )
            table.add_row(*row_cells, key=r.symbol)

    def _render_panels(self, snap: DashboardSnapshot) -> None:
        header = self.query_one(WatchlistHeader)
        header.snapshot = snap

        detail = self.query_one(DetailPanel)
        detail.snapshot = snap
        # Sync with current cursor on the table.
        table = self.query_one(DataTable)
        detail.row_index = table.cursor_row if table.row_count else 0
        self._sync_brief_for_cursor()

    def _sync_brief_for_cursor(self) -> None:
        """Reset / restore the AI brief footer based on the row under cursor.

        If the cache has an entry for the currently-selected (symbol, action),
        show it. Otherwise reset to idle so the previous brief doesn't linger
        once the user has moved off that row.
        """
        detail = self.query_one(DetailPanel)
        row = self._selected_row()
        if row is None or row.error:
            detail.brief_state = "idle"
            detail.brief_text = ""
            return
        key = (row.symbol, row.action.value)
        if key in self._brief_cache:
            detail.brief_state = "ready"
            detail.brief_text = self._brief_cache[key]
        else:
            detail.brief_state = "idle"
            detail.brief_text = ""

    def _render_alerts(self, snap: DashboardSnapshot) -> None:
        if not snap.alerts:
            return
        log = self.query_one("#alerts", RichLog)
        ring_bell = False
        for alert in snap.alerts:
            style = {
                "info": "cyan",
                "warn": "yellow",
                "critical": "bold red",
            }.get(alert.severity, "white")
            log.write(
                f"[dim]{alert.fired_at.strftime('%H:%M:%S')}[/] "
                f"[{style}]{alert.severity.upper():8}[/] "
                f"[bold]{alert.symbol}[/] · "
                f"[dim]{alert.rule}[/] · {rich_escape(alert.message)}"
            )
            if alert.severity == "critical":
                ring_bell = True
        if ring_bell:
            self.bell()

    def _render_events(self, snap: DashboardSnapshot) -> None:
        log = self.query_one("#events", RichLog)
        # Render only new events since last frame to avoid duplicate noise.
        existing = getattr(self, "_last_event_count", 0)
        for ev in snap.events[existing:]:
            colour = {"info": "green", "warn": "yellow", "error": "red"}.get(ev.level, "white")
            log.write(
                f"[dim]{ev.timestamp.strftime('%H:%M:%S')}[/] "
                f"[{colour}]{ev.level.upper():5}[/] {rich_escape(ev.message)}"
            )
        self._last_event_count = len(snap.events)

    # -- track row cursor for the detail pane ----------------------------

    @on(DataTable.RowHighlighted)
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        detail = self.query_one(DetailPanel)
        detail.row_index = event.cursor_row if event.cursor_row is not None else 0
        self._sync_brief_for_cursor()
