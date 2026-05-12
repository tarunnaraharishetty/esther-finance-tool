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
from src.strategy.base import SignalAction

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController


_ACTION_STYLES = {
    SignalAction.BUY: "bold green",
    SignalAction.SELL: "bold red",
    SignalAction.HOLD: "yellow",
}


def _fmt_signed(x: float) -> str:
    if x != x:  # NaN
        return "  -  "
    return f"{x:+.2f}"


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


class DetailPanel(Static):
    """Reasoning + indicator detail for the selected row."""

    row_index: reactive[int] = reactive(0)
    snapshot: reactive[DashboardSnapshot | None] = reactive(None)

    def render(self) -> str:
        snap = self.snapshot
        if snap is None or not snap.rows:
            return "[dim]select a row for details[/dim]"
        idx = max(0, min(self.row_index, len(snap.rows) - 1))
        r = snap.rows[idx]
        if r.error:
            return f"[bold]{r.symbol}[/bold] — [red]error:[/red] {rich_escape(r.error)}"
        explanation = _explain_row(r)
        header = (
            f"[bold]{r.symbol}[/bold]  →  {_action_text(r.action)}  "
            f"(combined [bold]{r.combined_score:+.2f}[/bold], "
            f"confidence [bold]{r.confidence:.2f}[/bold] "
            f"[dim]{explanation.confidence_label}[/dim])"
        )
        metrics = (
            f"[dim]rsi[/dim] {_fmt_signed(r.rsi)}  "
            f"[dim]macd[/dim] {_fmt_signed(r.macd)}  "
            f"[dim]bollinger[/dim] {_fmt_signed(r.bollinger)}  "
            f"[dim]news[/dim] {r.num_news_articles}  "
            f"[dim]last[/dim] {_fmt_price(r.last_price)}"
        )
        why = "\n".join(
            f"  • [bold]{c.name}[/bold] ({c.score:+.2f}): {rich_escape(c.note)}"
            for c in explanation.contributors
        )
        if not why:
            why = "  [dim]no contributing signals[/dim]"
        return (
            f"{header}\n"
            f"{metrics}\n"
            f"[italic]{rich_escape(explanation.headline)}[/italic]\n"
            f"{why}"
        )


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
    #detail { height: auto; padding: 0 1 1 1; border-top: solid $primary 30%; }
    #events { height: 12; border-top: solid $primary 30%; }
    DataTable { height: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("r", "refresh_now", "refresh"),
        Binding("p", "toggle_pause", "pause/resume"),
        Binding("up,down", "noop", "select", show=False),
    ]

    paused: reactive[bool] = reactive(False)

    def __init__(self, controller: "BaseController", refresh_seconds: float = 5.0) -> None:
        super().__init__()
        self.controller = controller
        self.refresh_seconds = refresh_seconds
        self._snapshot: DashboardSnapshot | None = None
        self._tick_handle = None

    # -- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            table = DataTable(zebra_stripes=True, cursor_type="row")
            table.add_columns(
                "SYM", "ACTION", "CONF", "BAR", "TECH", "SENT",
                "RSI", "MACD", "BBAND", "PRICE", "NEWS",
            )
            yield table
            yield DetailPanel(id="detail")
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
        detail = self.query_one(DetailPanel)
        detail.snapshot = snap
        # Sync with current cursor on the table.
        table = self.query_one(DataTable)
        detail.row_index = table.cursor_row if table.row_count else 0

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
