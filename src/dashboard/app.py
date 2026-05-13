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
from src.intelligence.opportunities import (
    Opportunity,
    RankedOpportunity,
    detect_opportunities,
    rank_opportunities,
)
from src.intelligence.pulse import MarketPulse, compute_pulse
from src.intelligence.signal_profile import SignalProfile
from src.intelligence.rankings import Rankings
from src.intelligence.rankings import compute as compute_rankings
from src.intelligence.watchlist import (
    action_breakdown,
    diff_snapshots,
)
from src.strategy.base import RecommendationTier, SignalAction

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


_TIER_STYLES: dict[RecommendationTier, str] = {
    RecommendationTier.STRONG_BUY: "bold green on grey15",
    RecommendationTier.BUY: "bold green",
    RecommendationTier.HOLD: "yellow",
    RecommendationTier.SELL: "bold red",
    RecommendationTier.STRONG_SELL: "bold red on grey15",
}


def _tier_text(tier: RecommendationTier) -> str:
    """Render a tier cell. STRONG variants get a subtle background tint
    so they pop against BUY/SELL without screaming."""
    style = _TIER_STYLES.get(tier, "")
    return f"[{style}]{tier.display:<11}[/]"


def _format_quality_tier(quality: str) -> str:
    style = {
        "high": "bold green",
        "moderate": "yellow",
        "low": "dim",
    }.get(quality, "white")
    return f"[{style}]{quality.upper()}[/]"


def _format_stability_tier(stability: str) -> str:
    style = {
        "stable": "bold green",
        "moderate": "yellow",
        "volatile": "bold red",
    }.get(stability, "white")
    return f"[{style}]{stability.upper()}[/]"


def _confidence_bar(conf: float, width: int = 10) -> str:
    """Mini ASCII bar — width chars filled proportional to conf in [0,1]."""
    filled = max(0, min(width, int(round(conf * width))))
    return "█" * filled + "·" * (width - filled)


class WatchlistHeader(Static):
    """Compact intelligence header above the watchlist.

    Status line + up to six ranked sections (momentum, sentiment,
    reversals, confidence, unusual movers, volatility). Empty sections
    are skipped so first-tick / no-history layouts stay tight.

    Holds an internal reference to the previous snapshot so it can render
    "Since last refresh" without coupling to the controller's state.
    """

    snapshot: reactive[DashboardSnapshot | None] = reactive(None)

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._prev_snapshot: DashboardSnapshot | None = None
        self._last_signature: tuple | None = None

    def watch_snapshot(self, _old: object, new: object) -> None:
        """When a new snapshot arrives, render against the previous one,
        then promote the new one to prev for the next refresh.

        Skips ``self.refresh()`` when the rendered content would be
        identical to the last frame — sidesteps a Rich-markup parse and
        a Textual redraw cycle when rankings + alerts + diff are
        stable.
        """
        if new is None:
            return
        signature = _header_signature(new, self._prev_snapshot)  # type: ignore[arg-type]
        if signature != self._last_signature:
            self.refresh()
            self._last_signature = signature
        # Promote *after* signature check so the diff line keeps working.
        self._prev_snapshot = new  # type: ignore[assignment]

    def render(self) -> str:
        snap = self.snapshot
        if snap is None:
            return "[dim]watchlist intel: loading…[/dim]"

        rankings = compute_rankings(snap, n=3)
        pulse = compute_pulse(snap)
        lines: list[str] = []

        # --- Pulse line (skip when no healthy rows) --------------------
        if not pulse.is_empty:
            lines.append(f"{_section_label('PULSE')}{_format_pulse(pulse)}")

        # --- Status line: action mix + changes since last refresh -------
        counts = action_breakdown(snap)
        mix = (
            f"[bold green]{counts[SignalAction.BUY]} BUY[/]  "
            f"[bold yellow]{counts[SignalAction.HOLD]} HOLD[/]  "
            f"[bold red]{counts[SignalAction.SELL]} SELL[/]"
        )
        changes = diff_snapshots(snap, self._prev_snapshot)
        if changes:
            change_str = "  ·  ".join(
                f"[bold]{c.symbol}[/bold] [dim]({c.kind})[/dim]"
                for c in changes[:3]
            )
        elif self._prev_snapshot is None:
            change_str = "[dim](first frame)[/dim]"
        else:
            change_str = "[dim](no changes)[/dim]"
        lines.append(
            f"{_section_label('MIX')}{mix}      "
            f"{_section_label('CHANGES')}{change_str}"
        )

        # --- Ranked sections (skip empties) ----------------------------
        sections: list[tuple[str, tuple[tuple[str, float], ...], str]] = [
            ("MOMENTUM", rankings.strongest_momentum, "signed"),
            ("SENTIMENT", rankings.strongest_sentiment, "signed"),
            ("CONFIDENCE", rankings.highest_confidence, "magnitude"),
            ("REVERSALS", rankings.biggest_reversals, "reversal"),
            ("UNUSUAL", rankings.unusual_movers, "magnitude"),
            ("VOLATILE", rankings.most_volatile, "int"),
        ]
        for label, entries, fmt in sections:
            if not entries:
                continue
            cells = "  ".join(_format_rank_cell(sym, score, fmt) for sym, score in entries)
            lines.append(f"{_section_label(label)}{cells}")

        # --- Session alert summary (skip when empty) -------------------
        alerts_summary = _format_alert_counts(snap.recent_alerts)
        if alerts_summary:
            lines.append(f"{_section_label('ALERTS')}{alerts_summary}")

        # --- Top opportunities (skip when none qualify) ---------------
        # Ranked-composite output replaces the older kind-based view —
        # same screen real estate, richer per-line content. Skipped
        # when no directional symbols rank.
        opportunities = rank_opportunities(snap, n=3)
        for opp in opportunities:
            lines.append(
                f"{_section_label('OPP')}{_format_ranked_opportunity(opp)}"
            )

        return "\n".join(lines)


def _section_label(text: str) -> str:
    """Fixed-width left-aligned section header. 11 chars keeps columns
    visually aligned across the panel."""
    return f"[bold dim]{text:<11}[/]"


def _detail_signature(
    row: RecommendationRow,
    snap: DashboardSnapshot,
    brief_state: str,
    brief_text: str,
) -> tuple:
    """Stable signature of what DetailPanel.render() would produce.

    Captures every input the render path reads. Rounded floats so
    pandas-level float jitter doesn't bust the cache.
    """
    history = (snap.signal_history or {}).get(row.symbol)
    hist_sig: tuple = ()
    if history is not None:
        hist_sig = (
            history.current.action.value,
            history.current.tick_count,
            round(history.current.confidence_first, 3),
            round(history.current.confidence_last, 3),
            tuple(
                (ep.action.value, ep.tick_count) for ep in history.recent
            ),
        )
    # Per-symbol alert count signature.
    symbol_alerts_sig = tuple(
        (a.fired_at.isoformat(), a.severity, a.rule, a.message)
        for a in snap.recent_alerts
        if a.symbol == row.symbol
    )
    return (
        row.symbol,
        row.action.value,
        row.tier.value,
        row.signal_quality,
        row.stability,
        row.quality_reasons,
        round(row.confidence, 3),
        round(row.combined_score, 3),
        round(row.technical_score, 3),
        round(row.sentiment_score, 3),
        round(row.rsi if row.rsi == row.rsi else 0.0, 3),
        round(row.macd if row.macd == row.macd else 0.0, 3),
        round(row.bollinger if row.bollinger == row.bollinger else 0.0, 3),
        round(row.last_price if row.last_price == row.last_price else 0.0, 2),
        row.num_news_articles,
        row.error,
        hist_sig,
        symbol_alerts_sig,
        brief_state,
        brief_text,
    )


def _header_signature(
    snap: DashboardSnapshot,
    prev_snap: DashboardSnapshot | None,
) -> tuple:
    """Stable signature of what the WatchlistHeader would render.

    Two snapshots that produce the same signature would render byte-
    identically, so we can short-circuit the redraw.
    """
    # Per-row identity for action mix + diff input.
    rows_sig = tuple(
        (r.symbol, r.action.value, round(r.confidence, 3), r.error)
        for r in snap.rows
    )
    prev_sig = (
        tuple(r.symbol + r.action.value for r in prev_snap.rows)
        if prev_snap is not None
        else None
    )
    # Recent-alerts count by severity drives the ALERTS line.
    from src.intelligence.alerts import Alert

    severity_counts: dict[str, int] = {"critical": 0, "warn": 0, "info": 0}
    for a in snap.recent_alerts:
        if isinstance(a, Alert):
            severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1
    # Round indicator scores so float noise doesn't bust the cache.
    indicators_sig = tuple(
        (r.symbol, round(r.macd if r.macd == r.macd else 0.0, 3),
         round(r.sentiment_score, 3), r.num_news_articles)
        for r in snap.rows
    )
    # Pulse output drives the PULSE line.
    pulse = compute_pulse(snap)
    pulse_sig = (pulse.sentiment, pulse.conviction, pulse.activity)
    # Opportunities drive the OPP lines.
    opp_sig = tuple(
        (o.symbol, round(o.composite_score, 3), o.profile)
        for o in rank_opportunities(snap, n=3)
    )
    return (
        rows_sig,
        prev_sig,
        tuple(sorted(severity_counts.items())),
        indicators_sig,
        pulse_sig,
        opp_sig,
    )


def _format_ranked_opportunity(opp: RankedOpportunity) -> str:
    """One dense OPP line for the ranked-composite output.

    Format:
      SYMBOL  TIER         0.82   [stable·strengthening·persistent]   rationale phrases
    """
    tier_styled = _tier_text(opp.tier)
    profile_chip = _format_profile_chip(opp.profile)
    score = f"[bold]{opp.composite_score:.2f}[/bold]"
    rationale = (
        "  [dim]·[/]  ".join(opp.rationale)
        if opp.rationale
        else "[dim](no driver above floor)[/dim]"
    )
    return (
        f"[bold]{opp.symbol:<6}[/]  {tier_styled}  {score}  "
        f"{profile_chip}  [dim]{rich_escape(rationale)}[/dim]"
    )


_PROFILE_AXIS_STYLES: dict[str, dict[str, str]] = {
    "stability": {"stable": "bold green", "noisy": "bold red"},
    "trend": {
        "strengthening": "bold green",
        "weakening": "bold red",
        "flat": "dim",
    },
    "persistence": {"persistent": "bold green", "flipping": "bold red"},
}


def _format_profile_chip(profile: SignalProfile) -> str:
    """Three colored axis labels separated by middle dots.

    Each axis is independently colored — green for the positive
    bucket (stable / strengthening / persistent), red for the
    negative (noisy / weakening / flipping), dim for flat. Compact
    enough to fit on one OPP line.
    """
    parts = [
        f"[{_PROFILE_AXIS_STYLES['stability'][profile.stability]}]"
        f"{profile.stability}[/]",
        f"[{_PROFILE_AXIS_STYLES['trend'][profile.trend]}]"
        f"{profile.trend}[/]",
        f"[{_PROFILE_AXIS_STYLES['persistence'][profile.persistence]}]"
        f"{profile.persistence}[/]",
    ]
    return "[dim][[/dim]" + "[dim]·[/dim]".join(parts) + "[dim]][/dim]"


def _format_opportunity(opp: Opportunity) -> str:
    """One opportunity line: SYMBOL  kind-tag  rationale."""
    kind_style = {
        "convergence": "bold green",
        "reversal": "bold yellow",
        "high_conviction": "bold cyan",
    }.get(opp.kind, "white")
    return (
        f"[bold]{opp.symbol}[/]  "
        f"[{kind_style}]{opp.kind}[/]  "
        f"[dim]{rich_escape(opp.rationale)}[/dim]"
    )


def _format_pulse(pulse: MarketPulse) -> str:
    """One dense line surfacing every pulse dimension.

    Chunks (· separated, omitted when not applicable):
      sentiment B/N  ·  conv  ·  mom %%  ·  sent %%  ·  vol  ·  ↻ revs  ·  ! alerts  ·  STRONG symbols
    """
    sentiment_style = {
        "bullish": "bold green",
        "bearish": "bold red",
        "mixed": "bold yellow",
        "neutral": "dim",
    }.get(pulse.sentiment, "white")
    conviction_style = {
        "strong": "bold",
        "moderate": "white",
        "weak": "dim",
    }.get(pulse.conviction, "white")
    activity_style = {
        "volatile": "bold red",
        "active": "yellow",
        "calm": "dim",
    }.get(pulse.activity, "white")

    chunks: list[str] = []

    # Sentiment chip with the bullish/bearish count proportion.
    directional = pulse.bullish_count + pulse.bearish_count
    if directional > 0 and pulse.healthy_count > 0:
        if pulse.sentiment == "bullish":
            chip = f"[{sentiment_style}]{pulse.sentiment}[/] [dim]{pulse.bullish_count}/{pulse.healthy_count}[/dim]"
        elif pulse.sentiment == "bearish":
            chip = f"[{sentiment_style}]{pulse.sentiment}[/] [dim]{pulse.bearish_count}/{pulse.healthy_count}[/dim]"
        else:
            chip = f"[{sentiment_style}]{pulse.sentiment}[/] [dim]{pulse.bullish_count}b/{pulse.bearish_count}s[/dim]"
    else:
        chip = f"[{sentiment_style}]{pulse.sentiment}[/]"
    chunks.append(chip)

    # Conviction tier (always present).
    chunks.append(f"[{conviction_style}]{pulse.conviction}[/] conv")

    # Breadth fractions — only meaningful when there's a denominator.
    if pulse.momentum_breadth > 0 or directional > 0:
        mom_pct = int(round(pulse.momentum_breadth * 100))
        mom_style = _breadth_style(pulse.momentum_breadth)
        chunks.append(f"[dim]mom[/dim] [{mom_style}]{mom_pct}%[/]")
    if pulse.sentiment_breadth > 0:
        sent_pct = int(round(pulse.sentiment_breadth * 100))
        sent_style = _breadth_style(pulse.sentiment_breadth)
        chunks.append(f"[dim]sent[/dim] [{sent_style}]{sent_pct}%[/]")

    # Volatility regime (always present — it's information even when calm).
    chunks.append(f"[{activity_style}]{pulse.activity}[/]")

    # Reversal intensity — surface only when non-zero (calm sessions stay tight).
    if pulse.reversal_intensity > 0:
        chunks.append(
            f"[dim]revs[/dim] [yellow]{pulse.reversal_intensity}[/yellow]"
        )

    # Alert intensity — only when non-zero.
    if pulse.alert_intensity > 0:
        chunks.append(
            f"[dim]alerts[/dim] [bold red]{pulse.alert_intensity}[/]"
        )

    # Strongest symbols — small inline list. Omitted entirely when none.
    if pulse.strongest_symbols:
        strong_chunks = [
            f"[bold]{sym}[/] [dim]{display}[/dim]"
            for sym, display in pulse.strongest_symbols
        ]
        chunks.append("STRONG " + " ".join(strong_chunks))

    return "  [dim]·[/]  ".join(chunks)


def _breadth_style(fraction: float) -> str:
    """Color a breadth fraction by tier — high agreement gets bolder."""
    if fraction >= 0.80:
        return "bold green"
    if fraction >= 0.50:
        return "yellow"
    return "dim"


def _format_alert_counts(alerts: tuple[object, ...]) -> str:
    """Compact severity summary for the watchlist header alerts line.

    Returns empty string if no alerts in the rolling window.
    """
    from src.intelligence.alerts import Alert

    if not alerts:
        return ""
    counts: dict[str, int] = {"critical": 0, "warn": 0, "info": 0}
    for a in alerts:
        if isinstance(a, Alert):
            counts[a.severity] = counts.get(a.severity, 0) + 1
    chunks: list[str] = []
    if counts["critical"]:
        chunks.append(f"[bold red]{counts['critical']} critical[/]")
    if counts["warn"]:
        chunks.append(f"[bold yellow]{counts['warn']} warn[/]")
    if counts["info"]:
        chunks.append(f"[cyan]{counts['info']} info[/]")
    return "  ".join(chunks) + f"  [dim]({len(alerts)} this session)[/dim]"


def _format_rank_cell(symbol: str, score: float, fmt: str) -> str:
    """Render one (symbol, score) cell consistent with the section's fmt."""
    if fmt == "signed":
        return f"[bold]{symbol}[/] [{_score_style(score)}]{score:+.2f}[/]"
    if fmt == "magnitude":
        # Non-negative score (confidence, unusual). Color by magnitude.
        style = "bold green" if score >= 0.6 else "yellow" if score >= 0.3 else "dim"
        return f"[bold]{symbol}[/] [{style}]{score:.2f}[/]"
    if fmt == "int":
        return f"[bold]{symbol}[/] [yellow]{int(score)}[/]"
    if fmt == "reversal":
        # Signed score: positive = bearish->bullish flip, negative = the reverse.
        arrow = "↑" if score > 0 else "↓"
        style = "bold green" if score > 0 else "bold red"
        return f"[bold]{symbol}[/] [{style}]{arrow}[/]"
    return f"[bold]{symbol}[/] {score:+.2f}"


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

    # In-render cache: when the input signature matches the last frame,
    # short-circuit and return the cached string. Saves a full Rich
    # markup build (header / metrics / signals / history / alerts /
    # brief) on every tick where the selected row hasn't changed.
    _last_signature: tuple | None = None
    _last_rendered: str = ""

    def render(self) -> str:
        snap = self.snapshot
        if snap is None or not snap.rows:
            return "[dim]select a row for details[/dim]"
        idx = max(0, min(self.row_index, len(snap.rows) - 1))
        r = snap.rows[idx]

        signature = _detail_signature(
            r, snap, self.brief_state, self.brief_text
        )
        if signature == self._last_signature:
            return self._last_rendered
        if r.error:
            return f"[bold cyan]{r.symbol}[/]  [red]error:[/]  {rich_escape(r.error)}"

        explanation = _explain_row(r)
        history = (snap.signal_history or {}).get(r.symbol)

        # Header line — symbol + tier + confidence + quality / stability tiers.
        header = (
            f"[bold cyan]{r.symbol}[/]  {_tier_text(r.tier)}  "
            f"[dim]conf[/] [bold]{r.confidence:.2f}[/]  "
            f"[dim]quality[/] {_format_quality_tier(r.signal_quality)}  "
            f"[dim]stability[/] {_format_stability_tier(r.stability)}"
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

        # Section: Tier reasons (the bullets that explain why the tier
        # is what it is). Only shows when promote_to_tier produced
        # reasons — i.e. some driver met the threshold to be mentioned.
        if r.quality_reasons:
            reason_lines = [f"  • [dim]{rich_escape(reason)}[/dim]" for reason in r.quality_reasons]
            tier_block = "[bold cyan]Why this tier[/]\n" + "\n".join(reason_lines)
        else:
            tier_block = ""

        # Section: History (this session).
        history_block = (
            "[bold cyan]History[/]\n" + _render_history_block(history)
            if history is not None
            else ""
        )

        # Section: Alerts (this symbol, this session).
        symbol_alerts = [a for a in snap.recent_alerts if a.symbol == r.symbol]
        alerts_block = (
            "[bold cyan]Alerts[/]\n" + _render_symbol_alerts(symbol_alerts)
            if symbol_alerts
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
        if tier_block:
            sections.append(tier_block)
        if history_block:
            sections.append(history_block)
        if alerts_block:
            sections.append(alerts_block)
        if brief_block:
            sections.append(brief_block)
        rendered = "\n".join(sections)
        self._last_signature = signature
        self._last_rendered = rendered
        return rendered


def _format_event_line(ev: object) -> str:
    """One line for the events RichLog. Appends ``× N`` when count > 1."""
    level = getattr(ev, "level", "info")
    message = getattr(ev, "message", "")
    timestamp = getattr(ev, "timestamp", None)
    count = getattr(ev, "count", 1)
    colour = {"info": "green", "warn": "yellow", "error": "red"}.get(level, "white")
    ts = timestamp.strftime("%H:%M:%S") if timestamp is not None else "?"
    suffix = f"  [dim]× {count}[/dim]" if count > 1 else ""
    return (
        f"[dim]{ts}[/] [{colour}]{level.upper():5}[/] "
        f"{rich_escape(message)}{suffix}"
    )


def _render_symbol_alerts(alerts: list[object], *, limit: int = 4) -> str:
    """Render up to ``limit`` recent alerts for one symbol, newest-first.

    Caller pre-filters by symbol; we just render. Each line is
    severity-colored to match the alerts pane's styling.
    """
    severity_style = {
        "critical": "bold red",
        "warn": "yellow",
        "info": "cyan",
    }
    lines: list[str] = []
    for alert in alerts[:limit]:
        # Duck-type: Alert has .severity .rule .message .fired_at attrs.
        sev = getattr(alert, "severity", "info")
        rule = getattr(alert, "rule", "?")
        message = getattr(alert, "message", "")
        fired_at = getattr(alert, "fired_at", None)
        style = severity_style.get(sev, "white")
        when = fired_at.strftime("%H:%M:%S") if fired_at is not None else "?"
        lines.append(
            f"  [dim]{when}[/dim]  [{style}]{sev.upper():8}[/]  "
            f"[dim]{rule}[/dim]  {rich_escape(message)}"
        )
    if len(alerts) > limit:
        lines.append(f"  [dim]+ {len(alerts) - limit} more this session[/dim]")
    return "\n".join(lines)


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


class StatusLine(Static):
    """One-line dense status anchored just above the footer.

    Always-visible at-a-glance state: sentiment chip · top STRONG
    symbol (if any) · critical alert count · tick + UTC time.
    Stateless — renders directly from the snapshot reactive.
    """

    snapshot: reactive[DashboardSnapshot | None] = reactive(None)

    def render(self) -> str:
        snap = self.snapshot
        if snap is None or not snap.rows:
            return "[dim]status: idle[/dim]"
        pulse = compute_pulse(snap)

        chunks: list[str] = []

        # Sentiment chip (color-coded by tier).
        if not pulse.is_empty:
            sent_style = {
                "bullish": "bold green",
                "bearish": "bold red",
                "mixed": "bold yellow",
                "neutral": "dim",
            }.get(pulse.sentiment, "white")
            chunks.append(f"[{sent_style}]{pulse.sentiment}[/]")

        # Top STRONG-tier symbol from the pulse's curated list. The pulse
        # already ranks STRONG symbols by confidence and caps the list —
        # the status line just picks the head.
        if pulse.strongest_symbols:
            sym, display = pulse.strongest_symbols[0]
            tier_style = "bold green" if "BUY" in display else "bold red"
            chunks.append(f"[{tier_style}]{display}[/] [bold]{sym}[/]")

        # Critical alert count.
        critical = sum(
            1 for a in snap.recent_alerts if a.severity == "critical"
        )
        if critical:
            chunks.append(f"[bold red]{critical} critical[/]")

        # Tick + clock.
        clock = snap.timestamp.strftime("%H:%M:%S")
        chunks.append(f"[dim]tick {snap.tick} · {clock} UTC[/dim]")

        return "  ·  ".join(chunks)


class DashboardApp(App[None]):
    """Esther terminal dashboard — observational, no order submission."""

    CSS = """
    Screen { layout: vertical; }
    #watchlist_header { height: auto; max-height: 8; padding: 0 1; }
    #detail { height: auto; padding: 0 1 1 1; border-top: solid $primary 30%; }
    #alerts { height: 6; border-top: solid $warning 50%; }
    #events { height: 10; border-top: solid $primary 30%; }
    #status { height: 1; padding: 0 1; background: $primary 8%; }
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
        burst_seconds: float = 1.5,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.refresh_seconds = refresh_seconds
        # Adaptive cadence: when a tick produces new alerts or any row's
        # action flips, schedule a one-shot follow-up at burst_seconds
        # instead of waiting for the next base interval. Cancelled and
        # replaced on each refresh.
        self.burst_seconds = burst_seconds
        self.summarizer = summarizer
        self._snapshot: DashboardSnapshot | None = None
        self._tick_handle = None
        self._tick_burst_handle = None
        # Per-symbol last-seen action; drives burst detection.
        self._previous_actions: dict[str, SignalAction] = {}
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
            yield StatusLine(id="status")
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
        self._maybe_schedule_burst(snap)

    def _should_burst(self, snap: DashboardSnapshot) -> bool:
        """True when this tick warrants a faster follow-up.

        Triggers: any fresh alert this tick OR any healthy row whose
        action differs from the prior snapshot's recorded action.
        Updates ``self._previous_actions`` as a side effect so the
        next call sees the right baseline.
        """
        burst = bool(snap.alerts)
        if not burst and self._previous_actions:
            for row in snap.rows:
                if row.error:
                    continue
                prev = self._previous_actions.get(row.symbol)
                if prev is not None and prev != row.action:
                    burst = True
                    break
        self._previous_actions = {
            r.symbol: r.action for r in snap.rows if not r.error
        }
        return burst

    def _maybe_schedule_burst(self, snap: DashboardSnapshot) -> None:
        """Cancel any pending burst timer; if conditions are met,
        schedule a new one-shot follow-up tick at ``burst_seconds``.

        Idempotent — calling this twice within one tick replaces the
        prior timer rather than stacking. The base ``set_interval``
        continues running on its own schedule.
        """
        if self._tick_burst_handle is not None:
            self._tick_burst_handle.stop()
            self._tick_burst_handle = None
        if self.paused:
            return
        if not self._should_burst(snap):
            return
        self._tick_burst_handle = self.set_timer(self.burst_seconds, self._tick)

    def _render_table(self, snap: DashboardSnapshot) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=False)
        for r in snap.rows:
            tier_cell = _tier_text(r.tier) if not r.error else "[red]ERR[/red]"
            bar = _confidence_bar(r.confidence)
            row_cells: tuple[object, ...] = (
                r.symbol,
                tier_cell,
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

        status = self.query_one(StatusLine)
        status.snapshot = snap

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
        events = snap.events
        if not events:
            return
        last_count: int = getattr(self, "_last_event_count", 0)
        last_top_count: int = getattr(self, "_last_top_count", 0)

        new_entries = events[last_count:]
        if new_entries:
            # The previous top (if any) is now "finalized" — its dedup
            # window closed when the new entry arrived. If its count grew
            # since we last rendered it, write an updated line first so
            # the trader sees the final count before the new entry below.
            if last_count > 0 and last_count <= len(events):
                prev_top = events[last_count - 1]
                if prev_top.count > last_top_count:
                    log.write(_format_event_line(prev_top))
            for ev in new_entries:
                log.write(_format_event_line(ev))
        else:
            # No new entries this frame, but the current top's count may
            # have bumped since last render. Show it once with the new
            # count; subsequent bumps stay silent until something else
            # finalizes or the next render catches up.
            top = events[-1]
            if top.count > last_top_count:
                log.write(_format_event_line(top))

        self._last_event_count = len(events)
        self._last_top_count = events[-1].count

    # -- track row cursor for the detail pane ----------------------------

    @on(DataTable.RowHighlighted)
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        detail = self.query_one(DetailPanel)
        detail.row_index = event.cursor_row if event.cursor_row is not None else 0
        self._sync_brief_for_cursor()
