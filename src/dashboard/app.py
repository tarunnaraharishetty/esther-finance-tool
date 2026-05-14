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
from typing import TYPE_CHECKING, ClassVar

from rich.markup import escape as rich_escape
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static

from src.dashboard.state import DashboardSnapshot, RecommendationRow
from src.intelligence.explain import Explanation, explain
from src.intelligence.history import SignalEpisode, SignalHistorySummary
from src.intelligence.opportunities import (
    Opportunity,
    RankedOpportunity,
    rank_opportunities,
)
from src.intelligence.opportunity_drilldown import (
    OpportunityDrilldown,
    build_drilldown,
)
from src.intelligence.opportunity_history import OpportunityHistory
from src.intelligence.pulse import MarketPulse, compute_pulse
from src.intelligence.pulse_evolution import TrajectoryPattern
from src.intelligence.pulse_history import PulseHistory
from src.intelligence.rankings import compute as compute_rankings
from src.intelligence.signal_profile import SignalProfile
from src.intelligence.watchlist import (
    action_breakdown,
    diff_snapshots,
)
from src.strategy.base import RecommendationTier, SignalAction
from src.strategy.multi_timeframe import IntradayRead, is_divergent

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController
    from src.intelligence.alerts import Alert
    from src.intelligence.opportunity_brief import (
        LLMOpportunityBriefer,
        OpportunityBriefContext,
    )
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
    filled = max(0, min(width, round(conf * width)))
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
        self._last_signature: tuple[object, ...] | None = None

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
        if not snap.rows:
            # Friendly empty-state — every section below assumes at
            # least one row, and zero-row stubs in the action mix /
            # rankings would just be visual noise.
            return (
                "[bold yellow]Watchlist is empty.[/]  "
                "[dim]Press [bold]a[/bold] to add a symbol  ·  "
                "run [bold]esther backfill[/bold] first to warm up cached bars.[/dim]"
            )

        rankings = compute_rankings(snap, n=3)
        pulse = snap.pulse if snap.pulse is not None else compute_pulse(snap)
        lines: list[str] = []

        # --- Pulse line (skip when no healthy rows) --------------------
        if not pulse.is_empty:
            lines.append(f"{_section_label('PULSE')}{_format_pulse(pulse)}")

        # --- HIST line (sparklines of recent pulses; needs >=2 ticks) --
        if snap.pulse_history is not None and snap.pulse_history.has_trend:
            lines.append(f"{_section_label('HIST')}{_format_pulse_history(snap.pulse_history)}")

        # --- REGIME + PATTERNS (trajectory-level synthesis) -------------
        # Both lines key off snap.pulse_evolution. REGIME is always
        # shown when the evolution exists and isn't indeterminate;
        # PATTERNS only when at least one fires.
        evolution = snap.pulse_evolution
        if evolution is not None and evolution.regime != "indeterminate":
            lines.append(f"{_section_label('REGIME')}{_format_regime(evolution.regime)}")
        if evolution is not None and evolution.patterns:
            lines.append(f"{_section_label('PATTERNS')}{_format_patterns(evolution.patterns)}")

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
                f"[bold]{c.symbol}[/bold] [dim]({c.kind})[/dim]" for c in changes[:3]
            )
        elif self._prev_snapshot is None:
            change_str = "[dim](first frame)[/dim]"
        else:
            change_str = "[dim](no changes)[/dim]"
        lines.append(f"{_section_label('MIX')}{mix}      {_section_label('CHANGES')}{change_str}")

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

        # --- Intraday divergence count (skip when no intraday data) ---
        # When at least one healthy row carries an intraday read, show
        # how many of those rows disagree with the daily action. Quiet
        # when intraday is disabled or every row aligns.
        intraday_summary = _format_intraday_summary(snap.rows)
        if intraday_summary:
            lines.append(f"{_section_label('INTRADAY')}{intraday_summary}")

        # --- Top opportunities (skip when none qualify) ---------------
        # Ranked-composite output replaces the older kind-based view —
        # same screen real estate, richer per-line content. Skipped
        # when no directional symbols rank.
        opportunities = rank_opportunities(snap, n=3)
        for opp in opportunities:
            history = snap.opp_history.get(opp.symbol)
            lines.append(f"{_section_label('OPP')}{_format_ranked_opportunity(opp, history)}")

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
    brief_kind: str = "row",
    opp_match: tuple[int, RankedOpportunity] | None = None,
) -> tuple[object, ...]:
    """Stable signature of what DetailPanel.render() would produce.

    Captures every input the render path reads. Rounded floats so
    pandas-level float jitter doesn't bust the cache.
    """
    history = (snap.signal_history or {}).get(row.symbol)
    hist_sig: tuple[object, ...] = ()
    if history is not None:
        hist_sig = (
            history.current.action.value,
            history.current.tick_count,
            round(history.current.confidence_first, 3),
            round(history.current.confidence_last, 3),
            tuple((ep.action.value, ep.tick_count) for ep in history.recent),
        )
    # Per-symbol alert count signature.
    symbol_alerts_sig = tuple(
        (a.fired_at.isoformat(), a.severity, a.rule, a.message)
        for a in snap.recent_alerts
        if a.symbol == row.symbol
    )
    # OPP membership signature — when the selected row ranks in top-N,
    # the drilldown is part of the rendered output and must invalidate
    # the cache when any driver score, profile axis, or history badge
    # changes. Rounded to 3 places so tick-over-tick float jitter on a
    # stable OPP doesn't bust the cache.
    opp_sig: tuple[object, ...] = ()
    if opp_match is not None:
        rank, opp = opp_match
        opp_history = snap.opp_history.get(row.symbol)
        opp_sig = (
            rank,
            round(opp.composite_score, 3),
            round(opp.technical_alignment, 3),
            round(opp.sentiment_alignment, 3),
            round(opp.confidence_acceleration, 3),
            round(opp.momentum_persistence, 3),
            round(opp.unusual_activity, 3),
            round(opp.reversal_strength, 3),
            round(opp.signal_quality_score, 3),
            opp.profile.stability,
            opp.profile.trend,
            opp.profile.persistence,
            opp.rationale,
            (opp_history.streak, opp_history.appearances) if opp_history is not None else None,
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
        opp_sig,
        # Intraday signature — when present, action + rounded scores
        # invalidate the cache on a divergence flip or material score
        # change. None signature when intraday is disabled or the
        # secondary fetch failed.
        (
            row.intraday.timeframe.value,
            row.intraday.action.value,
            round(row.intraday.confidence, 3),
            round(row.intraday.combined_score, 3),
        )
        if row.intraday is not None
        else None,
        brief_state,
        brief_text,
        brief_kind,
    )


def _header_signature(
    snap: DashboardSnapshot,
    prev_snap: DashboardSnapshot | None,
) -> tuple[object, ...]:
    """Stable signature of what the WatchlistHeader would render.

    Two snapshots that produce the same signature would render byte-
    identically, so we can short-circuit the redraw.
    """
    # Per-row identity for action mix + diff input.
    rows_sig = tuple((r.symbol, r.action.value, round(r.confidence, 3), r.error) for r in snap.rows)
    prev_sig = (
        tuple(r.symbol + r.action.value for r in prev_snap.rows) if prev_snap is not None else None
    )
    # Recent-alerts count by severity drives the ALERTS line.
    from src.intelligence.alerts import Alert

    severity_counts: dict[str, int] = {"critical": 0, "warn": 0, "info": 0}
    for a in snap.recent_alerts:
        if isinstance(a, Alert):
            severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1
    # Round indicator scores so float noise doesn't bust the cache.
    indicators_sig = tuple(
        (
            r.symbol,
            round(r.macd if r.macd == r.macd else 0.0, 3),
            round(r.sentiment_score, 3),
            r.num_news_articles,
        )
        for r in snap.rows
    )
    # Pulse output drives the PULSE line. Capture the full set of fields
    # the renderer surfaces — sentiment/conviction/activity AND breadth,
    # intensity, strong-symbol counts — so a breadth shift invalidates
    # the cache even when the categorical tier hasn't moved.
    pulse = snap.pulse if snap.pulse is not None else compute_pulse(snap)
    pulse_sig = (
        pulse.sentiment,
        pulse.conviction,
        pulse.activity,
        pulse.bullish_count,
        pulse.bearish_count,
        pulse.healthy_count,
        round(pulse.momentum_breadth, 3),
        round(pulse.sentiment_breadth, 3),
        pulse.reversal_intensity,
        pulse.alert_intensity,
        pulse.strongest_symbols,
    )
    # History drives the HIST line — last value per series suffices for
    # the cache key since adding a new tick is the only way the series
    # advances.
    hist_sig: tuple[object, ...] = ()
    if snap.pulse_history is not None and snap.pulse_history.has_trend:
        hist_sig = (
            snap.pulse_history.length,
            round(snap.pulse_history.momentum_breadth[-1], 3),
            round(snap.pulse_history.sentiment_breadth[-1], 3),
            snap.pulse_history.reversal_intensity[-1],
            snap.pulse_history.alert_intensity[-1],
        )
    # Opportunities + their history badges drive the OPP lines.
    opp_sig = tuple(
        (
            o.symbol,
            round(o.composite_score, 3),
            o.profile,
            # History badge is part of the rendered line — without it a
            # streak bump (2x → 3x) wouldn't invalidate the cache.
            (
                snap.opp_history[o.symbol].streak,
                snap.opp_history[o.symbol].appearances,
            )
            if o.symbol in snap.opp_history
            else None,
        )
        for o in rank_opportunities(snap, n=3)
    )
    # Intraday signature — the INTRADAY header line keys off per-row
    # divergence vs daily action. Any (daily action, intraday action)
    # pair change must invalidate the cache so the count line updates.
    intraday_sig = tuple(
        (r.symbol, r.action.value, r.intraday.action.value if r.intraday else None)
        for r in snap.rows
        if not r.error and r.intraday is not None
    )
    # Pulse-evolution signature — regime + pattern names drive the
    # REGIME / PATTERNS header lines. Detail strings (numeric grounding)
    # are derived from the same series the pattern-detector keyed off,
    # so the names suffice as a cache fingerprint.
    evolution_sig: tuple[object, ...] = ()
    if snap.pulse_evolution is not None:
        evolution_sig = (
            snap.pulse_evolution.regime,
            tuple(p.name for p in snap.pulse_evolution.patterns),
        )
    return (
        rows_sig,
        prev_sig,
        tuple(sorted(severity_counts.items())),
        indicators_sig,
        pulse_sig,
        hist_sig,
        opp_sig,
        intraday_sig,
        evolution_sig,
    )


def _format_ranked_opportunity(
    opp: RankedOpportunity,
    history: OpportunityHistory | None = None,
) -> str:
    """One dense OPP line for the ranked-composite output.

    Format:
      SYMBOL  TIER         BADGE  0.82   [stable·strengthening·persistent]   rationale phrases

    The history badge ("NEW" or "Nx") sits between the tier and the
    composite score so the eye picks up "fresh vs sticky" alongside the
    symbol identity. Width is fixed so OPP lines column-align even when
    badges differ in length.
    """
    tier_styled = _tier_text(opp.tier)
    history_badge = _format_history_badge(history)
    profile_chip = _format_profile_chip(opp.profile)
    score = f"[bold]{opp.composite_score:.2f}[/bold]"
    rationale = (
        "  [dim]·[/]  ".join(opp.rationale)
        if opp.rationale
        else "[dim](no driver above floor)[/dim]"
    )
    return (
        f"[bold]{opp.symbol:<6}[/]  {tier_styled}  {history_badge}  {score}  "
        f"{profile_chip}  [dim]{rich_escape(rationale)}[/dim]"
    )


def _format_history_badge(history: OpportunityHistory | None) -> str:
    """Fixed-width chip describing top-N membership recency.

    Width 4 keeps OPP lines column-aligned regardless of streak digits:
      None / streak == 0 → blank pad (no info)
      streak == 1        → "NEW " in bold yellow (fresh entry)
      streak >= 2        → "{N}x{pad}" in dim (sticky run)
    """
    if history is None or history.streak == 0:
        return "    "
    if history.streak == 1:
        return "[bold yellow]NEW [/]"
    label = f"{history.streak}x"
    return f"[dim]{label:<4}[/]"


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
        f"[{_PROFILE_AXIS_STYLES['stability'][profile.stability]}]{profile.stability}[/]",
        f"[{_PROFILE_AXIS_STYLES['trend'][profile.trend]}]{profile.trend}[/]",
        f"[{_PROFILE_AXIS_STYLES['persistence'][profile.persistence]}]{profile.persistence}[/]",
    ]
    return "[dim][[/dim]" + "[dim]·[/dim]".join(parts) + "[dim]][/dim]"


_QUALITY_LABEL_STYLES: dict[str, str] = {
    "high conviction": "bold green",
    "building momentum": "bold cyan",
    "reversal candidate": "bold yellow",
    "sentiment-driven": "bold magenta",
    "unstable / choppy": "bold red",
}


def _quality_label_style(label: str) -> str:
    return _QUALITY_LABEL_STYLES.get(label, "white")


def _driver_bar(score: float, width: int = 10) -> str:
    """Filled-block bar of width chars, proportional to score in ``[0, 1]``.

    Out-of-range scores clamp — defensive against any future driver
    whose normalization slips outside the bound.
    """
    filled = max(0, min(width, round(score * width)))
    return "█" * filled + "·" * (width - filled)


def _driver_style(score: float) -> str:
    """Color a driver score by tier — green for strong, yellow for
    moderate, dim for quiet. Mirrors the magnitude style used by the
    watchlist-header rank cells so the eye learns one palette."""
    if score >= 0.6:
        return "bold green"
    if score >= 0.3:
        return "yellow"
    return "dim"


def _render_opportunity_drilldown(drilldown: OpportunityDrilldown) -> str:
    """Multi-line render of the Opportunity Intelligence drilldown.

    Layout (skipping empty subsections):

      OPP #1  composite 0.78  NEW
      high conviction  ·  building momentum
      Drivers
        signal quality       ██████████  1.00  high signal quality
        momentum persistence ████████··  0.80  momentum holding across recent ticks
        ...
      Rationale
        ·  indicators aligned with action
        ·  high signal quality
    """
    header = (
        f"  [bold]OPP #{drilldown.rank}[/]  "
        f"[dim]composite[/] [bold]{drilldown.composite_score:.2f}[/]"
    )
    badge = _format_history_badge(drilldown.history)
    if badge.strip():
        header += f"  {badge}"

    lines: list[str] = [header]

    if drilldown.quality_labels:
        chips = "  [dim]·[/]  ".join(
            f"[{_quality_label_style(label)}]{label}[/]" for label in drilldown.quality_labels
        )
        lines.append(f"  {chips}")

    lines.append("  [bold dim]Drivers[/]")
    for driver in drilldown.drivers:
        bar = _driver_bar(driver.score)
        style = _driver_style(driver.score)
        descriptor = f"  [dim]{rich_escape(driver.descriptor)}[/dim]" if driver.descriptor else ""
        lines.append(
            f"    [dim]{driver.label:<22}[/dim]  "
            f"[{style}]{bar}[/]  "
            f"[{style}]{driver.score:.2f}[/]"
            f"{descriptor}"
        )

    if drilldown.state_phrases or drilldown.rationale:
        lines.append("  [bold dim]Rationale[/]")
        # State phrases first — they describe richer combinations
        # (strengthening, breadth, persistence). The source rationale
        # follows with per-driver specifics (tick counts, etc.) so the
        # reader gets the gist line first and the detail below.
        for phrase in drilldown.state_phrases:
            lines.append(f"    [dim]·[/]  [dim]{rich_escape(phrase)}[/dim]")
        for phrase in drilldown.rationale:
            lines.append(f"    [dim]·[/]  [dim]{rich_escape(phrase)}[/dim]")

    return "\n".join(lines)


def _format_intraday_summary(rows: list[RecommendationRow]) -> str:
    """Header summary of intraday alignment across the watchlist.

    Returns ``""`` when no row carries an intraday read (i.e. the
    feature is disabled or every fetch failed) so the header skips
    the line entirely. Otherwise reports counts:
      INTRADAY  3 diverging · 12 aligned · 4 neutral
    """
    diverging = 0
    aligned = 0
    neutral = 0
    has_any = False
    for r in rows:
        if r.error or r.intraday is None:
            continue
        has_any = True
        if is_divergent(r.action, r.intraday):
            diverging += 1
        elif r.action == SignalAction.HOLD or r.intraday.action == SignalAction.HOLD:
            neutral += 1
        else:
            aligned += 1
    if not has_any:
        return ""
    chunks: list[str] = []
    if diverging:
        chunks.append(f"[bold red]{diverging} diverging[/]")
    if aligned:
        chunks.append(f"[bold green]{aligned} aligned[/]")
    if neutral:
        chunks.append(f"[dim]{neutral} neutral[/]")
    return "  [dim]·[/]  ".join(chunks)


def _format_intraday_line(daily_action: SignalAction, intraday: IntradayRead) -> str:
    """One-line intraday alignment chip for the DetailPanel.

    Color reflects agreement with the daily action: green when both
    are directional and aligned, red when they're directional and
    opposed, dim when either side is HOLD (no divergence — just
    "intraday has no read").
    """
    if is_divergent(daily_action, intraday):
        alignment_style = "bold red"
        alignment_word = "diverging"
    elif daily_action == SignalAction.HOLD or intraday.action == SignalAction.HOLD:
        alignment_style = "dim"
        alignment_word = "neutral"
    else:
        alignment_style = "bold green"
        alignment_word = "aligned"
    action_text = _action_text(intraday.action)
    return (
        f"  [dim]{intraday.timeframe.value:<6}[/]  {action_text}  "
        f"[dim]conf[/] [bold]{intraday.confidence:.2f}[/]  "
        f"[dim]combined[/] {_fmt_signed(intraday.combined_score)}  "
        f"[dim]·[/]  [{alignment_style}]{alignment_word}[/]"
    )


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
        mom_pct = round(pulse.momentum_breadth * 100)
        mom_style = _breadth_style(pulse.momentum_breadth)
        chunks.append(f"[dim]mom[/dim] [{mom_style}]{mom_pct}%[/]")
    if pulse.sentiment_breadth > 0:
        sent_pct = round(pulse.sentiment_breadth * 100)
        sent_style = _breadth_style(pulse.sentiment_breadth)
        chunks.append(f"[dim]sent[/dim] [{sent_style}]{sent_pct}%[/]")

    # Volatility regime (always present — it's information even when calm).
    chunks.append(f"[{activity_style}]{pulse.activity}[/]")

    # Reversal intensity — surface only when non-zero (calm sessions stay tight).
    if pulse.reversal_intensity > 0:
        chunks.append(f"[dim]revs[/dim] [yellow]{pulse.reversal_intensity}[/yellow]")

    # Alert intensity — only when non-zero.
    if pulse.alert_intensity > 0:
        chunks.append(f"[dim]alerts[/dim] [bold red]{pulse.alert_intensity}[/]")

    # Strongest symbols — small inline list. Omitted entirely when none.
    if pulse.strongest_symbols:
        strong_chunks = [
            f"[bold]{sym}[/] [dim]{display}[/dim]" for sym, display in pulse.strongest_symbols
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


# ---------------------------------------------------------------------------
# Pulse history (HIST line)
# ---------------------------------------------------------------------------


_SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
"""Eight Unicode block-element heights, low to high. Index by
``round(value * 7)`` for a normalized [0, 1] series."""


def _sparkline_normalized(values: tuple[float, ...], *, lo: float = 0.0, hi: float = 1.0) -> str:
    """Sparkline with a fixed value range — best for breadth fractions
    where the absolute level matters (50% should look mid-height
    regardless of whether the series ever hit 80%).

    Empty series renders empty. Out-of-range values clamp.
    """
    if not values:
        return ""
    span = hi - lo
    if span <= 0:
        return _SPARKLINE_CHARS[0] * len(values)
    n = len(_SPARKLINE_CHARS) - 1
    return "".join(_SPARKLINE_CHARS[max(0, min(n, round((v - lo) / span * n)))] for v in values)


def _sparkline_relative(values: tuple[int, ...] | tuple[float, ...]) -> str:
    """Sparkline auto-scaled to the observed min/max — best for counts
    (reversal_intensity, alert_intensity) where there's no natural
    upper bound and what matters is the trajectory.

    Flat series renders as the lowest character so a quiet stretch
    reads visually quiet rather than mid-height.
    """
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return _SPARKLINE_CHARS[0] * len(values)
    span = hi - lo
    n = len(_SPARKLINE_CHARS) - 1
    return "".join(_SPARKLINE_CHARS[round((v - lo) / span * n)] for v in values)


def _format_pulse_history(history: PulseHistory) -> str:
    """One dense HIST line: sparkline + current value for each numeric
    pulse series. Each chunk omitted when its series carries no signal
    (all zeros) so quiet sessions stay tight.
    """
    chunks: list[str] = []

    # Momentum breadth — fixed [0, 1] scale.
    mom_curr = history.momentum_breadth[-1]
    if any(v > 0 for v in history.momentum_breadth):
        spark = _sparkline_normalized(history.momentum_breadth)
        style = _breadth_style(mom_curr)
        chunks.append(f"[dim]mom[/dim] [{style}]{spark}[/] [{style}]{round(mom_curr * 100)}%[/]")

    # Sentiment breadth — fixed [0, 1] scale.
    sent_curr = history.sentiment_breadth[-1]
    if any(v > 0 for v in history.sentiment_breadth):
        spark = _sparkline_normalized(history.sentiment_breadth)
        style = _breadth_style(sent_curr)
        chunks.append(f"[dim]sent[/dim] [{style}]{spark}[/] [{style}]{round(sent_curr * 100)}%[/]")

    # Reversal intensity — relative scale (no natural upper bound).
    if any(v > 0 for v in history.reversal_intensity):
        spark = _sparkline_relative(history.reversal_intensity)
        chunks.append(
            f"[dim]revs[/dim] [yellow]{spark}[/yellow] "
            f"[yellow]{history.reversal_intensity[-1]}[/yellow]"
        )

    # Alert intensity — relative scale.
    if any(v > 0 for v in history.alert_intensity):
        spark = _sparkline_relative(history.alert_intensity)
        chunks.append(
            f"[dim]alerts[/dim] [bold red]{spark}[/] [bold red]{history.alert_intensity[-1]}[/]"
        )

    if not chunks:
        # All-zero history — every series carries no signal. Render a
        # quiet placeholder so the HIST label doesn't visually orphan.
        return f"[dim]({history.length} ticks of quiet)[/dim]"

    return "  [dim]·[/]  ".join(chunks)


_REGIME_STYLES: dict[str, str] = {
    "risk-on": "bold green",
    "risk-off": "bold red",
    "mixed": "bold yellow",
    "indeterminate": "dim",
}


def _format_regime(regime: str) -> str:
    """Render the regime chip with regime-specific color."""
    style = _REGIME_STYLES.get(regime, "white")
    return f"[{style}]{regime}[/]"


def _format_patterns(patterns: tuple[TrajectoryPattern, ...]) -> str:
    """Render trajectory patterns as a · -separated row of chips.

    Each chip shows the human label and its numeric detail in dim
    text so the gate evidence sits right next to the call-out.
    """
    chunks = [
        f"[bold]{rich_escape(p.label)}[/]  [dim]{rich_escape(p.detail)}[/dim]" for p in patterns
    ]
    return "  [dim]·[/]  ".join(chunks)


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
    brief_kind: reactive[str] = reactive("row")  # "row" | "opp"

    # In-render cache: when the input signature matches the last frame,
    # short-circuit and return the cached string. Saves a full Rich
    # markup build (header / metrics / signals / history / alerts /
    # brief) on every tick where the selected row hasn't changed.
    _last_signature: tuple[object, ...] | None = None
    _last_rendered: str = ""

    def render(self) -> str:
        snap = self.snapshot
        if snap is None:
            return "[dim]select a row for details[/dim]"
        if not snap.rows:
            # Distinguish "no snapshot yet" from "snapshot has zero
            # rows" — the latter is the trader's empty-watchlist state
            # and warrants an actionable hint, not a passive prompt.
            return (
                "[bold yellow]No symbols on the watchlist.[/]\n"
                "[dim]Press [bold]a[/bold] to add one  ·  "
                "[bold]?[/bold] for the full keybinding list.[/dim]"
            )
        idx = max(0, min(self.row_index, len(snap.rows) - 1))
        r = snap.rows[idx]

        # OPP match: compute once so the signature builder and the
        # drilldown render see the same RankedOpportunity. Pure call,
        # negligible cost — same data the WatchlistHeader already
        # recomputes on its own render.
        opp_match: tuple[int, RankedOpportunity] | None = None
        for rank, opp in enumerate(rank_opportunities(snap, n=3), start=1):
            if opp.symbol == r.symbol:
                opp_match = (rank, opp)
                break

        signature = _detail_signature(
            r,
            snap,
            self.brief_state,
            self.brief_text,
            self.brief_kind,
            opp_match=opp_match,
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
            "[bold cyan]Alerts[/]\n" + _render_symbol_alerts(symbol_alerts) if symbol_alerts else ""
        )

        # Section: Intraday — secondary timeframe read when enabled.
        # The chip shows action/conf/combined plus an alignment word
        # against the daily action. Absent rows + intraday-disabled
        # controllers leave r.intraday=None and skip the section.
        intraday_block = ""
        if r.intraday is not None:
            intraday_block = "[bold cyan]Intraday[/]\n" + _format_intraday_line(
                r.action, r.intraday
            )

        # Section: Opportunity Intelligence (only when this symbol ranks
        # in the top-N). Additive — non-OPP rows render unchanged. The
        # row is threaded through so action-aware state phrases
        # (sentiment breadth polarity, stable BUY/SELL persistence) can
        # fire.
        opp_block = ""
        if opp_match is not None:
            rank, opp = opp_match
            drilldown = build_drilldown(
                opp,
                rank,
                snap.opp_history.get(r.symbol),
                row=r,
            )
            opp_block = "[bold cyan]Opportunity Intelligence[/]\n" + _render_opportunity_drilldown(
                drilldown
            )

        # Section: AI brief. Header reflects which kind of brief this is
        # so the trader can tell a per-row summary from an OPP brief at
        # a glance — the prose styles look similar enough otherwise.
        brief_block = ""
        brief_label = "AI brief — OPP" if self.brief_kind == "opp" else "AI brief"
        if self.brief_state == "loading":
            brief_block = f"[bold cyan]{brief_label}[/]\n  [dim italic]loading…[/dim italic]"
        elif self.brief_state == "ready":
            brief_block = (
                f"[bold cyan]{brief_label}[/]\n  [italic]{rich_escape(self.brief_text)}[/italic]"
            )
        elif self.brief_state == "error":
            brief_block = (
                f"[bold cyan]{brief_label}[/]\n  [red]failed:[/] {rich_escape(self.brief_text)}"
            )

        sections = [header, tagline, numbers, signals_block]
        if tier_block:
            sections.append(tier_block)
        if history_block:
            sections.append(history_block)
        if alerts_block:
            sections.append(alerts_block)
        if intraday_block:
            sections.append(intraday_block)
        if opp_block:
            sections.append(opp_block)
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
    return f"[dim]{ts}[/] [{colour}]{level.upper():5}[/] {rich_escape(message)}{suffix}"


def _render_symbol_alerts(alerts: list[Alert], *, limit: int = 4) -> str:
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
        pulse = snap.pulse if snap.pulse is not None else compute_pulse(snap)

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
        critical = sum(1 for a in snap.recent_alerts if a.severity == "critical")
        if critical:
            chunks.append(f"[bold red]{critical} critical[/]")

        # Tick + clock.
        clock = snap.timestamp.strftime("%H:%M:%S")
        chunks.append(f"[dim]tick {snap.tick} · {clock} UTC[/dim]")

        return "  ·  ".join(chunks)


class AddSymbolModal(ModalScreen[str | None]):
    """Modal prompting for a symbol to add to the watchlist.

    Returns the entered symbol (uppercased + stripped) on Enter, or
    ``None`` on Escape / empty input. The caller is responsible for
    duplicate / format validation — this modal just collects text.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "cancel", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="add_symbol_modal"):
            yield Static(
                "[bold]Add symbol to watchlist[/bold]\n[dim]e.g. NVDA · Esc to cancel[/dim]"
            )
            yield Input(placeholder="symbol", id="add_symbol_input")

    def on_mount(self) -> None:
        # Move focus to the input so the user can start typing immediately.
        self.query_one(Input).focus()

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted) -> None:
        value = event.value.strip().upper()
        self.dismiss(value or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


_HELP_TEXT = (
    "[bold]Esther dashboard — keybindings[/bold]\n\n"
    "[bold cyan]q[/]      quit the dashboard\n"
    "[bold cyan]r[/]      refresh now (force a tick)\n"
    "[bold cyan]p[/]      pause / resume auto-refresh\n"
    "[bold cyan]s[/]      AI brief for the selected row\n"
    "[bold cyan]o[/]      cycle through top-N opportunities\n"
    "[bold cyan]b[/]      AI brief for the selected opportunity\n"
    "[bold cyan]a[/]      add a symbol to the watchlist\n"
    "[bold cyan]x[/]      remove the selected symbol\n"
    "[bold cyan]?[/]      show / close this help overlay\n"
    "[bold cyan]↑ ↓[/]    select rows in the watchlist\n\n"
    "[dim]Esc or ? to close.[/dim]"
)


class HelpOverlay(ModalScreen[None]):
    """Modal listing every dashboard keybinding.

    Bound to ``?`` and ``Esc`` for symmetric open/close. Stateless —
    the text is constant and the modal closes via ``dismiss(None)``.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "close", show=False),
        Binding("question_mark", "close", "close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="help_overlay"):
            yield Static(_HELP_TEXT, id="help_text")

    def action_close(self) -> None:
        self.dismiss(None)


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

    AddSymbolModal { align: center middle; }
    AddSymbolModal #add_symbol_modal {
        width: 50;
        height: 7;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    AddSymbolModal Static { margin-bottom: 1; }

    HelpOverlay { align: center middle; }
    HelpOverlay #help_overlay {
        width: 56;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "quit"),
        Binding("r", "refresh_now", "refresh"),
        Binding("p", "toggle_pause", "pause/resume"),
        Binding("s", "summarize_selected", "AI brief"),
        Binding("o", "cycle_opportunity", "next OPP"),
        Binding("b", "brief_opportunity", "OPP brief"),
        Binding("a", "add_symbol", "add symbol"),
        Binding("x", "remove_selected", "remove symbol"),
        Binding("question_mark", "show_help", "help"),
        Binding("up,down", "noop", "select", show=False),
    ]

    paused: reactive[bool] = reactive(False)

    def __init__(
        self,
        controller: BaseController,
        refresh_seconds: float = 5.0,
        summarizer: Summarizer | None = None,
        opportunity_briefer: LLMOpportunityBriefer | None = None,
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
        self.opportunity_briefer = opportunity_briefer
        self._snapshot: DashboardSnapshot | None = None
        self._tick_handle: Timer | None = None
        self._tick_burst_handle: Timer | None = None
        # Per-symbol last-seen action; drives burst detection.
        self._previous_actions: dict[str, SignalAction] = {}
        # Brief cache keyed by (symbol, action_str). Action change invalidates.
        self._brief_cache: dict[tuple[str, str], str] = {}
        # OPP brief cache keyed by (symbol, composite_bucket). Bucket is
        # round(composite * 100) so a meaningful score shift re-bills,
        # but tick-over-tick float jitter on the same OPP doesn't.
        self._opp_brief_cache: dict[tuple[str, int], str] = {}

    # -- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield WatchlistHeader(id="watchlist_header")
            table: DataTable[object] = DataTable(zebra_stripes=True, cursor_type="row")
            table.add_columns(
                "SYM",
                "ACTION",
                "CONF",
                "BAR",
                "TECH",
                "SENT",
                "RSI",
                "MACD",
                "BBAND",
                "PRICE",
                "NEWS",
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
        self._update_sub_title()
        # Hydrate the in-memory brief caches from the controller's
        # persisted mirror so a restart surfaces saved briefs on the
        # very first keypress rather than re-billing the LLM.
        self._hydrate_briefs_from_controller()
        # First refresh immediately; then schedule recurring.
        await self._refresh_snapshot()
        self._tick_handle = self.set_interval(self.refresh_seconds, self._tick)

    def _hydrate_briefs_from_controller(self) -> None:
        """Copy the controller's persisted brief caches into local
        dict storage at mount.

        Keys are split on the controller's ``|`` delimiter back into
        the in-memory tuple form. Malformed keys (missing delimiter,
        empty halves) are skipped — defensive against a tampered
        snapshot file."""
        for raw_key, text in self.controller.brief_cache.items():
            symbol, sep, action = raw_key.partition("|")
            if not sep or not symbol or not action:
                continue
            self._brief_cache[(symbol, action)] = text
        for raw_key, text in self.controller.opp_brief_cache.items():
            symbol, sep, bucket_str = raw_key.partition("|")
            if not sep or not symbol:
                continue
            try:
                bucket = int(bucket_str)
            except ValueError:
                continue
            self._opp_brief_cache[(symbol, bucket)] = text

    def _update_sub_title(self) -> None:
        """Reflect the current watchlist in the dashboard sub-title.

        Called on mount and after any mutation so the header always
        matches reality.
        """
        watchlist = self.controller.watchlist
        if watchlist:
            self.sub_title = f"watchlist: {', '.join(watchlist)}"
        else:
            self.sub_title = "watchlist: (empty — press `a` to add)"

    # -- actions ----------------------------------------------------------

    def action_refresh_now(self) -> None:
        # Textual's run_worker is typed as Callable[..., Never] upstream, but
        # accepts any coroutine at runtime — the ignore is for the upstream
        # type signature, not a runtime concern.
        self.run_worker(self._refresh_snapshot, exclusive=True)  # type: ignore[arg-type]

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        msg = "paused" if self.paused else "resumed"
        self.query_one("#events", RichLog).write(
            f"[bold yellow]{datetime.now(UTC).strftime('%H:%M:%S')}[/] {msg}"
        )

    def action_noop(self) -> None:  # bound for footer hint only
        pass

    def action_show_help(self) -> None:
        """`?`: push the help overlay listing every keybinding."""
        self.push_screen(HelpOverlay())

    def action_add_symbol(self) -> None:
        """`a`: open the AddSymbolModal and append the entered symbol.

        Duplicates are silently rejected (no flash); empty / cancelled
        input is a no-op. After a successful add, kick a refresh worker
        so the new row appears immediately rather than on the next
        scheduled tick.
        """

        def on_dismiss(symbol: str | None) -> None:
            if symbol is None:
                return
            if not self.controller.add_symbol(symbol):
                # Already present or invalid — nothing more to do.
                return
            self._update_sub_title()
            self.query_one("#events", RichLog).write(
                f"[bold green]+[/] watchlist: added [bold]{symbol}[/]"
            )
            # Textual's run_worker is typed as Callable[..., Never] upstream, but

        # accepts any coroutine at runtime — the ignore is for the upstream
        # type signature, not a runtime concern.
        self.run_worker(self._refresh_snapshot, exclusive=True)  # type: ignore[arg-type]

        self.push_screen(AddSymbolModal(), on_dismiss)

    def action_remove_selected(self) -> None:
        """`x`: drop the currently-selected row's symbol from the watchlist.

        No confirmation — easily reversible via `a`. No-op when no row
        is selected or the watchlist is already empty.
        """
        row = self._selected_row()
        if row is None:
            return
        if not self.controller.remove_symbol(row.symbol):
            return
        self._update_sub_title()
        self.query_one("#events", RichLog).write(
            f"[bold red]−[/] watchlist: removed [bold]{row.symbol}[/]"
        )
        # Wipe brief caches keyed on this symbol so a re-add doesn't
        # show a stale brief from the prior session. Mirror the wipe
        # to the controller's persisted cache so a restart-after-
        # removal doesn't resurrect the brief either.
        self._brief_cache = {k: v for k, v in self._brief_cache.items() if k[0] != row.symbol}
        self._opp_brief_cache = {
            k: v for k, v in self._opp_brief_cache.items() if k[0] != row.symbol
        }
        self.controller.prune_briefs_for_symbol(row.symbol)
        # Textual's run_worker is typed as Callable[..., Never] upstream, but
        # accepts any coroutine at runtime — the ignore is for the upstream
        # type signature, not a runtime concern.
        self.run_worker(self._refresh_snapshot, exclusive=True)  # type: ignore[arg-type]

    def action_brief_opportunity(self) -> None:
        """`b`: generate or show an AI brief for one ranked opportunity.

        Target picking: the currently-selected row if it's in the top-N
        OPP set, else OPP #1. This pairs naturally with `o` (cycle) —
        press `o` to land on OPP #2, then `b` to brief that one. From a
        non-OPP row, `b` is a one-key shortcut to the top OPP brief.

        Caches by (symbol, composite_bucket) where bucket is composite
        rounded to two decimals * 100. Tick-over-tick float jitter on
        the same OPP doesn't re-bill, but a meaningful score shift does.

        No-op when no opportunities are ranked. Surfaces a friendly
        error in DetailPanel when the briefer isn't configured.
        """
        snap = self._snapshot
        if snap is None or not snap.rows:
            return

        from src.intelligence.opportunity_brief import OpportunityBriefContext

        ranked = rank_opportunities(snap, n=3)
        if not ranked:
            return

        opp_symbols = [opp.symbol for opp in ranked]
        selected = self._selected_row()
        if selected is not None and selected.symbol in opp_symbols:
            target_symbol = selected.symbol
        else:
            target_symbol = opp_symbols[0]

        target_opp = next(opp for opp in ranked if opp.symbol == target_symbol)

        detail = self.query_one(DetailPanel)
        detail.brief_kind = "opp"

        if self.opportunity_briefer is None:
            detail.brief_state = "error"
            detail.brief_text = "ANTHROPIC_API_KEY not configured — see ALPACA_SETUP.md."
            detail.refresh()
            return

        composite_bucket = round(target_opp.composite_score * 100)
        cache_key = (target_symbol, composite_bucket)
        if cache_key in self._opp_brief_cache:
            detail.brief_state = "ready"
            detail.brief_text = self._opp_brief_cache[cache_key]
            detail.refresh()
            return

        context = OpportunityBriefContext.from_snapshot(snap, target_symbol)
        if context is None:
            # Defensive: rank_opportunities found this symbol but
            # from_snapshot couldn't build a context. Bail silently.
            return

        detail.brief_state = "loading"
        detail.brief_text = ""
        detail.refresh()
        self.run_worker(
            self._fetch_opp_brief(context, cache_key),
            exclusive=False,
            group="opp_brief",
        )

    async def _fetch_opp_brief(
        self,
        context: OpportunityBriefContext,
        cache_key: tuple[str, int],
    ) -> None:
        """Compute the OPP brief off the UI thread, then push to DetailPanel."""
        import asyncio

        assert self.opportunity_briefer is not None  # checked by caller

        try:
            text = await asyncio.to_thread(self.opportunity_briefer.brief, context)
        except Exception as e:
            detail = self.query_one(DetailPanel)
            detail.brief_state = "error"
            detail.brief_text = str(e)
            detail.brief_kind = "opp"
            detail.refresh()
            return

        self._opp_brief_cache[cache_key] = text
        # Mirror into the controller so the next persistence tick
        # picks it up. Symbol + bucket are the cache key; the text
        # is opaque to the controller.
        symbol, bucket = cache_key
        self.controller.record_opp_brief(symbol, bucket, text)
        detail = self.query_one(DetailPanel)
        detail.brief_state = "ready"
        detail.brief_text = text
        detail.brief_kind = "opp"
        detail.refresh()

    def action_cycle_opportunity(self) -> None:
        """`o`: drill into the next ranked OPP by moving the table cursor.

        Behavior is keyed off the currently-selected row rather than an
        instance counter so the cycle stays predictable across snapshot
        refreshes:

        * Not on an OPP (or no row selected) → jump to OPP #1.
        * On OPP #N (N < last) → jump to OPP #N+1.
        * On the last OPP → wrap back to OPP #1.

        No-op when the snapshot has no rows or no symbols rank. Cursor
        movement triggers ``on_row_highlighted`` which refreshes the
        DetailPanel + AI brief sync — same cascade as arrow-key nav.
        """
        snap = self._snapshot
        if snap is None or not snap.rows:
            return
        ranked = rank_opportunities(snap, n=3)
        if not ranked:
            return

        opp_symbols = [opp.symbol for opp in ranked]
        selected = self._selected_row()
        if selected is not None and selected.symbol in opp_symbols:
            next_idx = (opp_symbols.index(selected.symbol) + 1) % len(opp_symbols)
        else:
            next_idx = 0
        target_symbol = opp_symbols[next_idx]

        row_idx = next(
            (i for i, r in enumerate(snap.rows) if r.symbol == target_symbol),
            None,
        )
        if row_idx is None:
            return
        self.query_one(DataTable).move_cursor(row=row_idx)

    def action_summarize_selected(self) -> None:
        """`s`: load the LLM brief for the currently-highlighted row."""
        row = self._selected_row()
        if row is None or row.error:
            return
        detail = self.query_one(DetailPanel)
        if self.summarizer is None:
            detail.brief_state = "error"
            detail.brief_text = "ANTHROPIC_API_KEY not configured — see ALPACA_SETUP.md."
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

    async def _fetch_brief(self, row: RecommendationRow, cache_key: tuple[str, str]) -> None:
        """Compute the AI brief off the UI thread, then push to DetailPanel."""
        import asyncio

        assert self.summarizer is not None  # checked by caller

        explanation = _explain_row(row)
        headlines = list(row.headlines) or None
        try:
            text = await asyncio.to_thread(
                self.summarizer.summarize, explanation, headlines=headlines
            )
        except Exception as e:
            detail = self.query_one(DetailPanel)
            detail.brief_state = "error"
            detail.brief_text = str(e)
            detail.refresh()
            return

        self._brief_cache[cache_key] = text
        # Mirror to the controller so the brief survives a restart.
        # Cache key is (symbol, action_value) — same shape the
        # controller persists.
        symbol, action_value = cache_key
        self.controller.record_row_brief(symbol, action_value, text)
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
        except Exception as e:
            self.query_one("#events", RichLog).write(f"[red]controller error:[/] {e}")
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
        self._previous_actions = {r.symbol: r.action for r in snap.rows if not r.error}
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

        If the row-brief cache has an entry for the currently-selected
        (symbol, action), show it. Otherwise reset to idle so a previous
        brief (row or OPP) doesn't linger once the cursor has moved.
        Always resets brief_kind to "row" since cursor navigation is a
        per-row concept; the user re-presses ``b`` to surface an OPP
        brief, which hits the OPP cache instantly if available.
        """
        detail = self.query_one(DetailPanel)
        detail.brief_kind = "row"
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
