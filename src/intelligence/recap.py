"""Watchlist-wide AI recap.

Generates a session-spanning brief across the whole watchlist — the
"I stepped away for an hour, what happened?" report. Same shape as
:class:`~src.intelligence.llm_summary.LLMSummarizer` but at the
snapshot level instead of per-symbol, with the same anti-hallucination
grounding rules.

Inputs (frozen :class:`RecapContext`):
- Action mix counts (BUY / HOLD / SELL)
- Ranked sections from :mod:`src.intelligence.rankings`
- Recent action flips with episode durations
- Top headlines per symbol (verbatim strings)
- Alert summary by severity

Output: 4–8 sentences of plain prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anthropic

from src.config import Settings, get_settings
from src.intelligence.grounding import GROUNDING_RULES_WATCHLIST
from src.intelligence.rankings import Rankings
from src.intelligence.rankings import compute as compute_rankings
from src.strategy.base import SignalAction
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot

log = get_logger(__name__)


_SYSTEM_PROMPT = f"""You are Esther, an AI trading research assistant in a terminal dashboard. \
This task is the session recap: a high-level brief summarizing what's notable across the \
trader's whole watchlist right now. The recap is decision support for a discretionary \
trader — never execution advice, never autonomous instruction.

You receive a structured snapshot: action mix, ranked sections (top movers by momentum / \
sentiment / confidence, biggest reversals, unusual movers, most volatile), recent action \
flips with how long each ran, top recent headlines per symbol, and alert counts.

{GROUNDING_RULES_WATCHLIST}

Structure your output (4–8 short sentences, plain prose, no markdown):

1. Open with the watchlist mix in one sentence (e.g., "The watchlist is leaning bullish: \
3 BUY, 1 HOLD, 1 SELL.").
2. Name the standout symbol(s) from the ranked sections — momentum, reversal, or \
unusual movement — citing the data that makes them stand out.
3. If there are action flips, mention one or two with their durations.
4. If alerts have fired, note the count and severity mix in one sentence.
5. Close with a watchful, observational framing — what to keep an eye on, not what to do. \
Never tell the user what to do.

Avoid: price targets, autonomous instructions, hype, hedging boilerplate, generic \
disclaimers, markdown formatting."""


@dataclass(frozen=True)
class RecapContext:
    """Frozen facts captured from a snapshot for recap generation.

    Constructed by :meth:`from_snapshot` from data already in the
    system; never holds free-form descriptions or out-of-band facts.
    The LLM literally cannot speculate about anything not in here.
    """

    action_mix: dict[SignalAction, int]
    rankings: Rankings
    recent_flips: tuple[tuple[str, SignalAction, SignalAction, int], ...]
    # (symbol, prior_action, current_action, prior_tick_count)
    headlines_by_symbol: dict[str, tuple[str, ...]] = field(default_factory=dict)
    alert_counts: dict[str, int] = field(default_factory=dict)
    # (severity → count)
    watchlist: tuple[str, ...] = ()
    tick: int = 0

    @classmethod
    def from_snapshot(
        cls,
        snapshot: DashboardSnapshot,
        *,
        max_headlines_per_symbol: int = 3,
        max_flips: int = 4,
    ) -> RecapContext:
        from src.intelligence.watchlist import action_breakdown

        rankings = compute_rankings(snapshot, n=3)
        action_mix = action_breakdown(snapshot)
        watchlist = tuple(r.symbol for r in snapshot.rows)

        # Flips: pull from signal_history's most-recent prior episode.
        flips: list[tuple[str, SignalAction, SignalAction, int]] = []
        for row in snapshot.rows:
            if row.error:
                continue
            summary = snapshot.signal_history.get(row.symbol)
            if summary is None or not summary.recent:
                continue
            prior = summary.recent[0]
            if prior.action == row.action:
                continue  # not actually a flip
            flips.append((row.symbol, prior.action, row.action, prior.tick_count))
        flips.sort(key=lambda f: f[3], reverse=True)  # longest prior run first

        headlines: dict[str, tuple[str, ...]] = {}
        for row in snapshot.rows:
            if row.headlines:
                headlines[row.symbol] = tuple(row.headlines[:max_headlines_per_symbol])

        alert_counts: dict[str, int] = {}
        for alert in snapshot.alerts:
            alert_counts[alert.severity] = alert_counts.get(alert.severity, 0) + 1

        return cls(
            action_mix=action_mix,
            rankings=rankings,
            recent_flips=tuple(flips[:max_flips]),
            headlines_by_symbol=headlines,
            alert_counts=alert_counts,
            watchlist=watchlist,
            tick=snapshot.tick,
        )


class LLMRecapGenerator:
    """Anthropic-backed recap generator. Same wire shape as
    :class:`~src.intelligence.llm_summary.LLMSummarizer`.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if client is not None:
            self.client = client
        else:
            key = self.settings.anthropic_api_key
            if key is None or not key.get_secret_value().strip():
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not configured. Set it in .env or pass an "
                    "explicit `client` argument to LLMRecapGenerator."
                )
            self.client = anthropic.Anthropic(api_key=key.get_secret_value())

    def generate(self, context: RecapContext) -> str:
        """Render the recap. Surfaces Anthropic SDK exceptions to the
        caller — the CLI wraps them in user-friendly errors."""
        user_message = _build_recap_user_message(context)
        log.info(
            "llm.recap.start",
            tick=context.tick,
            symbols=len(context.watchlist),
            flips=len(context.recent_flips),
        )
        response = self.client.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            system=_SYSTEM_PROMPT,
            output_config={"effort": "low"},
            cache_control={"type": "ephemeral"},
            messages=[{"role": "user", "content": user_message}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        log.info(
            "llm.recap.done",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read=getattr(response.usage, "cache_read_input_tokens", 0),
            cache_write=getattr(response.usage, "cache_creation_input_tokens", 0),
        )
        return text


def _build_recap_user_message(ctx: RecapContext) -> str:
    """Render the user message: structured, slot-filled, deterministic.

    Format kept stable so future-us can A/B against alternative wordings
    without surprising the model.
    """
    lines: list[str] = []
    lines.append(f"Tick: {ctx.tick}")
    if ctx.watchlist:
        lines.append(f"Watchlist: {', '.join(ctx.watchlist)}")
    lines.append(
        f"Action mix: "
        f"{ctx.action_mix.get(SignalAction.BUY, 0)} BUY, "
        f"{ctx.action_mix.get(SignalAction.HOLD, 0)} HOLD, "
        f"{ctx.action_mix.get(SignalAction.SELL, 0)} SELL"
    )

    sections = (
        ("Top momentum", ctx.rankings.strongest_momentum, "signed"),
        ("Top sentiment", ctx.rankings.strongest_sentiment, "signed"),
        ("Highest confidence", ctx.rankings.highest_confidence, "plain"),
        ("Biggest reversals", ctx.rankings.biggest_reversals, "signed"),
        ("Unusual movers", ctx.rankings.unusual_movers, "plain"),
        ("Most volatile", ctx.rankings.most_volatile, "int"),
    )
    for label, entries, fmt in sections:
        if not entries:
            continue
        rendered = ", ".join(_format_entry(sym, score, fmt) for sym, score in entries)
        lines.append(f"{label}: {rendered}")

    if ctx.recent_flips:
        lines.append("")
        lines.append("Recent action flips (most-recent-significant first):")
        for sym, prior, current, ticks in ctx.recent_flips:
            tick_word = "tick" if ticks == 1 else "ticks"
            lines.append(
                f"  {sym}: was {prior.value.upper()} for {ticks} {tick_word}, "
                f"now {current.value.upper()}"
            )

    if ctx.headlines_by_symbol:
        lines.append("")
        lines.append("Recent headlines (verbatim):")
        for sym, headlines in ctx.headlines_by_symbol.items():
            for h in headlines:
                lines.append(f'  {sym}: "{h}"')

    if ctx.alert_counts:
        lines.append("")
        parts = [f"{count} {sev}" for sev, count in sorted(ctx.alert_counts.items())]
        lines.append(f"Alerts this tick: {', '.join(parts)}")

    return "\n".join(lines)


def _format_entry(symbol: str, score: float, fmt: str) -> str:
    if fmt == "signed":
        return f"{symbol} {score:+.2f}"
    if fmt == "plain":
        return f"{symbol} {score:.2f}"
    if fmt == "int":
        return f"{symbol} {int(score)}"
    return f"{symbol} {score}"


__all__ = ["LLMRecapGenerator", "RecapContext"]
