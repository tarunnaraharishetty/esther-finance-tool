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


# Few-shot examples below pad the system prefix past Opus 4.7's ~4K-token
# cache minimum AND anchor output across three different watchlist moods
# (risk-on, risk-off, mixed/quiet). See shared/prompt-caching.md.
_FEW_SHOT_EXAMPLES = """EXAMPLES — these illustrate the structure and observational tone across \
three watchlist moods. Match the format: open with the mix, name the standouts citing the \
data that makes them stand out, mention any flips with durations, note alert counts, close \
with a watchful framing. Quote any headline you reference verbatim and stay within the \
watchlist symbols provided.

--- Example 1 (Risk-on watchlist, momentum-led) ---

Input:
Tick: 142
Watchlist: AAPL, MSFT, NVDA, GOOGL, TSLA
Action mix: 4 BUY, 1 HOLD, 0 SELL
Top momentum: NVDA +0.78, AAPL +0.62, MSFT +0.55
Top sentiment: NVDA +0.65, GOOGL +0.40
Highest confidence: NVDA 0.84, AAPL 0.78, MSFT 0.71
Biggest reversals: TSLA +0.45
Unusual movers: NVDA 0.62
Most volatile: TSLA 4

Recent action flips (most-recent-significant first):
  TSLA: was SELL for 6 ticks, now BUY

Recent headlines (verbatim):
  NVDA: "NVDA reports record quarterly revenue, beats estimates by 8%"
  NVDA: "Datacenter revenue growth accelerates on AI demand"
  AAPL: "Apple raises holiday-quarter guidance amid strong iPhone preorders"

Alerts this tick: 1 critical, 2 warn

Output:
The watchlist is leaning decisively bullish: 4 BUY, 1 HOLD, no SELLs. NVDA is the standout, \
topping momentum at +0.78 and confidence at 0.84 with the headline "NVDA reports record \
quarterly revenue, beats estimates by 8%" reinforcing the technical read. AAPL and MSFT \
follow on momentum at +0.62 and +0.55 respectively. TSLA flipped from SELL to BUY after \
6 ticks of selling pressure — worth watching whether the reversal sticks. One critical and \
two warn alerts have fired this tick. Worth keeping an eye on whether NVDA's lead extends \
or the rest of the watchlist catches up.

--- Example 2 (Risk-off watchlist, reversals + alerts) ---

Input:
Tick: 87
Watchlist: AAPL, NFLX, META, AMZN, COIN
Action mix: 0 BUY, 1 HOLD, 4 SELL
Top momentum: NFLX -0.65, COIN -0.58, META -0.42
Top sentiment: NFLX -0.70, META -0.55, COIN -0.48
Highest confidence: NFLX 0.82, COIN 0.74
Biggest reversals: AAPL -0.50, AMZN -0.40
Unusual movers: NFLX 0.55
Most volatile: COIN 5

Recent action flips (most-recent-significant first):
  AAPL: was BUY for 9 ticks, now SELL
  AMZN: was BUY for 4 ticks, now SELL

Recent headlines (verbatim):
  NFLX: "Netflix subscriber growth stalls, ad-tier traction weaker than expected"
  COIN: "Crypto exchanges face fresh regulatory scrutiny in two key jurisdictions"

Alerts this tick: 3 critical, 4 warn, 2 info

Output:
The watchlist has rolled over: 4 SELL, 1 HOLD, no BUYs. NFLX leads the bearish move with \
momentum at -0.65, sentiment at -0.70, and the highest confidence reading at 0.82, \
anchored by the headline "Netflix subscriber growth stalls, ad-tier traction weaker than \
expected". COIN is right behind on weak technicals and the headline "Crypto exchanges \
face fresh regulatory scrutiny in two key jurisdictions". AAPL flipped from BUY to SELL \
after a sustained 9-tick run, and AMZN flipped after 4 ticks — both worth tracking for \
follow-through. Nine alerts have fired this tick (3 critical, 4 warn, 2 info), so the \
move has breadth. Worth keeping an eye on whether the selling pressure stabilizes or \
spreads to the names still on HOLD.

--- Example 3 (Mixed / quiet watchlist) ---

Input:
Tick: 23
Watchlist: SPY, QQQ, IWM
Action mix: 1 BUY, 2 HOLD, 0 SELL
Top momentum: SPY +0.25
Highest confidence: SPY 0.55
Most volatile: IWM 2

Recent action flips (most-recent-significant first):

Alerts this tick: 1 info

Output:
The watchlist is quiet and tilted slightly bullish: 1 BUY (SPY), 2 HOLD, no SELLs. SPY is \
the only directional name, with momentum at +0.25 and confidence at 0.55 — constructive \
but not commanding. No action flips this tick and only one info-level alert has fired. \
Worth watching whether SPY's edge extends or the index ETFs converge on a clearer mood.

--- Example 4 (Mixed bias, divergent signals across the watchlist) ---

Input:
Tick: 215
Watchlist: AAPL, MSFT, NVDA, AMD, INTC
Action mix: 2 BUY, 1 HOLD, 2 SELL
Top momentum: NVDA +0.55, AAPL +0.40, INTC -0.45
Top sentiment: AAPL +0.60, NVDA +0.50, INTC -0.55
Highest confidence: NVDA 0.74, INTC 0.69
Biggest reversals: AMD -0.30
Unusual movers: NVDA 0.50, INTC 0.45
Most volatile: AMD 3, INTC 2

Recent action flips (most-recent-significant first):
  AMD: was BUY for 5 ticks, now SELL
  MSFT: was HOLD for 12 ticks, now BUY

Recent headlines (verbatim):
  NVDA: "NVDA reaffirms outlook ahead of GTC keynote"
  INTC: "Intel confirms further fab-build delays in Ohio program"
  AAPL: "Apple supplier signals improving production capacity for next quarter"

Alerts this tick: 0 critical, 3 warn, 1 info

Output:
The watchlist mix is split: 2 BUY, 1 HOLD, 2 SELL, with directional conviction on \
both ends. NVDA leads the bullish side at +0.55 momentum and 0.74 confidence with the \
headline "NVDA reaffirms outlook ahead of GTC keynote" supporting the read, while INTC \
anchors the bearish side at -0.45 momentum and 0.69 confidence on the headline "Intel \
confirms further fab-build delays in Ohio program". AMD just flipped from BUY to SELL \
after a 5-tick run, and MSFT flipped from HOLD to BUY after sitting flat for 12 ticks. \
Three warn-level alerts have fired this tick. Worth watching whether the chip-segment \
split widens or the bullish names start absorbing pressure from the bearish names.

--- Example 5 (Single-symbol watchlist, sparse data) ---

Input:
Tick: 4
Watchlist: SPY
Action mix: 0 BUY, 1 HOLD, 0 SELL
Highest confidence: SPY 0.32

Recent action flips (most-recent-significant first):

Alerts this tick:

Output:
The watchlist holds a single symbol on HOLD: SPY at 0.32 confidence. With only four ticks \
of data the picture is sparse — no flips, no alerts, and no ranked momentum or sentiment \
to call out. Worth watching whether SPY firms up into a directional read as more bars \
accumulate or stays in the neutral zone.

--- Example 6 (Alert-heavy watchlist, modest action mix) ---

Input:
Tick: 67
Watchlist: COIN, MSTR, SQ, PYPL, HOOD
Action mix: 1 BUY, 3 HOLD, 1 SELL
Top momentum: COIN +0.30, HOOD -0.35
Top sentiment: COIN +0.55, HOOD -0.40
Highest confidence: COIN 0.62, HOOD 0.58
Biggest reversals: MSTR +0.25
Most volatile: SQ 4, PYPL 3, HOOD 3

Recent action flips (most-recent-significant first):
  MSTR: was SELL for 3 ticks, now HOLD
  SQ: was BUY for 2 ticks, now HOLD
  HOOD: was HOLD for 4 ticks, now SELL
  PYPL: was BUY for 6 ticks, now HOLD

Recent headlines (verbatim):
  COIN: "Coinbase quarterly revenue beats consensus on trading-volume rebound"
  HOOD: "Robinhood transaction revenue softens for second consecutive quarter"

Alerts this tick: 1 critical, 5 warn, 3 info

Output:
The watchlist action mix is muted (1 BUY, 3 HOLD, 1 SELL) but the alert flow is heavy: \
nine alerts this tick split 1 critical, 5 warn, 3 info — the watchlist is moving even \
where the headline action mix isn't loud. COIN is the directional standout on the \
bullish side at +0.30 momentum, +0.55 sentiment, and 0.62 confidence with the headline \
"Coinbase quarterly revenue beats consensus on trading-volume rebound" supporting it; \
HOOD leans bearish at -0.35 momentum and 0.58 confidence on softening transaction \
revenue. Four flips just landed across MSTR, SQ, HOOD, and PYPL — none with a long \
prior run, which suggests the watchlist is churning rather than committing. Worth \
watching whether the alerts converge on a directional regime or stay scattered.

--- Example 7 (Reversal-heavy watchlist, several long-held positions flipping) ---

Input:
Tick: 198
Watchlist: AAPL, NFLX, NVDA, AMZN, GOOGL, META, TSLA
Action mix: 1 BUY, 3 HOLD, 3 SELL
Top momentum: NFLX -0.55, GOOGL -0.40, AMZN -0.35
Top sentiment: NFLX -0.60, GOOGL -0.45
Highest confidence: NFLX 0.78, NVDA 0.65
Biggest reversals: AAPL -0.62, NVDA +0.40, AMZN -0.38
Unusual movers: AAPL 0.55, NVDA 0.45
Most volatile: TSLA 4, META 3

Recent action flips (most-recent-significant first):
  AAPL: was BUY for 14 ticks, now SELL
  AMZN: was BUY for 9 ticks, now SELL
  NVDA: was HOLD for 7 ticks, now BUY
  GOOGL: was HOLD for 5 ticks, now SELL

Recent headlines (verbatim):
  AAPL: "Apple revises holiday-quarter outlook lower amid component cost pressure"
  NFLX: "Netflix subscriber growth stalls, ad-tier traction weaker than expected"
  NVDA: "NVDA reports record quarterly revenue, beats estimates by 8%"

Alerts this tick: 2 critical, 6 warn, 4 info

Output:
The watchlist has tilted decisively bearish: 3 SELL, 3 HOLD, 1 BUY — and the structure \
underneath is a wave of reversals from previously-stable longs. AAPL flipped from BUY \
to SELL after 14 ticks of sustained buying with the headline "Apple revises holiday-\
quarter outlook lower amid component cost pressure" anchoring the move; AMZN flipped \
after 9 ticks; GOOGL flipped from HOLD after 5 ticks. NVDA is the lone bullish \
standout, flipping from HOLD to BUY after 7 ticks of sitting on the sidelines, \
supported by "NVDA reports record quarterly revenue, beats estimates by 8%". NFLX leads \
the bearish read on momentum (-0.55) and confidence (0.78). Twelve alerts have fired \
this tick (2 critical, 6 warn, 4 info), reflecting the breadth of the move. Worth \
watching whether NVDA's bullish flip absorbs any pressure or whether the broader \
reversal continues to spread.

--- Example 8 (Sentiment-driven watchlist with technical lag) ---

Input:
Tick: 156
Watchlist: PFE, JNJ, MRK, LLY, ABBV
Action mix: 2 BUY, 3 HOLD, 0 SELL
Top momentum: LLY +0.40, MRK +0.25
Top sentiment: LLY +0.78, MRK +0.55, JNJ +0.40
Highest confidence: LLY 0.68, MRK 0.55
Biggest reversals: PFE +0.20
Unusual movers: LLY 0.50

Recent action flips (most-recent-significant first):
  LLY: was HOLD for 8 ticks, now BUY

Recent headlines (verbatim):
  LLY: "Eli Lilly oncology pipeline data exceeds analyst expectations at conference"
  LLY: "Lilly weight-loss segment posts another quarter of triple-digit growth"
  MRK: "Merck announces accelerated trial enrollment for blockbuster candidate"
  JNJ: "J&J unit spinoff produces stronger-than-expected initial standalone numbers"

Alerts this tick: 0 critical, 2 warn, 1 info

Output:
The watchlist is bullishly tilted in pharma: 2 BUY, 3 HOLD, no SELLs. LLY is the \
standout, with the strongest sentiment in the watchlist (+0.78) anchored by "Eli Lilly \
oncology pipeline data exceeds analyst expectations at conference" and a fresh BUY \
flip after 8 ticks of HOLD. MRK and JNJ also carry constructive headlines but their \
technical reads are quieter than the news flow suggests. Three alerts have fired this \
tick. Worth watching whether the technical setup catches up to the supportive news \
across the watchlist or whether the bullish read stays anchored mostly to LLY.

--- Example 9 (Macro-driven session with broad alerts but no flips) ---

Input:
Tick: 312
Watchlist: SPY, QQQ, IWM, DIA, EFA, EEM
Action mix: 0 BUY, 2 HOLD, 4 SELL
Top momentum: IWM -0.50, EEM -0.45, EFA -0.40, QQQ -0.30
Top sentiment: EEM -0.30, EFA -0.20
Highest confidence: IWM 0.62, EEM 0.58
Biggest reversals:
Unusual movers: IWM 0.40
Most volatile:

Recent action flips (most-recent-significant first):

Recent headlines (verbatim):
  SPY: "Risk-off tone broadens as macro data prints on the soft side"
  IWM: "Small-caps lag as rate-sensitive segments take outsized pressure"

Alerts this tick: 1 critical, 4 warn, 2 info

Output:
The watchlist has rolled over broadly: 4 SELL, 2 HOLD, no BUYs across major index ETFs. \
IWM leads the bearish move at -0.50 momentum and 0.62 confidence with the headline \
"Small-caps lag as rate-sensitive segments take outsized pressure" describing the \
mechanism, and EEM/EFA follow on weak momentum and modestly negative sentiment. No \
action flips this tick — the moves are extensions of existing positioning rather than \
fresh regime changes. Seven alerts have fired (1 critical, 4 warn, 2 info), \
consistent with a broad macro-driven session. Worth watching whether the breadth \
deepens into a sustained de-risking or stabilizes after the initial wave."""


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
disclaimers, markdown formatting.

{_FEW_SHOT_EXAMPLES}"""


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
