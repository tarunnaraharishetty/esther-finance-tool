# NEXT_STEPS

Pick-up notes for the next session. Read this before writing any code.

*Last touched: 2026-05-13 (news quality scoring shipped).*

---

## What Esther is

**An AI market intelligence workstation, a trader productivity system,
and a research assistant.** It helps a discretionary trader understand
what's happening across their watchlist faster and with more grounded
context than they could assemble by hand.

It is **not** an autonomous trading AI, a hedge-fund engine, or a
prediction machine. There is no order submission anywhere in the
codebase, the data layer points at `paper-api.alpaca.markets`, and
every LLM-driven feature ships with anti-hallucination grounding rules
that forbid forecasts and require headlines to be quoted verbatim.
Anything that nudges back toward those framings is wrong.

---

## Current architecture status

Stable. The intelligence layer has settled into a coherent set of
modules with one clear responsibility each, and the dashboard renders
all of them through a small set of named widgets.

```
src/
├── main.py            # CLI: status / doctor / initdb / backfill /
│                      # recommend / summarize / recap / dashboard
├── config/            # Pydantic Settings (Alpaca + Anthropic + sentiment)
├── data/              # Alpaca client, market data, news (deduped),
│                      # ORM, repositories, filesystem cache
├── sentiment/         # FinBERT scorer
├── indicators/        # RSI / MACD / Bollinger
├── strategy/          # Signal + RecommendationEngine + 5-tier promoter
├── risk/              # Sharpe + max-drawdown
├── dashboard/         # Textual app + controller + snapshot state +
│                      # WatchlistHeader / DetailPanel / StatusLine /
│                      # AddSymbolModal
├── intelligence/      # explain · summary · llm_summary · alerts ·
│                      # alert_prioritizer · watchlist · history · tier ·
│                      # rankings · recap · grounding · pulse ·
│                      # pulse_history · opportunities · signal_profile ·
│                      # opportunity_history · opportunity_brief ·
│                      # opportunity_drilldown
└── utils/             # logging, rate limiter, preflight
```

**Layering invariant:** `intelligence` depends on `strategy` + `data`;
it never imports from `dashboard` or `main`. The `Summarizer` protocol
lets the dashboard / CLI consume either `TemplateSummarizer` (offline,
deterministic) or `LLMSummarizer` (Claude-backed) without caring which.
`LLMRecapGenerator` and `LLMOpportunityBriefer` follow the same shape.

---

## Completed systems

Everything below is shipped, tested, and either rendered in the
dashboard or exposed via the CLI.

**Data + sentiment**
- Alpaca paper-feed client with rate limiting + retry
- Bars + news ingestion with filesystem cache (warm-start via `esther backfill`)
- News dedup at the source — defensive against SDK pagination quirks
- FinBERT sentiment scoring per article
- **News quality weighting** — per-article weight = recency_decay ×
  source_reputation feeds `weighted_sentiment_mean()` when
  aggregating per-symbol sentiment. Half-life + floor configurable
  via Settings; curated source-tier map (Reuters/AP/Bloomberg = 1.0,
  WSJ/FT/CNBC = 0.8, Benzinga/Motley Fool = 0.6, blogs = 0.4,
  unknown = 0.3). Every downstream feature (alerts, pulse, ranking,
  opportunities, briefs) inherits the improvement via
  `row.sentiment_score`.

**Strategy**
- RSI / MACD / Bollinger indicators
- Signal aggregator + `RecommendationEngine`
- 5-tier promoter (`STRONG_BUY` / `BUY` / `HOLD` / `SELL` / `STRONG_SELL`)
  with signal-quality grade and stability tier

**Intelligence layer**
- Per-row explanation builder (`explain.py`)
- Per-symbol Claude brief (`LLMSummarizer`) — `s` key in dashboard,
  `esther summarize` CLI
- Watchlist-wide Claude recap (`LLMRecapGenerator`) — `esther recap` CLI
- Per-OPP Claude brief (`LLMOpportunityBriefer`) — `b` key in dashboard
- All three LLM modules pad past Opus 4.7's 4K-token cache minimum
  with worked few-shot examples
- Anti-hallucination grounding rules centralized in `grounding.py`
- Signal history (`history.py`) — per-symbol episode tracking
- Watchlist diff + top-movers + action breakdown (`watchlist.py`)
- Six ranked sections (momentum / sentiment / confidence / reversals
  / unusual / volatile)
- Market pulse: sentiment / conviction / activity + breadth + intensity
  + STRONG-symbol callouts (`pulse.py`)
- Pulse history with rolling-window sparklines (`pulse_history.py`)
- Opportunity detection (`opportunities.py`) — kind-based +
  composite-ranked with 7 drivers
- Three-axis signal profile chip (stability · trend · persistence)
- Opportunity history badges (NEW / Nx) — top-N membership tracking
- Alert engine with 4 per-row rules + 1 snapshot rule (opportunity_entry)
- Alert prioritizer with cooldowns + composites + per-tick cap
- 4 OPP-related composite alerts shipped in the example YAML

**Dashboard**
- Textual TUI with adaptive refresh (5s base, 1.5s burst on flips/alerts)
- Render-skip cache on `WatchlistHeader` and `DetailPanel`
- Always-visible `StatusLine` at the bottom
- `EventBuffer` collapses consecutive duplicate log lines
- Per-symbol alerts surfaced inside `DetailPanel` for the cursor row
- **Expanded opportunity drilldown** in `DetailPanel`: when the
  cursor row is in top-N OPPs, surface rank, composite, NEW/Nx
  tenure, quality-label chips (high conviction / building momentum
  / reversal candidate / sentiment-driven / unstable·choppy),
  seven driver bars sorted strongest-first with observational
  descriptors, source rationale phrases verbatim, **plus richer
  composite state phrases** (momentum strengthening across recent
  ticks · positive/negative sentiment breadth · stable BUY/SELL
  persistence) derived strictly from driver + profile + row state.
  Non-OPP rows render unchanged.
- Keyboard shortcuts: `q` quit · `r` refresh · `p` pause · `s` brief
  · `o` cycle OPP · `b` OPP brief · `a` add symbol · `x` remove symbol

**Quality baseline**
- 534 passing tests, 1 deselected (`slow`/`integration`)
- `ruff check .` green across the repo
- `mypy src` (`--strict`) green across all 45 source files
- Documented exceptions live in `pyproject.toml`

---

## Remaining high-priority roadmap

Four themes for upcoming sessions. Listed with current understanding
of scope and what they unlock; sequencing recommendation is in the
next section. (Both prior roadmap items — expanded opportunity
drilldown and news quality scoring — shipped on 2026-05-13.)

### 1. Persistence / history improvements
Restart loses everything: alert history, signal history, opp tracker
membership, pulse history, brief caches. A `SessionStore` interface
(SQLite-backed or json-snapshot-per-session) would let the trader
resume mid-day. This is foundational for anything that wants to span
sessions — including multi-timeframe and smarter pulse evolution.

### 2. Multi-timeframe intelligence
Today the recommender runs on daily bars. Adding a second timeframe
(e.g., 15-min intraday) would let alerts and the pulse react inside
the trading day rather than tick-to-tick on daily-bar quirks.
Architectural lift — touches data layer (separate cache buckets),
indicators (already pure functions, work for any timeframe),
recommender (per-timeframe scores combined), UI (timeframe toggle
or both visible). Best done AFTER persistence so intraday state
survives restarts.

### 3. Smarter market pulse evolution
The pulse currently classifies one tick. With pulse_history we have
trajectory data. Natural next steps: pattern detection ("breadth
firming for 8 ticks"), regime classification (risk-on / risk-off /
mixed), or an LLM-driven read of the trajectory tied into the recap
brief. Build on top of pulse_history; doesn't add new state.

### 4. Dashboard refinement
Ongoing polish that doesn't fit a neat feature box: column tunables,
help-overlay (`?`), better empty-state messages, configurable burst
window, perhaps a command palette via Textual's built-in. Best done
in parallel with feature work — pick up small items between bigger
features.

---

## Technical debt

Real items, not manufactured ones. None block the product; clean them
when the surrounding code is touched.

- **SDK-boundary `# type: ignore` comments** in `market_data.py` and
  `news_ingestion.py`. They're legitimate (alpaca-py's `feed=` kwarg
  is typed too narrowly, BarSet's `.data` is loose, the
  `with_async_retry` decorator confuses override-checking, Textual's
  `run_worker` is typed `Callable[..., Never]`). Could be wrapped in
  a thin typed adapter layer if it grows.
- **`requirements.txt` and `pyproject.toml` are both maintained.**
  They're now in sync but drift is possible. Consider deleting
  `requirements.txt` in favor of `pip install -e .` only.
- **`detect_opportunities` (kind-based)** is exported but shadowed by
  `rank_opportunities` (composite). Decide whether to keep both as a
  dual-view or remove the kind-based one.
- **Brief cache keyed on `(symbol, action)`** — invalidates on action
  flip but not on, say, a sentiment swing that doesn't change action.
  Acceptable today; might want finer invalidation if briefs become
  more expensive.
- **`DashboardSnapshot` is a mutable `@dataclass`** — controller
  fills `pulse`, `pulse_history`, `opp_history`, `alerts`,
  `recent_alerts` after construction. Works, but a frozen variant
  with `replace()` would make snapshot identity safer to reason about.
- **The bare `tuple[object, ...]` signature type** for the render-skip
  cache keys is loose; a `NamedTuple` per signature kind would be
  more explicit + give autocomplete.
- **Two `compute_pulse(snap)` fallbacks remain** in
  `WatchlistHeader.render()` and `StatusLine.render()` for the
  case where `snap.pulse is None` (unit-test fixtures). The fallback
  is deliberate; a follow-up could either require all snapshots to
  carry pre-computed pulse OR factor the recompute into one helper.

---

## Recommended next implementation order

Sequenced for maximum compounding return.

**1. Persistence (`SessionStore`)** *(start here tomorrow)*
- Foundation for everything else. Multi-timeframe and smarter pulse
  evolution both want session state to survive restarts. Doing it
  here unblocks both.

**2. Multi-timeframe intelligence**
- Biggest architectural lift; do after persistence so intraday state
  is durable. Plan the schema with persistence in mind.

**3. Smarter market pulse evolution**
- Builds naturally on persistence + multi-timeframe + existing
  pulse_history. Likely an LLM-driven read of the trajectory.

**4. Dashboard refinement**
- In parallel with the above. Pick up small items between bigger
  features rather than as a dedicated session.

---

## Resume here tomorrow

```
git pull                                  # confirm sync
.venv/Scripts/python.exe -m pytest --no-cov -q   # confirm 534 passing
```

Open this doc and start with **Persistence / `SessionStore`**
(roadmap item #1, recommended order step 1). A storage interface
(SQLite or json-snapshot) that lets the dashboard survive restarts:
alert history, signal history, opp-tracker membership, pulse
history, brief caches. Foundation for multi-timeframe + smarter
pulse evolution downstream.

Suggested opening prompt to Claude:

> Design and implement `SessionStore` — a storage interface that
> survives restarts. Persist: signal_history per symbol, opp_history
> per symbol, pulse_history, recent alerts (last N), brief caches
> (row + OPP). Pick SQLite (already a dep) or json-per-session.
> The controller writes on each tick; on startup it hydrates the
> trackers. Keep the file path configurable via Settings. Don't
> persist live Alpaca data — only intelligence-layer state.

---

## Repo state at handoff

- **Branch:** `main` is clean at `e517980` (push to `origin/main`
  pending — harness blocks direct push to default branch unless
  the user runs it themselves).
- **Tests:** 534 passing, 1 deselected (`slow`/`integration` mark).
  Run with `pytest`.
- **Lint/type:** `ruff check .` green; `mypy src --strict` green
  across all 45 source files.
- **Dependencies installed in `.venv/`** (Python 3.14): all of
  `pyproject.toml`'s base set, plus `anthropic`, `pyyaml`, `textual`,
  `ruff`, `mypy`, `pytest`. `alembic` and `backtrader` removed.
- **`.env` is not present.** `.env.example` documents the schema;
  `esther doctor --init-env` scaffolds the file. Real keys live only
  on disk, gitignored.
