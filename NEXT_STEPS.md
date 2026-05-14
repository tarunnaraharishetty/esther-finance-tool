# NEXT_STEPS

Pick-up notes for the next session. Read this before writing any code.

*Last touched: 2026-05-14 (intraday alerts auto-wired into main.py).*

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
│                      # pulse_history · pulse_evolution · opportunities ·
│                      # signal_profile · opportunity_history ·
│                      # opportunity_brief · opportunity_drilldown
├── persistence/       # SessionStore (JSON snapshot for cross-restart
│                      # state: signal_history, opp_history,
│                      # pulse_history, alert_state, tick counter)
├── strategy/          # base · recommendation · multi_timeframe
│                      # (IntradayRead + is_divergent for the
│                      # secondary intraday alignment chip)
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

**Persistence**
- `SessionStore` — atomic JSON snapshot of intelligence-layer
  trackers (signal history, OPP membership, pulse history, alert
  state, tick counter). Written at the end of every tick; loaded at
  controller startup. Corruption-resilient (missing / invalid /
  schema-mismatched files cold-start cleanly). Opt-in via
  `SessionStore` injection into the controller; tests stay
  ephemeral by omitting it. Default path
  `data/session_state.json`, env-overridable.
- **Brief caches** (row + OPP) also persist in the same snapshot.
  Saved text survives restarts so the next `s` / `b` keypress
  surfaces the cached brief without re-billing the LLM. Schema
  unchanged — new pydantic dict fields default to empty so older
  snapshots load fine. Symbol removal prunes the controller mirror
  too, so re-adding a wiped symbol doesn't resurrect its brief.

**Pulse evolution**
- `PulseEvolution` data (regime + trajectory patterns) attached to
  every `DashboardSnapshot`. Regime is one of `risk-on` / `risk-off`
  / `mixed` / `indeterminate`, derived from net bull/bear tilt
  across the last 6 ticks. Patterns fire from numeric gates: half-
  window mean comparisons for breadth firming/fading, max-ratio for
  alert spikes, sum-ratio for reversal clusters.
- `WatchlistHeader` surfaces `REGIME` and `PATTERNS` lines when the
  synthesis is meaningful (quiet on `indeterminate` / empty).
- `RecapContext` carries the regime + patterns into the recap LLM
  prompt as deterministic facts; existing anti-hallucination rules
  already cover observational language so no new few-shot needed.

**Multi-timeframe**
- **Phase 1: per-row intraday read** alongside the daily pipeline.
  Opt-in via `Settings.intraday_enabled` (default off; doubles
  Alpaca bar fetches when on). `RecommendationEngine.recommend_
  intraday()` runs the indicator scoring path on the secondary
  bar dataframe. `IntradayRead` attaches to each
  `RecommendationRow`. `DetailPanel` + `WatchlistHeader` surface
  alignment chips + divergence counts.
- **Phase 2a: schema v2 + view toggle.** `SessionStore` schema
  bumped to v2 with v1 migration (v1 snapshots load with empty
  intraday state, persist as v2). `BaseController.intraday_signal_
  history` records from `row.intraday` each tick; surfaces on
  `DashboardSnapshot.intraday_signal_history`. View toggle `t`
  flips the table's ACTION / CONF / BAR / TECH cells between
  daily and intraday; `WatchlistHeader` shows a VIEW chip while
  intraday is active.
- **Phase 2b: intraday OPP ranker + tracker.** Schema v3 (additive
  — intraday OPP membership fields). `rank_opportunities_intraday`
  scores opportunities off `row.intraday` + intraday signal
  history; tier maps from `intraday.action`. Parallel
  `BaseController.intraday_opp_tracker` records top-N membership.
  `WatchlistHeader`'s OPP block flips to `OPP-I` label + intraday
  ranking in intraday view; daily view unchanged.
- **Phase 2c: intraday pulse + Intelligence blocks + alerts.**
  - `compute_pulse_intraday` + parallel `intraday_pulse_tracker`;
    full PULSE / HIST / REGIME / PATTERNS lines toggle to the
    intraday triad in intraday view (`-I` label suffix).
  - `src/intelligence/timeframe_compare.py` — per-symbol
    `TimeframeStance` (aligned_bullish / aligned_bearish /
    conflict / intraday_only / daily_only / neutral) with
    observational phrases ("strengthening intraday momentum",
    "intraday reversal against daily trend"). Header ALIGN line
    surfaces aggregate counts.
  - `DetailPanel` adds compact **Daily Intelligence** +
    **Intraday Intelligence** blocks per row (action tier,
    confidence, tenure, trend, reversal state, opp rank, top
    drivers).
  - `src/intelligence/intraday_alerts.py` — 5 deterministic
    rules: reversal acceleration, momentum collapse, timeframe
    disagreement (transition edge only), intraday OPP entry,
    rapid confidence decay. Cooldowns wired in
    `_DEFAULT_COOLDOWNS`. **Auto-registered** in `main.py`'s
    `dashboard` command when `settings.intraday_enabled` is on —
    works whether or not the trader has a custom `alerts.yaml`.

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
  · `?` help overlay (lists every keybinding)
- **Refinements:** empty-watchlist hint on header + DetailPanel
  (mentions `a` and `?`); `dashboard_refresh_seconds` and
  `dashboard_burst_seconds` are Settings fields, env-overridable
  and surfaced via the `--refresh-seconds` / `--burst-seconds` CLI
  flags. **Column tunables** via `Settings.dashboard_columns` (env-
  overridable JSON list; field validator rejects unknown names).
  **Command palette** (Ctrl+P) lists every keybinding action by
  name; provider reuses the live `action_*` methods so it can't
  drift from the keys.

**Quality baseline**
- 643 passing tests, 1 deselected (`slow`/`integration`)
- `ruff check .` green across the repo
- `mypy src` (`--strict`) green across all 51 source files
- Documented exceptions live in `pyproject.toml`

---

## Remaining high-priority roadmap

The roadmap is light — every major theme is shipped. Remaining
work is opportunistic polish + the two follow-ups noted below.

### Open follow-ups

- **Multi-timeframe Phase 2**: separate intraday `SignalHistory` /
  `OpportunityMembershipTracker` / `PulseHistoryTracker`, intraday-
  specific alerts and opportunities, `SessionStore` schema bump
  (`schema_version=2`) to persist intraday state. Justified when
  intraday becomes part of the trader's primary read rather than
  alignment context.

### Dashboard polish backlog

- Column tunables (let the trader hide / reorder watchlist
  columns).
- Textual command palette wired to controller actions.
- A `--dry-run` flag on `esther backfill` so the trader can
  preview what would be fetched without paying the Alpaca quota.
- Inline status footer for the SessionStore (last-write timestamp,
  size) so the persistence layer's health is visible.

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

The MT2 roadmap is now fully shipped (phases 2a, 2b, 2c). What
remains is opportunistic polish + small follow-ups around the
edges. None are critical-path.

**Open MT2 follow-ups (small, opportunistic):**
- `action_cycle_opportunity` + `action_brief_opportunity` could
  target the active view's ranking when `view_timeframe ==
  "intraday"` (currently both still use the daily ranker).
- `StatusLine` could flip to the intraday pulse's `sentiment` chip
  when in intraday view (currently always daily).
- DetailPanel "Opportunity Intelligence" drilldown could swap
  between daily and intraday `RankedOpportunity` sources (currently
  always daily).

**Polish backlog (opportunistic)**
- In-app column-toggle modal (bound to `c`), surfacing the same
  `Settings.dashboard_columns` list for live editing.
- `--dry-run` flag on `esther backfill` to preview without paying
  the Alpaca quota.
- Inline SessionStore status footer (last-write timestamp, size).

---

## Resume here tomorrow

```
git pull                                  # confirm sync
.venv/Scripts/python.exe -m pytest --no-cov -q   # confirm 643 passing
```

MT2 is fully shipped (phases 2a / 2b / 2c). Remaining work is
opportunistic polish — pick whichever follow-up the trader hits
first in practice. Most likely entry points:

- **Auto-register intraday alert rules** in `main.py` so they fire
  without manual `alerts.yaml` setup when `settings.intraday_enabled`.
- **Action keys `o` / `b`** target the active view's ranking.

Suggested opening prompt for whichever follow-up:

> Pick a small follow-up. The MT2 roadmap is fully shipped; the
> backlog in NEXT_STEPS.md describes opportunistic polish. Use
> whichever item is closest to a trader pain-point you've seen.
> Keep the commit narrow and well-tested.

---

## Repo state at handoff

- **Branch:** `main` is clean at `2dca97e` (push to `origin/main`
  pending — harness blocks direct push to default branch unless
  the user runs it themselves; one commit queued locally).
- **Tests:** 643 passing, 1 deselected (`slow`/`integration` mark).
  Run with `pytest`.
- **Lint/type:** `ruff check .` green; `mypy src --strict` green
  across all 49 source files.
- **Dependencies installed in `.venv/`** (Python 3.14): all of
  `pyproject.toml`'s base set, plus `anthropic`, `pyyaml`, `textual`,
  `ruff`, `mypy`, `pytest`. `alembic` and `backtrader` removed.
- **`.env` is not present.** `.env.example` documents the schema;
  `esther doctor --init-env` scaffolds the file. Real keys live only
  on disk, gitignored.
