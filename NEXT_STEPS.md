# NEXT_STEPS

Pick-up notes for the next session. Read this before writing any code.

*Last touched: 2026-05-13 (pulse evolution shipped).*

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

**Multi-timeframe (Phase 1)**
- **Per-row intraday read** alongside the daily pipeline. Opt-in
  via `Settings.intraday_enabled` (default off; doubles Alpaca bar
  fetches when on). Default secondary timeframe is 15-min;
  configurable.
- `RecommendationEngine.recommend_intraday()` runs the daily
  indicator-scoring path on the secondary bar dataframe — sentiment
  is skipped (timeframe-agnostic). Returns an `IntradayRead`
  attached to each `RecommendationRow`.
- `DetailPanel` renders an "Intraday" section with the alignment
  word (green `aligned` / red `diverging` / dim `neutral`) when the
  row carries an intraday read.
- `WatchlistHeader` adds an INTRADAY line with the
  diverging / aligned / neutral counts.

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
- 587 passing tests, 1 deselected (`slow`/`integration`)
- `ruff check .` green across the repo
- `mypy src` (`--strict`) green across all 49 source files
- Documented exceptions live in `pyproject.toml`

---

## Remaining high-priority roadmap

The remaining roadmap is small — most major themes shipped on
2026-05-13 (drilldown, news quality scoring, `SessionStore`
persistence, multi-timeframe Phase 1, pulse evolution). Two
follow-ups still open:
- **Brief caches** (row + OPP briefs in `DashboardApp`) aren't
  persisted yet — small separate-section hook on the SessionStore.
- **Multi-timeframe Phase 2**: separate intraday `SignalHistory` /
  `OpportunityMembershipTracker` / `PulseHistoryTracker`, intraday-
  specific alerts and opportunities, `SessionStore` schema bump
  (`schema_version=2`) to persist intraday state. Justified when
  intraday becomes part of the trader's primary read rather than
  alignment context.

### 1. Dashboard refinement
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

**1. Dashboard refinement** *(start here tomorrow)*
- Help overlay (`?`), command palette via Textual built-ins,
  configurable burst window, empty-state polish, column tunables.
  Pick up a few small items rather than chasing a single feature.

**2. Brief-cache persistence**
- Small follow-up. Extend `SessionStore` with a brief-caches
  section (or a sibling file); the `DashboardApp` reads at mount,
  writes on each successful brief generation. Saves Anthropic
  calls across restarts.

**3. Multi-timeframe Phase 2**
- Bigger lift; only justify when intraday becomes the trader's
  primary read. Plan the `schema_version=2` snapshot carefully.

---

## Resume here tomorrow

```
git pull                                  # confirm sync
.venv/Scripts/python.exe -m pytest --no-cov -q   # confirm 587 passing
```

Open this doc and start with **Dashboard refinement** (roadmap
item #1). The major features are all in; the next session is
polish work — a help overlay (`?`), Textual's built-in command
palette, configurable burst window, better empty-state messages.
None of these are individually large, so pick a small bundle that
adds visible UX wins.

Suggested opening prompt to Claude:

> Tackle dashboard refinement. Pick three items from the punch list:
> a help overlay bound to `?` that lists every keybinding;
> Textual's built-in command palette wired to controller actions;
> a friendlier empty-state when the watchlist is cleared. Keep each
> change small and well-tested; preserve render-skip caching.

---

## Repo state at handoff

- **Branch:** `main` is clean at `c9f0d66` (push to `origin/main`
  pending — harness blocks direct push to default branch unless
  the user runs it themselves; two commits queued locally).
- **Tests:** 587 passing, 1 deselected (`slow`/`integration` mark).
  Run with `pytest`.
- **Lint/type:** `ruff check .` green; `mypy src --strict` green
  across all 49 source files.
- **Dependencies installed in `.venv/`** (Python 3.14): all of
  `pyproject.toml`'s base set, plus `anthropic`, `pyyaml`, `textual`,
  `ruff`, `mypy`, `pytest`. `alembic` and `backtrader` removed.
- **`.env` is not present.** `.env.example` documents the schema;
  `esther doctor --init-env` scaffolds the file. Real keys live only
  on disk, gitignored.
