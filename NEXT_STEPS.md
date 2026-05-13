# NEXT_STEPS

Pick-up notes for the next session. Read this before writing any code.

*Last touched: 2026-05-13.*

---

## 1. What Esther is now

An **AI-powered trading dashboard and research assistant** that helps a
human trader understand what's happening in their watchlist — not an
autonomous trader. The earlier ambitions (Backtrader cerebro,
hedge-fund infrastructure, autonomous order submission, Monte Carlo VaR)
have been deliberately cut. Anything in this doc that nudges back
toward "beat SPY" or "submit orders for the user" is wrong.

The product is built around five feature axes (the fifth is new since
the last NEXT_STEPS rewrite):

1. **Market monitoring** — live Alpaca paper bars per watchlist symbol.
2. **Sentiment analysis** — FinBERT scoring on recent news, surfaced
   alongside technicals.
3. **Decision support** — RSI / MACD / Bollinger + sentiment fed into a
   `RecommendationEngine` that emits a 5-tier recommendation
   (`STRONG_BUY` / `BUY` / `HOLD` / `SELL` / `STRONG_SELL`) with
   confidence, signal-quality grade, stability tier, and a structured
   explanation.
4. **Explanation and alerts** — Claude-written prose briefs (per-symbol
   `s` brief, watchlist-wide `recap`, per-OPP `b` brief), rule-based
   alerts (action change, confidence cross, sentiment shift, tier
   change, opportunity-entry) prioritized through cooldowns + composite
   detection, surfaced in a dedicated dashboard pane with a terminal
   bell on critical.
5. **Pulse + opportunities** — a one-glance MarketPulse line
   (sentiment/conviction/activity + breadth + intensity + STRONG
   symbols), a ranked OPP list using a 7-driver composite score with a
   three-axis signal profile chip, top-N membership history badges
   (NEW / Nx), and `o` / `b` keyboard shortcuts to drill from the
   ranking into the watchlist row and the AI brief.

---

## 2. What got done since the last rewrite (2026-05-12 → 2026-05-13)

A long stack of commits. Bucketed by theme:

**Recommendation surface**
- 5-tier recommendation system with `promote_to_tier` engine
- Wired the 5-tier into the dashboard UI (`signal_quality`,
  `stability`, `quality_reasons` on rows; tier styling in renderers)

**Alerts**
- `TierChangedRule` (same-action tier shifts)
- `AlertPrioritizer` + `AlertState` with cooldowns, composites, and a
  per-tick cap; YAML schema extended for the prioritizer block
- `OpportunityEntryRule` — first snapshot-level rule; fires when a
  symbol newly enters the top-N OPPs; lives in a parallel
  `snapshot_rules:` YAML block
- `recent_alerts` wired to the dashboard for session-scoped review

**Intelligence**
- `MarketPulse` engine (sentiment / conviction / activity) and an
  extended pulse with breadth, intensity, and STRONG-symbol callouts
- Opportunity detection (kind-based: convergence / reversal /
  high_conviction) **and** opportunity ranking (composite-score top-N
  with seven drivers + three-axis profile + rationale phrases)
- `OpportunityMembershipTracker` — rolling-window top-N history;
  surfaces NEW / Nx badges on OPP lines
- `LLMRecapGenerator` + `esther recap` CLI
- `LLMOpportunityBriefer` + `b` keypress (per-OPP AI brief)
- Centralized anti-hallucination grounding rules
  (`src/intelligence/grounding.py`) shared by every LLM module
- LLMSummarizer system prompt hardened against hallucination

**UX / dashboard**
- `StatusLine` always-visible state strip at the bottom of the
  dashboard
- Adaptive refresh — burst-mode follow-up tick after a flip or alert
- Render-skip guards on `WatchlistHeader` and `DetailPanel` (signature
  caching)
- `EventBuffer` collapses consecutive duplicates into one entry
- `o` keybinding cycles cursor through ranked OPPs
- Six ranked watchlist sections above the table (momentum / sentiment
  / confidence / reversals / unusual / volatile)

**Test counts:** 449 passing, 1 deselected (`slow`/`integration`).

---

## 3. Repo layout (as of this handoff)

The detailed map lives in `CLAUDE.md`. Quick summary:

```
src/
├── main.py            # esther CLI: status / doctor / initdb / backfill /
│                      # recommend / summarize / recap / dashboard
├── config/            # Pydantic Settings (Alpaca + Anthropic + sentiment)
├── data/              # Alpaca client, market/news/streaming, ORM, repositories,
│                      # filesystem cache for warm dashboard start
├── sentiment/         # FinBERT scorer
├── indicators/        # RSI / MACD / Bollinger
├── strategy/          # base Signal + SignalAction + RecommendationEngine +
│                      # SignalAggregator + RecommendationTier (5-tier)
├── risk/              # Sharpe + max-drawdown (for future watchlist intel)
├── dashboard/         # Textual app + controller + snapshot state +
│                      # WatchlistHeader / DetailPanel / StatusLine
├── intelligence/      # explain / summary / llm_summary / alerts /
│                      # alert_prioritizer / watchlist / history / tier /
│                      # rankings / recap / grounding / pulse / opportunities /
│                      # signal_profile / opportunity_history /
│                      # opportunity_brief
└── utils/             # logging, rate limiter, preflight

config/
├── strategies.example.yaml   # universe + signals + indicators
└── alerts.example.yaml       # alert rules + snapshot_rules + prioritizer
```

Layering still holds: `intelligence` depends on `strategy` + `data`; it
does not import from `dashboard` or `main`. The `Summarizer` protocol
lets the dashboard / CLI consume either `TemplateSummarizer` (offline
deterministic) or `LLMSummarizer` (Claude-backed) without caring which.
`LLMRecapGenerator` and `LLMOpportunityBriefer` follow the same wire
shape.

---

## 4. Punch list for the next session

Ordered by impact. Stop and reprioritize after step 2 — at that point
the product has been validated against real data end-to-end, and the
rest is polish or new feature work.

### Critical path (do these first)

1. **End-to-end smoke against real Alpaca paper.** Still never run
   live in one go. Set `ALPACA_API_KEY`, `ALPACA_API_SECRET`, and
   `ANTHROPIC_API_KEY` in `.env`, then run:

   ```
   esther doctor                # all green?
   esther backfill -s AAPL -s MSFT --days 60   # warm the cache
   esther recommend             # real bars + news, no orders
   esther summarize AAPL        # Claude per-symbol brief
   esther recap                 # Claude watchlist-wide brief
   esther dashboard             # live TUI — exercise s, b, o, r, p
   ```

   *Done when:* every command completes without an unhandled exception
   and the dashboard runs through three or four ticks cleanly,
   including pressing `s`, `b`, and `o` against real symbols.

2. **Ruff + mypy cleanup.** Long-standing pre-existing caveat. Most
   findings are `UP037` quoted-annotation noise across `indicators/`,
   `strategy/`, `risk/metrics.py` plus stale `# type: ignore`
   comments. None block the product; cleaning them removes noise so
   future lint runs surface real regressions.

   *Done when:* `ruff check .` and `mypy src` are green or close to it
   (the alpaca-py SDK has some genuinely opaque types — those errors
   can stay if they're upstream).

### Highest-leverage feature work

3. **Pad the system prompts to hit prompt caching.** All three LLM
   modules (`llm_summary`, `recap`, `opportunity_brief`) set the
   ephemeral cache marker, but Opus 4.7's minimum cacheable prefix is
   ~4K tokens and the current system prompts sit closer to ~600. Add
   2–3 worked examples per module (one per action / one per OPP shape
   / one per recap mood) so the cache marker actually pays off **and**
   output quality gets a few-shot anchor.

   *Done when:* the `cache_read_input_tokens` field on the response is
   non-zero on the second call within a session.

4. **OPP-related composite alerts.** The prioritizer's composite-rule
   feature (`requires: [...]` → one collapsed alert) is wired but no
   default ships pairing OPP entries with other rules. Likely useful:
   `(opportunity_entry, action_changed)` → "fresh OPP from a flip" at
   `critical`, `(opportunity_entry, sentiment_shift)` → "fresh OPP
   from a sentiment swing." Document them in
   `config/alerts.example.yaml`.

5. **Pulse history + sparkline.** Currently `MarketPulse` is recomputed
   each tick with no memory. A small ring buffer (last N pulses) +
   ASCII sparklines on momentum_breadth / sentiment_breadth would
   surface "is the watchlist mood drifting?" — the natural next step
   after the static pulse line.

6. **In-app watchlist editing.** Today the watchlist is fixed at
   launch (CLI flag or default). A `:add SYM` / `:remove SYM` colon
   command (or modal) would let the trader iterate without restarting.
   Touches: `BaseController.watchlist`, the engine warm-up paths, and
   the dashboard composer.

### Small cleanups

7. **News dedup.** `AlpacaNewsSource.fetch` occasionally returns the
   same article id across pages. Repository upsert masks the issue at
   the storage layer, but in-memory dedup before scoring would cut
   FinBERT inference cost. One `dict[id, NewsArticle]` pass.

8. **Drop `alembic` from `pyproject.toml`** — was in for the Postgres
   migration path, which is deferred. SQLite + SQLAlchemy is enough.

9. **Audit `# type: ignore` comments.** Several are dead post-PR-1
   (e.g. in `main.py` around the recommend command, the
   opportunity_briefer wiring, the AlertEngine kwargs). Easier to do
   after step 2.

### Deferred (do NOT start without discussion)

- Persistent alerts/events storage. The current in-process buffer is
  fine for a session-scoped dashboard.
- WebSocket streaming pipeline (replace polling). Real-time delivery
  is nice but the current 5-second poll plus adaptive burst is
  responsive enough.
- Multi-strategy framework. One good strategy first.
- Postgres migration. SQLite is fine for a local terminal tool.

---

## 5. Day-one recipe for tomorrow

If you only have one hour:

```
git pull                       # sync with the current main
pip install -e ".[dev]"        # in case a dep moved
cp config/alerts.example.yaml config/alerts.yaml
# (edit .env with real ALPACA paper keys + ANTHROPIC_API_KEY)
esther doctor                  # confirm green
esther backfill -s AAPL -s MSFT -s NVDA --days 60
esther summarize AAPL          # confirm the Claude path works
esther recap                   # confirm the watchlist recap works
esther dashboard               # exercise s / b / o / r / p
```

That validates everything currently shipped. After that, work the
punch list top-down — the smoke test (step 1) usually surfaces at
least one bug worth fixing before any new feature work.

---

## 6. Repo state at handoff

- **Branch:** `main` is clean and pushed to `origin/main` at
  `e0608b0`.
- **Tests:** 449 passing, 1 deselected (`slow`/`integration` mark).
  Run with `pytest`.
- **Lint/type:** `ruff check .` and `mypy src` have lingering pre-
  existing findings (see punch-list step 2) — not introduced by recent
  sessions.
- **Dependencies installed in `.venv/`** (Python 3.14): all of
  `pyproject.toml`'s base set, plus `anthropic`, `pyyaml`, `textual`,
  `ruff`, `mypy`, `pytest`.
- **`.env` is not present.** `.env.example` documents the schema;
  `esther doctor --init-env` scaffolds the file. Real keys live only
  on disk, gitignored.
