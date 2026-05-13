# NEXT_STEPS

Pick-up notes for the next session. Read this before writing any code.

*Last touched: 2026-05-13 (second refresh — most of the prior punch list shipped this session).*

---

## 1. What Esther is now

An **AI-powered trading dashboard and research assistant** that helps a
human trader understand what's happening in their watchlist — not an
autonomous trader. The earlier ambitions (Backtrader cerebro,
hedge-fund infrastructure, autonomous order submission, Monte Carlo VaR)
have been deliberately cut. Anything in this doc that nudges back
toward "beat SPY" or "submit orders for the user" is wrong.

The product is built around five feature axes:

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
5. **Pulse + opportunities** — a one-glance `PULSE` line
   (sentiment/conviction/activity + breadth + intensity + STRONG
   symbols), a `HIST` line of rolling sparklines for momentum /
   sentiment breadth and reversal / alert intensity, a ranked OPP list
   using a 7-driver composite with three-axis profile chips and
   top-N membership history badges (NEW / Nx), and `o` / `b` / `a` /
   `x` keyboard shortcuts to cycle through ranked OPPs, generate AI
   briefs on them, and edit the watchlist live.

---

## 2. What got done this session

The prior NEXT_STEPS punch list ran 9 items deep; 8 of 9 shipped.
Bucketed by theme:

**LLM quality**
- `070f42c` — Padded all three LLM system prompts (`llm_summary`,
  `recap`, `opportunity_brief`) past Opus 4.7's ~4K-token cache
  minimum with 7–10 worked few-shot examples each. Anchors output
  format AND lets the cache marker actually pay off. Done when:
  verify in production by checking `cache_read_input_tokens > 0` on
  the second call within a session.

**Alerts**
- `6ed57ad` — Documented four OPP-related composites in the example
  YAML: `opp_flip_with_sentiment` (triple-confirm, critical),
  `fresh_opp_from_flip` (critical), `fresh_opp_from_sentiment_swing`
  (warn), `fresh_opp_from_tier_shift` (warn). Pinned the
  triple-before-pair ordering with regression tests so future edits
  can't silently break the contract.

**Intelligence**
- `b02d6e8` — `PulseHistoryTracker` rolling-window state on the
  controller; renderer adds a HIST line below PULSE with Unicode
  block-element sparklines. Drive-by: fixed a latent
  cache-invalidation gap where the header signature only hashed three
  categorical pulse fields.
- `e1e2b65` — `dedup_articles()` helper + defensive dedup pass inside
  `AlpacaNewsSource.fetch` to cut FinBERT inference cost when the
  SDK's pagination surfaces the same article id twice.

**UX**
- `58616e5` — In-app watchlist editing: `a` opens an `AddSymbolModal`
  with a focused Input, `x` removes the cursor row's symbol. Both
  trigger an immediate refresh; sub-title and brief caches stay in
  sync; per-symbol session state preserved across re-adds.

**Polish bundle**
- `dd42969` — Dropped `alembic` (deferred Postgres path) and the
  long-dead `backtrader` from requirements; synced `requirements.txt`
  with `pyproject.toml` for `anthropic` + `pyyaml`.
- `de008db` — Ruff: 102 findings → 0. Auto-fixed 70; manual fixes
  for the rest; added per-file ignores for deliberate patterns
  (typographic Unicode in dashboard prose, lazy E402 imports in
  `main.py`, blind asserts in tests).
- `34442fc` — Mypy `--strict`: 62 errors → 0. Found and fixed two
  latent bugs masked by `# type: ignore` comments:
  1. `DashboardController` was silently dropping
     `alert_prioritizer` and `alert_state` kwargs even though
     `main.py` was passing them — the live dashboard wasn't
     applying user-configured cooldowns or composites.
  2. `AlertPrioritizer` was using `set[Alert]` for a set of
     `id(a)` ints; would have broken if `Alert` ever became
     hash-incompatible.

**Test counts:** 488 passing, 1 deselected (`slow`/`integration`).

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
│                      # WatchlistHeader / DetailPanel / StatusLine /
│                      # AddSymbolModal
├── intelligence/      # explain / summary / llm_summary / alerts /
│                      # alert_prioritizer / watchlist / history / tier /
│                      # rankings / recap / grounding / pulse /
│                      # pulse_history / opportunities / signal_profile /
│                      # opportunity_history / opportunity_brief
└── utils/             # logging, rate limiter, preflight

config/
├── strategies.example.yaml   # universe + signals + indicators
└── alerts.example.yaml       # alert rules + snapshot_rules + prioritizer
                              # (now includes 4 OPP composites)
```

Layering still holds: `intelligence` depends on `strategy` + `data`; it
does not import from `dashboard` or `main`. The `Summarizer` protocol
lets the dashboard / CLI consume either `TemplateSummarizer` (offline
deterministic) or `LLMSummarizer` (Claude-backed) without caring which.
`LLMRecapGenerator` and `LLMOpportunityBriefer` follow the same wire
shape.

---

## 4. Punch list for the next session

The prior punch list has been worked down to one item. Anything below
that is forward-looking — not blocking, take or leave based on what's
interesting next.

### Critical path

1. **End-to-end smoke against real Alpaca paper.** Still never run
   live in one go. Set `ALPACA_API_KEY`, `ALPACA_API_SECRET`, and
   `ANTHROPIC_API_KEY` in `.env`, then run:

   ```
   esther doctor                # all green?
   esther backfill -s AAPL -s MSFT --days 60   # warm the cache
   esther recommend             # real bars + news, no orders
   esther summarize AAPL        # Claude per-symbol brief
   esther recap                 # Claude watchlist-wide brief
   esther dashboard             # live TUI — exercise s, b, o, a, x, r, p
   ```

   *Done when:* every command completes without an unhandled
   exception and the dashboard runs through three or four ticks
   cleanly, including pressing `s`, `b`, `o`, `a`, `x` against real
   symbols. **Bonus:** verify `cache_read_input_tokens > 0` on the
   second `s`/`b`/`recap` call within a session — confirms the
   prompt-cache padding from `070f42c` is firing.

### Forward-looking — high signal, no precondition

2. **Session persistence.** Restart loses everything: alert history,
   signal history, opp tracker membership, pulse history, brief
   caches. A small SQLite schema (or json snapshot per session) would
   let the trader resume mid-day. Touches: introduce a
   `SessionStore` interface in `src/intelligence/`, wire
   load/save into controller startup/teardown, decide what gets
   persisted vs is genuinely ephemeral.

3. **Multi-watchlist support.** Today the watchlist is fixed at
   launch (CLI flag, default, or runtime via `a`/`x`). A
   `--watchlist <name>` flag plus a `config/watchlists/*.yaml`
   directory would let the trader maintain themed sets (megacap,
   semis, banks, my-positions). Pairs naturally with item #2 — each
   watchlist could persist its own session state.

4. **OPP detail expansion.** When an OPP is selected, the DetailPanel
   shows row data. It could ALSO surface the seven-driver score
   breakdown — bar chart per driver (technical_alignment 1.0 ████ |
   sentiment_alignment 0.5 ██) so the trader sees WHY the composite
   is what it is, not just the rationale phrases. The drivers are
   already on `RankedOpportunity`; just needs render plumbing.

5. **News quality / freshness scoring.** All news sources are
   currently treated equally. A small heuristic that weights by
   recency (newer = more weight) and source reputation (Reuters >
   Benzinga > "blog") would tighten the FinBERT-driven sentiment
   scores. Lives in `src/sentiment/` as a pre-FinBERT weighting pass.

6. **OPP brief uses pulse history context.** The OPP briefer already
   gets the symbol's seven drivers + history; it could ALSO be
   handed the recent pulse trajectory ("the watchlist has been
   firming up over the last 8 ticks"). Free signal already on the
   snapshot; just thread it into `OpportunityBriefContext`.

### Small cleanups

7. **CLAUDE.md refresh.** The repo guide still describes the older
   shape (no `pulse_history`, no `opportunity_brief`, no `a`/`x`
   keys). Bring it current so a fresh onboarding agent doesn't have
   to reverse-engineer the layout.

8. **README.** There isn't one. A short README with screenshots /
   `asciinema` recording would make the project legible to anyone
   who lands on the GitHub page.

### Deferred (do NOT start without discussion)

- WebSocket streaming pipeline (replace polling). Real-time delivery
  is nice but the current 5-second poll plus adaptive burst is
  responsive enough for a discretionary trader.
- Multi-strategy framework. One good strategy first.
- Postgres migration. SQLite is fine for a local terminal tool
  (and `alembic` was just removed from deps for that reason).

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
esther dashboard               # exercise s / b / o / a / x / r / p
```

That validates everything currently shipped. After that, work the
punch list top-down — the smoke test (step 1) usually surfaces at
least one bug worth fixing before any new feature work.

---

## 6. Repo state at handoff

- **Branch:** `main` is clean and pushed to `origin/main` at
  `34442fc`.
- **Tests:** 488 passing, 1 deselected (`slow`/`integration` mark).
  Run with `pytest`.
- **Lint/type:** `ruff check .` is **green** across the repo;
  `mypy src` (`--strict`) is **green** across all 43 source files.
  Documented exceptions live in `pyproject.toml` (typographic
  Unicode in dashboard prose, lazy `E402` imports in `main.py`,
  blind `pytest.raises(Exception)` in tests).
- **Dependencies installed in `.venv/`** (Python 3.14): all of
  `pyproject.toml`'s base set, plus `anthropic`, `pyyaml`, `textual`,
  `ruff`, `mypy`, `pytest`. `alembic` and `backtrader` removed.
- **`.env` is not present.** `.env.example` documents the schema;
  `esther doctor --init-env` scaffolds the file. Real keys live only
  on disk, gitignored.
