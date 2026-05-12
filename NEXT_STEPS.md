# NEXT_STEPS

Pick-up notes for the next session. Read this before writing any code.

*Last touched: 2026-05-12.*

---

## 1. What Esther is now

An **AI-powered trading dashboard and research assistant** that helps a
human trader understand what's happening in their watchlist — not an
autonomous trader. The earlier ambitions (Backtrader cerebro,
hedge-fund infrastructure, autonomous order submission, Monte Carlo VaR)
have been deliberately cut. Anything in this doc that nudges back
toward "beat SPY" or "submit orders for the user" is wrong.

The product is built around four feature axes:

1. **Market monitoring** — live Alpaca paper bars per watchlist symbol.
2. **Sentiment analysis** — FinBERT scoring on recent news, surfaced
   alongside technicals.
3. **Decision support** — RSI / MACD / Bollinger + sentiment fed into a
   `RecommendationEngine` that emits BUY / HOLD / SELL with confidence
   and a structured explanation.
4. **Explanation and alerts** — Claude-written prose briefs, rule-based
   alerts (action change, confidence cross, sentiment shift) surfaced
   in a dedicated dashboard pane with a terminal bell on critical.

---

## 2. What got done this session (PRs 1–4)

Each is a single squashed commit on its own branch. None are pushed to
a remote yet; merge them into `main` (or whatever your trunk is) when
you're ready.

| PR | Branch | Commit | Summary |
|----|--------|--------|---------|
| 1 | `refactor/decision-support-cut` | `e5abf0a` | Stripped autonomous-trading + backtesting scaffolding. Removed `src/execution/`, `src/runner.py`, `src/backtesting/`, `src/risk/exposure.py`, `src/risk/var.py`, `scripts/demo_run.py`, plus `esther run` + `esther backtest` commands and risk-related settings. Dropped `backtrader` dep. Reframed `CLAUDE.md` around decision support; added a safety rule banning order submission anywhere in the codebase. |
| 2 | `feat/intelligence-scaffold` | `045f60e` | New `src/intelligence/` layer: `explain.py` (structured Explanation + Contributor types from raw scores), `summary.py` (`Summarizer` protocol + `TemplateSummarizer`), `alerts.py` (`Rule` protocol, `AlertEngine`, three concrete rules), `watchlist.py` (`diff_snapshots`, `top_movers`, `action_breakdown`). Dashboard `DetailPanel` now uses `explain()` to render bulleted contributor reasons. |
| 3 | `feat/llm-summaries` | `f1b712c` | Anthropic-backed `LLMSummarizer` on `claude-opus-4-7`, effort=low, prompt-cache marker in place (cache prefix is still below Opus 4.7's 4K-token minimum — marker pays off as the system prompt grows). New `esther summarize SYMBOL` CLI command maps Anthropic SDK exceptions to clean CLI errors. `anthropic>=0.55` added. |
| 4 | `feat/alerts-and-config` | `f400d4d` | `AlertEngine` wired into the dashboard — `BaseController` owns one, snapshot grows an `alerts` field. New dedicated alerts pane (color-coded by severity, terminal bell on critical). `config/alerts.yaml` YAML loader with strict validation. Latent `AlertEngine(rules=[])` falsy-default bug fixed. `pyyaml>=6.0` added. |

Test counts grew 58 → 99 → 135 → 144 → 154 across PRs 1–4. Run `pytest`
to confirm 154 pass and 1 slow/integration test is deselected.

---

## 3. Repo layout (as of this handoff)

The detailed map lives in `CLAUDE.md`. Quick summary:

```
src/
├── main.py            # esther CLI: status / doctor / initdb / recommend / summarize / dashboard
├── config/            # Pydantic Settings (Alpaca + Anthropic + sentiment)
├── data/              # Alpaca client, market/news/streaming, ORM, repositories
├── sentiment/         # FinBERT scorer
├── indicators/        # RSI / MACD / Bollinger
├── strategy/          # base Signal + SignalAction + RecommendationEngine + SignalAggregator
├── risk/              # Sharpe + max-drawdown (for future watchlist intel)
├── dashboard/         # Textual app + controller + snapshot state
├── intelligence/      # explain / summary / llm_summary / alerts / watchlist  (NEW)
└── utils/             # logging, rate limiter, preflight

config/
├── strategies.example.yaml   # universe + signals + indicators
└── alerts.example.yaml       # alert rules (NEW)
```

Layering still holds: `intelligence` depends on `strategy` + `data`; it
does not import from `dashboard` or `main`. The `Summarizer` protocol
lets the dashboard / CLI consume either `TemplateSummarizer` (offline
deterministic) or `LLMSummarizer` (Claude-backed) without caring which.

---

## 4. Punch list for the next session

Ordered by impact. Stop and reprioritize after step 3 — at that point
the product is genuinely usable end-to-end and the rest is polish.

### Critical path (do these first)

1. **End-to-end smoke against real Alpaca paper.** This pipeline has
   never been run live in one go. Set `ALPACA_API_KEY`,
   `ALPACA_API_SECRET`, and `ANTHROPIC_API_KEY` in `.env`, then run:

   ```
   esther doctor             # all green?
   esther recommend          # real bars + news, no orders
   esther summarize AAPL     # Claude brief on a real symbol
   esther dashboard          # live TUI with alerts pane
   ```

   *Done when:* every command completes without an unhandled exception
   and the dashboard runs through three or four ticks cleanly.

2. **`esther backfill --symbols X,Y,Z --days N`.** Cold-start solution
   so the indicators have history on the first dashboard tick. Pulls
   bars + news from Alpaca for the requested window and writes them
   into the SQLite cache via `BarRepository` / `NewsRepository` (both
   already exist). About 30 lines of CLI + a small async fetch loop.

   *Done when:* running it once then `esther dashboard` shows non-NaN
   RSI/MACD/Bollinger on the very first tick.

3. **Ruff + mypy cleanup.** Pre-existing caveat from PR 1 — ~50 ruff
   findings and ~25 mypy errors that predate this session. Most are
   `UP037` (quoted type annotations across `indicators/`,
   `strategy/`, `risk/metrics.py`) and stale `# type: ignore`
   comments. None block the product; cleaning them removes noise so
   future lint runs surface real regressions.

   *Done when:* `ruff check .` and `mypy src` are green or close to it
   (the alpaca-py SDK has some genuinely opaque types — those errors
   can stay if they're upstream).

### Highest-leverage feature work

4. **Integrate `summarize` into the dashboard.** Bind `s` to
   "summarize the currently-selected row." Open a modal or scrollable
   pane with the LLM brief. Cache the brief on the row so repeated
   presses don't re-bill Claude; invalidate when the recommendation
   action changes. This is the feature that most directly delivers on
   the "AI research assistant" framing.

   *Done when:* selecting AAPL in the dashboard and pressing `s`
   shows a Claude brief within ~3 seconds.

5. **Watchlist intelligence panel.** `src/intelligence/watchlist.py`
   has `diff_snapshots`, `top_movers`, and `action_breakdown` written
   and tested, but nothing in the UI uses them. A small panel above
   the watchlist table — "Top movers: NVDA (+0.8), MSFT (-0.6)" and
   "Since last refresh: AAPL flipped HOLD → BUY" — is high-signal and
   low-effort.

6. **Pad the system prompt to hit prompt caching.** Currently the
   `LLMSummarizer` system prompt is ~300 tokens; Opus 4.7's minimum
   cacheable prefix is 4K. Add 2–3 worked examples of good summaries
   in the system prompt (one per action: BUY / HOLD / SELL with
   different signal mixes). This both anchors output quality *and*
   makes the cache marker actually pay off.

### Small cleanups

7. **News dedup.** `AlpacaNewsSource.fetch` occasionally returns the
   same article id across pages. Repository upsert masks the issue at
   the storage layer, but in-memory dedup before scoring would cut
   FinBERT inference cost. One `dict[id, NewsArticle]` pass.

8. **Drop `alembic` from `pyproject.toml`** — was in for the Postgres
   migration path, which is deferred. SQLite + SQLAlchemy is enough.

9. **Audit `# type: ignore` comments.** Several are dead post-PR-1
   (e.g. in `main.py` around the recommend command). Easier to do
   after step 3.

### Deferred (do NOT start without discussion)

- Persistent alerts/events storage. The current in-process buffer is
  fine for a session-scoped dashboard.
- WebSocket streaming pipeline (replace polling). Real-time delivery
  is nice but the current 5-second poll is responsive enough.
- Multi-strategy framework. One good strategy first.
- Postgres migration. SQLite is fine for a local terminal tool.

---

## 5. Day-one recipe for tomorrow

If you only have one hour:

```
git checkout main             # or wherever you merged the PR train
pip install -e ".[dev]"       # PR 3 added anthropic, PR 4 added pyyaml
cp config/alerts.example.yaml config/alerts.yaml
# (edit .env with real ALPACA paper keys + ANTHROPIC_API_KEY)
esther doctor                 # confirm green
esther summarize AAPL         # confirm the Claude path works
esther dashboard              # confirm the full TUI loop works
```

That validates everything PRs 1–4 shipped. After that, work the punch
list top-down.

---

## 6. Repo state at handoff

- **Branches** (newest commit first):
  - `docs/next-steps-rewrite` — this commit
  - `feat/alerts-and-config` (`f400d4d`)
  - `feat/llm-summaries` (`f1b712c`)
  - `feat/intelligence-scaffold` (`045f60e`)
  - `refactor/decision-support-cut` (`e5abf0a`)
- **Tests:** 154 passing, 1 deselected (`slow`/`integration` mark).
  Run with `pytest`.
- **Lint/type:** ~50 ruff findings and ~25 mypy errors are
  pre-existing — not introduced by this session. See punch-list item 3.
- **Dependencies installed in `.venv/`** (Python 3.14): all of
  `pyproject.toml`'s base set, plus `anthropic`, `pyyaml`, `textual`,
  `ruff`, `mypy`, `pytest`.
- **`.env` is not present.** `.env.example` documents the schema;
  `esther doctor --init-env` scaffolds the file. Real keys live only
  on disk, gitignored.
