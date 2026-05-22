# CLAUDE.md

Guidance for Claude Code working in this repository. Read this before making changes.

## Project: Esther

Esther is an **AI market intelligence platform** for discretionary traders. It helps humans make better trading decisions — it does **not** trade on their behalf. The product combines:

- A multi-provider fundamentals pipeline with freshness contracts, cross-provider reconciliation, persistent health monitoring, and a durable retry queue.
- A grounded analyzer (technical scoring, multi-method valuation ensemble, AI explanations) with no-hallucination guarantees and per-claim citations.
- A Textual TUI and a React + Vite web frontend served by FastAPI, sharing the same SSE-streamed snapshot pipeline.
- A research-thesis surface (LLM-backed or deterministic template) with structured bull/bear/catalyst/valuation/technical/risk sections.

**In scope:** market monitoring, sentiment analysis, signal aggregation, AI summaries + theses, watchlist intelligence, alerts, explanations, provider observability, clean UX.

**Out of scope:** autonomous order submission, hedge-fund infrastructure, beating SPY, overengineered quant systems, excessive backtesting.

## Tech stack

- **Backend**: Python 3.12+, FastAPI, uvicorn, pydantic + pydantic-settings, SQLAlchemy + SQLite, structlog, tenacity, anthropic, httpx, pandas/numpy.
- **Sentiment**: transformers + torch (FinBERT).
- **TUI**: rich + textual.
- **Frontend**: React + TypeScript + Vite, Tailwind. Pages: Dashboard, Analyzer, AI, Charts, Movers, News, Research, Watchlist. SSE-driven snapshot updates.
- **Testing**: pytest + pytest-asyncio; vitest on the frontend.
- **Quality**: ruff (lint + format), mypy --strict.

## Layout

```
.
├── src/                          # all Python source
│   ├── main.py                   # CLI entry: status / doctor / initdb /
│   │                             # backfill / recommend / summarize / recap /
│   │                             # dashboard / serve
│   ├── api/                      # FastAPI app + routes
│   │   ├── app.py                # create_app, lifespan, SSE
│   │   ├── analyzer.py           # /api/analyzer/{symbol}
│   │   ├── broker.py             # SnapshotBroker (SSE tick loop)
│   │   ├── fundamentals.py       # /api/fundamentals/{symbol} + /health
│   │   ├── health.py             # /api/health/providers + /api/health/queue
│   │   └── research.py           # /api/research/{symbol}
│   ├── config/                   # Pydantic Settings, env loading
│   ├── data/                     # data layer (no business logic)
│   │   ├── alpaca_client.py      # Alpaca SDK wrapper, retry + rate limit
│   │   ├── cache.py              # bars + news filesystem cache
│   │   │                         # (mtime-TTL reads + envelope reads)
│   │   ├── envelope.py           # generic DataEnvelope[T] + Freshness tier
│   │   ├── freshness.py          # FreshnessPolicy table (fundamentals,
│   │   │                         # bars, news, analyst_targets)
│   │   ├── health_store.py       # SQLite-backed provider-health log
│   │   │                         # (lazy init, WAL, per-provider SLO summary)
│   │   ├── market_data.py        # bar / quote / trade fetching
│   │   ├── models.py             # Bar, NewsArticle, TimeFrame, etc.
│   │   ├── news_ingestion.py     # alpaca news pull + dedup
│   │   ├── orm.py / repositories # SQLAlchemy persistence
│   │   ├── retry_queue.py        # durable JSON-on-disk retry queue
│   │   ├── retry_worker.py       # async drain loop for the retry queue
│   │   ├── storage.py            # higher-level repo facade
│   │   └── streaming.py          # Alpaca websocket adapter
│   ├── dashboard/                # Textual TUI app + controller + state
│   ├── indicators/               # RSI, MACD, Bollinger, ATR, VolumeZScore
│   ├── intelligence/             # decision-support layer
│   │   ├── alerts.py             # alert engine
│   │   ├── alert_prioritizer.py  # cooldowns + composites + per-tick cap
│   │   ├── analyzer/             # technical scoring + valuation ensemble
│   │   │   ├── explanation.py    # grounded explanation builder + LLM
│   │   │   ├── prompts.py        # analyzer prompt assembly
│   │   │   ├── sector_medians.py # P/E, EV/EBITDA, P/S, PEG by sector
│   │   │   ├── technical.py      # 7 normalized [0,100] sub-scores
│   │   │   └── valuation.py      # 7-method ensemble (DCF, multiples,
│   │   │                         # PEG, historical band, analyst targets)
│   │   ├── fundamentals/         # provider abstraction (Phase 1+)
│   │   │   ├── alphavantage.py
│   │   │   ├── base.py           # Provider Protocol + error taxonomy
│   │   │   ├── finnhub.py
│   │   │   ├── fmp.py
│   │   │   ├── http.py           # shared HTTP error classification
│   │   │   ├── models.py         # NormalizedFundamentals + statements
│   │   │   ├── reconciliation.py # cross-provider field comparison
│   │   │   ├── sec_edgar.py
│   │   │   ├── service.py        # orchestrator: chain walk + envelope +
│   │   │   │                     # reconciliation + retry-enqueue
│   │   │   └── yahoo_fallback.py
│   │   ├── grounding.py          # canonical anti-hallucination rules
│   │   ├── history.py            # per-symbol signal episode tracking
│   │   ├── intraday_alerts.py    # 5-rule intraday alert pack
│   │   ├── llm_research.py       # Anthropic-backed thesis generator
│   │   ├── llm_summary.py        # Claude brief per symbol
│   │   ├── opportunities.py      # composite-ranked OPP detection
│   │   ├── opportunity_brief.py
│   │   ├── opportunity_drilldown.py
│   │   ├── opportunity_history.py
│   │   ├── pulse.py / pulse_history.py / pulse_evolution.py
│   │   ├── rankings.py
│   │   ├── recap.py              # watchlist-wide Claude recap
│   │   ├── research_thesis.py    # deterministic template thesis
│   │   ├── signal_profile.py
│   │   ├── summary.py            # template summary builder
│   │   ├── tier.py               # 5-tier promoter (STRONG_BUY..STRONG_SELL)
│   │   ├── timeframe_compare.py  # daily/intraday alignment stance
│   │   └── watchlist.py          # diff / top-movers / action breakdown
│   ├── persistence/              # SessionStore JSON snapshot
│   ├── risk/                     # Sharpe + max-drawdown
│   ├── sentiment/                # FinBERT analyzer + news quality
│   ├── strategy/                 # signal aggregator + RecommendationEngine +
│   │                             # multi-timeframe
│   └── utils/                    # logging, rate limiter, preflight
├── web/                          # React + Vite frontend
│   ├── src/
│   │   ├── lib/                  # API clients, types, helpers
│   │   └── pages/                # Dashboard, Analyzer, AI, Charts, Movers,
│   │                             # News, Research, Watchlist
│   └── vite.config.ts
├── config/                       # YAML/TOML configs (sector_medians.toml)
├── data/                         # SQLite DBs, OHLCV cache, news cache,
│                                 # health.db, retry_queue.json (gitignored)
├── logs/                         # rotating log files (gitignored)
├── tests/                        # pytest suite mirroring src/ layout
├── pyproject.toml
└── .env.example                  # copy to .env, never commit secrets
```

## Common commands

```bash
pip install -e ".[dev]"                       # install + dev tooling
python src/main.py                            # or: esther
python src/main.py serve                      # FastAPI on :8000
pytest                                        # full suite (integration deselected)
ruff check . && ruff format .                 # lint + format
mypy src                                      # strict type check
cd web && npm install && npm run dev          # Vite dev server
cd web && npm run typecheck                   # tsc --noEmit
```

## Architecture invariants

**Data freshness is a first-class field.** Every fetched record flows through a `DataEnvelope[T]` carrying `as_of`, `fetched_at`, `source_chain`, `freshness` tier (fresh/aging/stale/expired), and `provider_confidence`. Consumers never have to recompute "how old is this?". Freshness windows live in `src/data/freshness.POLICIES`.

**Provider chain with full attribution.** `FundamentalsService.fetch()` walks FMP → Finnhub → Alpha Vantage → SEC EDGAR → Yahoo, recording per-call `ProviderHealth` (status, latency, error) and surfacing the attempted chain on every result. Fall-through providers trigger **cross-provider reconciliation** on the four high-trust fields (revenue, net_income, eps_diluted, total_debt); divergences are reported, not silently picked.

**Persistent observability.** `HealthStore` (SQLite, WAL, lazy-init) sinks every health row. `/api/health/providers` returns rolling SLO summaries; `/api/health/queue` exposes in-flight retry state.

**Durable retry.** Transient-only chain exhaustions are enqueued in a JSON-on-disk `RetryQueue`. An opt-in `RetryWorker` drains with exponential backoff (30s → 30min) up to `max_attempts` (default 5).

**Layering.** `intelligence/` depends on `strategy/` + `data/`; it never imports from `dashboard/`, `main`, or `api/`. `data/` has no `intelligence/` imports (cycle-free). The `Summarizer` / `ThesisGenerator` protocols let the dashboard / CLI consume either template-based or LLM-backed implementations interchangeably.

## Conventions

- **Type hints everywhere.** mypy runs in `--strict`. No untyped defs. The repo is currently strict-clean across 80 source files.
- **Pydantic for config.** Never read `os.environ` directly outside `src/config/`. Import `get_settings()`.
- **Structured logging.** Use `structlog` via `src/utils/logging.py`. No `print()` in library code.
- **Async where it pays.** WebSocket streams, concurrent HTTP fetches → async. CPU-bound (indicators, sentiment scoring) → sync.
- **Rate-limit-aware.** All Alpaca/news API calls go through the wrapper in `src/data/alpaca_client.py` (uses `tenacity` for backoff). HTTP for fundamentals goes through `src/intelligence/fundamentals/http.py`.
- **Small files, small functions.** No god-modules. Split when a file grows past ~300 lines.
- **Secrets via env only.** Never hard-code keys. `.env` is gitignored; use `.env.example` as the schema.

## Safety rules

- **No order submission anywhere in the codebase.** Esther is decision-support; if a future module needs to talk to the broker side of Alpaca, raise it with the user first.
- **Paper feed only.** Even though we don't submit orders, the data layer points at `paper-api.alpaca.markets`. Live URLs are refused at startup by `DashboardController` / `recommend`.
- **No network in unit tests.** Mock provider/Alpaca at the SDK boundary. Real-API tests must be marked `@pytest.mark.integration` and are skipped by default.
- **All LLM outputs grounded.** Every Claude call routes through `src/intelligence/grounding.py`. Forecast language banned, headlines quoted verbatim, sourced fields only. Citation validator drops claims that don't trace to an input fact.

## Testing notes

- Mock external APIs at the SDK boundary; use `pytest-mock`. Don't hit the network in unit tests.
- Indicator tests should use deterministic fixture OHLCV data, not random.
- Headless Textual smoke tests use `App.run_test()`.
- HealthStore, RetryQueue, and DataEnvelope all have lazy-init invariants — tests rely on "constructing the object does not touch disk".
- The fundamentals test suite uses hand-rolled fake providers; the orchestrator's state machine is fully exercised without HTTP.

## Status

**Data-quality foundation complete (P1.1–P1.5):**
- P1.1 — `DataEnvelope[T]` + `FreshnessPolicy` table + freshness fields on `/api/fundamentals` and `/api/analyzer`.
- P1.2 — SQLite `HealthStore` + `/api/health/providers` rolling SLO endpoint.
- P1.3 — Cache TTL + freshness envelope on bars / news caches.
- P1.4 — Cross-provider reconciliation on high-trust fields with confidence penalty.
- P1.5 — Durable retry queue (JSON-on-disk) + opt-in async drain worker + `/api/health/queue`.

**Repo state:** 988 passing tests (1 deselected integration), `mypy src --strict` green across 80 source files, `ruff check .` green, `npm run typecheck` green.

**Next focus areas** (see `NEXT_STEPS.md` for the punch list):
- Priority #2 — Probabilistic analyzer (scenario tail probabilities, signal-history calibration).
- Priority #3 — Grounded research validator (per-claim source attribution + drop policy).
- Priority #4 — Trader workflow features (historical outcomes view, compare two stocks, sector-relative ranking).
- Priority #5 — Institutional-grade UX polish (loading states, freshness badges, mobile readability).
