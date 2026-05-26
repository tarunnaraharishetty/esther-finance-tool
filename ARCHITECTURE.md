# ARCHITECTURE.md — Esther as it actually is

This document describes how the system is built *today*, not how it is
pitched. Where the codebase makes a load-bearing claim that the
implementation does not honor, that gap is called out inline.

## 1. Bird's-eye view

```
                ┌────────────────────────────────────────────────────────┐
                │  React + Vite SPA (web/) — served same-origin in prod  │
                │  Pages: Dashboard, Analyzer, AI, Charts, Movers,       │
                │         News, Research, Watchlist, Compare, Providers  │
                │  Auth: cookie-based session, HttpOnly + SameSite=lax   │
                │  Realtime: SSE via /api/stream (one EventSource)       │
                └───────────────────────┬────────────────────────────────┘
                                        │ HTTP/JSON + SSE (same origin)
                ┌───────────────────────▼────────────────────────────────┐
                │  FastAPI app (src/api/app.py)                          │
                │  Routes: auth, watchlist, snapshot, stream,            │
                │   fundamentals, analyzer, research, compare,           │
                │   sector, movement, spotlight, history, health         │
                │  Lifespan builds: controller, FundamentalsService,     │
                │   HealthStore, AccuracyStore, RetryQueue, UserStore,   │
                │   WatchlistStore, SnapshotBroker                      │
                └─┬──────────────┬──────────────┬────────────────────────┘
                  │              │              │
       ┌──────────▼─┐   ┌────────▼────┐   ┌─────▼───────────────────┐
       │ Intelligence│   │ Strategy   │   │ Data / Provider chain  │
       │ analyzer,   │   │ signal_    │   │ FMP→Finnhub→AV→SEC→YH  │
       │ valuation,  │   │  aggregator│   │ Cache, Envelope,        │
       │ alerts,     │   │ Recommender│   │ HealthStore, AccuracyStr│
       │ research,   │   │            │   │ RetryQueue + Worker     │
       │ sentiment   │   │            │   │ Alpaca SDK wrapper      │
       └─────────────┘   └────────────┘   └────────────────────────┘
                                                  │
                  ┌───────────────────────────────┴───────────────┐
                  │ SQLite files (data/):                         │
                  │  esther.db (bars, news, sentiment)            │
                  │  users.db  (users, sessions, watchlists)      │
                  │  health.db (provider health + accuracy ledger)│
                  │ JSON-on-disk: retry_queue.json, session JSON  │
                  │ Filesystem cache: bars/, news/ envelopes      │
                  └───────────────────────────────────────────────┘
                                ▲                ▲
                                │                │
                        ┌───────┴────┐     ┌─────┴───────────┐
                        │ Anthropic  │     │ Alpaca (paper)  │
                        │ (LLM       │     │ + FMP, Finnhub, │
                        │  summary,  │     │ Alpha Vantage,  │
                        │  research, │     │ SEC EDGAR,      │
                        │  compare)  │     │ Yahoo HTML      │
                        └────────────┘     └─────────────────┘
```

## 2. Backend layers (`src/`)

### 2.1 `api/`

| Route                         | Auth?      | Source of truth                                |
| ----------------------------- | ---------- | ----------------------------------------------- |
| `/api/auth/{signup,login,logout,me}` | n/a    | UserStore (bcrypt), itsdangerous cookie |
| `/api/watchlist/*`            | **required** (`require_current_user`) | WatchlistStore |
| `/api/health`                 | public     | Cheap liveness                                  |
| `/api/snapshot`               | public     | One-shot `controller.fetch_snapshot()`          |
| `/api/stream`                 | **public** | SnapshotBroker (SSE, heartbeat ≤15 s)           |
| `/api/fundamentals/{symbol}`  | public     | `FundamentalsService` chain walk + cache        |
| `/api/analyzer/{symbol}`      | public     | technical + valuation + sentiment + optional LLM|
| `/api/research/{symbol}`      | public     | `LLMResearchGenerator` + `validate_thesis()`    |
| `/api/compare/{l}/{r}`        | public     | `comparison.compare()` deterministic verdicts   |
| `/api/compare/{l}/{r}/narrative` | public  | LLM compare narrator (no validator)             |
| `/api/sector`, `/api/movement`, `/api/spotlight` | public | composites |
| `/api/history/{symbol}`       | public     | Calibration store                                |
| `/api/health/providers`, `/api/health/queue` | public | HealthStore + RetryQueue summaries |

**Only watchlist mutations are authenticated.** Everything else — including
the live SSE snapshot stream — is fully public. The intent (per
`watchlist.py:26–28`) is that the frontend filters the shared snapshot per
user; this means snapshot isolation is a client-side concern, not a
server-side one.

### 2.2 `data/`

* **`orm.py` + `storage.py`** — SQLAlchemy on SQLite. Tables: `bars`,
  `news_articles`, `news_symbols`, `sentiment_scores`. Module-level
  `_engine` / `_SessionLocal` cached with lazy init. Race between
  thread-first init is theoretically possible but harmless under
  normal startup.
* **`cache.py`** — File-backed cache for OHLCV (CSV) and news (JSON).
  Dual freshness signal: mtime *and* in-band `as_of`. Merge logic
  prefers live data on overlaps. **No size cap; no eviction.**
* **`envelope.py` + `freshness.py`** — Generic `DataEnvelope[T]` with
  `as_of`, `fetched_at`, `source_chain`, `freshness` tier,
  `provider_confidence`. Policy table per domain
  (fundamentals/bars/news/analyst_targets). Strong design.
* **`health_store.py`** — SQLite (WAL, parameterized), `threading.Lock`
  serialized writes. Best-effort: a DB failure never kills a fetch.
  Holds per-call `ProviderHealth`.
* **`accuracy_store.py`** — Sister table to health. Records every
  reconciliation event (agreements and divergences) to power the
  Accuracy Ledger / `ProviderTrust`. **Ground truth is "last writer
  wins via reconciliation"; there is no external benchmark.** This
  weakens the moat claim — the ledger measures *self-consistency*, not
  truth.
* **`retry_queue.py` + `retry_worker.py`** — JSON-on-disk queue with
  exponential backoff (30 s → 30 min, cap 5 attempts). Atomic
  write-via-rename. Operations are O(n) per enqueue/drain.
  `PERMANENTLY_FAILED` rows never expire automatically.
* **`alpaca_client.py`** — Thin wrapper around `alpaca-py`. Subclients
  are `@cached_property`. Rate limiter hard-set to 200 req/min.
  Logs a warning (does *not* refuse) if the live API URL is set.
* **`streaming.py`** — Wraps Alpaca's WS streams in worker threads.
  Handler exceptions are logged but not propagated.
* **`news_ingestion.py:100`** — Non-default providers raise
  `NotImplementedError`. Only Alpaca is wired.
* **`user_store.py`, `watchlist_store.py`** — bcrypt (cost 12),
  `secrets.token_urlsafe(32)` session tokens, FK + cascade. Same lazy +
  lock pattern.

### 2.3 `intelligence/`

The "brain" of Esther. Six concerns live here, plus several adapters.

**Analyzer (`analyzer/`)**

* `technical.py` — Seven sub-scores (RSI overbought/oversold, Bollinger
  extension, MA distance, volume z, ATR ratio, momentum exhaustion).
  Every threshold (RSI 70/30, Bollinger 1 σ, MA 30 %, vol 3 σ, ATR 2×,
  momentum 5/20 pt) is a hardcoded magic number with no calibration
  source. NaN-safe.
* `valuation.py` — Seven-method ensemble (DCF, P/E, EV/EBITDA, P/S,
  PEG, historical band, analyst targets). DCF uses settings-default
  discount and growth rates — there is no WACC computation. Confidence
  formulas use unjustified constants (0.25, 5, 20, 0.3/0.6). Dispersion
  metric (`std(values) / max(abs(base), 1.0)`) can blow up when
  `base ≈ 0`.
* `sector_medians.py` + `config/sector_medians.toml` — Static TOML,
  with an inline comment that says "seed the analyzer until we backfill
  from real market data." The seed is in production. Unknown sectors
  fall back to PE = 20 / EVEB = 13 / PS = 2.5 / PEG = 2.0, which
  neutralizes the valuation signal entirely.
* `explanation.py` + `llm_explanation.py` — Grounded explanation
  builder. **This** is where the citation discipline is real: every
  claim must carry a `[technicals]`/`[fundamentals]`/`[news]` tag, and
  `validate_citations()` raises on missing tags.
* `prompts.py` — Prompt assembly.

**Other intelligence modules**

* `grounding.py` — A string of *guidelines* injected into the system
  prompt. **It is not a code-level validator.** "No forecasts" is a
  prose instruction, not a regex.
* `llm_summary.py` / `llm_research.py` / `comparison_narrative.py` —
  Three separate Anthropic call sites. All read API key via
  `SecretStr.get_secret_value()` (safe). **None pass `timeout=` or
  `temperature=`.** Two of three (research, compare) hardcode
  `max_tokens` instead of taking it from settings.
* `research_thesis.py` — Deterministic template fallback. Used when LLM
  is off or fails.
* `research_validator.py` — Tokenises numeric strings out of the
  thesis and substring-matches them against a corpus of facts pulled
  from the input row. Drops sentences that contain unsupported
  numbers. **Qualitative hallucinations** ("the company is gaining
  share") pass through.
* `alerts.py` / `alert_prioritizer.py` / `intraday_alerts.py` — Rules
  + cooldowns + per-tick cap. Several rules (action changed, tier
  changed) have *zero* default cooldown; per-tick cap is optional.
* `opportunities.py` / `opportunity_brief.py` / `opportunity_drilldown.py`
  / `opportunity_history.py` — Composite-ranked OPP detection.
  Weighted sum of seven drivers; weights hardcoded, no
  independence/correlation check.
* `pulse.py` / `pulse_history.py` / `pulse_evolution.py` — Watchlist
  pulse with regime labels. The "regime" here is a label on trend
  state, not a market-regime classifier.
* `tier.py` — Five-tier promotion (STRONG_BUY..STRONG_SELL). Quality
  and stability thresholds are all hand-set constants; the 4-episode
  veto is sensible, the 0.65 confidence floor is arbitrary.
* `timeframe_compare.py` — Real daily vs intraday alignment, not
  cosmetic.
* `trust_score.py` — Composite report grade. Weights are policy, not
  calibration. A+/A/B/C/D/F cliffs at 95/90/80/70/60.
* `comparison.py` + `comparison_narrative.py` — Two-symbol compare.
  Deterministic per-metric verdicts plus LLM narrator. The narrator
  has **no post-hoc validator** — it relies on the system prompt.
* `provider_trust.py` + `accuracy_store.py` — Per-provider trust
  weight in `[0.5, 1.2]`, computed from accuracy (65 %), uptime
  (25 %), latency (10 %). Cold start = 1.0.

**Fundamentals service (`intelligence/fundamentals/`)**

* `service.py` — Chain walker: FMP → Finnhub → Alpha Vantage → SEC
  EDGAR → Yahoo. Tracks per-call `ProviderHealth`, computes
  confidence by chain position, applies a multiplicative penalty per
  reconciliation divergence (`0.85^N`, floor 0.5).
* `base.py` + `http.py` — Error taxonomy:
  `ProviderRateLimited`/`Unavailable`/`Transient`/`NotFound`. httpx
  with explicit timeouts (good). **`Retry-After` header is ignored.**
* `reconciliation.py` — Compares revenue, net income, EPS diluted,
  total debt across providers. 5 % default divergence threshold,
  hardcoded. Sign-flip always fires. Strict fiscal-date match — no
  cross-quarter harmonization.
* Per-provider adapters (`fmp.py`, `finnhub.py`, `alphavantage.py`,
  `sec_edgar.py`, `yahoo_fallback.py`) — All use safe `.get()` access.
  Alpha Vantage's quota detection relies on substring matching a
  prose `Information` field.

### 2.4 `strategy/`

* `signal_aggregator.py` — Linear weighted vote. Default weight 1.0 if
  source not in dict. Threshold hardcoded at 0.2 (no config path).
  Confidence = `min(1.0, abs(net))`, which does *not* model
  disagreement.
* `RecommendationEngine` — Composes aggregator output with sentiment.

### 2.5 `persistence/`

* `session_store.py` — Atomic JSON snapshot of the dashboard session
  state. Schema-versioned (current v3) with backward-compat migrations.
  `signal_episodes` capped at 10, `pulse_records` 20, `alert_log` 200.
  **`brief_cache` and `opp_brief_cache` are uncapped dicts** — grow
  forever.

### 2.6 `dashboard/`, `main.py`, `sentiment/`, `indicators/`, `risk/`, `config/`

* `dashboard/app.py` is large (~2.8 k lines). It is the Textual TUI and
  remains a god-module; the controller and state layers around it are
  clean.
* `main.py` — CLI entrypoint (`esther` script). Lazy heavy imports
  inside command handlers. Refuses to start `recommend`/`dashboard`
  against the live Alpaca URL.
* `sentiment/analyzer.py` — FinBERT. Loaded once via
  `@cached_property` (good). **Inference is per-article, not batched.**
* `indicators/` — Deterministic RSI/MACD/Bollinger/ATR/VolumeZScore.
  Clean.
* `risk/` — Sharpe and max drawdown only.
* `config/settings.py` — Pydantic settings. Single source for env
  config. Several dangerous defaults documented in §4.

## 3. Frontend (`web/`)

* Pure prop-drilling + local hooks. No Redux / Context / Zustand.
* Routing is a hand-rolled `NavKey` enum (`App.tsx` → `RouteSwitch`).
  No URL state — refresh drops the active symbol.
* SSE client (`lib/stream.ts`) is the most polished module — exponential
  reconnect, stale-data detection (12 s), proper cleanup, `disposedRef`
  guard.
* Auth (`lib/auth.ts`) — fetches `/api/auth/me` on mount, swaps
  `LoginPage` for `AppShell`. Logout doesn't verify server success.
* All hooks (`research`, `compare`, `analyzer`, `watchlist`, …) use a
  `reqRef` race guard but no `AbortController` — old fetches complete
  and are discarded.
* **Templated "AI" surfaces**: `lib/aiSummary.ts` (used by `AiPage`,
  `Dashboard`) and `lib/sparkline.ts` are deterministic mocks marketed
  as AI. `PriceChart.tsx` generates `chartDataFor()` fixtures except in
  the TradingView tab.
* Provenance: `ProvenanceProse.tsx` re-tokenises text in the browser
  and looks up `ProvenanceMap` entries supplied by the server. Real,
  not cosmetic — but the regex is duplicated on both sides and must be
  kept in sync manually.

## 4. Configuration & deployment

* **Settings** (`config/settings.py`): Pydantic, env-loaded, sane field
  validators. Dangerous defaults:
  * `session_secret_key = "esther-dev-session-key-change-in-production"`
    — *not* enforced at startup; deployments that forget to set the env
    var silently use the placeholder.
  * `cors_origins = [http://localhost:5173, http://127.0.0.1:5173]` —
    fine for same-origin Railway; must be explicitly set for split
    deployments.
* **`api/app.py:406`**: `secure_cookies=False` is hardcoded. The
  comment literally says "exposed via Settings.app_env once we wire
  that." Until then, cookies ship without `Secure`.
* **Deployment artifacts**: `Procfile`, `railway.toml`, `nixpacks.toml`.
  Railway-only. No `Dockerfile`. The Procfile launches with `--mock`,
  which is correct for paper-trading safety but worth knowing.
* **Migrations**: none. Every schema is created from inline SQL on
  first connection. Adding a column needs a code change + restart.

## 5. Request lifecycle (example: `GET /api/research/AAPL`)

1. FastAPI route handler in `api/research.py`.
2. Looks for a hit in the research cache (file on disk, keyed by symbol +
   day).
3. Builds a `ResearchPayload` from the latest snapshot row +
   `FundamentalsService.fetch(AAPL)` + recent headlines.
4. Chooses the deterministic `ThesisGenerator` (template) or the
   `LLMResearchGenerator` based on settings + LLM availability.
5. `LLMResearchGenerator.generate(payload)` calls Anthropic. **No
   timeout, default temperature.** Response parsed as XML into a
   `ResearchThesis` object.
6. `validate_thesis(thesis, payload)` runs: builds a corpus from labeled
   fields (numbers from row + headlines), tokenises every claim,
   drops sentences whose numeric tokens are not in the corpus. Returns
   `(validated_thesis, validation_report)`.
7. Response is written to cache and returned to the client. Client
   renders prose with `ProvenanceProse` hover tooltips.

## 6. What the architecture is good at

* **Lazy init everywhere.** Stores construct without touching disk.
  Tests rely on this.
* **Freshness as a first-class field.** `DataEnvelope[T]` keeps
  "how old is this?" out of every consumer.
* **Best-effort observability.** Health/accuracy writes never break a
  fetch. Lock-serialised SQLite writes with WAL reads.
* **Atomic JSON persistence.** Session and retry queues use
  temp + rename.
* **Type discipline.** mypy `--strict` clean across ~80 files. Frontend
  is strict TypeScript.

## 7. What the architecture is bad at

* **Authorisation is bolt-on**, not pervasive. Only watchlist is
  protected.
* **Snapshot stream is global.** Per-user filtering is a frontend
  concern; the server hands the same blob to everyone.
* **Cost / latency surface is unmodelled.** No request budget, no
  per-provider rate-limit honoring, no Anthropic timeout.
* **Scoring is uncalibrated.** Every threshold and weight is a magic
  number with no walk-forward backtest.
* **"AI" UX surfaces (AiPage, AiSummary) are templated**, not LLM,
  despite the branding.
* **No migrations, no Docker, no CI artifacts** in the repo — Railway
  + Nixpacks is the only path.
