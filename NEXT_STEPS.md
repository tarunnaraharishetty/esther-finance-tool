# NEXT_STEPS

Pick-up notes for the next session. Read this before writing any code.

*Last touched: 2026-05-21 (Priority #1 data-quality foundation fully landed).*

---

## What Esther is

**An AI market intelligence platform** for discretionary traders. It
helps a discretionary trader understand what's happening across their
watchlist faster and with more grounded context than they could
assemble by hand. The decision is always theirs.

It is **not** an autonomous trading AI, a hedge-fund engine, or a
prediction machine. There is no order submission anywhere in the
codebase. The data layer points at `paper-api.alpaca.markets`. Every
LLM-driven feature ships with anti-hallucination grounding rules that
forbid forecasts and require headlines to be quoted verbatim.
Anything that nudges back toward those framings is wrong.

---

## Current architecture status

Stable. Each layer has a coherent responsibility and a stable contract
to the layer above:

```
src/
├── main.py             # CLI: status / doctor / initdb / backfill /
│                       # recommend / summarize / recap / dashboard / serve
├── api/                # FastAPI + SSE + JSON routes
│   ├── app.py          # create_app, lifespan, SSE
│   ├── analyzer.py     # /api/analyzer/{symbol}
│   ├── broker.py       # SnapshotBroker for SSE
│   ├── fundamentals.py # /api/fundamentals/{symbol} + /health + status
│   ├── health.py       # /api/health/providers + /api/health/queue
│   └── research.py     # /api/research/{symbol}
├── config/             # Pydantic Settings (single env reader)
├── data/               # Alpaca client, market data, news (deduped),
│                       # cache (mtime-TTL + envelope variants),
│                       # envelope.py + freshness.py (DataEnvelope / policies),
│                       # health_store.py (SQLite SLO log),
│                       # retry_queue.py + retry_worker.py
├── dashboard/          # Textual app + controller + snapshot state
├── indicators/         # RSI, MACD, Bollinger, ATR, VolumeZScore
├── intelligence/       # explain · summary · llm_summary · alerts ·
│                       # alert_prioritizer · watchlist · history · tier ·
│                       # rankings · recap · grounding · pulse · pulse_history ·
│                       # pulse_evolution · opportunities · signal_profile ·
│                       # opportunity_history · opportunity_brief ·
│                       # opportunity_drilldown · llm_research ·
│                       # research_thesis · timeframe_compare · intraday_alerts ·
│                       # analyzer/ (technical + valuation + explanation +
│                       # sector_medians + LLM explanation) ·
│                       # fundamentals/ (FMP / Finnhub / AlphaVantage /
│                       # SEC EDGAR / Yahoo + base / http / models /
│                       # reconciliation / service)
├── persistence/        # SessionStore (JSON snapshot for cross-restart state)
├── risk/               # Sharpe + max-drawdown
├── sentiment/          # FinBERT analyzer + news_quality
├── strategy/           # signal_aggregator + RecommendationEngine + multi_timeframe
└── utils/              # logging, rate limiter, preflight
web/                    # React + TypeScript + Vite frontend
├── src/lib/            # API clients, types, helpers
└── src/pages/          # Dashboard / Analyzer / AI / Charts / Movers /
                        # News / Research / Watchlist
```

**Layering invariants:**
- `intelligence/` depends on `strategy/` + `data/`; never imports from
  `dashboard/`, `main`, or `api/`.
- `data/` has no `intelligence/` imports (cycle-free; cross-cutting
  models like `ProviderHealth` are imported by `data/health_store.py`
  from `intelligence/fundamentals/models.py`, with the inverse via
  `TYPE_CHECKING` to avoid a cycle).
- The `Summarizer` / `ThesisGenerator` protocols let the dashboard /
  CLI consume either deterministic or LLM-backed implementations
  without caring which.

---

## Completed systems

### Data + sentiment
- Alpaca paper-feed client with rate limiting + retry.
- Bars + news ingestion with **freshness-aware filesystem cache**:
  - `read_bars` / `read_news` with mtime-based TTL (for warmup paths).
  - `read_bars_envelope` / `read_news_envelope` with in-band as_of and
    `DataEnvelope` wrapping (for analyzer/UI consumers).
  - `merge_with_cache` refuses to merge an expired cache prefix.
- News dedup at the source — defensive against SDK pagination quirks.
- FinBERT sentiment scoring per article with quality weighting
  (recency × source reputation).

### Fundamentals data quality (Priority #1 — fully shipped)

- **Provider chain** with typed error taxonomy
  (`ProviderUnavailable`/`RateLimited`/`NotFound`/`Transient`).
- **DataEnvelope[T]** generic wrapper (P1.1):
  - `as_of`, `fetched_at`, `source_chain`, `freshness`,
    `provider_confidence`. Lives at `src/data/envelope.py`.
  - `FreshnessPolicy` table for fundamentals.quarterly/.annual,
    bars.intraday/.daily, news, analyst_targets at `src/data/freshness.py`.
- **Persistent provider health** (P1.2):
  - `HealthStore` (SQLite, WAL, lazy-init) at `src/data/health_store.py`.
  - Best-effort sink writes on every `FundamentalsService.fetch()`
    (success or chain-exhausted).
  - `GET /api/health/providers?window=<int>(s|m|h|d)` rolling SLO
    summary + 20 most recent failure rows.
- **Cache TTL** on bars/news (P1.3) — see above.
- **Cross-provider reconciliation** (P1.4):
  - When primary chain position > 0, second provider is called for the
    same symbol and high-trust fields compared
    (revenue / net_income / eps_diluted / total_debt).
  - Sign-flip rule fires regardless of threshold; fiscal-date mismatch
    surfaces as a warning, not a divergence.
  - Divergences penalize `provider_confidence` multiplicatively
    (`0.85^N`, floored at `0.5x`).
- **Durable retry queue** (P1.5):
  - JSON-on-disk queue at `src/data/retry_queue.py` (atomic writes,
    corrupt-resilient, schema-versioned).
  - Service auto-enqueues on *transient-only* chain exhaustion; mixed
    failures don't queue.
  - Opt-in async `RetryWorker` (off by default; set
    `retry_queue_worker_enabled=true` to enable).
  - `GET /api/health/queue` returns pending + permanently-failed counts
    and full entry list.

### Analyzer + valuation

- Normalized technical scoring (`src/intelligence/analyzer/technical.py`):
  7 sub-scores in [0,100], combiners for overbought / oversold /
  pullback risk / rebound potential, coverage × agreement confidence.
- Multi-method valuation ensemble (`src/intelligence/analyzer/valuation.py`):
  DCF, P/E, EV/EBITDA, P/S, PEG, historical band, analyst targets.
  Per-method confidence; bear/base/bull at 25/50/75 percentile; weighted
  AI fair value; overall confidence in [0,100].
- Grounded explanation builder with citation validator
  (`src/intelligence/analyzer/explanation.py`).
- `/api/analyzer/{symbol}` ties it all together, including a
  `fundamentals_freshness` block with divergence + warning counts.

### Strategy + dashboard + AI summaries

- RSI / MACD / Bollinger / ATR / Volume indicators.
- 5-tier promoter (`STRONG_BUY`..`STRONG_SELL`) with signal-quality
  grade and stability tier.
- Per-symbol Claude brief (`s` key in dashboard, `esther summarize` CLI).
- Watchlist-wide Claude recap (`esther recap` CLI).
- Per-OPP Claude brief (`b` key in dashboard).
- Research thesis surface (`/api/research/{symbol}`): structured
  bull/bear/catalysts/valuation/technical sections; template or LLM mode.
- Anti-hallucination grounding rules in `src/intelligence/grounding.py`
  (regression-asserted verbatim in prompt construction).
- Six ranked sections (momentum / sentiment / confidence / reversals /
  unusual / volatile), opportunity detection + history, alerts +
  prioritizer with cooldowns.

### Multi-timeframe + pulse

- Per-row intraday read with view toggle (`t`) flipping ACTION/CONF/
  BAR/TECH cells between daily and intraday.
- `TimeframeStance` (aligned_bullish / aligned_bearish / conflict /
  intraday_only / daily_only / neutral) with trader-facing labels.
- Market pulse: sentiment / conviction / activity + breadth + intensity
  + STRONG-symbol callouts; rolling-window history with sparklines;
  regime + trajectory patterns.

### Quality baseline

- **988 passing tests**, 1 deselected (`slow`/`integration`).
- `ruff check .` green across the repo.
- `mypy src --strict` green across **80 source files**.
- Documented exceptions (single ASYNC109 noqa on the shared HTTP
  helper) live at the use site.

---

## Remaining roadmap

The data-quality foundation is complete. Remaining work is sequenced
priority-by-priority.

### Priority #2 — Probabilistic analyzer (not started)

Goal: estimate ranges and probabilities, not predictions.

- **`ScenarioModel`** layered on top of `ValuationEnsemble`: closed-form
  lognormal price distribution from realized vol → tail probabilities
  for "price below bear case in N days" and "price above bull case in
  N days". Deterministic, no Monte Carlo.
- **`SignalHistoryCalibrator`** offline build from `history.py`: maps
  `(score_bucket, action) → (n, hit_rate)` so pullback_risk = 73 maps
  to "64% of similar setups closed lower over 5d (412 observations)".
- UI surfacing: replace 25/50/75 percentile framing with calibrated
  probability + observation count. Refuse to publish a probability
  with < N observations.
- Calibration-drift check; auto-pause publishing when drift exceeds
  threshold.

### Priority #3 — Grounded research validator (not started)

Goal: institutional-grade AI research with per-claim source attribution.

- Move `LLMThesisGenerator` to tool_use structured output with a
  `ResearchRequest` → `ThesisDraft` schema where each claim carries
  `source_category` and `source_excerpt`.
- `ClaimValidator` drops any claim whose excerpt doesn't substring-
  match an input fact. Drops are counted and surfaced on the response.
- UI shows "AI declined to make 3 unsupported claims" as a trust
  signal, not as a defect.

### Priority #4 — Trader workflow features (partial)

What exists today: watchlists, opportunity scanner, alerts, "why is
this moving?" via `explain.py`, signal history storage, sector-median
lookups. What's missing:

- **Historical signal outcomes view** (depends on P2 calibration table).
- **Compare two stocks** side-by-side: analyzer / valuation / technical
  / sentiment. Cheap to build (frontend composition).
- **Sector-relative ranking** ("AAPL is #3/24 in Tech on overbought_score").
- **Causal "why is this moving?"** — extend `explain.py` to rank the
  contributing factors and surface the top 1–2.

Deferred deliberately: **backtesting-lite**. Correct backtesting is a
6-month project; a shallow version actively erodes trust.

### Priority #5 — Product quality (in flight)

Loading states and freshness badges on the web frontend are the
highest-leverage next moves now that the freshness envelope is wired
end-to-end. Then accessibility, mobile readability, performance.

---

## Recommended next implementation order

1. **Priority #2.1 — `ScenarioModel`** (closed-form tail probabilities).
   ~1 day. Depends on nothing; pure addition on top of `ValuationEnsemble`.
2. **Priority #2.2 — `SignalHistoryCalibrator`** offline build.
   ~1.5 days. Reads `history.py` output; writes calibration table to
   SessionStore.
3. **Priority #2.3 — UI surfacing**: probability framing + observation
   counts on analyzer cards. ~0.5 day.
4. **Priority #5 freshness UI** (in parallel with P2 if desired): badge
   component consuming `fundamentals_freshness` + recent failures from
   `/api/health/providers`. ~0.5 day.
5. **Priority #3.1 — `ResearchRequest`/`ThesisDraft` schema** + tool_use
   migration. ~2 days.
6. **Priority #3.2 — `ClaimValidator`** + drop counters. ~1 day.
7. **Priority #4 features** ordered by leverage: historical outcomes →
   compare two stocks → sector-relative → causal "why is it moving?".

---

## Resume here next session

```bash
git pull
.venv/Scripts/python.exe -m pytest --no-cov -q     # confirm 988 passing
.venv/Scripts/python.exe -m mypy src               # confirm strict-clean
.venv/Scripts/python.exe -m ruff check .           # confirm green
cd web && npm run typecheck                        # confirm tsc green
```

If the build is clean, pick up at Priority #2.1 (`ScenarioModel`). The
design doc is the natural place to start — write the seven sections
(goal, non-goals, data model, files, risks, tests, acceptance) before
touching code, then implement incrementally. See the
"feedback_workflow" memory for the cadence.

---

## Repo state at handoff (2026-05-21)

- **Branch:** `main`.
- **Tests:** 988 passing, 1 deselected (`slow`/`integration`).
  Run with `pytest`.
- **Lint/type:** `ruff check .` green; `mypy src --strict` green across
  all 80 source files; `npm run typecheck` green.
- **Dependencies installed in `.venv/`** (Python 3.14).
- **`.env` is not present.** `.env.example` documents the schema;
  `esther doctor --init-env` scaffolds the file. Real keys live only on
  disk, gitignored.
