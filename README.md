# Esther

**AI market intelligence platform for discretionary traders.**
Esther is decision-support software — it helps humans research and
monitor symbols. It does **not** trade on your behalf and has no
order-submission code path.

The product combines:

- a multi-provider fundamentals pipeline (FMP → Finnhub → Alpha Vantage
  → SEC EDGAR → Yahoo) with freshness envelopes, cross-provider
  reconciliation, and a persistent provider health + accuracy ledger;
- a grounded analyzer (seven technical sub-scores, a seven-method
  valuation ensemble, AI-written explanations with per-claim citation
  validation);
- a research thesis surface (LLM-backed with grounded validation, or a
  deterministic template fallback);
- a Textual TUI and a React + Vite web frontend, served by FastAPI and
  sharing the same SSE-streamed snapshot pipeline.

Esther is currently a **single-tenant prototype with a usable MVP
shape**. Several surfaces are still templated rather than LLM-driven
(clearly labelled in the UI). Several scoring thresholds are hand-set
constants without backtest provenance. See `ARCHITECTURE.md`,
`BUGS.md`, `TODO.md`, and `ROADMAP.md` for the honest state of the
codebase.

## What works today

- Web app: Dashboard, Analyzer, AI, Charts, Movers, News, Research,
  Watchlist, Compare, Providers pages. SSE-streamed snapshots, optimistic
  watchlist CRUD, cookie-based auth, per-user watchlists.
- Textual TUI: per-symbol grid, alerts, pulse, opportunity feed,
  research, recap.
- Backend: FastAPI app with multi-provider fundamentals chain,
  reconciliation, retry queue, health observability, accuracy ledger,
  trust score, calibration store.
- Grounded research thesis: every numeric token in an LLM-written
  thesis is validated against an input corpus; unsupported sentences
  are dropped.

## What does *not* work today

- **No order submission, anywhere.** Out of scope by design.
- **No backtesting / strategy harness.** The roadmap calls for one in
  Phase 4.
- Dashboard "AI Briefing" and per-symbol "Auto Reads" are *templated
  prose* generated client-side from snapshot fields. They are clearly
  labelled in-product as heuristic until the real LLM pipeline is
  wired through to those surfaces.
- Sparklines and the inline price chart (`PriceChart`) render
  synthetic data outside the dedicated TradingView tab. They are
  labelled "synthetic preview."
- Sector medians used by the multiples valuation are static seed data
  from `config/sector_medians.toml`. The valuation methods that depend
  on them carry that limitation forward.

## Tech stack

- **Backend** — Python 3.12+, FastAPI, uvicorn, pydantic + pydantic-settings,
  SQLAlchemy + SQLite, structlog, tenacity, anthropic, httpx,
  pandas/numpy, transformers + torch (FinBERT), rich/textual.
- **Frontend** — React + TypeScript + Vite, Tailwind, Recharts, cmdk.
- **Tests** — pytest + pytest-asyncio (backend), vitest (frontend).
- **Quality** — ruff (lint + format), mypy `--strict`, strict TS.

## Requirements

- Python **3.12+**
- Node 18+ for the web frontend.
- An Alpaca account (paper data only — live URL is refused at startup).
- API keys for the fundamentals providers you want to enable
  (FMP / Finnhub / Alpha Vantage / SEC EDGAR User-Agent). All are
  optional; the chain skips unconfigured providers.
- Optional: Anthropic API key for LLM research / analyzer explanations
  / compare narrative. Without it, deterministic templates are used.

## Setup

```bash
git clone <repo-url>
cd "esther finance tool"

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"
cp .env.example .env            # then edit .env with your keys

cd web && npm install && cd ..
```

## Run

```bash
# Web app + API on http://127.0.0.1:8000
python src/main.py serve

# Textual TUI
python src/main.py dashboard

# Vite dev server (proxies /api to :8000)
cd web && npm run dev
```

## Development

```bash
pytest                          # unit tests (excludes @integration)
pytest -m integration           # against real Alpaca paper API
ruff check . && ruff format .   # lint + format
mypy src                        # strict type-check
cd web && npm run typecheck && npm test
```

## Layout

```
src/
  api/           FastAPI app + routes + SSE broker
  config/        Pydantic settings & env loading
  data/          Alpaca client, market/news ingestion, storage, streaming,
                 cache, envelopes, freshness, retry queue, health/accuracy
                 stores, user + watchlist stores
  dashboard/     Textual TUI app, controller, state
  indicators/    RSI, MACD, Bollinger, ATR, VolumeZScore
  intelligence/  Analyzer (technical + valuation), alerts, opportunities,
                 sentiment-aware brief, LLM summary/research/compare,
                 grounding + validators, trust score, calibration
  persistence/   JSON session-snapshot store
  risk/          Sharpe, max drawdown
  sentiment/     FinBERT analyzer, news quality scoring
  strategy/      Signal aggregator + recommendation engine
  utils/         logging, rate limiter, preflight
web/             React + Vite frontend
config/          YAML/TOML configs (sector_medians.toml)
data/            SQLite DBs, OHLCV/news cache, retry_queue.json (gitignored)
logs/            Rotating log files (gitignored)
tests/           pytest suite mirroring src/
```

See `CLAUDE.md` for engineering conventions and safety rules and
`ARCHITECTURE.md` for a current architecture map.

## Safety

- **Paper data only.** `Settings.alpaca_base_url` defaults to
  `paper-api.alpaca.markets`; `AlpacaClient` refuses to instantiate
  against the live URL.
- **No order submission anywhere.** This is enforced by the absence of
  a broker / execution module, not by a runtime check.
- **All LLM outputs grounded.** Every Anthropic call routes through
  `src/intelligence/grounding.py` for the system prompt AND through a
  post-hoc dropper that removes sentences whose numeric tokens are not
  present in the input corpus. Research thesis + compare narrative use
  full-corpus validators with provenance; summary, recap, and OPP brief
  use the lightweight shared `text_grounding.validate_prose`. Drop
  counts are logged and surfaced on the research + compare wire so the
  UI can render "AI declined N unsupported claims" as a trust signal.
- **No network in unit tests.** Provider/Alpaca calls are mocked at the
  SDK boundary; real-API tests are marked `@pytest.mark.integration`
  and skipped by default.

## License

Proprietary — all rights reserved.
