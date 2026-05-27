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

## Deploying

Esther ships as a single bundled image (Dockerfile is multi-stage:
Node builds `web/dist/`, Python runtime serves it alongside the API).

```bash
docker compose build
docker compose up               # local prod simulation, port 8000
```

Before pushing to a real host (Render / Railway / Fly / Cloud Run),
fill `.env` with at minimum:

| Variable | Required when | Notes |
|---|---|---|
| `APP_ENV` | always | Set to `prod`. Triggers the safety validator. |
| `SESSION_SECRET_KEY` | `APP_ENV=prod` | ≥32 random bytes. Validator refuses the dev placeholder. |
| `SECURE_COOKIES` | `APP_ENV=prod` | Must be `true`. Validator refuses `false` in prod. |
| `TRUSTED_HOSTS` | `APP_ENV=prod` | Comma-list of hostnames; validator refuses `*` in prod. |
| `CORS_ORIGINS` | If frontend is on a different origin | Validator refuses `*` in prod (breaks credentialed CORS). |
| `TRUST_PROXY_HEADERS` | Behind a reverse proxy | `true` if the host terminates TLS for you. |
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | always | Paper-trading keys. |
| `ANTHROPIC_API_KEY` | For LLM surfaces | Unset falls back to deterministic templates. |

The model validator in `src/config/settings.py` rejects unsafe
defaults at startup — a misconfigured prod deploy fails fast with a
human-readable error instead of silently serving with weak signing
or wildcard hosts.

For persistent storage under Docker, bind-mount the host's `./data`
to `/app/data` (the compose file already does this). SQLite databases
for users, sessions, watchlists, provider health, accuracy ledger,
and retry queue all live there.

### Deploying to Render

`render.yaml` is a Blueprint — Render reads it directly so the
manual steps shrink to:

1. **Connect the repo.** Render dashboard → **New** → **Blueprint** →
   connect this GitHub repo. Render parses `render.yaml` and
   proposes one web service + a 1GB persistent disk.
2. **Fill in the secret env vars.** Render prompts for every variable
   declared `sync: false`:
   - `ALPACA_API_KEY` / `ALPACA_API_SECRET` (required)
   - `ANTHROPIC_API_KEY` (optional; unset = template fallback)
   - `FMP_API_KEY` / `FINNHUB_API_KEY` / `ALPHAVANTAGE_API_KEY` (all optional)
   - `SEC_EDGAR_USER_AGENT` (optional; format: `"App Name contact@you.com"`)
3. **Deploy.** Render builds the multi-stage Dockerfile, mounts the
   disk at `/app/data`, and starts the service. The model validator
   refuses to boot if any prod-safety setting is wrong — `SESSION_SECRET_KEY`,
   `SECURE_COOKIES`, `TRUSTED_HOSTS`, `CORS_ORIGINS` are all wired
   correctly in `render.yaml` so the first deploy passes.
4. **Open the URL.** Render assigns `<service-name>.onrender.com`.
   Sign up an account; the dashboard streams over SSE same as local.

Notes: `TRUSTED_HOSTS` auto-resolves to the assigned Render hostname
via `fromService.property: host` — if you later attach a custom
domain, edit `TRUSTED_HOSTS` in the dashboard to include both,
comma-separated. Render terminates TLS at its load balancer, so
`TRUST_PROXY_HEADERS=true` is set so `request.url.scheme` reflects
HTTPS correctly.

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
