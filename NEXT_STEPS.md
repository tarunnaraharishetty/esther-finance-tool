# NEXT_STEPS

End-of-day snapshot for the Esther project. Read this first when picking the work back up.

---

## 1. Simplified project vision

The original `PROJECT_CONTEXT.md.txt` was very ambitious (full backtesting, Monte Carlo VaR, multi-broker abstraction, Docker, Postgres, the works). Scope is now narrowed to a focused MVP:

> **An AI-powered terminal trading assistant that watches a market watchlist, fuses technical signals with FinBERT news sentiment, and surfaces clear BUY / HOLD / SELL recommendations live in a Textual dashboard.**

Concretely, the MVP loop is:

1. **Market tracking** — pull live OHLCV bars for a configurable watchlist (Alpaca paper account).
2. **Sentiment analysis** — pull recent news for those symbols, score with FinBERT.
3. **Recommendation** — combine technical indicators (RSI/MACD/Bollinger) with sentiment via the existing `SignalAggregator` to produce one BUY / HOLD / SELL per symbol with a confidence score and short rationale.
4. **Terminal dashboard** — Textual TUI showing the watchlist, recommendations, last-updated timestamps, and most recent headlines per symbol.

Order execution, full backtesting, deep risk management, and Monte Carlo VaR are **deferred** — they exist as scaffolding but are not on the critical path to shipping the assistant.

---

## 2. Current completed features

### Foundation (all green: 58 tests passing, 1 slow integration test deselected by default)

- **Project scaffolding**: pyproject.toml (3.12+, ruff/mypy strict), virtualenv, requirements.txt, `.env.example`, `.gitignore`, CLAUDE.md (engineering + trading-safety rules).
- **Config (`src/config/`)**: Pydantic `Settings` with `SecretStr` for Alpaca keys, paper-trading guard (`is_paper_trading`), enum-typed env loading, cached singleton via `get_settings()`.
- **Logging + utilities (`src/utils/`)**: structlog-based structured logging, token-bucket rate limiter, sync + async tenacity retry decorators.

### Data layer (real, not stubbed)

- **`AlpacaClient`** (`src/data/alpaca_client.py`): lazy facade over alpaca-py's `TradingClient`, `StockHistoricalDataClient`, `NewsClient`, `StockDataStream`, `NewsDataStream`. Single auth point, single rate-limit bucket.
- **`MarketDataService`** (`src/data/market_data.py`): async `get_bars()` / `get_latest_bar()`, sync SDK wrapped in `asyncio.to_thread`, retry-decorated, `TimeFrame` enum translation, Decimal precision.
- **`AlpacaNewsSource`** (`src/data/news_ingestion.py`): provider-pluggable; Alpaca implementation maps to our `NewsArticle` model.
- **`MarketStream`** (`src/data/streaming.py`): async wrapper over both stock and news WebSocket streams. Multi-handler fan-out per event type, domain-model conversion, error isolation (a raising handler doesn't kill the loop), `start()` / `stop()` / `run_forever()`.

### Sentiment

- **`SentimentAnalyzer`** (`src/sentiment/analyzer.py`): FinBERT pipeline lazy-loaded on first use. `score_text()` and `score_article()` return `SentimentScore` with a `signed` confidence value (positive ⇒ +conf, negative ⇒ –conf, neutral ⇒ 0). Verified end-to-end against a real headline.

### Indicators + strategy

- **Indicators** (`src/indicators/`): working `RSI`, `MACD`, `BollingerBands` with `Indicator` ABC.
- **`SignalAggregator`** (`src/strategy/signal_aggregator.py`): weighted-vote combiner over per-source signals → single BUY / HOLD / SELL per symbol with confidence + threshold.
- **`Signal` / `SignalAction`** dataclasses (`src/strategy/base.py`).

### Risk + execution (scaffolded; mostly used for safety gates)

- **`RiskGate`** (`src/risk/exposure.py`): pre-trade checks (positive equity, positive price, max position notional).
- **`AlpacaBroker`** (`src/execution/broker.py`): real `submit()` (with `OrderClass.BRACKET` + `TakeProfitRequest`/`StopLossRequest`), `cancel()`, `get_account_equity()`.
- **`OrderManager`** (`src/execution/order_manager.py`): `Signal` → risk-gated bracket order, `FixedFractionSizer` for position sizing.
- **Risk metrics** (`src/risk/metrics.py`, `var.py`): Sharpe, Sortino, max-drawdown; historical / parametric / Monte Carlo VaR.

### Storage

- **ORM models** (`src/data/orm.py`): `BarORM`, `NewsArticleORM`, `NewsSymbolORM`, `SentimentScoreORM` (composite PK on bars; many-to-many news/symbol; sentiment keyed by `(news_id, model)` so re-scoring with the same model overwrites and a new model coexists).
- **Repositories** (`src/data/repositories.py`): `BarRepository`, `NewsRepository`, `SentimentRepository`. SQLite + Postgres `ON CONFLICT` upsert, merge fallback elsewhere. UTC re-attached on read (SQLite is tz-naive).
- **Engine + session** (`src/data/storage.py`): `get_engine()`, `session_scope()`, `init_db()`, `reset_engine()` (for tests). SQLite FK pragma enabled per connection so `ON DELETE CASCADE` fires.

### CLI

- **`src/main.py`** (Click + Rich): `esther status`, `esther initdb`, `esther run` (stub). Path bootstrap so `python src/main.py` works alongside `python -m src.main`. ASCII output (works in Windows cp1252 console).

### Backtesting

- **`BacktestEngine` + `compare_to_spy`** (`src/backtesting/`): scaffolded with `NotImplementedError` — Backtrader is installed but the cerebro wiring is intentionally deferred.

---

## 3. Current architecture

```
                       ┌───────────────────┐
                       │       CLI         │  src/main.py (Click + Rich)
                       └─────────┬─────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ▼                ▼                ▼
       ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
       │   config     │ │    utils     │ │   storage      │
       │   Settings   │ │ logging,     │ │ engine,        │
       │   (Pydantic) │ │ rate_limiter │ │ session_scope  │
       └──────────────┘ └──────────────┘ └────────┬───────┘
                                                  │
                                                  ▼
                                        ┌─────────────────┐
                                        │  ORM + repos    │
                                        │  Bar / News /   │
                                        │  Sentiment      │
                                        └─────────────────┘

  ┌────────────────── data layer (alpaca-backed) ──────────────────┐
  │                                                                │
  │  AlpacaClient ── MarketDataService    AlpacaNewsSource         │
  │       │                │                       │               │
  │       │                ▼                       ▼               │
  │       │            historical bars         historical news     │
  │       │                                                        │
  │       └─── MarketStream ── live bars / quotes / trades / news  │
  └───────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────────┐
       │                alpha pipeline                    │
       │                                                  │
       │  Indicators (RSI, MACD, Bollinger) ─┐            │
       │  Sentiment (FinBERT)  ──────────────┤            │
       │                                     ▼            │
       │                          SignalAggregator        │
       │                                     │            │
       │                                     ▼            │
       │                            BUY / HOLD / SELL     │
       └─────────────────────────────────────┬────────────┘
                                             │
                                             ▼
       ┌──────────────────────────────────────────────────┐
       │  execution  (deferred for MVP — wired but unused)│
       │  RiskGate ── OrderManager ── AlpacaBroker        │
       └──────────────────────────────────────────────────┘
```

**Layering rules** (enforced by code structure + CLAUDE.md):

- `config` and `utils` have no other intra-project deps.
- `data` depends only on `config`, `utils`.
- `sentiment`, `indicators` depend only on `data`, `config`, `utils`.
- `strategy` depends on `data`, `sentiment`, `indicators`.
- `execution` depends on `strategy`, `risk`, `data`.
- Nothing imports from `main` or test code.

**Async boundary**: alpaca-py's SDK is sync; we wrap blocking calls in `asyncio.to_thread`. WebSocket streams run in a worker thread (alpaca-py's `run()` spins up its own asyncio loop).

---

## 4. What still needs to be built

### Critical path to MVP (the assistant)

1. **`RecommendationService`** (`src/strategy/recommendation.py` — does not yet exist).
   For each symbol in a watchlist: pull recent bars + news, compute indicator signals, score sentiment, feed everything into `SignalAggregator`, return a `Recommendation` (symbol, action, confidence, rationale, last-updated, contributing signals).
2. **Textual dashboard** (`src/ui/dashboard.py` — does not yet exist).
   Watchlist table with columns: symbol, last price, change %, recommendation (color-coded), confidence, latest headline, sentiment.
3. **`esther watch` CLI command** that boots the dashboard with a configurable refresh cadence and watchlist (read from `config/strategies.yaml`).
4. **Backfill command** (`esther backfill --symbols AAPL,MSFT --days 30`) to populate `bars` + `news_articles` so indicators have history on first run.

### Useful but not on the MVP critical path

- WebSocket → recommendation pipeline so the dashboard updates on tick instead of polling.
- News deduplication (Alpaca occasionally returns the same article id twice across pages).
- Persistent watchlist (currently env/YAML; could be a `watchlist` table).
- A `daily summary` command: top movers + sentiment shift over last N days.

### Deferred from the original `PROJECT_CONTEXT.md`

- Backtesting engine (Backtrader cerebro wiring) — needed only when we want to validate strategies historically.
- Live order execution — `AlpacaBroker` + `OrderManager` are wired but have no caller in the MVP path. Keep them out until the assistant is solid.
- Postgres migration — SQLite is fine for the local-terminal use case.
- Monte Carlo VaR / Sharpe portfolio analytics — interesting but premature.
- Multi-strategy framework — one good strategy first.

---

## 5. Exact next steps for tomorrow

Do these in order. Each step has a clear definition-of-done.

### Step 1 — `RecommendationService` (~1–2 hours)

Create `src/strategy/recommendation.py`:

```python
@dataclass(frozen=True)
class Recommendation:
    symbol: str
    action: SignalAction       # BUY / HOLD / SELL
    confidence: float          # 0..1
    last_price: Decimal
    rationale: str             # e.g. "RSI=72 (overbought), sentiment +0.7 over 12 articles"
    components: list[Signal]   # per-source signals that fed the aggregator
    generated_at: datetime
```

Service responsibilities:
- Inputs: `symbols: list[str]`, optional `lookback_days` (default 30), optional `news_lookback_hours` (default 24).
- For each symbol:
  - Fetch bars (`MarketDataService.get_bars`) + cache via `BarRepository.upsert`.
  - Fetch news (`AlpacaNewsSource.fetch`) + cache via `NewsRepository.upsert`.
  - Compute RSI, MACD, Bollinger → produce `Signal`s. (Encode the rules: RSI>70 → SELL, RSI<30 → BUY, MACD cross, BB break, etc.)
  - Score each new article with `SentimentAnalyzer`, persist via `SentimentRepository.upsert`. Average signed sentiment → one `Signal`.
  - Run all signals through `SignalAggregator`.
- Tests: mock `MarketDataService` + `AlpacaNewsSource` + `SentimentAnalyzer`; assert correct aggregation under known inputs (RSI overbought + negative sentiment ⇒ SELL with high confidence; mixed inputs ⇒ HOLD; etc.).

### Step 2 — `EstherDashboard` Textual app (~2–3 hours)

Create `src/ui/__init__.py` and `src/ui/dashboard.py`:

- Use `textual.app.App`, `textual.widgets.DataTable`, `Header`, `Footer`.
- Columns: Symbol, Price, Δ%, Action (color: green/yellow/red), Confidence, Latest headline (truncated), Updated.
- A `RefreshTimer` that calls `RecommendationService.recommend(watchlist)` every N seconds (default 60s; configurable in `config/strategies.yaml`).
- Keybindings: `r` to refresh now, `q` to quit.

### Step 3 — Wire `esther watch` (~30 min)

Add to `src/main.py`:

```python
@cli.command()
@click.option("--symbols", default=None, help="Comma-separated; overrides config.")
@click.option("--refresh", default=60, type=int, help="Refresh seconds.")
def watch(symbols: str | None, refresh: int) -> None:
    """Launch the live recommendation dashboard."""
    from src.ui.dashboard import EstherDashboard
    EstherDashboard(symbols=_resolve_watchlist(symbols), refresh_seconds=refresh).run()
```

Read default watchlist from `config/strategies.yaml` (the file already exists as `strategies.example.yaml`).

### Step 4 — `esther backfill` (~30 min)

```python
@cli.command()
@click.option("--symbols", required=True)
@click.option("--days", default=30, type=int)
def backfill(symbols: str, days: int) -> None:
    """Fetch historical bars + news for symbols and store locally."""
    # MarketDataService.get_bars + BarRepository.upsert
    # AlpacaNewsSource.fetch + NewsRepository.upsert
```

Run once per session before `esther watch` so the indicators have history immediately.

### Step 5 — End-to-end smoke test against paper Alpaca (~30 min)

1. Copy `.env.example` → `.env`, fill in real Alpaca paper API keys.
2. `python src/main.py initdb`
3. `python src/main.py backfill --symbols AAPL,MSFT,NVDA,SPY --days 30`
4. `python src/main.py watch --symbols AAPL,MSFT,NVDA,SPY`
5. Verify the table renders, recommendations update on the timer, no API errors.

### Files you'll touch tomorrow

- **Create**: `src/strategy/recommendation.py`, `src/ui/__init__.py`, `src/ui/dashboard.py`, `tests/test_recommendation.py`, `tests/test_dashboard.py` (if Textual snapshot testing is reasonable).
- **Modify**: `src/main.py` (`watch` and `backfill` subcommands), `src/strategy/__init__.py` (export `Recommendation` + `RecommendationService`).
- **Read first**: `config/strategies.example.yaml` (you'll likely need to extend it with a `watchlist:` section).

### Definition of done for the day

`python src/main.py watch --symbols AAPL,MSFT,NVDA,SPY` opens a terminal dashboard, shows live prices, and emits sensible BUY/HOLD/SELL recommendations updated every minute, without hitting the network more than necessary.

---

## Repo state at handoff

- Branch: `feat/initial-scaffold`
- Last commit: `f1bec12` — Implement alpaca client, streaming, and storage layers
- Tests: 58 passing, 1 slow integration deselected by default. Run with `pytest`.
- Lint/type: not yet run. Try `ruff check .` and `mypy src` in the morning before adding code.
- All deps installed in `.venv/` (Python 3.14, torch CPU-only, transformers, alpaca-py, backtrader, etc.).
