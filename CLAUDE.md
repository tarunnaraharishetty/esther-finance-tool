# CLAUDE.md

Guidance for Claude Code working in this repository. Read this before making changes.

## Project: Esther

Terminal-based AI trading **dashboard and research assistant** built in Python on the Alpaca API. The product helps humans make better trading decisions — it does **not** trade on their behalf. Combines real-time market + news ingestion, FinBERT sentiment scoring, technical indicators, AI summaries, watchlist intelligence, and alerts in a Textual TUI. Target users: discretionary traders, research-driven retail investors, anyone who wants context before pressing the button themselves.

**In scope:** market monitoring, sentiment analysis, signal aggregation, AI summaries, watchlist intelligence, alerts, explanations, clean UX.

**Out of scope:** autonomous order submission, hedge-fund infrastructure, beating SPY, overengineered quant systems, excessive backtesting.

## Tech stack

- Python 3.12+
- alpaca-py (market data + news — *not* the broker side)
- pydantic + pydantic-settings (config)
- pandas / numpy (analytics)
- transformers + torch (FinBERT sentiment)
- SQLAlchemy + SQLite
- rich / textual (terminal UI)
- structlog (structured logging)
- pytest + pytest-asyncio (testing)
- ruff (lint + format), mypy --strict (typing)

## Layout

```
.
├── src/                    # all Python source
│   ├── main.py             # CLI entry point (`python src/main.py` or `esther`)
│   ├── config/             # Pydantic settings, env loading
│   ├── data/               # Alpaca client, market/news ingestion, storage, streaming
│   ├── sentiment/          # FinBERT analyzer, news scoring
│   ├── indicators/         # RSI, MACD, Bollinger, base Indicator
│   ├── strategy/           # signal aggregation, recommendation engine
│   ├── risk/               # Sharpe + max-drawdown only (watchlist intel use)
│   ├── dashboard/          # Textual app, controller, snapshot state
│   └── utils/              # logging, rate limiter, preflight checks
├── config/                 # YAML/TOML configs (not Python)
├── data/                   # SQLite DBs, OHLCV cache, news cache (gitignored)
├── logs/                   # rotating log files (gitignored)
├── tests/                  # pytest suite mirroring src/ layout
├── pyproject.toml
├── requirements.txt
└── .env.example            # copy to .env, never commit secrets
```

## Common commands

```bash
pip install -e ".[dev]"        # install + dev tooling
python src/main.py             # or: esther
pytest                         # full suite (integration tests deselected by default)
ruff check . && ruff format .  # lint + format
mypy src                       # strict type check
```

## Conventions

- **Type hints everywhere.** mypy runs in `--strict`. No untyped defs.
- **Pydantic for config.** Never read `os.environ` directly outside `src/config/`. Import `get_settings()`.
- **Structured logging.** Use `structlog` via `src/utils/logging.py`. No `print()` in library code (Rich is fine in `main.py` / TUI).
- **Async where it pays.** WebSocket streams, concurrent HTTP fetches → async. CPU-bound (indicators, sentiment scoring) → sync.
- **Rate-limit-aware.** All Alpaca/news API calls go through the wrapper in `src/data/alpaca_client.py` (uses `tenacity` for backoff).
- **Small files, small functions.** No god-modules. Split when a file grows past ~300 lines.
- **Secrets via env only.** Never hard-code keys. `.env` is gitignored; use `.env.example` as the schema.

## Safety rules

- **No order submission anywhere in the codebase.** Esther is decision-support; if a future module needs to talk to the broker side of Alpaca, raise it with the user first.
- **Paper feed only.** Even though we don't submit orders, the data layer points at `paper-api.alpaca.markets`. Live URLs are refused at startup by `DashboardController` / `recommend`.
- **No network in unit tests.** Mock Alpaca at the SDK boundary. Real-API tests must be marked `@pytest.mark.integration` and are skipped by default.

## Testing notes

- Mock Alpaca API responses with `pytest-mock`. Don't hit the network in unit tests.
- Indicator tests should use deterministic fixture OHLCV data, not random.
- Headless Textual smoke tests use `App.run_test()`.

## Status

The autonomous-trading and backtesting scaffolding has been removed (PR `refactor/decision-support-cut`). Next focus areas: AI summaries (`src/intelligence/summary.py`, planned), alert rules (`src/intelligence/alerts.py`, planned), watchlist intelligence (`src/intelligence/watchlist.py`, planned). See `NEXT_STEPS.md` for the current punch list.
