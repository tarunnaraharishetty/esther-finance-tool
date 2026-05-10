# CLAUDE.md

Guidance for Claude Code working in this repository. Read this before making changes.

## Project: Esther

Production-grade terminal-based quantitative trading platform built in Python on the Alpaca API. Combines real-time market/news ingestion, AI sentiment scoring (FinBERT), technical indicators, risk management, backtesting, and paper/live execution into a single modular system. Target users: quant traders, algo researchers, retail traders building systematic strategies.

Source of truth for goals: `PROJECT_CONTEXT.md.txt`.

## Tech stack

- Python 3.12+
- alpaca-py (market data, news, broker)
- pydantic + pydantic-settings (config)
- pandas / numpy (analytics)
- transformers + torch (FinBERT sentiment)
- backtrader (backtesting)
- SQLAlchemy + SQLite now, Postgres later
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
│   ├── strategy/           # signal aggregation, base Strategy
│   ├── execution/          # broker, order manager, position sizing, bracket orders
│   ├── risk/               # VaR, Sharpe, drawdown, exposure
│   ├── backtesting/        # backtrader engine, benchmarks, analytics
│   └── utils/              # logging setup, rate limiter, helpers
├── config/                 # YAML/TOML strategy + runtime configs (not Python)
├── data/                   # SQLite DBs, OHLCV cache, news cache (gitignored)
├── logs/                   # rotating log files (gitignored)
├── tests/                  # pytest suite mirroring src/ layout
├── pyproject.toml
├── requirements.txt
└── .env.example            # copy to .env, never commit secrets
```

## Common commands

```bash
# Install (creates editable install + dev tooling)
pip install -e ".[dev]"

# Run the platform
python src/main.py            # or: esther

# Tests
pytest                        # full suite (excludes integration by default)
pytest -m integration         # hit real Alpaca paper API
pytest tests/indicators       # one module
pytest --cov=src              # coverage

# Lint / format / type-check
ruff check .
ruff format .
mypy src
```

## Conventions

- **Type hints everywhere.** mypy runs in `--strict`. No untyped defs.
- **Pydantic for config.** Never read `os.environ` directly outside `src/config/`. Import `get_settings()`.
- **Structured logging.** Use `structlog` via `src/utils/logging.py`. No `print()` in library code (Rich is fine in `main.py` / TUI).
- **Async where it pays.** WebSocket streams, concurrent HTTP fetches → async. CPU-bound (indicators, backtests) → sync.
- **Rate-limit-aware.** All Alpaca/news API calls go through the wrapper in `src/data/alpaca_client.py` (uses `tenacity` for backoff).
- **Small files, small functions.** No god-modules. Split when a file grows past ~300 lines.
- **OOP for strategies/indicators** (inherit from base classes); functional for pure transforms.
- **Secrets via env only.** Never hard-code keys. `.env` is gitignored; use `.env.example` as the schema.
- **Paper trading is the default.** Live trading must be an explicit, deliberate config flip — never the default in any code path or test.

## Safety rules (trading-specific)

- Never call live-trading endpoints in tests or examples. Use the paper URL or mocks.
- Risk checks (`src/risk/`) must run **before** any order submission in `src/execution/`. Don't bypass for "quick fixes."
- All new strategies require backtests against SPY benchmark before being wired into live/paper execution.
- Position sizing decisions must reference `MAX_POSITION_PCT` from settings — no hardcoded sizes.

## Testing notes

- Mock Alpaca API responses with `pytest-mock`. Don't hit the network in unit tests.
- Mark tests that hit real APIs with `@pytest.mark.integration` (skipped by default).
- Indicator tests should use deterministic fixture OHLCV data, not random.

## Status

Foundation phase. Modules are scaffolded with type-hinted stubs and `NotImplementedError`. Fill them in module-by-module; keep tests green as you go.
