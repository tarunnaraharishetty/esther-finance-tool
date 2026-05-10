# Esther

Production-grade terminal-based quantitative trading platform built in Python on the Alpaca API.

Esther centralizes the building blocks most retail trading bots leave fragmented: real-time market and news ingestion, AI-powered sentiment scoring, technical indicators, signal aggregation, risk management, backtesting, and paper/live execution — all behind a clean, modular architecture.

## Features

- Real-time market data and news streaming (Alpaca WebSockets + REST)
- FinBERT-based news sentiment scoring
- Technical indicators (RSI, MACD, Bollinger Bands) with a pluggable base class
- Hybrid signal aggregation across sentiment + technicals
- Risk engine: VaR, Sharpe, drawdown, exposure caps
- Bracket orders with stop-loss / take-profit and config-driven position sizing
- Backtesting via Backtrader with SPY benchmark comparison
- Rich/Textual terminal UI
- Structured logging, rate-limit-aware API clients, Pydantic-typed configuration

## Requirements

- Python **3.12+**
- An Alpaca account (paper trading is the default)
- A news provider API key (Alpaca News, NewsAPI, or Benzinga)

## Setup

```bash
git clone <repo-url>
cd "esther finance tool"

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"

cp .env.example .env            # then edit .env with your keys
```

## Run

```bash
python src/main.py              # or: esther
```

## Development

```bash
pytest                          # tests (excludes @integration)
pytest -m integration           # against real Alpaca paper API
ruff check . && ruff format .   # lint + format
mypy src                        # strict type-check
```

## Project layout

```
src/
  config/        Pydantic settings & env loading
  data/          Alpaca client, market/news ingestion, storage, streaming
  sentiment/     FinBERT analyzer & news scoring
  indicators/    RSI, MACD, Bollinger
  strategy/      Signal aggregation & strategy base classes
  execution/     Broker, order manager, position sizing, bracket orders
  risk/          VaR, Sharpe, drawdown, exposure
  backtesting/   Backtrader engine, benchmarks, analytics
  utils/         Logging, rate limiter, helpers
config/          YAML/TOML strategy & runtime configs
data/            SQLite DBs, OHLCV/news cache (gitignored)
logs/            Rotating log files (gitignored)
tests/           Pytest suite mirroring src/
```

See `CLAUDE.md` for engineering conventions and safety rules (trading-specific guardrails apply).

## Safety

- Paper trading is the default in every code path and test.
- Risk checks must run before any order submission.
- New strategies require a backtest against SPY before going live.

## License

Proprietary — all rights reserved.
