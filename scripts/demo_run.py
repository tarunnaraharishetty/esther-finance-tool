"""Live demo of `esther run` with mocked data — no Alpaca credentials needed.

This is a one-shot demo, not part of the package. It builds a TradingLoop
with mocked market data + news + broker, runs a few ticks in both dry-run
and --execute modes, and prints the per-symbol TickResult.

Run: ``py -3.14 -X utf8 scripts/demo_run.py``
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd

# Allow `py scripts/demo_run.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("ALPACA_API_KEY", "demo-key")
os.environ.setdefault("ALPACA_API_SECRET", "demo-secret")
os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from src.data.models import NewsArticle, TimeFrame  # noqa: E402
from src.execution.broker import OrderResult  # noqa: E402
from src.execution.order_manager import OrderManager  # noqa: E402
from src.risk.exposure import GateResult, RiskGate  # noqa: E402
from src.runner import TradingLoop, TradingLoopConfig  # noqa: E402
from src.sentiment.analyzer import SentimentAnalyzer, SentimentLabel, SentimentScore  # noqa: E402
from src.strategy.recommendation import RecommendationEngine  # noqa: E402

console = Console()


def synthetic_df(seed: int, drift: float = 0.012, noise: float = 0.01, n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=noise, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range(end=datetime.now(UTC), periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=int),
        },
        index=idx,
    )


class StubSentiment(SentimentAnalyzer):
    """Returns +0.85 positive — no model download."""

    def __init__(self) -> None:
        pass

    def score_text(self, text: str) -> SentimentScore:  # type: ignore[override]
        return SentimentScore(SentimentLabel.POSITIVE, 0.85)

    def score_article(self, article: NewsArticle) -> SentimentScore:  # type: ignore[override]
        return self.score_text(article.headline)


def build_loop(*, execute: bool) -> TradingLoop:
    """Build a TradingLoop with all I/O mocked but the engine real."""
    # Different dynamics per symbol so we see different actions.
    dfs = {
        "AAPL": synthetic_df(seed=7, drift=0.018, noise=0.01),  # bullish
        "MSFT": synthetic_df(seed=11, drift=-0.015, noise=0.012),  # bearish
        "NVDA": synthetic_df(seed=42, drift=0.001, noise=0.012),  # flat-ish
    }

    market = MagicMock()

    async def fake_get_bars(symbols: list[str], *_args: object, **_kw: object) -> list[object]:
        # Return a sentinel; to_dataframe maps it to the right df by symbol.
        return [{"symbol": symbols[0]}]

    market.get_bars = AsyncMock(side_effect=fake_get_bars)
    market.to_dataframe = MagicMock(
        side_effect=lambda bars: dfs.get(bars[0]["symbol"], pd.DataFrame()) if bars else pd.DataFrame()
    )

    news_source = MagicMock()
    sample_article = NewsArticle(
        id="demo-1",
        headline="Strong quarterly earnings beat expectations",
        source="demo",
        symbols=["AAPL"],
        published_at=datetime.now(UTC),
    )
    news_source.fetch = AsyncMock(return_value=[sample_article])

    broker = MagicMock()
    broker.get_account_equity = AsyncMock(return_value=Decimal("100000"))
    broker.submit = AsyncMock(
        return_value=OrderResult(
            id="paper-ord-42", status="accepted", filled_qty=Decimal(0), filled_avg_price=None
        )
    )
    broker.cancel = AsyncMock()

    risk_gate = RiskGate()  # real risk gate
    order_manager = OrderManager(broker=broker, risk_gate=risk_gate)

    engine = RecommendationEngine(sentiment_analyzer=StubSentiment())

    cfg = TradingLoopConfig(
        watchlist=["AAPL", "MSFT", "NVDA"],
        timeframe=TimeFrame.DAY_1,
        execute=execute,
        min_confidence=0.15,
        max_iterations=1,
        interval_seconds=0.0,
    )

    return TradingLoop(
        config=cfg,
        engine=engine,
        market=market,
        news_source=news_source,
        order_manager=order_manager,
    )


def render(title: str, results: list[object]) -> None:
    table = Table(title=title, header_style="bold cyan")
    table.add_column("symbol")
    table.add_column("action")
    table.add_column("confidence", justify="right")
    table.add_column("tech", justify="right")
    table.add_column("sent", justify="right")
    table.add_column("combined", justify="right")
    table.add_column("order_id")
    table.add_column("note", overflow="fold")
    style = {"buy": "green", "sell": "red", "hold": "yellow"}
    for r in results:  # type: ignore[assignment]
        rec = r.recommendation  # type: ignore[attr-defined]
        order_id = r.order.id if r.order else "-"  # type: ignore[attr-defined]
        if rec is None:
            table.add_row(r.symbol, "-", "-", "-", "-", "-", order_id, r.note or r.error or "")  # type: ignore[attr-defined]
            continue
        table.add_row(
            r.symbol,  # type: ignore[attr-defined]
            f"[bold {style.get(rec.action.value, 'white')}]{rec.action.value.upper()}[/]",
            f"{rec.confidence:.2f}",
            f"{rec.technical_score:+.2f}",
            f"{rec.sentiment_score:+.2f}",
            f"{rec.combined_score:+.2f}",
            order_id,
            r.note,  # type: ignore[attr-defined]
        )
    console.print(table)


async def main() -> None:
    console.rule("[bold cyan]esther run — DRY-RUN demo (default mode)")
    loop = build_loop(execute=False)
    dry_results = await loop.tick()
    render("dry-run tick (no broker calls)", dry_results)

    console.rule("[bold yellow]esther run --execute — paper order submission")
    loop2 = build_loop(execute=True)
    exec_results = await loop2.tick()
    render("execute tick (orders submitted via mocked broker)", exec_results)

    submitted = [r for r in exec_results if r.order is not None]  # type: ignore[attr-defined]
    console.print(
        f"\n[green]{len(submitted)} order(s) submitted[/green] "
        f"to the (mocked) paper broker. Reasonings:"
    )
    for r in exec_results:  # type: ignore[assignment]
        if r.recommendation:  # type: ignore[attr-defined]
            console.print(f"  • [bold]{r.symbol}[/bold] — {r.recommendation.reasoning}")  # type: ignore[attr-defined]


if __name__ == "__main__":
    asyncio.run(main())
