"""Esther CLI entry point.

Run: ``python src/main.py`` or, after install, ``esther``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python src/main.py` (script invocation) in addition to `python -m src.main`.
# When run as a script, the project root is not on sys.path, so `import src` fails.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from src import __version__  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402

console = Console()


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="esther")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Esther — quantitative trading platform."""
    settings = get_settings()
    configure_logging(settings)

    if ctx.invoked_subcommand is None:
        _print_banner(settings)
        ctx.exit(0)


@cli.command()
def status() -> None:
    """Print effective configuration (secrets masked)."""
    settings = get_settings()
    table = Table(title="Esther - runtime config", show_header=True, header_style="bold cyan")
    table.add_column("setting")
    table.add_column("value")
    table.add_row("app_env", settings.app_env.value)
    table.add_row("alpaca_base_url", settings.alpaca_base_url)
    table.add_row("paper_trading", "yes" if settings.is_paper_trading else "NO (LIVE)")
    table.add_row("data_feed", settings.alpaca_data_feed.value)
    table.add_row("news_provider", settings.news_provider.value)
    table.add_row("database_url", settings.database_url)
    table.add_row("log_level", settings.log_level.value)
    table.add_row("max_position_pct", f"{settings.max_position_pct:.2%}")
    table.add_row("max_daily_drawdown_pct", f"{settings.max_daily_drawdown_pct:.2%}")
    table.add_row("sentiment_model", settings.sentiment_model)
    console.print(table)


@cli.command()
def initdb() -> None:
    """Create database tables."""
    from src.data.storage import init_db

    init_db()
    console.print("[green]database initialised[/green]")


@cli.command()
@click.option(
    "-s",
    "--symbol",
    "symbols",
    multiple=True,
    help="Symbol to evaluate. Repeat for multiple, e.g. -s AAPL -s MSFT. "
    "Defaults to AAPL, MSFT, NVDA, SPY.",
)
@click.option("--lookback-days", default=60, show_default=True, help="OHLCV history window.")
@click.option("--news-hours", default=48, show_default=True, help="News lookback window.")
@click.option(
    "--interval",
    "interval_seconds",
    default=60.0,
    show_default=True,
    help="Seconds between ticks.",
)
@click.option(
    "--max-iterations",
    type=int,
    default=None,
    help="Stop after N ticks (omit to run until Ctrl-C).",
)
@click.option(
    "--min-confidence",
    default=0.3,
    show_default=True,
    help="Skip orders below this confidence even on BUY/SELL.",
)
@click.option(
    "--execute",
    is_flag=True,
    default=False,
    help="Actually submit paper orders. Default is dry-run (no orders).",
)
@click.option(
    "--no-sentiment",
    is_flag=True,
    help="Skip FinBERT — sentiment score forced to zero (no model download).",
)
def run(
    symbols: tuple[str, ...],
    lookback_days: int,
    news_hours: int,
    interval_seconds: float,
    max_iterations: int | None,
    min_confidence: float,
    execute: bool,
    no_sentiment: bool,
) -> None:
    """Start the main trading loop: data → engine → risk → paper broker."""
    import asyncio

    from src.runner import TradingLoop, TradingLoopConfig
    from src.strategy.recommendation import RecommendationEngine

    settings = get_settings()
    log = get_logger("main")

    if not settings.is_paper_trading:
        console.print(
            "[red]refusing to run: ALPACA_BASE_URL is not a paper-trading URL[/red]"
        )
        raise click.exceptions.Exit(1)

    watchlist = list(symbols) if symbols else ["AAPL", "MSFT", "NVDA", "SPY"]
    cfg = TradingLoopConfig(
        watchlist=watchlist,
        lookback_days=lookback_days,
        news_hours=news_hours,
        interval_seconds=interval_seconds,
        max_iterations=max_iterations,
        min_confidence=min_confidence,
        execute=execute,
    )
    engine = RecommendationEngine()
    if no_sentiment:
        from src.sentiment.analyzer import SentimentAnalyzer, SentimentLabel, SentimentScore

        class _Neutral(SentimentAnalyzer):
            def __init__(self) -> None:
                pass

            def score_text(self, text: str) -> SentimentScore:  # type: ignore[override]
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

            def score_article(self, article: object) -> SentimentScore:  # type: ignore[override]
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

        engine.sentiment_analyzer = _Neutral()

    loop = TradingLoop.build(cfg, engine=engine, settings=settings)
    mode = "EXECUTE (paper orders)" if execute else "DRY-RUN (no orders)"
    log.info("esther.run.start", version=__version__, watchlist=watchlist, mode=mode)
    console.print(
        f"[cyan]esther run[/cyan]  watchlist={watchlist}  interval={interval_seconds}s  "
        f"min_conf={min_confidence}  mode=[bold]{mode}[/bold]\n"
        f"[dim]Ctrl-C to stop[/dim]"
    )
    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]stopped[/yellow]")


@cli.command()
@click.option(
    "-s",
    "--symbol",
    "symbols",
    multiple=True,
    help="Symbol to evaluate. Repeat for multiple, e.g. -s AAPL -s MSFT. "
    "Defaults to AAPL, MSFT, NVDA, SPY.",
)
@click.option("--lookback-days", default=60, show_default=True, help="OHLCV history window.")
@click.option("--news-hours", default=48, show_default=True, help="News lookback window.")
@click.option(
    "--no-sentiment",
    is_flag=True,
    help="Skip FinBERT (no model download). Useful for smoke tests.",
)
def recommend(
    symbols: tuple[str, ...],
    lookback_days: int,
    news_hours: int,
    no_sentiment: bool,
) -> None:
    """Show BUY / HOLD / SELL recommendations for a watchlist.

    Paper-trading only: this command never submits orders.
    """
    import asyncio
    from datetime import UTC, datetime, timedelta

    from rich.text import Text

    from src.data.market_data import MarketDataService
    from src.data.models import TimeFrame
    from src.data.news_ingestion import get_news_source
    from src.strategy.recommendation import RecommendationEngine

    settings = get_settings()
    log = get_logger("recommend")

    if not settings.is_paper_trading:
        console.print(
            "[red]refusing to run: ALPACA_BASE_URL is not a paper-trading URL[/red]"
        )
        raise click.exceptions.Exit(1)

    watchlist = list(symbols) if symbols else ["AAPL", "MSFT", "NVDA", "SPY"]
    end = datetime.now(UTC)
    bars_start = end - timedelta(days=lookback_days)
    news_start = end - timedelta(hours=news_hours)

    market = MarketDataService()
    news_source = get_news_source()
    engine = RecommendationEngine()
    if no_sentiment:
        # Drive sentiment to 0 by routing through a neutral analyzer.
        from src.sentiment.analyzer import SentimentAnalyzer, SentimentLabel, SentimentScore

        class _Neutral(SentimentAnalyzer):
            def __init__(self) -> None:
                pass

            def score_text(self, text: str) -> SentimentScore:  # type: ignore[override]
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

            def score_article(self, article: object) -> SentimentScore:  # type: ignore[override]
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

        engine.sentiment_analyzer = _Neutral()

    async def _gather() -> list[tuple[str, object]]:
        results: list[tuple[str, object]] = []
        for sym in watchlist:
            log.info("recommend.fetch", symbol=sym)
            try:
                bars = await market.get_bars([sym], TimeFrame.DAY_1, bars_start, end)
                news = (
                    [] if no_sentiment else await news_source.fetch([sym], news_start, end, limit=20)
                )
                df = market.to_dataframe(bars)
                rec = engine.recommend(sym, df, news=news)
                results.append((sym, rec))
            except Exception as e:  # noqa: BLE001 — surface as a per-row error
                log.warning("recommend.error", symbol=sym, error=str(e))
                results.append((sym, e))
        return results

    rows = asyncio.run(_gather())

    table = Table(
        title=f"Esther — recommendations ({end.strftime('%Y-%m-%d %H:%M %Z')})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("symbol")
    table.add_column("action")
    table.add_column("confidence", justify="right")
    table.add_column("tech", justify="right")
    table.add_column("sent", justify="right")
    table.add_column("news", justify="right")
    table.add_column("reasoning", overflow="fold")

    action_style = {"buy": "green", "sell": "red", "hold": "yellow"}

    for sym, item in rows:
        if isinstance(item, Exception):
            table.add_row(sym, "[red]ERROR[/red]", "-", "-", "-", "-", str(item))
            continue
        rec = item  # TradingRecommendation
        style = action_style.get(rec.action.value, "white")
        table.add_row(
            sym,
            Text(rec.action.value.upper(), style=f"bold {style}"),
            f"{rec.confidence:.2f}",
            f"{rec.technical_score:+.2f}",
            f"{rec.sentiment_score:+.2f}",
            str(rec.num_news_articles),
            rec.reasoning,
        )

    console.print(table)
    console.print(
        "[dim]paper-trading only — recommendations are advisory; no orders submitted.[/dim]"
    )


@cli.command()
@click.option("--symbol", default="SPY", show_default=True, help="Symbol to backtest.")
@click.option(
    "--start",
    "start_str",
    default=None,
    help="ISO date (YYYY-MM-DD). Defaults to 3 years ago.",
)
@click.option(
    "--end",
    "end_str",
    default=None,
    help="ISO date (YYYY-MM-DD). Defaults to today.",
)
@click.option("--lookback-bars", default=60, show_default=True)
@click.option("--starting-cash", default=100_000.0, show_default=True)
@click.option("--commission-pct", default=0.0005, show_default=True, help="Per-fill commission.")
@click.option("--min-confidence", default=0.3, show_default=True)
@click.option("--allow-short", is_flag=True, help="Allow short positions on SELL signals.")
@click.option("--save-csv", type=click.Path(), default=None, help="Write equity curve CSV.")
def backtest(
    symbol: str,
    start_str: str | None,
    end_str: str | None,
    lookback_bars: int,
    starting_cash: float,
    commission_pct: float,
    min_confidence: float,
    allow_short: bool,
    save_csv: str | None,
) -> None:
    """Walk-forward backtest of the recommendation engine vs buy-and-hold.

    Fetches daily bars from Alpaca (paper feed) for ``--symbol`` between
    ``--start`` and ``--end``, runs the engine bar-by-bar with a rolling
    window, then prints strategy stats and alpha vs buy-and-hold.
    """
    import asyncio
    from datetime import UTC, date, datetime, timedelta

    from src.backtesting.recommendation_backtest import (
        BacktestRunConfig,
        RecommendationBacktest,
    )
    from src.data.market_data import MarketDataService
    from src.data.models import TimeFrame
    from src.strategy.recommendation import RecommendationEngine

    settings = get_settings()
    log = get_logger("backtest")

    if not settings.is_paper_trading:
        console.print("[red]refusing to run: ALPACA_BASE_URL is not paper[/red]")
        raise click.exceptions.Exit(1)

    end_dt = (
        datetime.combine(date.fromisoformat(end_str), datetime.min.time(), tzinfo=UTC)
        if end_str
        else datetime.now(UTC)
    )
    start_dt = (
        datetime.combine(date.fromisoformat(start_str), datetime.min.time(), tzinfo=UTC)
        if start_str
        else end_dt - timedelta(days=365 * 3)
    )

    log.info("backtest.fetch", symbol=symbol, start=start_dt.isoformat(), end=end_dt.isoformat())
    console.print(
        f"[cyan]backtest[/cyan] {symbol}  "
        f"{start_dt.date()} → {end_dt.date()}  "
        f"lookback={lookback_bars}  min_conf={min_confidence}  "
        f"commission={commission_pct * 100:.2f}%"
    )

    md = MarketDataService()
    bars = asyncio.run(md.get_bars([symbol], TimeFrame.DAY_1, start_dt, end_dt))
    df = md.to_dataframe(bars)
    if df.empty or len(df) <= lookback_bars + 1:
        console.print(
            f"[red]not enough bars ({len(df)} rows) for lookback={lookback_bars}[/red]"
        )
        raise click.exceptions.Exit(2)

    # Numeric columns from Bar.model_dump() will be Decimal — coerce for pandas.
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(int)

    engine = RecommendationEngine()  # sentiment off below via use_sentiment flag
    cfg = BacktestRunConfig(
        starting_cash=starting_cash,
        lookback_bars=lookback_bars,
        min_confidence=min_confidence,
        commission_pct=commission_pct,
        allow_short=allow_short,
        use_sentiment=False,
    )
    bt = RecommendationBacktest(engine, cfg)
    result = bt.run(df, symbol=symbol)

    table = Table(title=f"Backtest — {symbol}", header_style="bold cyan")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("starting cash", f"${starting_cash:,.0f}")
    table.add_row("final value", f"${result.final_value:,.0f}")
    table.add_row("total return", f"{result.total_return_pct:+.2f}%")
    table.add_row("buy & hold return", f"{result.buy_and_hold_return_pct:+.2f}%")
    alpha_style = "green" if result.alpha_pct > 0 else "red"
    table.add_row("alpha vs B&H", f"[{alpha_style}]{result.alpha_pct:+.2f}%[/]")
    table.add_row("Sharpe (annualised)", f"{result.sharpe:.2f}")
    table.add_row("max drawdown", f"{result.max_drawdown_pct:.2f}%")
    table.add_row("trades", str(result.trades))
    table.add_row("bars evaluated", str(len(df)))
    console.print(table)

    if save_csv:
        import pandas as pd

        out = pd.DataFrame(
            {
                "equity": result.equity_curve,
                "returns": result.returns,
                "action": result.actions,
                "position": result.positions,
                "confidence": result.confidence,
            }
        )
        out.to_csv(save_csv)
        console.print(f"[dim]equity curve written → {save_csv}[/dim]")


def _print_banner(settings: object) -> None:
    body = (
        f"[bold]Esther[/bold] v{__version__}\n"
        f"env: {getattr(settings, 'app_env').value}  "
        f"paper: {'yes' if getattr(settings, 'is_paper_trading') else 'NO'}\n\n"
        "Commands:\n"
        "  esther status     show effective config\n"
        "  esther initdb     create database tables\n"
        "  esther recommend  show BUY/HOLD/SELL for a watchlist (paper)\n"
        "  esther backtest   walk-forward backtest vs buy-and-hold\n"
        "  esther run        start trading loop\n"
    )
    console.print(Panel(body, title="quant trading platform", border_style="cyan"))


if __name__ == "__main__":
    sys.exit(cli())
