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


def _print_banner(settings: object) -> None:
    body = (
        f"[bold]Esther[/bold] v{__version__}\n"
        f"env: {getattr(settings, 'app_env').value}  "
        f"paper: {'yes' if getattr(settings, 'is_paper_trading') else 'NO'}\n\n"
        "Commands:\n"
        "  esther status     show effective config\n"
        "  esther initdb     create database tables\n"
        "  esther recommend  show BUY/HOLD/SELL for a watchlist (paper)\n"
        "  esther run        start trading loop\n"
    )
    console.print(Panel(body, title="quant trading platform", border_style="cyan"))


if __name__ == "__main__":
    sys.exit(cli())
