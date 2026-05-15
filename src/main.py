"""Esther CLI entry point.

Run: ``python src/main.py`` or, after install, ``esther``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Allow `python src/main.py` (script invocation) in addition to `python -m src.main`.
# When run as a script, the project root is not on sys.path, so `import src` fails.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src import __version__
from src.config import Settings, get_settings
from src.utils.logging import configure_logging, get_logger

if TYPE_CHECKING:
    # Imported only for the helper's type annotation. The runtime import
    # lives inside _build_controller so `esther --help` stays fast — the
    # dashboard.controller module transitively pulls in pandas, alpaca-py,
    # and other heavy deps.
    from src.dashboard.controller import BaseController

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
    help="Symbol to backfill. Repeat for each ticker, e.g. -s AAPL -s MSFT. "
    "Defaults to AAPL, MSFT, NVDA, SPY.",
)
@click.option("--days", default=90, show_default=True, help="OHLCV history window in days.")
@click.option(
    "--news-hours",
    default=48,
    show_default=True,
    help="News lookback window in hours.",
)
@click.option(
    "--skip-news",
    is_flag=True,
    help="Bars only — skip news fetch (useful on Alpaca tiers without news entitlement).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the symbol/date plan without calling Alpaca. Useful for previewing "
    "what would be fetched against your monthly quota.",
)
def backfill(
    symbols: tuple[str, ...],
    days: int,
    news_hours: int,
    skip_news: bool,
    dry_run: bool,
) -> None:
    """Cache historical bars + news to ``data/cache/`` so the dashboard
    has warm history on first tick.

    Run once before ``esther dashboard`` for any symbol with sparse Alpaca
    history. The cache is plain CSV + JSON — ``cat`` it any time to
    inspect, ``rm -rf data/cache/`` to wipe.
    """
    import asyncio
    from datetime import UTC, datetime, timedelta

    from src.data import cache
    from src.data.market_data import MarketDataService
    from src.data.models import TimeFrame
    from src.data.news_ingestion import get_news_source

    settings = get_settings()
    log = get_logger("backfill")

    if not settings.is_paper_trading:
        console.print("[red]refusing to run: ALPACA_BASE_URL is not paper[/red]")
        raise click.exceptions.Exit(1)

    watchlist = [s.upper() for s in symbols] if symbols else ["AAPL", "MSFT", "NVDA", "SPY"]
    end = datetime.now(UTC)
    bars_start = end - timedelta(days=days)
    news_start = end - timedelta(hours=news_hours)

    if dry_run:
        plan = Table(
            title=f"Backfill plan (dry-run · no Alpaca calls) → {cache.cache_dir()}",
            header_style="bold cyan",
        )
        plan.add_column("symbol")
        plan.add_column("bars window", overflow="fold")
        plan.add_column("news window", overflow="fold")
        bars_window = (
            f"{bars_start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} "
            f"({days}d, 1D bars)"
        )
        news_window = (
            "[dim]skipped[/dim]"
            if skip_news
            else f"{news_start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} "
            f"UTC ({news_hours}h)"
        )
        for sym in watchlist:
            plan.add_row(sym, bars_window, news_window)
        console.print(plan)
        console.print(
            f"[dim]would fetch {len(watchlist)} symbol(s) × "
            f"({'bars only' if skip_news else 'bars + news'}). "
            "Re-run without --dry-run to execute.[/dim]"
        )
        return

    market = MarketDataService()
    news_source = None if skip_news else get_news_source()

    async def _backfill_one(sym: str) -> tuple[int, int]:
        log.info("backfill.symbol", symbol=sym)
        bars = await market.get_bars([sym], TimeFrame.DAY_1, bars_start, end)
        df = market.to_dataframe(bars)
        bars_written = 0
        if not df.empty:
            for col in ("open", "high", "low", "close"):
                df[col] = df[col].astype(float)
            df["volume"] = df["volume"].astype(int)
            cache.write_bars(sym, TimeFrame.DAY_1, df)
            bars_written = len(df)

        news_written = 0
        if news_source is not None:
            articles = await news_source.fetch([sym], news_start, end, limit=50)
            cache.write_news(sym, articles)
            news_written = len(articles)
        return bars_written, news_written

    async def _gather() -> list[tuple[str, int, int, Exception | None]]:
        results: list[tuple[str, int, int, Exception | None]] = []
        for sym in watchlist:
            try:
                bars_n, news_n = await _backfill_one(sym)
                results.append((sym, bars_n, news_n, None))
            except Exception as e:
                log.warning("backfill.error", symbol=sym, error=str(e))
                results.append((sym, 0, 0, e))
        return results

    rows = asyncio.run(_gather())

    table = Table(
        title=f"Backfill ({days}d bars / {news_hours}h news -> data/cache/)",
        header_style="bold cyan",
    )
    table.add_column("symbol")
    table.add_column("bars", justify="right")
    table.add_column("news", justify="right")
    table.add_column("status")
    for sym, bars_n, news_n, err in rows:
        if err is not None:
            table.add_row(sym, "-", "-", f"[red]error: {err}[/red]")
        else:
            table.add_row(
                sym,
                str(bars_n),
                "-" if skip_news else str(news_n),
                "[green]ok[/green]",
            )
    console.print(table)


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
        console.print("[red]refusing to run: ALPACA_BASE_URL is not a paper-trading URL[/red]")
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

            def score_text(self, text: str) -> SentimentScore:
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

            def score_article(self, article: object) -> SentimentScore:
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

        engine.sentiment_analyzer = _Neutral()

    from src.strategy.recommendation import TradingRecommendation

    async def _gather() -> list[tuple[str, TradingRecommendation | Exception]]:
        results: list[tuple[str, TradingRecommendation | Exception]] = []
        for sym in watchlist:
            log.info("recommend.fetch", symbol=sym)
            try:
                bars = await market.get_bars([sym], TimeFrame.DAY_1, bars_start, end)
                news = (
                    []
                    if no_sentiment
                    else await news_source.fetch([sym], news_start, end, limit=20)
                )
                df = market.to_dataframe(bars)
                rec = engine.recommend(sym, df, news=news)
                results.append((sym, rec))
            except Exception as e:
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
        rec = item
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
    console.print("[dim]decision-support output — no orders are submitted.[/dim]")


@cli.command()
@click.argument("symbol")
@click.option("--lookback-days", default=60, show_default=True, help="OHLCV history window.")
@click.option("--news-hours", default=48, show_default=True, help="News lookback window.")
@click.option(
    "--no-sentiment",
    is_flag=True,
    help="Skip FinBERT (sentiment forced to zero). Doesn't affect the LLM brief itself.",
)
def summarize(symbol: str, lookback_days: int, news_hours: int, no_sentiment: bool) -> None:
    """Generate a plain-English AI brief for one SYMBOL.

    Fetches bars + news, runs the recommendation engine, then asks Claude
    to write a 2-4 sentence research brief. Requires ANTHROPIC_API_KEY.
    """
    import asyncio
    from datetime import UTC, datetime, timedelta

    import anthropic

    from src.data.market_data import MarketDataService
    from src.data.models import TimeFrame
    from src.data.news_ingestion import get_news_source
    from src.intelligence.explain import explain
    from src.intelligence.llm_summary import LLMSummarizer
    from src.strategy.recommendation import RecommendationEngine

    settings = get_settings()
    log = get_logger("summarize")

    if not settings.is_paper_trading:
        console.print("[red]refusing to run: ALPACA_BASE_URL is not paper[/red]")
        raise click.exceptions.Exit(1)

    try:
        summarizer = LLMSummarizer(settings=settings)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise click.exceptions.Exit(2) from None

    sym = symbol.upper()
    end = datetime.now(UTC)
    bars_start = end - timedelta(days=lookback_days)
    news_start = end - timedelta(hours=news_hours)

    market = MarketDataService()
    news_source = get_news_source()
    engine = RecommendationEngine()
    if no_sentiment:
        from src.sentiment.analyzer import SentimentAnalyzer, SentimentLabel, SentimentScore

        class _Neutral(SentimentAnalyzer):
            def __init__(self) -> None:
                pass

            def score_text(self, text: str) -> SentimentScore:
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

            def score_article(self, article: object) -> SentimentScore:
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

        engine.sentiment_analyzer = _Neutral()

    from src.data.models import NewsArticle as _NewsArticle
    from src.strategy.recommendation import TradingRecommendation as _TradingRec

    async def _gather() -> tuple[_TradingRec, list[_NewsArticle]]:
        bars = await market.get_bars([sym], TimeFrame.DAY_1, bars_start, end)
        df = market.to_dataframe(bars)
        if df.empty:
            raise RuntimeError(f"no bars returned for {sym}")
        news = [] if no_sentiment else await news_source.fetch([sym], news_start, end, limit=20)
        rec = engine.recommend(sym, df, news=news)
        return rec, list(news)

    try:
        rec, news = asyncio.run(_gather())
    except Exception as e:
        log.warning("summarize.fetch_failed", symbol=sym, error=str(e))
        console.print(f"[red]failed to fetch data for {sym}: {e}[/red]")
        raise click.exceptions.Exit(3) from None

    explanation = explain(
        symbol=rec.symbol,
        action=rec.action,
        confidence=rec.confidence,
        combined_score=rec.combined_score,
        indicator_scores=rec.indicator_scores,
        sentiment_score=rec.sentiment_score,
        num_news_articles=rec.num_news_articles,
    )

    try:
        brief = summarizer.summarize(
            explanation,
            headlines=[a.headline for a in news[:5]],
        )
    except anthropic.AuthenticationError:
        console.print("[red]Anthropic auth failed — check ANTHROPIC_API_KEY in .env.[/red]")
        raise click.exceptions.Exit(2) from None
    except anthropic.RateLimitError:
        console.print("[yellow]Anthropic rate-limited — try again in a minute.[/yellow]")
        raise click.exceptions.Exit(4) from None
    except anthropic.APIStatusError as e:
        console.print(f"[red]Anthropic API error ({e.status_code}): {e.message}[/red]")
        raise click.exceptions.Exit(5) from None

    console.print(
        Panel(
            brief,
            title=f"{sym} — AI brief ({end.strftime('%Y-%m-%d %H:%M %Z')})",
            border_style="cyan",
        )
    )
    console.print("[dim]decision support — research only, not execution advice.[/dim]")


@cli.command()
@click.option(
    "-s",
    "--symbol",
    "symbols",
    multiple=True,
    help="Symbol to include. Repeat for each ticker. Defaults to AAPL, MSFT, NVDA, SPY.",
)
@click.option("--lookback-days", default=60, show_default=True, help="OHLCV history window.")
@click.option("--news-hours", default=48, show_default=True, help="News lookback window.")
@click.option(
    "--no-sentiment",
    is_flag=True,
    help="Skip FinBERT (sentiment forced to zero). News headlines still go into the recap.",
)
def recap(
    symbols: tuple[str, ...],
    lookback_days: int,
    news_hours: int,
    no_sentiment: bool,
) -> None:
    """Generate a session recap across the watchlist.

    One-shot: fetches bars + news for each symbol, runs the
    recommendation engine, computes the ranked sections, then asks
    Claude for a 4-8 sentence brief grounded in the structured data.
    Requires ANTHROPIC_API_KEY.
    """
    import asyncio
    from datetime import UTC, datetime, timedelta

    import anthropic

    from src.dashboard.state import DashboardSnapshot
    from src.data.market_data import MarketDataService
    from src.data.models import TimeFrame
    from src.data.news_ingestion import get_news_source
    from src.intelligence.recap import LLMRecapGenerator, RecapContext
    from src.strategy.recommendation import RecommendationEngine

    settings = get_settings()
    log = get_logger("recap")

    if not settings.is_paper_trading:
        console.print("[red]refusing to run: ALPACA_BASE_URL is not paper[/red]")
        raise click.exceptions.Exit(1)

    try:
        generator = LLMRecapGenerator(settings=settings)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise click.exceptions.Exit(2) from None

    watchlist = [s.upper() for s in symbols] if symbols else ["AAPL", "MSFT", "NVDA", "SPY"]
    end = datetime.now(UTC)
    bars_start = end - timedelta(days=lookback_days)
    news_start = end - timedelta(hours=news_hours)

    market = MarketDataService()
    news_source = get_news_source()
    engine = RecommendationEngine()
    if no_sentiment:
        from src.sentiment.analyzer import SentimentAnalyzer, SentimentLabel, SentimentScore

        class _Neutral(SentimentAnalyzer):
            def __init__(self) -> None:
                pass

            def score_text(self, text: str) -> SentimentScore:
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

            def score_article(self, article: object) -> SentimentScore:
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

        engine.sentiment_analyzer = _Neutral()

    async def _build_rows() -> list[object]:
        from src.dashboard.controller import _row_from_recommendation

        rows: list[object] = []
        for sym in watchlist:
            log.info("recap.fetch", symbol=sym)
            try:
                bars = await market.get_bars([sym], TimeFrame.DAY_1, bars_start, end)
                df = market.to_dataframe(bars)
                if df.empty:
                    continue
                for col in ("open", "high", "low", "close"):
                    df[col] = df[col].astype(float)
                df["volume"] = df["volume"].astype(int)
                news = await news_source.fetch([sym], news_start, end, limit=20)
                rec = engine.recommend(sym, df, news=news, now=end)
                top_headlines = tuple(
                    a.headline for a in sorted(news, key=lambda a: a.published_at, reverse=True)[:5]
                )
                rows.append(
                    _row_from_recommendation(
                        rec,
                        last_price=float(df["close"].iloc[-1]),
                        headlines=top_headlines,
                    )
                )
            except Exception as e:
                log.warning("recap.error", symbol=sym, error=str(e))
        return rows

    rows = asyncio.run(_build_rows())
    if not rows:
        console.print("[red]no rows could be built — Alpaca returned empty for all symbols[/red]")
        raise click.exceptions.Exit(3)

    snapshot = DashboardSnapshot(
        tick=1,
        rows=rows,  # type: ignore[arg-type]
        events=[],
        alerts=[],
        signal_history={},
        timestamp=end,
    )
    context = RecapContext.from_snapshot(snapshot)

    try:
        text = generator.generate(context)
    except anthropic.AuthenticationError:
        console.print("[red]Anthropic auth failed — check ANTHROPIC_API_KEY in .env.[/red]")
        raise click.exceptions.Exit(2) from None
    except anthropic.RateLimitError:
        console.print("[yellow]Anthropic rate-limited — try again in a minute.[/yellow]")
        raise click.exceptions.Exit(4) from None
    except anthropic.APIStatusError as e:
        console.print(f"[red]Anthropic API error ({e.status_code}): {e.message}[/red]")
        raise click.exceptions.Exit(5) from None

    console.print(
        Panel(
            text,
            title=f"Watchlist recap ({end.strftime('%Y-%m-%d %H:%M %Z')})",
            border_style="cyan",
        )
    )
    console.print("[dim]decision support — research only, not execution advice.[/dim]")


def _render_preflight_report(report: object) -> None:
    """Print a Rich table for a PreflightReport. Imported here to keep
    the doctor / dashboard handlers thin."""
    from src.utils.preflight import CheckStatus, PreflightReport

    assert isinstance(report, PreflightReport)
    table = Table(title="Esther — preflight", header_style="bold cyan")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    style_for = {
        CheckStatus.OK: "green",
        CheckStatus.WARN: "yellow",
        CheckStatus.FAIL: "red",
    }
    for r in report.results:
        style = style_for.get(r.status, "white")
        table.add_row(
            r.name,
            f"[bold {style}]{r.status.value.upper()}[/]",
            r.detail,
        )
    console.print(table)
    # Print remediation hints separately so they don't clutter the table.
    for r in report.results:
        if r.hint and not r.ok:
            console.print(f"  [dim]-> {r.name}:[/dim] [yellow]{r.hint}[/yellow]")


@cli.command()
@click.option(
    "--no-online",
    is_flag=True,
    help="Skip the network check that pings Alpaca /v2/clock.",
)
@click.option(
    "--init-env",
    is_flag=True,
    help="Create .env from .env.example if it doesn't exist, then re-check.",
)
@click.option(
    "--force",
    is_flag=True,
    help="With --init-env, overwrite an existing .env (DESTRUCTIVE).",
)
def doctor(no_online: bool, init_env: bool, force: bool) -> None:
    """Diagnose your local setup for live Alpaca paper usage.

    Checks: .env presence, alpaca-py installed, API keys filled in,
    paper URL configured, FinBERT deps installed, and an optional online
    request to Alpaca /v2/clock to confirm the keys authenticate.
    """
    from src.utils.preflight import init_env_from_example, run_preflight

    settings = get_settings()

    if init_env:
        try:
            dst = init_env_from_example(settings.project_root, force=force)
            console.print(f"[green]created {dst}[/green]")
            console.print(
                "[yellow]Edit it now: replace the placeholder Alpaca keys with your "
                "paper keys from https://app.alpaca.markets/paper/dashboard/overview[/yellow]"
            )
        except FileExistsError as e:
            console.print(f"[yellow]{e}[/yellow]")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            raise click.exceptions.Exit(1) from None
        # Re-load settings now that .env may have changed.
        from src.config import settings as settings_mod

        settings_mod.get_settings.cache_clear()
        settings = get_settings()

    report = run_preflight(settings, online=not no_online)
    _render_preflight_report(report)

    if not report.ok:
        console.print(
            "\n[red bold]preflight failed[/red bold] — fix the items above before "
            "running `esther dashboard`."
        )
        raise click.exceptions.Exit(2)
    console.print("\n[green]all checks passed — ready to launch[/green]")


@cli.command()
@click.option(
    "-s",
    "--symbol",
    "symbols",
    multiple=True,
    help="Repeat for each ticker, e.g. -s AAPL -s MSFT. Defaults to AAPL MSFT NVDA TSLA SPY.",
)
@click.option(
    "--refresh-seconds",
    default=None,
    type=float,
    help="Seconds between dashboard refreshes. Falls back to "
    "Settings.dashboard_refresh_seconds (5.0).",
)
@click.option(
    "--burst-seconds",
    default=None,
    type=float,
    help="Seconds for the adaptive burst follow-up after flips / alerts. "
    "Falls back to Settings.dashboard_burst_seconds (1.5).",
)
@click.option(
    "--lookback-days",
    default=60,
    show_default=True,
    help="OHLCV history window passed to the engine each tick.",
)
@click.option(
    "--news-hours",
    default=48,
    show_default=True,
    help="News lookback window per tick.",
)
@click.option(
    "--mock",
    is_flag=True,
    help="Use synthetic data — no Alpaca credentials required.",
)
@click.option(
    "--no-sentiment",
    is_flag=True,
    help="Skip FinBERT — sentiment score forced to zero (no model download).",
)
@click.option(
    "--skip-preflight",
    is_flag=True,
    help="Skip the preflight checks. Not recommended.",
)
def dashboard(
    symbols: tuple[str, ...],
    refresh_seconds: float | None,
    burst_seconds: float | None,
    lookback_days: int,
    news_hours: int,
    mock: bool,
    no_sentiment: bool,
    skip_preflight: bool,
) -> None:
    """Launch the Textual dashboard: live watchlist + recommendations + events.

    Decision-support only — no orders are submitted from this UI.
    With --mock you can run offline using synthetic data; without --mock
    the dashboard fetches real Alpaca paper bars + news and (if sentiment
    is enabled) scores headlines with FinBERT.
    """
    from src.dashboard.app import DashboardApp

    settings = get_settings()
    watchlist = list(symbols) if symbols else ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]

    # CLI flags override; otherwise pull from Settings so env-config
    # is the single tuning source for the cadence pair.
    effective_refresh_seconds = (
        refresh_seconds if refresh_seconds is not None else settings.dashboard_refresh_seconds
    )
    effective_burst_seconds = (
        burst_seconds if burst_seconds is not None else settings.dashboard_burst_seconds
    )

    controller = _build_controller(
        watchlist=watchlist,
        mock=mock,
        no_sentiment=no_sentiment,
        skip_preflight=skip_preflight,
        lookback_days=lookback_days,
        news_hours=news_hours,
        settings=settings,
        surface="dashboard",
    )

    # Try to build the LLM summarizer + OPP briefer; if ANTHROPIC_API_KEY
    # isn't set the dashboard still launches but the `s` and `b` keypresses
    # surface a clean error.
    summarizer: object | None
    opportunity_briefer: object | None
    try:
        from src.intelligence.llm_summary import LLMSummarizer
        from src.intelligence.opportunity_brief import LLMOpportunityBriefer

        summarizer = LLMSummarizer(settings=settings)
        opportunity_briefer = LLMOpportunityBriefer(settings=settings)
    except RuntimeError:
        summarizer = None
        opportunity_briefer = None
        console.print(
            "[yellow]ANTHROPIC_API_KEY not set — AI briefs disabled. "
            "Press `s` or `b` for setup hint.[/yellow]"
        )

    DashboardApp(
        controller,
        refresh_seconds=effective_refresh_seconds,
        burst_seconds=effective_burst_seconds,
        summarizer=summarizer,
        opportunity_briefer=opportunity_briefer,
        columns=settings.dashboard_columns,
    ).run()


@cli.command()
@click.option(
    "-s",
    "--symbol",
    "symbols",
    multiple=True,
    help="Repeat for each ticker, e.g. -s AAPL -s MSFT. Defaults to AAPL MSFT NVDA TSLA SPY.",
)
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind.")
@click.option("--port", default=8000, show_default=True, type=int, help="TCP port to bind.")
@click.option(
    "--lookback-days", default=60, show_default=True, help="OHLCV history window per snapshot."
)
@click.option(
    "--news-hours", default=48, show_default=True, help="News lookback window per snapshot."
)
@click.option(
    "--mock",
    is_flag=True,
    help="Use synthetic data — no Alpaca credentials required.",
)
@click.option(
    "--no-sentiment",
    is_flag=True,
    help="Skip FinBERT — sentiment score forced to zero (no model download).",
)
@click.option(
    "--skip-preflight",
    is_flag=True,
    help="Skip the preflight checks. Not recommended.",
)
def serve(
    symbols: tuple[str, ...],
    host: str,
    port: int,
    lookback_days: int,
    news_hours: int,
    mock: bool,
    no_sentiment: bool,
    skip_preflight: bool,
) -> None:
    """Launch the read-only HTTP/JSON API mirror of the dashboard.

    Boots uvicorn against the FastAPI factory in ``src.api``. The server
    exposes ``GET /api/health`` and ``GET /api/snapshot`` — the same
    snapshot the Textual dashboard renders, JSON-encoded for a future
    web / mobile / external client.

    Decision-support data only — no order-submission endpoints, and the
    upstream Alpaca client is paper-feed locked. Bind defaults to
    localhost; only flip ``--host 0.0.0.0`` after wiring auth.
    """
    import uvicorn

    from src.api import create_app

    settings = get_settings()
    watchlist = list(symbols) if symbols else ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]

    controller = _build_controller(
        watchlist=watchlist,
        mock=mock,
        no_sentiment=no_sentiment,
        skip_preflight=skip_preflight,
        lookback_days=lookback_days,
        news_hours=news_hours,
        settings=settings,
        surface="API server",
    )
    app = create_app(controller)

    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            f"[yellow]warning: binding to {host} — API has no auth yet, "
            "exposing to a non-loopback address is risky.[/yellow]"
        )
    console.print(
        f"[green]Esther API live at[/green] http://{host}:{port}/api/snapshot · "
        f"[dim]watchlist: {', '.join(watchlist) or '(empty)'}[/dim]"
    )
    console.print("[dim]Ctrl+C to stop.[/dim]")
    # log_level=warning keeps uvicorn's per-request access logs out of
    # the terminal — Esther's own structlog setup handles request
    # observability if needed.
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _build_controller(
    *,
    watchlist: list[str],
    mock: bool,
    no_sentiment: bool,
    skip_preflight: bool,
    lookback_days: int,
    news_hours: int,
    settings: Settings,
    surface: str,
) -> BaseController:
    """Construct the snapshot-producing controller for ``dashboard`` /
    ``serve``.

    Both commands need exactly the same setup: load ``alerts.yaml`` if
    present, auto-register intraday rules when the feature is on,
    optionally run the preflight gate on live mode, and patch the
    sentiment analyzer to neutral on ``--no-sentiment``. Keeping this
    in one place means a future tweak to either command's pipeline
    can't drift apart from the other.

    ``surface`` only colors the preflight-failure error message
    ("Cannot launch live <surface> — preflight failed.") — the rest
    of the pipeline is identical regardless of caller.
    """
    from src.dashboard.controller import (
        DashboardController,
        MockDashboardController,
    )
    from src.intelligence.alert_prioritizer import AlertPrioritizer
    from src.intelligence.alerts import (
        AlertEngine,
        load_prioritizer_config_from_yaml,
        load_rules_from_yaml,
        load_snapshot_rules_from_yaml,
    )
    from src.strategy.recommendation import RecommendationEngine

    # Load user-defined alert rules + prioritizer knobs from
    # config/alerts.yaml if present. Missing file = both default.
    alerts_path = settings.config_dir / "alerts.yaml"
    if alerts_path.exists():
        try:
            rules = load_rules_from_yaml(alerts_path)
            snapshot_rules = load_snapshot_rules_from_yaml(alerts_path)
            prio_config = load_prioritizer_config_from_yaml(alerts_path)
            # YAML with no `snapshot_rules:` block returns an empty list,
            # which would silence opportunity_entry. Fall back to defaults
            # so the OPP alert stays on unless explicitly muted.
            engine_kwargs: dict[str, object] = {"rules": rules}
            if snapshot_rules:
                engine_kwargs["snapshot_rules"] = snapshot_rules
            alert_engine = AlertEngine(**engine_kwargs)  # type: ignore[arg-type]
            alert_prioritizer = AlertPrioritizer(config=prio_config)  # type: ignore[arg-type]
            console.print(
                f"[dim]loaded {len(rules)} alert rule(s) + "
                f"{len(snapshot_rules)} snapshot rule(s) from {alerts_path}[/dim]"
            )
        except ValueError as e:
            console.print(f"[red]invalid {alerts_path}: {e}[/red]")
            raise click.exceptions.Exit(2) from None
    else:
        alert_engine = AlertEngine()
        alert_prioritizer = AlertPrioritizer()

    # Auto-register intraday alert rules when the feature is on.
    if settings.intraday_enabled:
        from dataclasses import replace as dc_replace

        from src.intelligence.intraday_alerts import (
            default_intraday_cooldowns,
            default_intraday_rules,
        )

        extra_rules, extra_snapshot_rules = default_intraday_rules()
        alert_engine.rules.extend(extra_rules)  # type: ignore[arg-type]
        alert_engine.snapshot_rules.extend(extra_snapshot_rules)  # type: ignore[arg-type]
        merged_cooldowns = {
            **default_intraday_cooldowns(),
            **alert_prioritizer.config.cooldowns,
        }
        alert_prioritizer.config = dc_replace(alert_prioritizer.config, cooldowns=merged_cooldowns)
        console.print(
            "[dim]intraday alerts wired: "
            f"{len(extra_rules)} per-row + {len(extra_snapshot_rules)} snapshot rule(s)[/dim]"
        )

    if mock:
        return MockDashboardController(
            watchlist=watchlist,
            use_sentiment=not no_sentiment,
            alert_engine=alert_engine,
            alert_prioritizer=alert_prioritizer,
        )

    if not skip_preflight:
        from src.utils.preflight import PreflightError, require_live_ok

        try:
            require_live_ok(settings, online=True)
        except PreflightError as e:
            console.print(
                f"[red bold]Cannot launch live {surface} — preflight failed.[/red bold]"
            )
            _render_preflight_report(e.report)
            console.print(
                "\n[dim]Run `esther doctor` for the same diagnostics, or "
                "pass `--mock` to run with synthetic data.[/dim]"
            )
            raise click.exceptions.Exit(2) from None

    engine = RecommendationEngine()
    if no_sentiment:
        from src.sentiment.analyzer import (
            SentimentAnalyzer,
            SentimentLabel,
            SentimentScore,
        )

        class _Neutral(SentimentAnalyzer):
            def __init__(self) -> None:
                pass

            def score_text(self, text: str) -> SentimentScore:
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

            def score_article(self, article: object) -> SentimentScore:
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

        engine.sentiment_analyzer = _Neutral()
    return DashboardController(
        watchlist=watchlist,
        engine=engine,
        settings=settings,
        lookback_days=lookback_days,
        news_hours=news_hours,
        alert_engine=alert_engine,
        alert_prioritizer=alert_prioritizer,
    )


def _print_banner(settings: Settings) -> None:
    body = (
        f"[bold]Esther[/bold] v{__version__}\n"
        f"env: {settings.app_env.value}  "
        f"paper: {'yes' if settings.is_paper_trading else 'NO'}\n\n"
        "Commands:\n"
        "  esther status      show effective config\n"
        "  esther doctor      diagnose Alpaca + FinBERT setup\n"
        "  esther initdb      create database tables\n"
        "  esther backfill    cache historical bars + news to data/cache/\n"
        "  esther recommend   show BUY/HOLD/SELL for a watchlist\n"
        "  esther summarize   AI-written brief for one symbol (needs ANTHROPIC_API_KEY)\n"
        "  esther recap       AI-written brief across the whole watchlist\n"
        "  esther dashboard   live Textual dashboard (use --mock for demo)\n"
        "  esther serve       HTTP/JSON API mirror (use --mock for demo)\n"
    )
    console.print(Panel(body, title="market intelligence + decision support", border_style="cyan"))


if __name__ == "__main__":
    sys.exit(cli())
