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
@click.option(
    "--refresh-seconds",
    default=None,
    type=float,
    help="Cadence of the SSE stream's tick loop. Falls back to "
    "Settings.dashboard_refresh_seconds (5.0). Pass 0 (or any "
    "non-positive value) to disable /api/stream entirely.",
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
    refresh_seconds: float | None,
) -> None:
    """Launch the read-only HTTP/JSON API mirror of the dashboard.

    Boots uvicorn against the FastAPI factory in ``src.api``. The server
    exposes:

    \b
      GET /api/health    — liveness probe
      GET /api/snapshot  — one-shot fetch (request → fresh snapshot)
      GET /api/stream    — Server-Sent Events stream of every tick

    Decision-support data only — no order-submission endpoints, and the
    upstream Alpaca client is paper-feed locked. Bind defaults to
    localhost; only flip ``--host 0.0.0.0`` after wiring auth.
    """
    import uvicorn

    from src.api import create_app
    from src.intelligence.calibration import CalibrationStore
    from src.intelligence.calibration_worker import CalibrationRecorder

    settings = get_settings()
    watchlist = list(symbols) if symbols else ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]

    effective_refresh = (
        refresh_seconds if refresh_seconds is not None else settings.dashboard_refresh_seconds
    )
    # 0 / negative cadence disables the stream — useful when running a
    # REST-only deployment that doesn't want the broker spinning at all.
    stream_interval: float | None = effective_refresh if effective_refresh > 0 else None

    # Calibration wiring: a single store backs both the snapshot-loop
    # recorder (writes per-tick observations) and the analyzer endpoint
    # (reads materialized hit-rate buckets). Both must point at the same
    # file or the wire contract publishes the wrong probabilities. When
    # ``calibration_store_path`` is None (operator opt-out via
    # ``CALIBRATION_STORE_PATH=``), both stay None and the snapshot loop
    # runs without ingestion.
    calibration_store: CalibrationStore | None = None
    calibration_recorder: CalibrationRecorder | None = None
    if settings.calibration_store_path is not None:
        calibration_store = CalibrationStore(settings.calibration_store_path)
        calibration_recorder = CalibrationRecorder(
            store=calibration_store,
            horizon_days=settings.calibration_horizon_days,
        )

    controller = _build_controller(
        watchlist=watchlist,
        mock=mock,
        no_sentiment=no_sentiment,
        skip_preflight=skip_preflight,
        lookback_days=lookback_days,
        news_hours=news_hours,
        settings=settings,
        surface="API server",
        calibration_recorder=calibration_recorder,
    )
    # Same-origin frontend serving: when `web/dist/` exists (built via
    # `npm run build`), FastAPI serves the React bundle alongside the
    # API on this single port. Missing → API-only mode (dev workflow
    # where Vite serves the frontend on :5173 separately).
    frontend_dir = settings.project_root / "web" / "dist"
    app = create_app(
        controller,
        stream_interval=stream_interval,
        cors_origins=settings.cors_origins,
        frontend_dir=frontend_dir,
        calibration_store=calibration_store,
    )
    serving_frontend = frontend_dir.exists()

    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            f"[yellow]warning: binding to {host} — API has no auth yet, "
            "exposing to a non-loopback address is risky.[/yellow]"
        )
    stream_status = (
        f"stream every {stream_interval:.1f}s" if stream_interval else "stream disabled"
    )
    frontend_status = "+frontend" if serving_frontend else "api-only (run `cd web && npm run dev`)"
    # One-line operator-visible signal: is the calibration system
    # actually accumulating observations on this deployment, and is
    # the maturation worker draining outcomes? Bug report triage
    # gets noticeably faster when the answer's printed at startup
    # instead of inferred from `ls data/`.
    if calibration_store is None:
        calibration_status = "calibration off"
    elif settings.calibration_maturation_enabled:
        calibration_status = (
            f"calibration on ({settings.calibration_store_path}) + worker"
        )
    else:
        calibration_status = (
            f"calibration on ({settings.calibration_store_path}) ingest-only"
        )
    console.print(
        f"[green]Esther API live at[/green] http://{host}:{port} · "
        f"[dim]{stream_status} · {frontend_status} · {calibration_status} · "
        f"watchlist: {', '.join(watchlist) or '(empty)'}[/dim]"
    )
    if serving_frontend:
        console.print(
            f"[dim]Open http://{host}:{port}/ in a browser — "
            "frontend + API on the same origin.[/dim]"
        )
    else:
        console.print(
            f"[dim]Endpoints: /api/health · /api/snapshot · /api/stream "
            f"(http://{host}:{port}/api/stream)[/dim]"
        )
    console.print("[dim]Ctrl+C to stop.[/dim]")
    # log_level=warning keeps uvicorn's per-request access logs out of
    # the terminal — Esther's own structlog setup handles request
    # observability if needed.
    uvicorn.run(app, host=host, port=port, log_level="warning")


# -----------------------------------------------------------------------------
# Calibration admin
# -----------------------------------------------------------------------------


@cli.group()
def calibrate() -> None:
    """Manage the signal-history calibration store.

    Subcommands:

      \b
      build    materialize hit-rate buckets from settled observations
      status   print observation + bucket counts for operator review
      cleanup  drop permanently-failed observations older than --older-than

    The calibration store accumulates observations from the snapshot
    loop and settles them via the maturation worker. ``build`` is the
    operator-triggered step that turns the raw observation history
    into the materialized buckets the analyzer endpoint reads. Run on
    a cron after the maturation worker has had a chance to settle
    recent horizons.
    """


def _require_calibration_store(settings: Settings) -> object:
    """Return a ``CalibrationStore`` or exit 2 if calibration is disabled.

    The CLI subcommands all need a store; this helper centralizes the
    "disabled in settings" message + exit code so every subcommand
    fails identically when ``CALIBRATION_STORE_PATH=``.
    """
    from src.intelligence.calibration import CalibrationStore

    if settings.calibration_store_path is None:
        console.print(
            "[red]Calibration store disabled in settings "
            "(CALIBRATION_STORE_PATH=).[/red]"
        )
        raise click.exceptions.Exit(2)
    return CalibrationStore(settings.calibration_store_path)


def _pairings_by_outcome() -> dict[str, list[str]]:
    """Group the analyzer's score↔outcome pairings by outcome.

    ``build`` walks the result so a single CLI invocation materializes
    every outcome the analyzer publishes against.
    """
    from src.intelligence.analyzer.calibration import PAIRINGS

    out: dict[str, list[str]] = {}
    for score_name, outcome_name in PAIRINGS:
        out.setdefault(outcome_name, []).append(score_name)
    return out


@calibrate.command("build")
@click.option(
    "--horizon",
    "horizon_days",
    type=int,
    default=None,
    help="Outcome horizon in days. Defaults to "
    "Settings.calibration_horizon_days (5).",
)
@click.option(
    "--bucket-width",
    type=float,
    default=None,
    help="Score-bucket width in [0,100]. Defaults to "
    "Settings.calibration_bucket_width (10.0).",
)
def calibrate_build(horizon_days: int | None, bucket_width: float | None) -> None:
    """Materialize hit-rate buckets from settled observations.

    Rebuilds atomically: every (score, horizon, outcome) triple is
    DELETE'd and re-INSERT'd inside one transaction, so concurrent
    readers never see a half-built table.

    Idempotent — running ``build`` twice in a row produces the same
    result as running it once.
    """
    settings = get_settings()
    store = _require_calibration_store(settings)
    effective_horizon = (
        horizon_days if horizon_days is not None else settings.calibration_horizon_days
    )
    effective_width = (
        bucket_width if bucket_width is not None else settings.calibration_bucket_width
    )

    all_obs = store.all_observations()  # type: ignore[attr-defined]
    settled = [o for o in all_obs if o.outcome_value is not None]
    unsettled = [o for o in all_obs if o.outcome_value is None]
    console.print(
        f"Loaded {len(all_obs)} observation(s) "
        f"({len(settled)} settled · {len(unsettled)} pending maturation)."
    )

    if not settled:
        console.print(
            "[yellow]No settled observations to bucket yet.[/yellow] "
            "Let the maturation worker run for at least one horizon "
            "window, then re-run."
        )
        return

    console.print("Building tables...")
    for outcome_name, score_names in sorted(_pairings_by_outcome().items()):
        table = store.build_table(  # type: ignore[attr-defined]
            score_names=score_names,
            horizon_days=effective_horizon,
            outcome_name=outcome_name,
            bucket_width=effective_width,
        )
        console.print(
            f"  [green]{outcome_name}[/green]: "
            f"{len(table.buckets)} bucket(s) across {len(score_names)} score(s)"
        )
    console.print(
        f"Done. Calibration table written to [bold]{settings.calibration_store_path}[/bold]."
    )


@calibrate.command("status")
def calibrate_status() -> None:
    """Print observation + bucket counts for operator review."""
    settings = get_settings()
    store = _require_calibration_store(settings)

    console.print(
        f"Calibration store: [bold]{settings.calibration_store_path}[/bold]"
    )
    obs = store.all_observations()  # type: ignore[attr-defined]
    settled = [o for o in obs if o.outcome_value is not None]
    unsettled = [o for o in obs if o.outcome_value is None]
    console.print(
        f"Observations: {len(obs)} total · "
        f"{len(settled)} settled · {len(unsettled)} pending maturation"
    )
    if obs:
        last_obs = max(obs, key=lambda o: o.observed_at)
        console.print(f"Last observation: {last_obs.observed_at.isoformat()}")
    if settled:
        # ``Observation`` carries observed_at, not a settled_at column —
        # the latest settled is the latest observation we've graded.
        last_settled = max(settled, key=lambda o: o.observed_at)
        console.print(
            f"Last settled observation: {last_settled.observed_at.isoformat()}"
        )

    table = store.load_table()  # type: ignore[attr-defined]
    from src.intelligence.analyzer.calibration import PAIRINGS

    console.print(
        f"\nMaterialized buckets (horizon={settings.calibration_horizon_days}d):"
    )
    for score_name, outcome_name in PAIRINGS:
        matching = [
            b
            for b in table.buckets
            if b.score_name == score_name
            and b.outcome_name == outcome_name
            and b.horizon_days == settings.calibration_horizon_days
        ]
        if not matching:
            console.print(
                f"  [dim]{score_name} → {outcome_name}: "
                "0 buckets (table not built yet)[/dim]"
            )
            continue
        total_obs = sum(b.n_observations for b in matching)
        best = max(matching, key=lambda b: b.n_observations)
        console.print(
            f"  [bold]{score_name}[/bold] → {outcome_name} · "
            f"{len(matching)} bucket(s) · {total_obs} obs · "
            f"best n={best.n_observations} hit={best.hit_rate:.2f}"
        )


@calibrate.command("cleanup")
@click.option(
    "--older-than",
    "older_than_days",
    type=int,
    default=30,
    show_default=True,
    help="Drop permanently-failed observations older than this many days.",
)
def calibrate_cleanup(older_than_days: int) -> None:
    """Drop unsettled observations older than ``--older-than`` days.

    When a symbol rotates off the watchlist mid-horizon, its
    observations get stuck unsettleable — the maturation worker can't
    grade them because no fresh bar will arrive. After a sufficient
    quiet period (typically weeks past the configured horizon), it's
    safe to drop them.

    **Settled observations are never removed by cleanup.** They are
    the calibration data; their value compounds over time. This
    command targets only the "horizon expired without ever grading"
    long tail.
    """
    from datetime import timedelta

    if older_than_days < 0:
        console.print("[red]--older-than must be non-negative.[/red]")
        raise click.exceptions.Exit(2)

    settings = get_settings()
    store = _require_calibration_store(settings)
    removed = store.cleanup(older_than=timedelta(days=older_than_days))  # type: ignore[attr-defined]
    console.print(
        f"Removed {removed} permanently-failed observation(s) "
        f"older than {older_than_days} day(s)."
    )


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
    calibration_recorder: object | None = None,
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
            calibration_recorder=calibration_recorder,  # type: ignore[arg-type]
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
        calibration_recorder=calibration_recorder,  # type: ignore[arg-type]
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
