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
def backfill(
    symbols: tuple[str, ...],
    days: int,
    news_hours: int,
    skip_news: bool,
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
            except Exception as e:  # noqa: BLE001 — per-symbol surface
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
        "[dim]decision-support output — no orders are submitted.[/dim]"
    )


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

            def score_text(self, text: str) -> SentimentScore:  # type: ignore[override]
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

            def score_article(self, article: object) -> SentimentScore:  # type: ignore[override]
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

        engine.sentiment_analyzer = _Neutral()

    async def _gather() -> tuple[object, list[object]]:
        bars = await market.get_bars([sym], TimeFrame.DAY_1, bars_start, end)
        df = market.to_dataframe(bars)
        if df.empty:
            raise RuntimeError(f"no bars returned for {sym}")
        news = (
            []
            if no_sentiment
            else await news_source.fetch([sym], news_start, end, limit=20)
        )
        rec = engine.recommend(sym, df, news=news)
        return rec, list(news)

    try:
        rec, news = asyncio.run(_gather())
    except Exception as e:  # noqa: BLE001 — top-level CLI error surface
        log.warning("summarize.fetch_failed", symbol=sym, error=str(e))
        console.print(f"[red]failed to fetch data for {sym}: {e}[/red]")
        raise click.exceptions.Exit(3) from None

    explanation = explain(
        symbol=rec.symbol,  # type: ignore[attr-defined]
        action=rec.action,  # type: ignore[attr-defined]
        confidence=rec.confidence,  # type: ignore[attr-defined]
        combined_score=rec.combined_score,  # type: ignore[attr-defined]
        indicator_scores=rec.indicator_scores,  # type: ignore[attr-defined]
        sentiment_score=rec.sentiment_score,  # type: ignore[attr-defined]
        num_news_articles=rec.num_news_articles,  # type: ignore[attr-defined]
    )

    try:
        brief = summarizer.summarize(
            explanation,
            headlines=[a.headline for a in news[:5]],
        )
    except anthropic.AuthenticationError:
        console.print(
            "[red]Anthropic auth failed — check ANTHROPIC_API_KEY in .env.[/red]"
        )
        raise click.exceptions.Exit(2) from None
    except anthropic.RateLimitError:
        console.print(
            "[yellow]Anthropic rate-limited — try again in a minute.[/yellow]"
        )
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
    console.print(
        "[dim]decision support — research only, not execution advice.[/dim]"
    )


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

            def score_text(self, text: str) -> SentimentScore:  # type: ignore[override]
                return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

            def score_article(self, article: object) -> SentimentScore:  # type: ignore[override]
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
                    a.headline
                    for a in sorted(news, key=lambda a: a.published_at, reverse=True)[:5]
                )
                rows.append(
                    _row_from_recommendation(
                        rec,
                        last_price=float(df["close"].iloc[-1]),
                        headlines=top_headlines,
                    )
                )
            except Exception as e:  # noqa: BLE001 — recap is best-effort per symbol
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
    console.print(
        "[dim]decision support — research only, not execution advice.[/dim]"
    )


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
    default=5.0,
    show_default=True,
    help="Seconds between dashboard refreshes.",
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
    refresh_seconds: float,
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
    from src.dashboard.controller import (
        DashboardController,
        MockDashboardController,
    )
    from src.intelligence.alert_prioritizer import AlertPrioritizer
    from src.intelligence.alerts import (
        AlertEngine,
        load_prioritizer_config_from_yaml,
        load_rules_from_yaml,
    )
    from src.strategy.recommendation import RecommendationEngine

    settings = get_settings()
    watchlist = list(symbols) if symbols else ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]

    # Load user-defined alert rules + prioritizer knobs from
    # config/alerts.yaml if present. Missing file = both default.
    alerts_path = settings.config_dir / "alerts.yaml"
    if alerts_path.exists():
        try:
            rules = load_rules_from_yaml(alerts_path)
            prio_config = load_prioritizer_config_from_yaml(alerts_path)
            alert_engine = AlertEngine(rules=rules)
            alert_prioritizer = AlertPrioritizer(config=prio_config)  # type: ignore[arg-type]
            console.print(
                f"[dim]loaded {len(rules)} alert rule(s) from {alerts_path}[/dim]"
            )
        except ValueError as e:
            console.print(f"[red]invalid {alerts_path}: {e}[/red]")
            raise click.exceptions.Exit(2) from None
    else:
        alert_engine = AlertEngine()
        alert_prioritizer = AlertPrioritizer()

    if mock:
        controller = MockDashboardController(
            watchlist=watchlist,
            use_sentiment=not no_sentiment,
            alert_engine=alert_engine,
            alert_prioritizer=alert_prioritizer,
        )
    else:
        if not skip_preflight:
            from src.utils.preflight import PreflightError, require_live_ok

            try:
                require_live_ok(settings, online=True)
            except PreflightError as e:
                console.print(
                    "[red bold]Cannot launch live dashboard — preflight failed.[/red bold]"
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

                def score_text(self, text: str) -> SentimentScore:  # type: ignore[override]
                    return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

                def score_article(self, article: object) -> SentimentScore:  # type: ignore[override]
                    return SentimentScore(SentimentLabel.NEUTRAL, 0.0)

            engine.sentiment_analyzer = _Neutral()
        controller = DashboardController(
            watchlist=watchlist,
            engine=engine,
            settings=settings,
            lookback_days=lookback_days,
            news_hours=news_hours,
            alert_engine=alert_engine,
            alert_prioritizer=alert_prioritizer,
        )

    # Try to build the LLM summarizer; if ANTHROPIC_API_KEY isn't set the
    # dashboard still launches but the `s` keypress will surface a clean error.
    summarizer: object | None
    try:
        from src.intelligence.llm_summary import LLMSummarizer

        summarizer = LLMSummarizer(settings=settings)
    except RuntimeError:
        summarizer = None
        console.print(
            "[yellow]ANTHROPIC_API_KEY not set — AI briefs disabled. "
            "Press `s` for setup hint.[/yellow]"
        )

    DashboardApp(
        controller,
        refresh_seconds=refresh_seconds,
        summarizer=summarizer,  # type: ignore[arg-type]
    ).run()


def _print_banner(settings: object) -> None:
    body = (
        f"[bold]Esther[/bold] v{__version__}\n"
        f"env: {getattr(settings, 'app_env').value}  "
        f"paper: {'yes' if getattr(settings, 'is_paper_trading') else 'NO'}\n\n"
        "Commands:\n"
        "  esther status      show effective config\n"
        "  esther doctor      diagnose Alpaca + FinBERT setup\n"
        "  esther initdb      create database tables\n"
        "  esther backfill    cache historical bars + news to data/cache/\n"
        "  esther recommend   show BUY/HOLD/SELL for a watchlist\n"
        "  esther summarize   AI-written brief for one symbol (needs ANTHROPIC_API_KEY)\n"
        "  esther recap       AI-written brief across the whole watchlist\n"
        "  esther dashboard   live Textual dashboard (use --mock for demo)\n"
    )
    console.print(Panel(body, title="market intelligence + decision support", border_style="cyan"))


if __name__ == "__main__":
    sys.exit(cli())
