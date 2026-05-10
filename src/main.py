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
def run() -> None:
    """Start the main trading loop. (Stub — fill in once strategies land.)"""
    log = get_logger("main")
    log.info("esther.start", version=__version__)
    console.print(
        "[yellow]run loop not yet implemented — wire up data → strategy → execution here[/yellow]"
    )


def _print_banner(settings: object) -> None:
    body = (
        f"[bold]Esther[/bold] v{__version__}\n"
        f"env: {getattr(settings, 'app_env').value}  "
        f"paper: {'yes' if getattr(settings, 'is_paper_trading') else 'NO'}\n\n"
        "Commands:\n"
        "  esther status     show effective config\n"
        "  esther initdb     create database tables\n"
        "  esther run        start trading loop\n"
    )
    console.print(Panel(body, title="quant trading platform", border_style="cyan"))


if __name__ == "__main__":
    sys.exit(cli())
