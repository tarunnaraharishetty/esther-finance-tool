"""Coverage for the `esther backfill` CLI command.

We only exercise the planning path (``--dry-run``) and the paper-feed
guard here. The live fetch is exercised by ``test_alpaca_client`` /
``test_news_ingestion`` against the underlying services.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner


def _ensure_paper_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a paper-URL config so ``is_paper_trading`` is True regardless
    of the user's local .env. Tests that exercise the guard explicitly
    flip this back to a live URL.
    """
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()


def test_backfill_dry_run_prints_plan_without_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--dry-run`` should print the symbol/date plan and exit cleanly
    without ever instantiating ``MarketDataService`` or the news source.
    We assert by patching both constructors to raise — if the plan path
    touches them, the test fails loudly.
    """
    _ensure_paper_env(monkeypatch)

    def _explode(*_a: object, **_k: object) -> None:
        raise AssertionError("dry-run path should not construct fetch services")

    monkeypatch.setattr("src.data.market_data.MarketDataService.__init__", _explode)
    monkeypatch.setattr("src.data.news_ingestion.get_news_source", _explode)

    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["backfill", "-s", "AAPL", "-s", "MSFT", "--days", "30", "--dry-run"],
    )
    assert res.exit_code == 0, res.output
    # Plan table headers + both symbols appear in the output.
    assert "Backfill plan" in res.output
    assert "AAPL" in res.output and "MSFT" in res.output
    # Window reflects the --days flag.
    assert "30d" in res.output
    # Hint about re-running without --dry-run.
    assert "without --dry-run" in res.output


def test_backfill_dry_run_skip_news_marks_news_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``--skip-news --dry-run`` the news column should read
    ``skipped`` instead of a real window range."""
    _ensure_paper_env(monkeypatch)

    def _explode(*_a: object, **_k: object) -> None:
        raise AssertionError("dry-run path should not construct fetch services")

    monkeypatch.setattr("src.data.market_data.MarketDataService.__init__", _explode)
    monkeypatch.setattr("src.data.news_ingestion.get_news_source", _explode)

    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(
        cli, ["backfill", "-s", "NVDA", "--skip-news", "--dry-run"]
    )
    assert res.exit_code == 0, res.output
    assert "skipped" in res.output
    assert "bars only" in res.output


def test_backfill_refuses_live_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with --dry-run, the paper-feed guard fires before any plan
    print so a misconfigured live URL gets caught early."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()

    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["backfill", "--dry-run"])
    assert res.exit_code == 1
    assert "not paper" in res.output
