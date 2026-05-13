"""Tests for the preflight checks + `esther doctor` CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.utils.preflight import (
    CheckResult,
    CheckStatus,
    PreflightError,
    PreflightReport,
    check_alpaca_credentials,
    check_alpaca_sdk_installed,
    check_env_file,
    check_paper_url,
    check_sentiment_dependencies,
    first_fatal,
    init_env_from_example,
    require_live_ok,
    run_preflight,
)

# Many tests below assume the SDK check is OK. If alpaca-py isn't installed
# in the test environment (as in this sandbox), we patch the SDK check to
# return OK so the *other* logic under test isn't masked by an unrelated FAIL.
_ALPACA_SDK_INSTALLED = importlib.util.find_spec("alpaca") is not None


def _force_sdk_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    if _ALPACA_SDK_INSTALLED:
        return
    monkeypatch.setattr(
        "src.utils.preflight.check_alpaca_sdk_installed",
        lambda: CheckResult(
            name="alpaca-py SDK",
            status=CheckStatus.OK,
            detail="patched OK in tests",
        ),
    )


def _settings(**overrides: Any) -> Settings:
    """Build a Settings object with overridable defaults, bypassing .env."""
    defaults: dict[str, Any] = {
        "alpaca_api_key": "real-paper-key-1234567890",
        "alpaca_api_secret": "real-paper-secret-1234567890",
        "alpaca_base_url": "https://paper-api.alpaca.markets",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def test_env_file_ok_when_present(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("ALPACA_API_KEY=x", encoding="utf-8")
    r = check_env_file(tmp_path)
    assert r.status == CheckStatus.OK


def test_env_file_warn_when_missing(tmp_path: Path) -> None:
    r = check_env_file(tmp_path)
    assert r.status == CheckStatus.WARN
    assert r.hint is not None


def test_credentials_fail_on_placeholder() -> None:
    s = _settings(alpaca_api_key="your_alpaca_key_here", alpaca_api_secret="real-secret-1234567890")
    r = check_alpaca_credentials(s)
    assert r.status == CheckStatus.FAIL
    assert "placeholder" in r.detail.lower()


def test_credentials_fail_on_test_value() -> None:
    """Test fixtures use test-key/test-secret; this should also flag."""
    s = _settings(alpaca_api_key="test-key", alpaca_api_secret="test-secret")
    r = check_alpaca_credentials(s)
    assert r.status == CheckStatus.FAIL


def test_credentials_warn_on_short_value() -> None:
    s = _settings(alpaca_api_key="short", alpaca_api_secret="alsoshrt")
    r = check_alpaca_credentials(s)
    assert r.status == CheckStatus.WARN


def test_credentials_ok_on_realistic_values() -> None:
    s = _settings()  # defaults are realistic-length
    r = check_alpaca_credentials(s)
    assert r.status == CheckStatus.OK


def test_paper_url_ok() -> None:
    s = _settings(alpaca_base_url="https://paper-api.alpaca.markets")
    r = check_paper_url(s)
    assert r.status == CheckStatus.OK


def test_paper_url_fail_on_live() -> None:
    s = _settings(alpaca_base_url="https://api.alpaca.markets")
    r = check_paper_url(s)
    assert r.status == CheckStatus.FAIL
    assert "refuses" in (r.hint or "")


def test_paper_url_warn_on_non_canonical_paper() -> None:
    s = _settings(alpaca_base_url="https://staging-paper-api.alpaca.markets")
    r = check_paper_url(s)
    assert r.status == CheckStatus.WARN


def test_alpaca_sdk_check_returns_a_result() -> None:
    """Just confirm the function runs and produces a usable result.
    The actual status depends on whether `alpaca` is installed in the test env."""
    r = check_alpaca_sdk_installed()
    assert r.status in {CheckStatus.OK, CheckStatus.FAIL}


def test_sentiment_deps_check_returns_a_result() -> None:
    r = check_sentiment_dependencies()
    assert r.status in {CheckStatus.OK, CheckStatus.WARN}


# ---------------------------------------------------------------------------
# Composed report
# ---------------------------------------------------------------------------


def test_run_preflight_offline_produces_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_sdk_ok(monkeypatch)
    s = _settings()
    report = run_preflight(s, project_root=tmp_path, online=False)
    assert isinstance(report, PreflightReport)
    names = {r.name for r in report.results}
    assert "Alpaca credentials" in names
    assert "Paper trading URL" in names
    # `online=False` skips the live check entirely (no entry added).
    assert "Alpaca connectivity" not in names


def test_run_preflight_skips_online_when_creds_bad(tmp_path: Path) -> None:
    s = _settings(alpaca_api_key="your_alpaca_key_here")
    report = run_preflight(s, project_root=tmp_path, online=True)
    online = next(r for r in report.results if r.name == "Alpaca connectivity")
    assert online.status == CheckStatus.WARN
    assert "skipped" in online.detail


def test_run_preflight_hits_clock_when_creds_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_sdk_ok(monkeypatch)
    s = _settings()
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"content-type": "application/json"}
    fake_response.json = MagicMock(return_value={"is_open": True})
    monkeypatch.setattr(
        "src.utils.preflight.httpx.get", MagicMock(return_value=fake_response)
    )
    report = run_preflight(s, project_root=tmp_path, online=True)
    online = next(r for r in report.results if r.name == "Alpaca connectivity")
    assert online.status == CheckStatus.OK
    assert "market_open=True" in online.detail


def test_run_preflight_flags_auth_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_sdk_ok(monkeypatch)
    s = _settings()
    fake_response = MagicMock()
    fake_response.status_code = 401
    monkeypatch.setattr(
        "src.utils.preflight.httpx.get", MagicMock(return_value=fake_response)
    )
    report = run_preflight(s, project_root=tmp_path, online=True)
    online = next(r for r in report.results if r.name == "Alpaca connectivity")
    assert online.status == CheckStatus.FAIL
    assert "401" in online.detail


def test_first_fatal_returns_first_fail() -> None:
    s = _settings(
        alpaca_api_key="your_alpaca_key_here",
        alpaca_base_url="https://api.alpaca.markets",
    )
    report = run_preflight(s, online=False)
    fatal = first_fatal(report)
    assert fatal is not None
    assert fatal.fatal


def test_first_fatal_none_when_clean() -> None:
    s = _settings()
    report = run_preflight(s, online=False)
    # may have WARNs (e.g. .env missing in tmp), but no FAILs
    assert first_fatal(report) is None or all(
        f.name not in {"Alpaca credentials", "Paper trading URL"} for f in report.fatal_failures
    )


def test_require_live_ok_raises_on_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_sdk_ok(monkeypatch)
    s = _settings(alpaca_base_url="https://api.alpaca.markets")
    with pytest.raises(PreflightError) as exc:
        require_live_ok(s, online=False)
    assert exc.value.first.name == "Paper trading URL"
    assert isinstance(exc.value.report, PreflightReport)


def test_require_live_ok_returns_report_when_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_sdk_ok(monkeypatch)
    s = _settings()
    report = require_live_ok(s, online=False)
    assert isinstance(report, PreflightReport)


# ---------------------------------------------------------------------------
# init_env_from_example
# ---------------------------------------------------------------------------


def test_init_env_copies_example(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("ALPACA_API_KEY=your_alpaca_key_here\n", encoding="utf-8")
    dst = init_env_from_example(tmp_path)
    assert dst == tmp_path / ".env"
    assert "your_alpaca_key_here" in dst.read_text(encoding="utf-8")


def test_init_env_refuses_overwrite_without_force(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("x", encoding="utf-8")
    (tmp_path / ".env").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        init_env_from_example(tmp_path)


def test_init_env_force_overwrites(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("new", encoding="utf-8")
    (tmp_path / ".env").write_text("old", encoding="utf-8")
    dst = init_env_from_example(tmp_path, force=True)
    assert dst.read_text(encoding="utf-8") == "new"


def test_init_env_missing_template_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        init_env_from_example(tmp_path)


# ---------------------------------------------------------------------------
# `esther doctor` CLI
# ---------------------------------------------------------------------------


def test_doctor_cli_runs_offline() -> None:
    """`esther doctor --no-online` should not require network access."""
    from click.testing import CliRunner

    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["doctor", "--no-online"])
    # Exit code 2 if a fatal check fails (test env uses placeholder keys),
    # 0 if all green. Either way it shouldn't raise.
    assert res.exit_code in (0, 2)
    # Should always print the diagnostic table.
    assert "preflight" in res.output.lower()


def test_doctor_cli_init_env_creates_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`esther doctor --init-env` should create .env when missing."""
    from click.testing import CliRunner

    # Stage a .env.example in tmp_path and point settings at it.
    (tmp_path / ".env.example").write_text(
        "ALPACA_API_KEY=your_alpaca_key_here\n"
        "ALPACA_API_SECRET=your_alpaca_secret_here\n"
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n",
        encoding="utf-8",
    )
    # Patch PROJECT_ROOT so the helper writes into tmp_path.
    monkeypatch.setattr("src.utils.preflight.Path", Path)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    # Force settings.project_root to the tmp path
    monkeypatch.setattr(
        settings_mod.Settings, "project_root", tmp_path, raising=False
    )
    s = settings_mod.get_settings()
    s.__dict__["project_root"] = tmp_path  # bypass model frozen-ness

    runner = CliRunner()
    from src.main import cli

    res = runner.invoke(cli, ["doctor", "--init-env", "--no-online"])
    assert (tmp_path / ".env").exists()
    assert res.exit_code in (0, 2)
