"""Coverage for the ``esther serve`` CLI command.

We mock ``uvicorn.run`` so the test never actually binds a socket;
the goal is to verify the command exists, accepts its flags, wires
the FastAPI app to a controller, and respects the paper-feed /
preflight gates.
"""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner


def _set_paper_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the settings module to a paper URL so the live-mode guard
    inside ``_build_controller`` doesn't bail before we get to the
    serve-specific assertions."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()


def _stub_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Replace ``uvicorn.run`` with a recorder.

    Returns a list that will be populated with one dict per call,
    capturing ``(app, host, port, kwargs)``. Tests inspect it to assert
    the right wiring without needing a real HTTP server.
    """
    calls: list[dict[str, Any]] = []

    def _record(app: object, **kw: Any) -> None:
        calls.append({"app": app, **kw})

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", _record)
    return calls


def test_serve_mock_boots_uvicorn_with_factory_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``esther serve --mock`` should build a FastAPI app via the
    factory and hand it to uvicorn with the configured host/port."""
    _set_paper_env(monkeypatch)
    calls = _stub_uvicorn(monkeypatch)

    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["serve", "--mock", "--host", "127.0.0.1", "--port", "8123", "-s", "AAPL", "-s", "MSFT"],
    )
    assert res.exit_code == 0, res.output
    assert len(calls) == 1, "uvicorn.run should be called exactly once"
    call = calls[0]
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 8123
    # The app must expose the two expected routes.
    paths = {route.path for route in call["app"].routes}  # type: ignore[attr-defined]
    assert "/api/health" in paths
    assert "/api/snapshot" in paths


def test_serve_warns_on_non_loopback_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binding to a non-loopback host should print a warning because
    the API ships with no auth in Phase 0."""
    _set_paper_env(monkeypatch)
    _stub_uvicorn(monkeypatch)

    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["serve", "--mock", "--host", "0.0.0.0"])
    assert res.exit_code == 0, res.output
    assert "warning" in res.output.lower()
    assert "no auth" in res.output


def test_serve_loopback_bind_does_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loopback bind is the safe default — no warning should appear."""
    _set_paper_env(monkeypatch)
    _stub_uvicorn(monkeypatch)

    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["serve", "--mock"])
    assert res.exit_code == 0, res.output
    assert "warning" not in res.output.lower()


def test_serve_appears_in_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """``esther --help`` should list the new subcommand alongside
    dashboard / backfill / etc."""
    _set_paper_env(monkeypatch)

    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["serve", "--help"])
    assert res.exit_code == 0
    assert "Launch the read-only HTTP/JSON API" in res.output
    # All advertised flags should appear in --help output.
    for flag in ("--host", "--port", "--mock", "--no-sentiment", "--skip-preflight"):
        assert flag in res.output, f"missing flag in help: {flag}"
