"""Preflight checks: verify the local environment can talk to Alpaca paper
and run the sentiment pipeline before launching a long-running command.

Each check is small, side-effect-free (unless explicitly marked online),
and returns a :class:`CheckResult` with a clear remediation hint when it
fails. Checks are composable so `esther doctor` and `esther dashboard`
can both rely on them.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx

from src.config import Settings, get_settings

# These are the placeholder strings shipped in .env.example. If a real env
# still contains them, the user forgot to fill the file in.
_PLACEHOLDER_KEYS = {
    "your_alpaca_key_here",
    "your_alpaca_secret_here",
    "your_news_api_key_here",
    "",
    "test-key",
    "test-secret",
}

# Hard-coded canonical paper URL — used for defence-in-depth in the live
# preflight (we already trust ``Settings.is_paper_trading``, but this is
# explicit and ungameable).
PAPER_URL = "https://paper-api.alpaca.markets"


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single preflight check."""

    name: str
    status: CheckStatus
    detail: str
    hint: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == CheckStatus.OK

    @property
    def fatal(self) -> bool:
        return self.status == CheckStatus.FAIL


@dataclass
class PreflightReport:
    """All check results from one run. ``ok`` only when zero FAILs."""

    results: list[CheckResult]

    @property
    def ok(self) -> bool:
        return not any(r.fatal for r in self.results)

    @property
    def fatal_failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.fatal]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_env_file(project_root: Path) -> CheckResult:
    """`.env` exists in the project root."""
    env_path = project_root / ".env"
    if env_path.exists():
        return CheckResult(
            name=".env file",
            status=CheckStatus.OK,
            detail=f"found at {env_path}",
        )
    # Not strictly required if env vars are exported elsewhere, so warn.
    return CheckResult(
        name=".env file",
        status=CheckStatus.WARN,
        detail=f"no .env at {env_path}",
        hint=(
            "Copy .env.example -> .env and fill in your Alpaca paper keys. "
            "(You can also export the env vars in your shell instead.)"
        ),
    )


def check_alpaca_credentials(settings: Settings) -> CheckResult:
    """API key + secret are present and not the placeholder values."""
    key = settings.alpaca_api_key.get_secret_value().strip()
    secret = settings.alpaca_api_secret.get_secret_value().strip()
    if not key or not secret:
        return CheckResult(
            name="Alpaca credentials",
            status=CheckStatus.FAIL,
            detail="ALPACA_API_KEY or ALPACA_API_SECRET is empty",
            hint=(
                "Generate paper-trading keys at "
                "https://app.alpaca.markets/paper/dashboard/overview "
                "-> 'View' next to 'Your API Keys' -> 'Generate New Key'. "
                "Paste both into .env."
            ),
        )
    if key in _PLACEHOLDER_KEYS or secret in _PLACEHOLDER_KEYS:
        return CheckResult(
            name="Alpaca credentials",
            status=CheckStatus.FAIL,
            detail="keys still contain placeholder values from .env.example",
            hint="Edit .env and replace 'your_alpaca_key_here' / 'your_alpaca_secret_here'.",
        )
    # Quick sanity check on length — Alpaca paper keys are typically ~20+ chars.
    if len(key) < 16 or len(secret) < 16:
        return CheckResult(
            name="Alpaca credentials",
            status=CheckStatus.WARN,
            detail=f"keys look short (key={len(key)} chars, secret={len(secret)} chars)",
            hint="Double-check you pasted the full key / secret from Alpaca.",
        )
    return CheckResult(
        name="Alpaca credentials",
        status=CheckStatus.OK,
        detail=f"key length {len(key)}, secret length {len(secret)}",
    )


def check_paper_url(settings: Settings) -> CheckResult:
    """Hard-stop if ALPACA_BASE_URL isn't the paper endpoint."""
    url = settings.alpaca_base_url.rstrip("/")
    if url == PAPER_URL:
        return CheckResult(
            name="Paper trading URL",
            status=CheckStatus.OK,
            detail=f"using {url}",
        )
    if "paper" in url.lower():
        return CheckResult(
            name="Paper trading URL",
            status=CheckStatus.WARN,
            detail=f"non-canonical paper URL: {url}",
            hint=f"Expected {PAPER_URL}. Make sure this is still a paper endpoint.",
        )
    return CheckResult(
        name="Paper trading URL",
        status=CheckStatus.FAIL,
        detail=f"non-paper URL configured: {url}",
        hint=(
            f"This tool refuses to run against live endpoints. Set "
            f"ALPACA_BASE_URL={PAPER_URL} in .env."
        ),
    )


def check_alpaca_sdk_installed() -> CheckResult:
    """`alpaca-py` package is importable."""
    if importlib.util.find_spec("alpaca") is not None:
        return CheckResult(
            name="alpaca-py SDK",
            status=CheckStatus.OK,
            detail="alpaca package is importable",
        )
    return CheckResult(
        name="alpaca-py SDK",
        status=CheckStatus.FAIL,
        detail="alpaca-py is not installed in this Python environment",
        hint=(
            "Install dependencies: `pip install -e \".[dev]\"` from the project root. "
            "Make sure you're using the same Python that has the venv activated."
        ),
    )


def check_sentiment_dependencies() -> CheckResult:
    """`transformers` + `torch` are importable. Loading the model is *not* done here."""
    missing: list[str] = []
    for pkg in ("transformers", "torch"):
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if not missing:
        return CheckResult(
            name="FinBERT dependencies",
            status=CheckStatus.OK,
            detail="transformers + torch importable",
        )
    return CheckResult(
        name="FinBERT dependencies",
        status=CheckStatus.WARN,
        detail=f"missing: {', '.join(missing)}",
        hint=(
            "Install with `pip install -e \".[dev]\"`. "
            "Or run the dashboard with --no-sentiment to skip FinBERT."
        ),
    )


def check_alpaca_connection(settings: Settings, *, timeout: float = 10.0) -> CheckResult:
    """Online: hit /v2/clock to confirm the keys actually authenticate.

    Network-dependent. Safe — /v2/clock is read-only and doesn't move money.
    """
    url = f"{settings.alpaca_base_url.rstrip('/')}/v2/clock"
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key.get_secret_value(),
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret.get_secret_value(),
    }
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        return CheckResult(
            name="Alpaca connectivity",
            status=CheckStatus.FAIL,
            detail=f"network error: {e}",
            hint="Check your internet connection and ALPACA_BASE_URL.",
        )

    if response.status_code == 200:
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        is_open = body.get("is_open")
        return CheckResult(
            name="Alpaca connectivity",
            status=CheckStatus.OK,
            detail=f"authenticated · market_open={is_open}",
        )
    if response.status_code in (401, 403):
        return CheckResult(
            name="Alpaca connectivity",
            status=CheckStatus.FAIL,
            detail=f"HTTP {response.status_code} — keys rejected by Alpaca",
            hint=(
                "Verify ALPACA_API_KEY / ALPACA_API_SECRET in .env match the *paper* "
                "(not live) keys from https://app.alpaca.markets/paper/dashboard/overview"
            ),
        )
    return CheckResult(
        name="Alpaca connectivity",
        status=CheckStatus.WARN,
        detail=f"unexpected HTTP {response.status_code}",
        hint="Alpaca returned a non-2xx, non-auth response. Check Alpaca status page.",
    )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def run_preflight(
    settings: Settings | None = None,
    *,
    project_root: Path | None = None,
    online: bool = True,
) -> PreflightReport:
    """Run every check and return a report.

    Set ``online=False`` to skip the network call (useful in CI / offline)."""
    s = settings or get_settings()
    root = project_root or s.project_root
    results: list[CheckResult] = [
        check_env_file(root),
        check_alpaca_sdk_installed(),
        check_alpaca_credentials(s),
        check_paper_url(s),
        check_sentiment_dependencies(),
    ]
    if online:
        # Only attempt the live check if credentials look usable; otherwise
        # the 401 noise drowns out the real problem.
        creds_ok = next(
            (r for r in results if r.name == "Alpaca credentials"), None
        )
        url_ok = next((r for r in results if r.name == "Paper trading URL"), None)
        if creds_ok and creds_ok.ok and url_ok and url_ok.ok:
            results.append(check_alpaca_connection(s))
        else:
            results.append(
                CheckResult(
                    name="Alpaca connectivity",
                    status=CheckStatus.WARN,
                    detail="skipped — fix credentials/URL first",
                )
            )
    return PreflightReport(results=results)


def first_fatal(report: PreflightReport) -> CheckResult | None:
    """Return the first FAIL check, or None if all clear."""
    failures = report.fatal_failures
    return failures[0] if failures else None


# ---------------------------------------------------------------------------
# Public helper for use inside other commands
# ---------------------------------------------------------------------------


def require_live_ok(settings: Settings | None = None, *, online: bool = True) -> PreflightReport:
    """Run preflight and raise :class:`PreflightError` on any fatal check.

    Use in CLI handlers to fail fast with a structured exception instead of
    letting the SDK throw an opaque error deep in the stack.
    """
    report = run_preflight(settings, online=online)
    fatal = first_fatal(report)
    if fatal is not None:
        raise PreflightError(report=report, first=fatal)
    return report


class PreflightError(RuntimeError):
    """Raised by :func:`require_live_ok` when a fatal check fails."""

    def __init__(self, report: PreflightReport, first: CheckResult) -> None:
        super().__init__(f"preflight failed: {first.name} — {first.detail}")
        self.report = report
        self.first = first


# ---------------------------------------------------------------------------
# .env scaffolding helper (used by `esther doctor --init-env`)
# ---------------------------------------------------------------------------


def init_env_from_example(project_root: Path, *, force: bool = False) -> Path:
    """Copy .env.example -> .env. Returns the destination path.

    Raises FileExistsError if .env exists and ``force`` is False.
    """
    src = project_root / ".env.example"
    dst = project_root / ".env"
    if not src.exists():
        raise FileNotFoundError(f"missing template: {src}")
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} already exists (pass --force to overwrite)")
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    # Lock down on POSIX; on Windows the OS already gates this to the user.
    if os.name != "nt":
        try:
            dst.chmod(0o600)
        except OSError:
            pass
    return dst
