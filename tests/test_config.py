"""Settings sanity checks."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from src.config import get_settings
from src.config.settings import AlpacaDataFeed, AppEnv, Settings


def test_settings_load_from_env() -> None:
    s = get_settings()
    assert s.alpaca_api_key.get_secret_value() == "test-key"
    assert s.app_env == AppEnv.DEV
    assert s.is_paper_trading is True
    assert s.alpaca_data_feed == AlpacaDataFeed.IEX


def test_dashboard_cadence_defaults_match_legacy_constants() -> None:
    """The previously-hardcoded refresh + burst cadence (5.0s / 1.5s)
    now lives in Settings. Defaults must stay the same so existing
    behavior is preserved when callers don't override."""
    s = get_settings()
    assert s.dashboard_refresh_seconds == 5.0
    assert s.dashboard_burst_seconds == 1.5


def test_dashboard_columns_default_lists_every_column() -> None:
    """The default column list must enumerate every column the
    dashboard knows about. If someone adds a new column to
    _COLUMN_DEFS without updating the default, the dashboard would
    silently hide it on fresh installs — catch that here."""
    from src.dashboard.app import _COLUMN_NAMES

    s = get_settings()
    assert set(s.dashboard_columns) == _COLUMN_NAMES


def test_dashboard_columns_rejects_unknown_names() -> None:
    """A typo in the env-config should fail fast at startup rather
    than silently hiding every column or crashing later in compose."""
    with pytest.raises(Exception, match="unknown name"):
        Settings(
            alpaca_api_key="x",  # type: ignore[arg-type]
            alpaca_api_secret="y",  # type: ignore[arg-type]
            dashboard_columns=["SYM", "NOPE", "PRICE"],
        )


# ---------------------------------------------------------------------------
# Production-safety model validator
# ---------------------------------------------------------------------------

# A non-default secret that lets the production validator pass on each
# of the field-specific tests below. ``itsdangerous`` accepts arbitrary
# bytes so any non-placeholder string is fine here.
_PROD_SECRET = "n" * 64


def _prod_kwargs(**overrides: object) -> dict[str, object]:
    """Build a minimum prod-mode Settings kwargs dict.

    Each individual test below overrides exactly the one field it
    cares about (e.g. flip ``trusted_hosts`` back to ``["*"]``) so
    the validator failure can be attributed unambiguously.
    """
    base: dict[str, object] = {
        "alpaca_api_key": "x",
        "alpaca_api_secret": "y",
        "app_env": AppEnv.PROD,
        "session_secret_key": SecretStr(_PROD_SECRET),
        "secure_cookies": True,
        "trusted_hosts": ["esther.example.com"],
        "cors_origins": ["https://esther.example.com"],
    }
    base.update(overrides)
    return base


def test_prod_safety_rejects_default_session_secret() -> None:
    """The committed dev placeholder must not survive into prod."""
    with pytest.raises(ValidationError, match="SESSION_SECRET_KEY"):
        Settings(**_prod_kwargs(session_secret_key=SecretStr(
            "esther-dev-session-key-change-in-production"
        )))  # type: ignore[arg-type]


def test_prod_safety_requires_secure_cookies() -> None:
    """Plain-HTTP cookies in prod = trivial session theft on any LAN."""
    with pytest.raises(ValidationError, match="SECURE_COOKIES"):
        Settings(**_prod_kwargs(secure_cookies=False))  # type: ignore[arg-type]


def test_prod_safety_rejects_wildcard_trusted_hosts() -> None:
    """Host-header injection (cache poisoning, password-reset link
    forgery) is unmitigated unless we pin the hostnames the server
    answers for."""
    with pytest.raises(ValidationError, match="TRUSTED_HOSTS"):
        Settings(**_prod_kwargs(trusted_hosts=["*"]))  # type: ignore[arg-type]


def test_prod_safety_rejects_wildcard_cors() -> None:
    """FastAPI's CORS middleware silently breaks ``allow_credentials=True``
    paired with a wildcard origin — the browser receives ``*`` as the
    Access-Control-Allow-Origin and refuses the response. Fail fast
    at startup instead of producing a frontend that silently can't
    reach the API."""
    with pytest.raises(ValidationError, match="CORS_ORIGINS"):
        Settings(**_prod_kwargs(cors_origins=["*"]))  # type: ignore[arg-type]


def test_prod_safety_passes_with_all_overrides_set() -> None:
    """The clean prod config the deploy docs describe must validate."""
    s = Settings(**_prod_kwargs())  # type: ignore[arg-type]
    assert s.app_env == AppEnv.PROD
    assert s.is_prod is True


def test_trust_proxy_headers_defaults_false() -> None:
    """XFF is attacker-controlled when the server isn't behind a
    trusted proxy. Default must be opt-in, not opt-out."""
    s = get_settings()
    assert s.trust_proxy_headers is False


def test_trusted_hosts_parses_comma_separated_env() -> None:
    """Deploy hosts pass env vars as plain strings — comma-separated
    must round-trip through the validator alongside JSON lists."""
    parsed = Settings(
        alpaca_api_key="x",  # type: ignore[arg-type]
        alpaca_api_secret="y",  # type: ignore[arg-type]
        trusted_hosts="esther.example.com, api.esther.example.com",  # type: ignore[arg-type]
    )
    assert parsed.trusted_hosts == [
        "esther.example.com",
        "api.esther.example.com",
    ]
