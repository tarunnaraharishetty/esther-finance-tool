"""Settings sanity checks."""

from __future__ import annotations

from src.config import get_settings
from src.config.settings import AlpacaDataFeed, AppEnv


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
    import pytest

    from src.config.settings import Settings

    with pytest.raises(Exception, match="unknown name"):
        Settings(
            alpaca_api_key="x",  # type: ignore[arg-type]
            alpaca_api_secret="y",  # type: ignore[arg-type]
            dashboard_columns=["SYM", "NOPE", "PRICE"],
        )
