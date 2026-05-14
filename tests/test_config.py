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
