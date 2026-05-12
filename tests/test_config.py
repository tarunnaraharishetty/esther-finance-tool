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

