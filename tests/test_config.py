"""Settings sanity checks."""

from __future__ import annotations

import pytest

from src.config import get_settings
from src.config.settings import AlpacaDataFeed, AppEnv


def test_settings_load_from_env() -> None:
    s = get_settings()
    assert s.alpaca_api_key.get_secret_value() == "test-key"
    assert s.app_env == AppEnv.DEV
    assert s.is_paper_trading is True
    assert s.alpaca_data_feed == AlpacaDataFeed.IEX


def test_max_position_pct_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_POSITION_PCT", "1.5")
    from src.config import settings as m

    m.get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()
