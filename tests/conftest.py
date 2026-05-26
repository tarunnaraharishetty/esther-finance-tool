"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _safe_test_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force tests to use paper-trading credentials.

    Guards against accidentally hitting live endpoints if a developer's local
    .env is loaded.
    """
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    # Neutralize any real Anthropic key from the developer's local .env so
    # tests that assert "no key configured" stay deterministic. Tests that
    # need a key should setenv explicitly.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    # Disable the calibration store by default — tests that need it
    # construct a tmp_path-rooted CalibrationStore explicitly and pass
    # it to create_app. Otherwise the analyzer endpoint would write to
    # data/calibration.db under the project root from any test that
    # constructs an app.
    monkeypatch.setenv("CALIBRATION_STORE_PATH", "")
    # Disable the user store + auth gate by default. Tests that exercise
    # auth (test_api_auth, test_api_watchlist, and the new stream-auth
    # tests) construct a tmp_path-rooted UserStore explicitly and pass
    # it to create_app. Without this, the default user_store_path under
    # the repo root would be wired and the auth gate middleware would
    # 401 every API-contract test that doesn't sign up first.
    monkeypatch.setenv("USER_STORE_PATH", "")
    # Bust the lru_cache on Settings between tests
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    # Clear any computed sector-median overrides so a test that
    # installs them via refresh_computed_medians can't leak the
    # override into an unrelated test that expects the static seed.
    from src.intelligence.analyzer.sector_medians import register_computed_medians

    register_computed_medians(None)
    yield
    settings_mod.get_settings.cache_clear()
    register_computed_medians(None)


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """100 deterministic daily bars for SYNTH symbol."""
    rng = np.random.default_rng(seed=42)
    n = 100
    base = 100.0
    rets = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = base * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = np.concatenate([[base], close[:-1]])
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    idx = pd.date_range(end=datetime.now(UTC), periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def synthetic_returns() -> pd.Series:
    # Strong positive drift so the test asserting "Sharpe > 0" isn't fragile
    # to a particular RNG draw.
    rng = np.random.default_rng(seed=7)
    return pd.Series(rng.normal(loc=0.005, scale=0.005, size=252))


@pytest.fixture
def equity_curve() -> pd.Series:
    rng = np.random.default_rng(seed=11)
    rets = rng.normal(loc=0.0004, scale=0.01, size=252)
    return pd.Series(100_000.0 * np.exp(np.cumsum(rets)))
