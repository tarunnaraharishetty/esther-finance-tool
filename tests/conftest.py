"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

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
    # Bust the lru_cache on Settings between tests
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield
    settings_mod.get_settings.cache_clear()


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
