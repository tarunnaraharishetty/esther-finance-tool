"""Filesystem cache for backfilled bars and news.

Bars live in ``{symbol}_bars_{timeframe}.csv`` with a UTC datetime index;
news lives in ``{symbol}_news.json`` as a list of serialized
:class:`~src.data.models.NewsArticle`. The cache directory is
``Settings.data_dir / "cache"`` (typically ``data/cache/``).

This module is intentionally tiny — no ORM, no schema, no migrations.
A trader can ``cat`` either file and see the data. If a cache layout
ever needs to evolve, the user can ``rm -rf data/cache`` and re-run
``esther backfill``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.config import get_settings
from src.data.models import NewsArticle, TimeFrame

if TYPE_CHECKING:
    pass

_BARS_COLUMNS = ("open", "high", "low", "close", "volume")


def cache_dir() -> Path:
    """Resolve the cache directory from settings. Created on first write."""
    return get_settings().data_dir / "cache"


def bars_path(symbol: str, timeframe: TimeFrame) -> Path:
    return cache_dir() / f"{symbol.upper()}_bars_{timeframe.value}.csv"


def news_path(symbol: str) -> Path:
    return cache_dir() / f"{symbol.upper()}_news.json"


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------


def write_bars(symbol: str, timeframe: TimeFrame, df: pd.DataFrame) -> Path:
    """Overwrite the cache file for one (symbol, timeframe).

    The dataframe must have a datetime index and the standard OHLCV columns;
    extra columns are written too but downstream code only reads OHLCV.
    """
    if df.empty:
        raise ValueError(f"refusing to write empty bars cache for {symbol}")
    missing = set(_BARS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"bars df missing columns: {sorted(missing)}")
    path = bars_path(symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="timestamp")
    return path


def read_bars(symbol: str, timeframe: TimeFrame) -> pd.DataFrame | None:
    """Read cached bars for ``(symbol, timeframe)``. Returns None if absent."""
    path = bars_path(symbol, timeframe)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    # Coerce timezones consistently — Alpaca returns UTC.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


def write_news(symbol: str, articles: list[NewsArticle]) -> Path:
    """Overwrite the news cache for one symbol."""
    path = news_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [a.model_dump(mode="json") for a in articles]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_news(symbol: str) -> list[NewsArticle]:
    """Read cached news for ``symbol``. Returns ``[]`` if absent."""
    path = news_path(symbol)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [NewsArticle(**item) for item in raw]


# ---------------------------------------------------------------------------
# Merge helper for the dashboard hot path
# ---------------------------------------------------------------------------


def merge_with_cache(
    symbol: str, timeframe: TimeFrame, live: pd.DataFrame
) -> pd.DataFrame:
    """Prepend cached bars to a live bar window, keeping the live values
    on overlapping timestamps (cache may be stale).

    Used by the dashboard controller so the indicator engine sees a longer
    warm-up window without paying for a deeper Alpaca request every tick.
    """
    cached = read_bars(symbol, timeframe)
    if cached is None or cached.empty:
        return live
    if live.empty:
        return cached
    # ``combine_first`` keeps caller's non-null values, fills from the other.
    # live wins on overlap; cached fills the prefix gap.
    return live.combine_first(cached).sort_index()
