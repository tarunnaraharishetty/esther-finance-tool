"""Tests for src.data.cache — the filesystem cache for backfilled data."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import cache
from src.data.models import NewsArticle, TimeFrame


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the cache directory to a tmp path for the test.

    Patches ``data_dir`` on the cached Settings so cache.cache_dir() resolves
    here instead of the real project ``data/`` folder.
    """
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()
    monkeypatch.setattr(s, "data_dir", tmp_path, raising=False)
    return tmp_path / "cache"


def _df(n: int = 30, start: datetime | None = None) -> pd.DataFrame:
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    idx = pd.date_range(start=start, periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=int),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------


def test_write_then_read_bars_roundtrip(cache_root: Path) -> None:
    df = _df(60)
    path = cache.write_bars("AAPL", TimeFrame.DAY_1, df)
    assert path.parent == cache_root
    assert path.name == "AAPL_bars_1Day.csv"

    loaded = cache.read_bars("AAPL", TimeFrame.DAY_1)
    assert loaded is not None
    assert len(loaded) == 60
    assert loaded.index.tz is not None  # tz preserved
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]


def test_read_bars_missing_returns_none(cache_root: Path) -> None:
    assert cache.read_bars("XXXX", TimeFrame.DAY_1) is None


def test_write_bars_uppercases_symbol(cache_root: Path) -> None:
    cache.write_bars("aapl", TimeFrame.DAY_1, _df(5))
    assert (cache_root / "AAPL_bars_1Day.csv").exists()
    # And read works case-insensitively too.
    assert cache.read_bars("aapl", TimeFrame.DAY_1) is not None


def test_write_bars_refuses_empty(cache_root: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        cache.write_bars("AAPL", TimeFrame.DAY_1, pd.DataFrame())


def test_write_bars_validates_columns(cache_root: Path) -> None:
    bad = pd.DataFrame({"open": [1, 2, 3]})  # missing high/low/close/volume
    with pytest.raises(ValueError, match="missing columns"):
        cache.write_bars("AAPL", TimeFrame.DAY_1, bad)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


def _article(article_id: str, when: datetime) -> NewsArticle:
    return NewsArticle(
        id=article_id,
        headline=f"headline {article_id}",
        source="test",
        symbols=["AAPL"],
        published_at=when,
    )


def test_write_then_read_news_roundtrip(cache_root: Path) -> None:
    when = datetime(2026, 5, 1, tzinfo=UTC)
    articles = [_article("n1", when), _article("n2", when)]
    path = cache.write_news("AAPL", articles)
    assert path.name == "AAPL_news.json"

    loaded = cache.read_news("AAPL")
    assert [a.id for a in loaded] == ["n1", "n2"]
    assert loaded[0].headline == "headline n1"


def test_read_news_missing_returns_empty(cache_root: Path) -> None:
    assert cache.read_news("XXXX") == []


def test_write_news_handles_empty_list(cache_root: Path) -> None:
    path = cache.write_news("AAPL", [])
    assert path.exists()
    assert cache.read_news("AAPL") == []


# ---------------------------------------------------------------------------
# merge_with_cache — the hot-path helper
# ---------------------------------------------------------------------------


def test_merge_no_cache_returns_live_unchanged(cache_root: Path) -> None:
    live = _df(10, start=datetime(2026, 4, 1, tzinfo=UTC))
    merged = cache.merge_with_cache("AAPL", TimeFrame.DAY_1, live)
    pd.testing.assert_frame_equal(merged, live, check_names=False)


def test_merge_prepends_cached_history(cache_root: Path) -> None:
    cached = _df(60, start=datetime(2026, 1, 1, tzinfo=UTC))
    cache.write_bars("AAPL", TimeFrame.DAY_1, cached)
    live = _df(10, start=datetime(2026, 4, 1, tzinfo=UTC))
    merged = cache.merge_with_cache("AAPL", TimeFrame.DAY_1, live)
    # 60 cached + 10 live with no overlap = 70 rows.
    assert len(merged) == 70
    assert merged.index.min() == cached.index.min()
    assert merged.index.max() == live.index.max()


def test_merge_overlap_keeps_live_values(cache_root: Path) -> None:
    """On overlapping timestamps, live wins (cache may be stale)."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    cached = _df(20, start=base)
    cached["close"] = 100.0  # marker for "cached"
    cache.write_bars("AAPL", TimeFrame.DAY_1, cached)

    # Live overlaps the last 5 cached days with a different close value.
    live_start = cached.index[-5]
    live = _df(15, start=live_start)
    live["close"] = 999.0  # marker for "live"

    merged = cache.merge_with_cache("AAPL", TimeFrame.DAY_1, live)
    # Overlap rows should carry the live value, not the cached one.
    overlap = merged.loc[live_start : cached.index[-1]]
    assert (overlap["close"] == 999.0).all()


def test_merge_empty_live_returns_cache(cache_root: Path) -> None:
    cached = _df(30)
    cache.write_bars("AAPL", TimeFrame.DAY_1, cached)
    merged = cache.merge_with_cache("AAPL", TimeFrame.DAY_1, pd.DataFrame())
    # Reading from CSV restores values + tz-aware index, but not the original
    # index name; compare on values, not metadata.
    pd.testing.assert_frame_equal(
        merged.reset_index(drop=True),
        cached.reset_index(drop=True),
    )
    assert (merged.index == cached.index).all()
