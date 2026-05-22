"""Tests for src.data.cache — the filesystem cache for backfilled data."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
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


# ---------------------------------------------------------------------------
# TTL (mtime-driven) on the simple reads
# ---------------------------------------------------------------------------


def _age_file(path: Path, delta: timedelta) -> None:
    """Mutate a file's mtime backwards by ``delta``.

    Used to simulate a long-stale cache without having to actually
    wait. Touching atime as well keeps platforms that surface both
    consistent.
    """
    new_mtime = (datetime.now(UTC) - delta).timestamp()
    os.utime(path, (new_mtime, new_mtime))


def test_read_bars_max_age_returns_none_when_file_too_old(cache_root: Path) -> None:
    path = cache.write_bars("AAPL", TimeFrame.DAY_1, _df(30))
    _age_file(path, timedelta(hours=2))

    assert cache.read_bars("AAPL", TimeFrame.DAY_1, max_age=timedelta(hours=1)) is None
    # Sanity: without max_age, the cache still reads.
    assert cache.read_bars("AAPL", TimeFrame.DAY_1) is not None


def test_read_bars_max_age_returns_data_when_fresh(cache_root: Path) -> None:
    cache.write_bars("AAPL", TimeFrame.DAY_1, _df(30))
    # File was just written; well within a 1h budget.
    loaded = cache.read_bars(
        "AAPL", TimeFrame.DAY_1, max_age=timedelta(hours=1)
    )
    assert loaded is not None
    assert len(loaded) == 30


def test_read_news_max_age_returns_empty_when_file_too_old(cache_root: Path) -> None:
    when = datetime(2026, 5, 1, tzinfo=UTC)
    path = cache.write_news("AAPL", [_article("n1", when)])
    _age_file(path, timedelta(hours=12))

    assert cache.read_news("AAPL", max_age=timedelta(hours=6)) == []
    # Without max_age, the cache still reads.
    assert len(cache.read_news("AAPL")) == 1


def test_merge_with_cache_skips_expired_cache(cache_root: Path) -> None:
    """A stale CSV must NOT contaminate a fresh live-data merge.

    Pre-P1.3 this was the silent failure mode: a multi-day-old warmup
    cache would get prepended to fresh bars, with the indicator engine
    happily computing on what was effectively two disjoint windows.
    """
    cached = _df(60, start=datetime(2026, 1, 1, tzinfo=UTC))
    cache_path = cache.write_bars("AAPL", TimeFrame.DAY_1, cached)
    _age_file(cache_path, timedelta(days=10))

    live = _df(10, start=datetime(2026, 4, 1, tzinfo=UTC))
    merged = cache.merge_with_cache(
        "AAPL", TimeFrame.DAY_1, live, max_age=timedelta(hours=24)
    )
    # Cache was too old → live unchanged.
    pd.testing.assert_frame_equal(merged, live, check_names=False)


def test_merge_with_cache_uses_cache_when_within_max_age(cache_root: Path) -> None:
    cached = _df(60, start=datetime(2026, 1, 1, tzinfo=UTC))
    cache.write_bars("AAPL", TimeFrame.DAY_1, cached)
    # No artificial age — file is fresh.
    live = _df(10, start=datetime(2026, 4, 1, tzinfo=UTC))
    merged = cache.merge_with_cache(
        "AAPL", TimeFrame.DAY_1, live, max_age=timedelta(hours=24)
    )
    assert len(merged) == 70


# ---------------------------------------------------------------------------
# Envelope reads (in-band as_of)
# ---------------------------------------------------------------------------


def test_read_bars_envelope_uses_last_bar_timestamp_as_as_of(cache_root: Path) -> None:
    start = datetime(2026, 5, 1, tzinfo=UTC)
    df = _df(10, start=start)
    cache.write_bars("AAPL", TimeFrame.DAY_1, df)

    # Pin "now" to one day after the last bar so freshness is
    # deterministic. With the bars.daily policy (25h fresh window)
    # this should be ``fresh``.
    last_ts = df.index.max().to_pydatetime()
    env = cache.read_bars_envelope("AAPL", TimeFrame.DAY_1, now=last_ts + timedelta(hours=1))
    assert env is not None
    assert env.as_of == last_ts
    assert env.freshness == "fresh"
    assert env.source_chain == ("cache",)
    assert env.provider_confidence == 1.0


def test_read_bars_envelope_freshness_falls_through_tiers(cache_root: Path) -> None:
    """Walk a single fixture across fresh → aging → stale → expired."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    df = _df(5, start=start)
    cache.write_bars("AAPL", TimeFrame.DAY_1, df)
    last_ts = df.index.max().to_pydatetime()

    # bars.daily policy: fresh ≤25h, aging ≤3d, stale ≤7d, then expired.
    fresh = cache.read_bars_envelope(
        "AAPL", TimeFrame.DAY_1, now=last_ts + timedelta(hours=1)
    )
    aging = cache.read_bars_envelope(
        "AAPL", TimeFrame.DAY_1, now=last_ts + timedelta(days=2)
    )
    stale = cache.read_bars_envelope(
        "AAPL", TimeFrame.DAY_1, now=last_ts + timedelta(days=5)
    )
    expired = cache.read_bars_envelope(
        "AAPL", TimeFrame.DAY_1, now=last_ts + timedelta(days=10)
    )
    assert fresh is not None and fresh.freshness == "fresh"
    assert aging is not None and aging.freshness == "aging"
    assert stale is not None and stale.freshness == "stale"
    assert expired is not None and expired.freshness == "expired"


def test_read_bars_envelope_intraday_policy_for_subdaily_timeframes(
    cache_root: Path,
) -> None:
    """A 5-minute timeframe should grade against bars.intraday by default."""
    start = datetime(2026, 5, 1, 9, 30, tzinfo=UTC)
    idx = pd.date_range(start=start, periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1},
        index=idx,
    )
    cache.write_bars("AAPL", TimeFrame.MIN_5, df)
    last_ts = df.index.max().to_pydatetime()
    # bars.intraday: fresh ≤90s. 30 seconds out should be fresh.
    env = cache.read_bars_envelope(
        "AAPL", TimeFrame.MIN_5, now=last_ts + timedelta(seconds=30)
    )
    assert env is not None and env.freshness == "fresh"
    # 6 minutes out → aging window (≤5m fresh / ≤30m expired? actually
    # ≤90s fresh, ≤5m aging, ≤30m stale, >30m expired).
    env = cache.read_bars_envelope(
        "AAPL", TimeFrame.MIN_5, now=last_ts + timedelta(minutes=6)
    )
    assert env is not None and env.freshness == "stale"


def test_read_bars_envelope_returns_none_when_file_missing(cache_root: Path) -> None:
    assert cache.read_bars_envelope("ZZZZ", TimeFrame.DAY_1) is None


def test_read_news_envelope_uses_latest_published_at(cache_root: Path) -> None:
    base = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    articles = [
        _article("n1", base),
        _article("n2", base + timedelta(hours=2)),
        _article("n3", base + timedelta(hours=1)),
    ]
    cache.write_news("AAPL", articles)
    env = cache.read_news_envelope("AAPL", now=base + timedelta(hours=3))
    assert env is not None
    # Latest article wins.
    assert env.as_of == base + timedelta(hours=2)
    assert env.freshness == "fresh"  # 1h < 6h news fresh window


def test_read_news_envelope_falls_back_to_mtime_when_no_articles(
    cache_root: Path,
) -> None:
    path = cache.write_news("AAPL", [])
    env = cache.read_news_envelope("AAPL")
    assert env is not None
    # as_of should match the file mtime within a small tolerance.
    mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    assert abs((env.as_of - mtime).total_seconds()) < 1.0


def test_read_news_envelope_returns_none_when_file_missing(cache_root: Path) -> None:
    assert cache.read_news_envelope("ZZZZ") is None


def test_read_news_envelope_freshness_against_news_policy(cache_root: Path) -> None:
    base = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    cache.write_news("AAPL", [_article("n1", base)])
    # news policy: fresh ≤6h, aging ≤24h, stale ≤3d, expired beyond.
    fresh = cache.read_news_envelope("AAPL", now=base + timedelta(hours=2))
    aging = cache.read_news_envelope("AAPL", now=base + timedelta(hours=12))
    stale = cache.read_news_envelope("AAPL", now=base + timedelta(days=2))
    expired = cache.read_news_envelope("AAPL", now=base + timedelta(days=5))
    assert fresh is not None and fresh.freshness == "fresh"
    assert aging is not None and aging.freshness == "aging"
    assert stale is not None and stale.freshness == "stale"
    assert expired is not None and expired.freshness == "expired"
