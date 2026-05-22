"""Filesystem cache for backfilled bars and news, with freshness-aware reads.

Bars live in ``{symbol}_bars_{timeframe}.csv`` with a UTC datetime index;
news lives in ``{symbol}_news.json`` as a list of serialized
:class:`~src.data.models.NewsArticle`. The cache directory is
``Settings.data_dir / "cache"`` (typically ``data/cache/``).

Two read paths
--------------
Each cached data class exposes a pair of reads that answer different
questions:

* :func:`read_bars` / :func:`read_news` — *simple* reads. Take an
  optional ``max_age`` kwarg that compares against the file's mtime.
  Answers "did we write this recently?" — the question warm-start
  paths and the hot :func:`merge_with_cache` helper actually care
  about.

* :func:`read_bars_envelope` / :func:`read_news_envelope` —
  freshness-graded reads. Wrap the data in a
  :class:`~src.data.envelope.DataEnvelope` whose ``as_of`` reflects
  the *in-band* timestamp (last bar's timestamp, latest article
  ``published_at``). Answers "should the analyzer / UI trust this?"
  — the question the freshness contract introduced in P1.1 already
  centralizes for fundamentals.

Both questions are legitimate and orthogonal. mtime is honest about
when WE acted; in-band is honest about when the WORLD said it.

Why no source attribution on cache reads
----------------------------------------
The bars/news cache is *content-addressed by filename*; we don't
persist a sidecar with the original provider chain. The envelope's
``source_chain`` is therefore the opaque marker ``("cache",)`` and
``provider_confidence`` is ``1.0`` — cache integrity is binary, and
the *data quality* concern is captured by the freshness tier. A
follow-on could write a sidecar JSON if the trader needs to see
"this bar window came from Alpaca at 10:31 ET"; for now the
freshness tier carries the load.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.config import get_settings
from src.data.envelope import DataEnvelope
from src.data.freshness import evaluate_freshness, policy_for
from src.data.models import NewsArticle, TimeFrame

if TYPE_CHECKING:
    pass

_BARS_COLUMNS = ("open", "high", "low", "close", "volume")

# Opaque source marker for cache-originated envelopes. Keeps the
# envelope's source_chain semantically meaningful (it's not a
# provider name) without lying about which upstream answered.
_CACHE_SOURCE = "cache"


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


def read_bars(
    symbol: str,
    timeframe: TimeFrame,
    *,
    max_age: timedelta | None = None,
) -> pd.DataFrame | None:
    """Read cached bars for ``(symbol, timeframe)``.

    Returns ``None`` when:

    * No cache file exists, OR
    * ``max_age`` was provided and the file's mtime is older than that.

    ``max_age`` is compared against file mtime — it answers "did WE
    write this recently?" In the hot warmup path the question is
    "should I trust my prior cache as warmup, or refetch?", which
    mtime captures cleanly. Use :func:`read_bars_envelope` when the
    question is "is the underlying market data fresh?".
    """
    path = bars_path(symbol, timeframe)
    if not path.exists():
        return None
    if max_age is not None and _file_age(path) > max_age:
        return None
    df = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    # Coerce timezones consistently — Alpaca returns UTC.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def read_bars_envelope(
    symbol: str,
    timeframe: TimeFrame,
    *,
    policy_key: str | None = None,
    now: datetime | None = None,
) -> DataEnvelope[pd.DataFrame] | None:
    """Return cached bars wrapped in a freshness :class:`DataEnvelope`.

    ``as_of`` is the *last bar's timestamp* — the in-band marker of
    when the market data was true. ``fetched_at`` is the cache file's
    mtime (UTC). Returns ``None`` when the cache file is missing or
    contains no rows.

    The policy defaults to ``bars.intraday`` for sub-daily timeframes
    and ``bars.daily`` for ``DAY_1`` — chosen automatically from
    ``timeframe`` so callers don't have to repeat the mapping.
    """
    path = bars_path(symbol, timeframe)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if df.empty:
        return None

    last_ts = df.index.max()
    # pandas Timestamp → python datetime; the index is UTC-aware above.
    as_of = last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts
    fetched_at = _file_mtime(path)
    moment = now or datetime.now(UTC)
    key = policy_key or _default_bars_policy_key(timeframe)
    freshness = evaluate_freshness(as_of, moment, policy_for(key))

    return DataEnvelope(
        data=df,
        as_of=as_of,
        fetched_at=fetched_at,
        source_chain=(_CACHE_SOURCE,),
        freshness=freshness,
        provider_confidence=1.0,
    )


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


def read_news(
    symbol: str,
    *,
    max_age: timedelta | None = None,
) -> list[NewsArticle]:
    """Read cached news for ``symbol``.

    Returns ``[]`` when:

    * No cache file exists, OR
    * ``max_age`` was provided and the file's mtime is older than that.

    Returning an empty list (not ``None``) preserves the contract that
    pre-dated this signature: existing callers do
    ``articles = read_news(symbol)`` and iterate without a None-check.
    """
    path = news_path(symbol)
    if not path.exists():
        return []
    if max_age is not None and _file_age(path) > max_age:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [NewsArticle(**item) for item in raw]


def read_news_envelope(
    symbol: str,
    *,
    policy_key: str = "news",
    now: datetime | None = None,
) -> DataEnvelope[list[NewsArticle]] | None:
    """Return cached news wrapped in a freshness :class:`DataEnvelope`.

    ``as_of`` is the most recent article's ``published_at``. When all
    articles lack that field (or the file is empty) we fall back to
    the file mtime so an envelope is still returned — losing the
    envelope on a partial-data file would force every consumer to
    branch on absence.

    Returns ``None`` only when the cache file itself is missing.
    """
    path = news_path(symbol)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    articles = [NewsArticle(**item) for item in raw]
    fetched_at = _file_mtime(path)
    moment = now or datetime.now(UTC)

    publish_times = [
        _ensure_utc(a.published_at) for a in articles if a.published_at is not None
    ]
    # Fall back to mtime when the cache has no datable articles. An
    # article-less file then grades as "we recently checked this
    # symbol and the world had nothing to say" rather than expired.
    as_of = max(publish_times) if publish_times else fetched_at

    freshness = evaluate_freshness(as_of, moment, policy_for(policy_key))
    return DataEnvelope(
        data=articles,
        as_of=as_of,
        fetched_at=fetched_at,
        source_chain=(_CACHE_SOURCE,),
        freshness=freshness,
        provider_confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Merge helper for the dashboard hot path
# ---------------------------------------------------------------------------


def merge_with_cache(
    symbol: str,
    timeframe: TimeFrame,
    live: pd.DataFrame,
    *,
    max_age: timedelta | None = None,
) -> pd.DataFrame:
    """Prepend cached bars to a live bar window, keeping the live values
    on overlapping timestamps (cache may be stale).

    Used by the dashboard controller so the indicator engine sees a longer
    warm-up window without paying for a deeper Alpaca request every tick.

    When ``max_age`` is provided and the cache file is older than that,
    behave as if no cache existed — the live frame is returned
    unchanged. This prevents a long-stale CSV from contaminating the
    indicator warm-up after a multi-day pause.
    """
    cached = read_bars(symbol, timeframe, max_age=max_age)
    if cached is None or cached.empty:
        return live
    if live.empty:
        return cached
    # ``combine_first`` keeps caller's non-null values, fills from the other.
    # live wins on overlap; cached fills the prefix gap.
    return live.combine_first(cached).sort_index()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_mtime(path: Path) -> datetime:
    """Return the file's mtime as a tz-aware UTC datetime."""
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)


def _file_age(path: Path) -> timedelta:
    """How long ago the file was last written. Always non-negative."""
    age = datetime.now(UTC) - _file_mtime(path)
    if age.total_seconds() < 0:
        return timedelta(0)
    return age


def _ensure_utc(value: datetime) -> datetime:
    """Coerce a naive datetime to UTC; pass tz-aware through unchanged.

    Article ``published_at`` values are inconsistent across providers
    — Alpaca returns aware, NewsAPI returns naive. The envelope's
    freshness math assumes UTC, so we coerce here.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _default_bars_policy_key(timeframe: TimeFrame) -> str:
    """Pick the appropriate :data:`POLICIES` key for a bar timeframe.

    ``DAY_1`` → ``bars.daily``; everything sub-daily → ``bars.intraday``.
    Mirrors the trader-facing distinction (overnight cadence vs.
    intraday cadence).
    """
    if timeframe is TimeFrame.DAY_1:
        return "bars.daily"
    return "bars.intraday"
