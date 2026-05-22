"""Unit tests for :class:`HealthStore`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.data.health_store import HealthStore, _percentile
from src.intelligence.fundamentals.models import ProviderHealth, ProviderName


def _row(
    *,
    provider: ProviderName = ProviderName.FMP,
    symbol: str = "AAPL",
    status: str = "ok",
    latency_ms: float = 100.0,
    checked_at: datetime | None = None,
    error_message: str | None = None,
) -> ProviderHealth:
    return ProviderHealth(
        provider=provider,
        symbol=symbol,
        status=status,
        latency_ms=latency_ms,
        checked_at=checked_at or datetime.now(UTC),
        error_message=error_message,
    )


# -----------------------------------------------------------------------------
# Construction + lazy init
# -----------------------------------------------------------------------------


def test_init_does_not_touch_disk(tmp_path: Path) -> None:
    """Constructing a HealthStore must NOT create the db file.

    Tests that never call record()/summarize()/list_recent_failures()
    should not leak a SQLite file onto disk. Lazy initialization is a
    behavioral guarantee, not an implementation detail.
    """
    db = tmp_path / "nested" / "health.db"
    HealthStore(db)
    assert not db.exists()
    assert not db.parent.exists()


def test_first_record_creates_db_and_parent_dirs(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "health.db"
    store = HealthStore(db)
    store.record([_row()])
    assert db.exists()
    assert db.parent.exists()


def test_record_returns_count_written(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    n = store.record([_row(), _row(provider=ProviderName.FINNHUB)])
    assert n == 2


def test_record_empty_iterable_is_no_op(tmp_path: Path) -> None:
    db = tmp_path / "h.db"
    store = HealthStore(db)
    written = store.record([])
    assert written == 0
    # Still no disk touch — record() on an empty list shouldn't open
    # the connection.
    assert not db.exists()


# -----------------------------------------------------------------------------
# Summarize: breakdown + success rate
# -----------------------------------------------------------------------------


def test_summarize_returns_one_entry_per_provider(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(provider=ProviderName.FMP, checked_at=now),
            _row(provider=ProviderName.FINNHUB, checked_at=now),
            _row(provider=ProviderName.FMP, checked_at=now),
        ]
    )
    summaries = store.summarize(timedelta(hours=1), now=now + timedelta(minutes=5))
    by_provider = {s.provider: s for s in summaries}
    assert by_provider["fmp"].total == 2
    assert by_provider["finnhub"].total == 1


def test_summarize_excludes_rows_outside_window(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(checked_at=now - timedelta(hours=2)),  # outside 1h window
            _row(checked_at=now - timedelta(minutes=30)),
        ]
    )
    summaries = store.summarize(timedelta(hours=1), now=now)
    assert len(summaries) == 1
    assert summaries[0].total == 1


def test_summarize_success_rate_excludes_skipped(tmp_path: Path) -> None:
    """Success rate denominator excludes skipped rows.

    Skipped means "we didn't try this provider this call" — it's not
    a real observation of provider behavior and shouldn't deflate the
    success rate.
    """
    store = HealthStore(tmp_path / "h.db")
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(status="ok", checked_at=now),
            _row(status="ok", checked_at=now),
            _row(status="ok", checked_at=now),
            _row(status="rate_limited", checked_at=now),
            _row(status="skipped", checked_at=now),  # excluded
            _row(status="skipped", checked_at=now),  # excluded
        ]
    )
    summary = store.summarize(timedelta(hours=1), now=now)[0]
    # 3 ok / 4 non-skipped = 0.75
    assert summary.success_rate == pytest.approx(0.75)
    assert summary.ok == 3
    assert summary.rate_limited == 1
    assert summary.skipped == 2
    assert summary.total == 6


def test_summarize_success_rate_zero_when_only_skipped(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(status="skipped", checked_at=now),
            _row(status="skipped", checked_at=now),
        ]
    )
    summary = store.summarize(timedelta(hours=1), now=now)[0]
    assert summary.success_rate == 0.0


def test_summarize_breakdown_dict_carries_all_observed_statuses(
    tmp_path: Path,
) -> None:
    store = HealthStore(tmp_path / "h.db")
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(status="ok", checked_at=now),
            _row(status="ok", checked_at=now),
            _row(status="rate_limited", checked_at=now),
            _row(status="empty", checked_at=now),
        ]
    )
    summary = store.summarize(timedelta(hours=1), now=now)[0]
    assert summary.breakdown == {"ok": 2, "rate_limited": 1, "empty": 1}


def test_summarize_last_seen_and_last_error_at(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    base = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(status="ok", checked_at=base),
            _row(status="rate_limited", checked_at=base + timedelta(minutes=10)),
            _row(status="ok", checked_at=base + timedelta(minutes=30)),
        ]
    )
    summary = store.summarize(timedelta(hours=1), now=base + timedelta(hours=1))[0]
    # last_seen_at is the latest row regardless of status.
    assert summary.last_seen_at == base + timedelta(minutes=30)
    # last_error_at is the latest *failure* row.
    assert summary.last_error_at == base + timedelta(minutes=10)


def test_summarize_last_error_at_none_when_no_failures(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(status="ok", checked_at=now),
            _row(status="empty", checked_at=now),  # NOT counted as failure
        ]
    )
    summary = store.summarize(timedelta(hours=1), now=now)[0]
    assert summary.last_error_at is None


def test_summarize_rejects_non_positive_window(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    with pytest.raises(ValueError, match="positive"):
        store.summarize(timedelta(0))
    with pytest.raises(ValueError, match="positive"):
        store.summarize(timedelta(seconds=-1))


def test_summarize_results_are_deterministically_ordered(tmp_path: Path) -> None:
    """Higher total first; alphabetic tie-break."""
    store = HealthStore(tmp_path / "h.db")
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(provider=ProviderName.YAHOO, checked_at=now),
            _row(provider=ProviderName.FMP, checked_at=now),
            _row(provider=ProviderName.FMP, checked_at=now),
            _row(provider=ProviderName.FINNHUB, checked_at=now),
            _row(provider=ProviderName.FINNHUB, checked_at=now),
        ]
    )
    summaries = store.summarize(timedelta(hours=1), now=now)
    # fmp and finnhub tie at 2 → alphabetical ("finnhub" < "fmp"
    # because 'i' < 'm'). yahoo at 1 is last by count.
    assert [s.provider for s in summaries] == ["finnhub", "fmp", "yahoo"]


# -----------------------------------------------------------------------------
# Latency percentiles
# -----------------------------------------------------------------------------


def test_percentile_known_values() -> None:
    # 10 evenly-spaced values: linear-interp p50 = 5.5, p95 = 9.55.
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert _percentile(values, 0.5) == pytest.approx(5.5)
    assert _percentile(values, 0.95) == pytest.approx(9.55)


def test_percentile_singleton_returns_value() -> None:
    assert _percentile([42.0], 0.5) == 42.0
    assert _percentile([42.0], 0.95) == 42.0


def test_percentile_empty_is_none() -> None:
    assert _percentile([], 0.5) is None


def test_summarize_p50_p95_from_latencies(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    store.record(
        [_row(status="ok", latency_ms=float(ms), checked_at=now) for ms in latencies]
    )
    summary = store.summarize(timedelta(hours=1), now=now)[0]
    assert summary.p50_latency_ms == pytest.approx(55.0)
    assert summary.p95_latency_ms == pytest.approx(95.5)


def test_summarize_latency_excludes_skipped_rows(tmp_path: Path) -> None:
    """A skipped row carries latency=0 in the wire row. Including it
    in the latency percentile would falsely drag p50 toward 0."""
    store = HealthStore(tmp_path / "h.db")
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(status="ok", latency_ms=100.0, checked_at=now),
            _row(status="ok", latency_ms=200.0, checked_at=now),
            _row(status="skipped", latency_ms=0.0, checked_at=now),
        ]
    )
    summary = store.summarize(timedelta(hours=1), now=now)[0]
    # p50 across {100, 200} = 150; if skipped row leaked in, p50 = 100.
    assert summary.p50_latency_ms == pytest.approx(150.0)


# -----------------------------------------------------------------------------
# Recent failures
# -----------------------------------------------------------------------------


def test_list_recent_failures_excludes_ok_and_empty(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(status="ok", checked_at=now),
            _row(status="empty", checked_at=now),
            _row(status="rate_limited", checked_at=now),
            _row(status="unavailable", checked_at=now),
            _row(status="transient", checked_at=now),
        ]
    )
    failures = store.list_recent_failures(limit=10)
    statuses = {f.status for f in failures}
    assert statuses == {"rate_limited", "unavailable", "transient"}


def test_list_recent_failures_orders_newest_first(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    base = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(status="rate_limited", symbol="A", checked_at=base),
            _row(
                status="unavailable",
                symbol="B",
                checked_at=base + timedelta(minutes=10),
            ),
            _row(
                status="transient",
                symbol="C",
                checked_at=base + timedelta(minutes=5),
            ),
        ]
    )
    failures = store.list_recent_failures(limit=10)
    assert [f.symbol for f in failures] == ["B", "C", "A"]


def test_list_recent_failures_limit_is_honored(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    base = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    store.record(
        [
            _row(
                status="rate_limited",
                symbol=f"S{i:02d}",
                checked_at=base + timedelta(minutes=i),
            )
            for i in range(10)
        ]
    )
    failures = store.list_recent_failures(limit=3)
    assert len(failures) == 3


def test_list_recent_failures_zero_limit_returns_empty(tmp_path: Path) -> None:
    store = HealthStore(tmp_path / "h.db")
    store.record([_row(status="rate_limited")])
    assert store.list_recent_failures(limit=0) == ()


# -----------------------------------------------------------------------------
# Durability / concurrency
# -----------------------------------------------------------------------------


def test_records_persist_across_store_instances(tmp_path: Path) -> None:
    """A new HealthStore on the same path sees prior writes."""
    db = tmp_path / "h.db"
    HealthStore(db).record([_row(status="ok")])
    store2 = HealthStore(db)
    summaries = store2.summarize(timedelta(days=1))
    assert summaries[0].ok == 1
