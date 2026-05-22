"""Tests for :class:`CalibrationRecorder` and :class:`CalibrationMaturationWorker`.

The recorder is exercised against a real :class:`CalibrationStore`
(SQLite, lazy-init — no slowdown). The worker is exercised through
``tick()`` so we don't depend on the asyncio scheduler for the
behavioral assertions; one lifecycle test covers ``start``/``stop``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.intelligence.analyzer.technical import (
    TechnicalScores,
    TechnicalSubscores,
)
from src.intelligence.calibration import CalibrationStore
from src.intelligence.calibration_worker import (
    CalibrationMaturationWorker,
    CalibrationRecorder,
    bars_cache_price_lookup,
    derive_outcome,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _subscores() -> TechnicalSubscores:
    return TechnicalSubscores(
        rsi_overbought_score=None,
        rsi_oversold_score=None,
        bollinger_extension_score=None,
        ma_extension_score=None,
        volume_spike_score=None,
        atr_volatility_score=None,
        momentum_exhaustion_score=None,
    )


def _technicals(
    *,
    overbought: float | None = None,
    oversold: float | None = None,
    pullback: float | None = None,
    rebound: float | None = None,
) -> TechnicalScores:
    return TechnicalScores(
        subscores=_subscores(),
        overbought_score=overbought,
        oversold_score=oversold,
        pullback_risk=pullback,
        rebound_potential=rebound,
        confidence_score=50.0,
        raw_rsi=None,
        raw_bollinger_z=None,
        raw_ma_distance_pct=None,
        raw_volume_z=None,
        raw_atr_ratio=None,
    )


def _store(tmp_path: Path) -> CalibrationStore:
    return CalibrationStore(tmp_path / "calibration.db")


# =============================================================================
# derive_outcome — pure-function truth table
# =============================================================================


@pytest.mark.parametrize(
    "start, end, outcome_name, expected",
    [
        # Positive return cases
        (100.0, 105.0, "return_positive", True),
        (100.0, 105.0, "return_negative", False),
        # Negative return cases
        (100.0, 95.0, "return_negative", True),
        (100.0, 95.0, "return_positive", False),
        # Zero return is False for BOTH directional outcomes
        (100.0, 100.0, "return_positive", False),
        (100.0, 100.0, "return_negative", False),
    ],
)
def test_derive_outcome_truth_table(
    start: float, end: float, outcome_name: str, expected: bool
) -> None:
    assert derive_outcome(start, end, outcome_name) is expected


def test_derive_outcome_returns_none_on_missing_inputs() -> None:
    assert derive_outcome(None, 105.0, "return_positive") is None
    assert derive_outcome(100.0, None, "return_positive") is None


def test_derive_outcome_returns_none_on_non_positive_start() -> None:
    """Zero or negative starting price can't anchor a percentage return."""
    assert derive_outcome(0.0, 100.0, "return_positive") is None
    assert derive_outcome(-1.0, 100.0, "return_positive") is None


def test_derive_outcome_returns_none_on_non_finite_end() -> None:
    assert derive_outcome(100.0, float("nan"), "return_positive") is None
    assert derive_outcome(100.0, float("inf"), "return_positive") is None


def test_derive_outcome_returns_none_on_unknown_outcome_name() -> None:
    """Unknown name → grade refused; worker will skip the observation."""
    assert derive_outcome(100.0, 105.0, "nonsense") is None


# =============================================================================
# CalibrationRecorder
# =============================================================================


def test_recorder_emits_one_observation_per_populated_pairing(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    recorder = CalibrationRecorder(store=store, horizon_days=5)
    technicals = _technicals(pullback=73.0, rebound=12.0)
    when = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)

    ids = recorder.record_scores(
        symbol="aapl",  # lowercase → store should uppercase
        technicals=technicals,
        observed_at=when,
        starting_price=180.50,
    )
    assert len(ids) == 2  # pullback + rebound; the other two are None
    rows = store.all_observations()
    by_score = {r.score_name: r for r in rows}
    assert "pullback_risk" in by_score
    assert "rebound_potential" in by_score
    assert by_score["pullback_risk"].symbol == "AAPL"
    assert by_score["pullback_risk"].score_value == 73.0
    assert by_score["pullback_risk"].outcome_name == "return_negative"
    assert by_score["pullback_risk"].starting_price == 180.50
    assert by_score["rebound_potential"].outcome_name == "return_positive"


def test_recorder_skips_pairings_with_none_score(tmp_path: Path) -> None:
    """If a TechnicalScores field is None, no observation for that pairing."""
    store = _store(tmp_path)
    recorder = CalibrationRecorder(store=store, horizon_days=5)
    # Only pullback set; the other three pairings drop out.
    technicals = _technicals(pullback=73.0)
    ids = recorder.record_scores(
        symbol="AAPL",
        technicals=technicals,
        observed_at=datetime.now(UTC),
        starting_price=180.0,
    )
    assert len(ids) == 1
    rows = store.all_observations()
    assert {r.score_name for r in rows} == {"pullback_risk"}


def test_recorder_uses_configured_horizon(tmp_path: Path) -> None:
    store = _store(tmp_path)
    recorder = CalibrationRecorder(store=store, horizon_days=10)
    technicals = _technicals(pullback=73.0)
    recorder.record_scores(
        symbol="AAPL",
        technicals=technicals,
        observed_at=datetime.now(UTC),
        starting_price=180.0,
    )
    rows = store.all_observations()
    assert rows[0].horizon_days == 10


# =============================================================================
# Schema migration / starting_price round-trip
# =============================================================================


def test_observation_round_trips_starting_price(tmp_path: Path) -> None:
    store = _store(tmp_path)
    when = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    oid = store.record_observation(
        symbol="AAPL",
        score_name="pullback_risk",
        score_value=73.0,
        observed_at=when,
        horizon_days=5,
        outcome_name="return_negative",
        starting_price=180.50,
    )
    obs = next(o for o in store.all_observations() if o.id == oid)
    assert obs.starting_price == 180.50


def test_observation_starting_price_optional(tmp_path: Path) -> None:
    store = _store(tmp_path)
    oid = store.record_observation(
        symbol="AAPL",
        score_name="pullback_risk",
        score_value=73.0,
        observed_at=datetime.now(UTC),
        horizon_days=5,
        outcome_name="return_negative",
    )
    obs = next(o for o in store.all_observations() if o.id == oid)
    assert obs.starting_price is None


def test_schema_migration_adds_starting_price_column(tmp_path: Path) -> None:
    """A pre-v2 observations table gets the starting_price column on connect.

    We forge a v1-shaped DB and confirm the v2 connect path migrates
    in place without losing the existing row.
    """
    import sqlite3

    path = tmp_path / "calibration.db"
    legacy = sqlite3.connect(path, isolation_level=None)
    legacy.executescript(
        """
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            score_name TEXT NOT NULL,
            score_value REAL NOT NULL,
            observed_at TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            outcome_name TEXT NOT NULL,
            outcome_value INTEGER
        );
        """
    )
    legacy.execute(
        """
        INSERT INTO observations
            (symbol, score_name, score_value, observed_at,
             horizon_days, outcome_name, outcome_value)
        VALUES (?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            "AAPL",
            "pullback_risk",
            73.0,
            "2026-05-01T00:00:00+00:00",
            5,
            "return_negative",
        ),
    )
    legacy.close()

    # New store on the same path triggers the migration.
    store = CalibrationStore(path)
    rows = store.all_observations()
    assert len(rows) == 1
    assert rows[0].symbol == "AAPL"
    assert rows[0].starting_price is None  # v1 row, no starting_price


# =============================================================================
# CalibrationMaturationWorker.tick — happy path
# =============================================================================


def _make_observation(
    store: CalibrationStore,
    *,
    symbol: str = "AAPL",
    score_value: float = 73.0,
    starting_price: float | None = 100.0,
    outcome_name: str = "return_negative",
    observed_at: datetime | None = None,
    horizon_days: int = 5,
) -> int:
    return store.record_observation(
        symbol=symbol,
        score_name="pullback_risk",
        score_value=score_value,
        observed_at=observed_at or datetime(2026, 5, 1, tzinfo=UTC),
        horizon_days=horizon_days,
        outcome_name=outcome_name,
        starting_price=starting_price,
    )


def _price_lookup(prices: dict[str, float]) -> object:
    """Build a deterministic async price_lookup callable."""

    async def lookup(symbol: str, at: datetime) -> float | None:
        return prices.get(symbol)

    return lookup


@pytest.mark.asyncio
async def test_worker_settles_matured_observations_with_a_hit(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    oid = _make_observation(
        store, starting_price=100.0, outcome_name="return_negative"
    )
    # End price below start → return_negative fires.
    worker = CalibrationMaturationWorker(
        store, _price_lookup({"AAPL": 95.0}), interval_seconds=60
    )
    now = datetime(2026, 5, 1, tzinfo=UTC) + timedelta(days=10)

    # Drive the lookup via the published tick API (using the stored
    # matures_at for the underlying observation).
    settled = await worker.tick()
    assert settled == 1
    obs = next(o for o in store.all_observations() if o.id == oid)
    assert obs.outcome_value is True
    del now  # only used to document the maturation timeline


@pytest.mark.asyncio
async def test_worker_settles_matured_observation_with_a_miss(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    oid = _make_observation(
        store, starting_price=100.0, outcome_name="return_negative"
    )
    # End price above start → return_negative doesn't fire.
    worker = CalibrationMaturationWorker(
        store, _price_lookup({"AAPL": 105.0}), interval_seconds=60
    )
    await worker.tick()
    obs = next(o for o in store.all_observations() if o.id == oid)
    assert obs.outcome_value is False


@pytest.mark.asyncio
async def test_worker_handles_each_pairing_independently(
    tmp_path: Path,
) -> None:
    """A single tick can settle multiple symbols / pairings in one pass."""
    store = _store(tmp_path)
    _make_observation(
        store, symbol="UP", starting_price=100.0, outcome_name="return_positive"
    )
    _make_observation(
        store, symbol="DOWN", starting_price=100.0, outcome_name="return_negative"
    )
    worker = CalibrationMaturationWorker(
        store,
        _price_lookup({"UP": 110.0, "DOWN": 90.0}),
        interval_seconds=60,
    )
    settled = await worker.tick()
    assert settled == 2
    rows = {o.symbol: o for o in store.all_observations()}
    assert rows["UP"].outcome_value is True
    assert rows["DOWN"].outcome_value is True


# =============================================================================
# CalibrationMaturationWorker.tick — degradation
# =============================================================================


@pytest.mark.asyncio
async def test_worker_skips_observations_without_starting_price(
    tmp_path: Path,
) -> None:
    """v1-schema rows can't be settled. Worker no-ops, doesn't crash."""
    store = _store(tmp_path)
    oid = _make_observation(store, starting_price=None)
    worker = CalibrationMaturationWorker(
        store, _price_lookup({"AAPL": 95.0}), interval_seconds=60
    )
    settled = await worker.tick()
    assert settled == 0
    obs = next(o for o in store.all_observations() if o.id == oid)
    assert obs.outcome_value is None  # stays unsettled


@pytest.mark.asyncio
async def test_worker_skips_when_price_lookup_returns_none(
    tmp_path: Path,
) -> None:
    """No price at maturation (weekend, gap, delisted) → observation stays open."""
    store = _store(tmp_path)
    oid = _make_observation(store)
    worker = CalibrationMaturationWorker(
        store, _price_lookup({}), interval_seconds=60  # empty mapping
    )
    settled = await worker.tick()
    assert settled == 0
    obs = next(o for o in store.all_observations() if o.id == oid)
    assert obs.outcome_value is None


@pytest.mark.asyncio
async def test_worker_swallows_lookup_exceptions(tmp_path: Path) -> None:
    """A lookup that raises must not kill the tick. Observation stays unsettled."""
    store = _store(tmp_path)
    oid = _make_observation(store)

    async def boom(symbol: str, at: datetime) -> float | None:
        raise RuntimeError("simulated upstream failure")

    worker = CalibrationMaturationWorker(store, boom, interval_seconds=60)
    settled = await worker.tick()
    assert settled == 0
    obs = next(o for o in store.all_observations() if o.id == oid)
    assert obs.outcome_value is None


@pytest.mark.asyncio
async def test_worker_continues_past_one_failed_observation(
    tmp_path: Path,
) -> None:
    """One bad observation must not block the rest of the queue."""
    store = _store(tmp_path)
    bad_oid = _make_observation(store, symbol="BAD")
    good_oid = _make_observation(store, symbol="GOOD")

    async def lookup(symbol: str, at: datetime) -> float | None:
        if symbol == "BAD":
            raise RuntimeError("boom")
        return 90.0  # negative return → return_negative fires

    worker = CalibrationMaturationWorker(store, lookup, interval_seconds=60)
    settled = await worker.tick()
    assert settled == 1
    rows = {o.id: o for o in store.all_observations()}
    assert rows[bad_oid].outcome_value is None
    assert rows[good_oid].outcome_value is True


@pytest.mark.asyncio
async def test_worker_ignores_observations_not_yet_matured(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    future = datetime.now(UTC) + timedelta(days=30)
    _make_observation(store, observed_at=future)
    worker = CalibrationMaturationWorker(
        store, _price_lookup({"AAPL": 95.0}), interval_seconds=60
    )
    settled = await worker.tick()
    assert settled == 0


# =============================================================================
# CalibrationMaturationWorker lifecycle
# =============================================================================


@pytest.mark.asyncio
async def test_start_then_stop_drains_one_pass(tmp_path: Path) -> None:
    """Smoke test: start the loop, see it run a tick, stop cleanly."""
    store = _store(tmp_path)
    _make_observation(store)
    drained = asyncio.Event()

    async def lookup(symbol: str, at: datetime) -> float | None:
        drained.set()
        return 95.0

    worker = CalibrationMaturationWorker(store, lookup, interval_seconds=0.05)
    await worker.start()
    await asyncio.wait_for(drained.wait(), timeout=2.0)
    await worker.stop()
    obs = store.all_observations()[0]
    assert obs.outcome_value is True


@pytest.mark.asyncio
async def test_double_start_is_no_op(tmp_path: Path) -> None:
    store = _store(tmp_path)

    async def lookup(symbol: str, at: datetime) -> float | None:
        return None

    worker = CalibrationMaturationWorker(store, lookup, interval_seconds=0.1)
    await worker.start()
    first_task = worker._task
    await worker.start()
    assert worker._task is first_task
    await worker.stop()


@pytest.mark.asyncio
async def test_stop_when_not_running_is_no_op(tmp_path: Path) -> None:
    store = _store(tmp_path)

    async def lookup(symbol: str, at: datetime) -> float | None:
        return None

    worker = CalibrationMaturationWorker(store, lookup, interval_seconds=0.1)
    await worker.stop()
    assert not worker.is_running


def test_invalid_interval_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)

    async def lookup(symbol: str, at: datetime) -> float | None:
        return None

    with pytest.raises(ValueError, match="positive"):
        CalibrationMaturationWorker(store, lookup, interval_seconds=0)
    with pytest.raises(ValueError, match="positive"):
        CalibrationMaturationWorker(store, lookup, interval_seconds=-1.0)


# =============================================================================
# bars_cache_price_lookup — production callsite
# =============================================================================


@pytest.mark.asyncio
async def test_price_lookup_returns_close_at_or_after_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first bar at-or-after ``at`` wins. The bar that covers
    maturity is the one whose close reflects the horizon's outcome."""
    import numpy as np
    import pandas as pd

    from src.config import settings as settings_mod
    from src.data import cache as bar_cache
    from src.data.models import TimeFrame

    settings_mod.get_settings.cache_clear()
    monkeypatch.setattr(settings_mod.get_settings(), "data_dir", tmp_path, raising=False)

    idx = pd.date_range(
        start=datetime(2026, 5, 15, tzinfo=UTC), periods=5, freq="D"
    )
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    df = pd.DataFrame(
        {
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": np.full(5, 1_000_000),
        },
        index=idx,
    )
    bar_cache.write_bars("AAPL", TimeFrame.DAY_1, df)

    # ``at`` is between bar 2 and 3 → the first bar at-or-after is
    # 2026-05-17 (close=102.0).
    price = await bars_cache_price_lookup(
        "AAPL", datetime(2026, 5, 17, 6, 0, tzinfo=UTC)
    )
    assert price == 102.0


@pytest.mark.asyncio
async def test_price_lookup_returns_none_when_cache_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    monkeypatch.setattr(settings_mod.get_settings(), "data_dir", tmp_path, raising=False)
    assert (
        await bars_cache_price_lookup(
            "UNKNOWN", datetime(2026, 5, 17, tzinfo=UTC)
        )
        is None
    )


@pytest.mark.asyncio
async def test_price_lookup_returns_none_when_no_bar_past_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache exists but every bar is older than ``at`` — maturation
    can't be graded yet. None is correct, not an exception."""
    import pandas as pd

    from src.config import settings as settings_mod
    from src.data import cache as bar_cache
    from src.data.models import TimeFrame

    settings_mod.get_settings.cache_clear()
    monkeypatch.setattr(settings_mod.get_settings(), "data_dir", tmp_path, raising=False)

    idx = pd.date_range(
        start=datetime(2026, 1, 1, tzinfo=UTC), periods=3, freq="D"
    )
    df = pd.DataFrame(
        {
            "open": [100.0] * 3, "high": [100.0] * 3, "low": [100.0] * 3,
            "close": [100.0] * 3, "volume": [1_000_000] * 3,
        },
        index=idx,
    )
    bar_cache.write_bars("AAPL", TimeFrame.DAY_1, df)
    assert (
        await bars_cache_price_lookup(
            "AAPL", datetime(2026, 6, 1, tzinfo=UTC)
        )
        is None
    )


@pytest.mark.asyncio
async def test_price_lookup_works_through_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: store observation, write a bar past maturation,
    worker tick settles the outcome."""
    import numpy as np
    import pandas as pd

    from src.config import settings as settings_mod
    from src.data import cache as bar_cache
    from src.data.models import TimeFrame

    settings_mod.get_settings.cache_clear()
    monkeypatch.setattr(settings_mod.get_settings(), "data_dir", tmp_path, raising=False)

    store = CalibrationStore(tmp_path / "calibration.db")
    # Observed 10 days ago at price 100; matures at observed_at + 5d.
    observed_at = datetime.now(UTC) - timedelta(days=10)
    oid = store.record_observation(
        symbol="AAPL",
        score_name="pullback_risk",
        score_value=73.0,
        observed_at=observed_at,
        horizon_days=5,
        outcome_name="return_negative",
        starting_price=100.0,
    )
    # Bar at maturation date is below start → return_negative fires.
    matures_at = observed_at + timedelta(days=5)
    idx = pd.date_range(start=matures_at, periods=3, freq="D")
    closes = np.array([95.0, 96.0, 97.0])
    df = pd.DataFrame(
        {
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [1_000_000] * 3,
        },
        index=idx,
    )
    bar_cache.write_bars("AAPL", TimeFrame.DAY_1, df)

    worker = CalibrationMaturationWorker(
        store, bars_cache_price_lookup, interval_seconds=60.0
    )
    settled = await worker.tick()
    assert settled == 1
    obs = next(o for o in store.all_observations() if o.id == oid)
    assert obs.outcome_value is True
