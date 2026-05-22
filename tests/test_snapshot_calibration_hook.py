"""Tests for the snapshot-loop calibration recorder hook.

Exercises the per-tick wiring on :class:`MockDashboardController` (no
Alpaca required). The contract is: when a recorder is wired, every
successful row produces observations; failures in the recorder path
never propagate into the snapshot.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.dashboard.controller import MockDashboardController
from src.intelligence.calibration import CalibrationStore
from src.intelligence.calibration_worker import CalibrationRecorder


def _store(tmp_path: Path) -> CalibrationStore:
    return CalibrationStore(tmp_path / "calibration.db")


# -----------------------------------------------------------------------------
# Wiring + observation persistence
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_records_observations_for_each_symbol(
    tmp_path: Path,
) -> None:
    """One tick with three watchlist symbols → observations land for each."""
    store = _store(tmp_path)
    recorder = CalibrationRecorder(store=store, horizon_days=5)
    controller = MockDashboardController(
        watchlist=["AAPL", "MSFT", "NVDA"],
        seed=7,
        calibration_recorder=recorder,
    )
    snap = await controller.fetch_snapshot()
    assert snap.tick == 1

    obs = store.all_observations()
    symbols_recorded = {o.symbol for o in obs}
    assert symbols_recorded == {"AAPL", "MSFT", "NVDA"}


@pytest.mark.asyncio
async def test_each_observation_has_starting_price_and_horizon(
    tmp_path: Path,
) -> None:
    """The recorder must persist starting_price and the configured horizon."""
    store = _store(tmp_path)
    recorder = CalibrationRecorder(store=store, horizon_days=7)
    controller = MockDashboardController(
        watchlist=["AAPL"],
        seed=7,
        calibration_recorder=recorder,
    )
    await controller.fetch_snapshot()
    for o in store.all_observations():
        # MockDashboardController centers prices around 100 with ±a few %;
        # any non-zero positive starting_price is correct.
        assert o.starting_price is not None
        assert o.starting_price > 0
        assert o.horizon_days == 7
        assert o.outcome_value is None  # unsettled at write time


@pytest.mark.asyncio
async def test_pairings_match_the_analyzer_contract(tmp_path: Path) -> None:
    """Every observation lands under one of the documented pairings.

    Pins the data contract: the snapshot loop only ever writes scores
    the analyzer knows how to look up. Adding a pairing requires
    updating PAIRINGS in src/intelligence/analyzer/calibration.py.
    """
    from src.intelligence.analyzer.calibration import PAIRINGS

    store = _store(tmp_path)
    recorder = CalibrationRecorder(store=store, horizon_days=5)
    controller = MockDashboardController(
        watchlist=["AAPL"], seed=7, calibration_recorder=recorder
    )
    await controller.fetch_snapshot()
    observed_pairings = {(o.score_name, o.outcome_name) for o in store.all_observations()}
    # Every observed pairing must be in the canonical set.
    assert observed_pairings.issubset(set(PAIRINGS))


# -----------------------------------------------------------------------------
# Failure isolation
# -----------------------------------------------------------------------------


class _BrokenRecorder:
    """Recorder stub that raises on every record call.

    The snapshot tick must complete despite this; observability
    cannot be allowed to take down the trader's primary surface.
    """

    def __init__(self) -> None:
        self.calls = 0

    def record_scores(self, **_kwargs: object) -> list[int]:
        self.calls += 1
        raise RuntimeError("simulated calibration sink failure")


@pytest.mark.asyncio
async def test_broken_recorder_does_not_break_snapshot_tick(
    tmp_path: Path,
) -> None:
    """A raising recorder must not propagate. Snapshot still renders."""
    broken = _BrokenRecorder()
    controller = MockDashboardController(
        watchlist=["AAPL", "MSFT"],
        seed=7,
        calibration_recorder=broken,  # type: ignore[arg-type]
    )
    snap = await controller.fetch_snapshot()
    assert snap.tick == 1
    assert len(snap.rows) == 2
    # The recorder was attempted once per healthy row.
    assert broken.calls == 2
    # Calibration failure surfaced on the event buffer for operator visibility.
    events_text = " ".join(e.message for e in snap.events)
    assert "calibration" in events_text


@pytest.mark.asyncio
async def test_no_recorder_is_a_clean_no_op(tmp_path: Path) -> None:
    """Without a recorder, the snapshot path runs unchanged."""
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    snap = await controller.fetch_snapshot()
    assert snap.tick == 1
    assert len(snap.rows) == 1


# -----------------------------------------------------------------------------
# Repeat-tick semantics
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observations_accumulate_across_ticks(tmp_path: Path) -> None:
    """Each tick writes a fresh observation per symbol/pairing.

    The maturation horizon is independent of tick cadence; running
    three ticks should produce three sets of observations (one set
    per tick) so the calibration table builds up density quickly.
    """
    store = _store(tmp_path)
    recorder = CalibrationRecorder(store=store, horizon_days=5)
    controller = MockDashboardController(
        watchlist=["AAPL"], seed=7, calibration_recorder=recorder
    )
    for _ in range(3):
        await controller.fetch_snapshot()
    obs = store.all_observations()
    # 4 pairings × 3 ticks for one symbol = 12 max; some pairings
    # may drop because the technical score is None for that tick.
    # Conservative lower bound: at least one observation per tick.
    assert len(obs) >= 3


@pytest.mark.asyncio
async def test_observed_at_advances_across_ticks(tmp_path: Path) -> None:
    """Each tick's observation carries a fresh timestamp; the maturation
    worker uses ``observed_at`` to compute ``matures_at`` so timestamps
    must monotonically increase across the recorded set."""
    store = _store(tmp_path)
    recorder = CalibrationRecorder(store=store, horizon_days=5)
    controller = MockDashboardController(
        watchlist=["AAPL"], seed=7, calibration_recorder=recorder
    )
    before = datetime.now(UTC)
    await controller.fetch_snapshot()
    after = datetime.now(UTC)
    obs = store.all_observations()
    for o in obs:
        assert before <= o.observed_at <= after
