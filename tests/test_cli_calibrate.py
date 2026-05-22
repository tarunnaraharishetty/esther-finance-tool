"""Coverage for the ``esther calibrate`` admin CLI group."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner


def _enable_calibration(
    monkeypatch: pytest.MonkeyPatch, db_path: Path
) -> None:
    """Point Settings.calibration_store_path at a tmp file + clear cache."""
    monkeypatch.setenv("CALIBRATION_STORE_PATH", str(db_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()


def _disable_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIBRATION_STORE_PATH", "")
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()


def _seed_settled(
    store: object,
    *,
    score_name: str,
    score_value: float,
    n: int,
    hits: int,
    outcome_name: str,
    horizon_days: int = 5,
) -> None:
    """Helper: write ``n`` settled observations with given hit count."""
    base = datetime.now(UTC) - timedelta(days=horizon_days + 1)
    for i in range(n):
        oid = store.record_observation(  # type: ignore[attr-defined]
            symbol=f"S{i % 5:02d}",
            score_name=score_name,
            score_value=score_value,
            observed_at=base + timedelta(hours=i),
            horizon_days=horizon_days,
            outcome_name=outcome_name,
            starting_price=100.0,
        )
        store.record_outcome(oid, i < hits)  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Discovery / disabled-store
# -----------------------------------------------------------------------------


def test_calibrate_group_appears_in_help(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``esther calibrate --help`` lists build/status/cleanup."""
    _enable_calibration(monkeypatch, tmp_path / "cal.db")
    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["calibrate", "--help"])
    assert res.exit_code == 0, res.output
    for sub in ("build", "status", "cleanup"):
        assert sub in res.output, f"missing subcommand in help: {sub}"


@pytest.mark.parametrize("sub", ["build", "status", "cleanup"])
def test_disabled_store_exits_two_with_clear_message(
    monkeypatch: pytest.MonkeyPatch, sub: str
) -> None:
    """``CALIBRATION_STORE_PATH=`` → exit 2 + 'disabled' message."""
    _disable_calibration(monkeypatch)
    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["calibrate", sub])
    assert res.exit_code == 2
    assert "disabled" in res.output.lower()


# -----------------------------------------------------------------------------
# build
# -----------------------------------------------------------------------------


def test_build_on_empty_store_prints_no_settled_observations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _enable_calibration(monkeypatch, tmp_path / "cal.db")
    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["calibrate", "build"])
    assert res.exit_code == 0, res.output
    assert "no settled observations" in res.output.lower()


def test_build_materializes_buckets_for_both_outcomes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "cal.db"
    _enable_calibration(monkeypatch, db)
    from src.intelligence.calibration import CalibrationStore

    store = CalibrationStore(db)
    # Seed both outcomes with enough observations to produce non-empty
    # buckets. The CLI must run build for both in one invocation.
    _seed_settled(
        store,
        score_name="pullback_risk",
        score_value=75.0,
        n=50,
        hits=35,
        outcome_name="return_negative",
    )
    _seed_settled(
        store,
        score_name="rebound_potential",
        score_value=15.0,
        n=40,
        hits=20,
        outcome_name="return_positive",
    )
    store.close()

    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["calibrate", "build"])
    assert res.exit_code == 0, res.output
    assert "Building tables" in res.output
    # Both outcome names should appear in the output as separate lines.
    assert "return_negative" in res.output
    assert "return_positive" in res.output

    # Verify the bucket cache landed in the DB.
    store2 = CalibrationStore(db)
    table = store2.load_table()
    pullback = [b for b in table.buckets if b.score_name == "pullback_risk"]
    rebound = [b for b in table.buckets if b.score_name == "rebound_potential"]
    assert pullback and rebound
    # Hit rate is preserved from the seeded ratio.
    assert any(abs(b.hit_rate - 0.70) < 1e-6 for b in pullback)
    assert any(abs(b.hit_rate - 0.50) < 1e-6 for b in rebound)


def test_build_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two consecutive builds produce the same bucket set."""
    db = tmp_path / "cal.db"
    _enable_calibration(monkeypatch, db)
    from src.intelligence.calibration import CalibrationStore

    store = CalibrationStore(db)
    _seed_settled(
        store,
        score_name="pullback_risk",
        score_value=75.0,
        n=50,
        hits=35,
        outcome_name="return_negative",
    )
    store.close()

    from src.main import cli

    runner = CliRunner()
    runner.invoke(cli, ["calibrate", "build"])
    runner.invoke(cli, ["calibrate", "build"])

    store2 = CalibrationStore(db)
    table = store2.load_table()
    pullback = [b for b in table.buckets if b.score_name == "pullback_risk"]
    # Idempotent: still exactly one bucket at [70, 80) with the
    # original count (not 100, which would indicate duplication).
    assert len(pullback) == 1
    assert pullback[0].n_observations == 50


def test_build_respects_horizon_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "cal.db"
    _enable_calibration(monkeypatch, db)
    from src.intelligence.calibration import CalibrationStore

    store = CalibrationStore(db)
    _seed_settled(
        store,
        score_name="pullback_risk",
        score_value=75.0,
        n=50,
        hits=35,
        outcome_name="return_negative",
        horizon_days=10,
    )
    store.close()

    from src.main import cli

    runner = CliRunner()
    # Default horizon is 5 — won't match the 10-day observations.
    res = runner.invoke(cli, ["calibrate", "build"])
    assert res.exit_code == 0
    store2 = CalibrationStore(db)
    default_table = store2.load_table()
    assert all(
        b.score_name != "pullback_risk" for b in default_table.buckets
    ), "default horizon should not match the 10-day observations"

    # Now build at horizon=10 → match.
    res = runner.invoke(cli, ["calibrate", "build", "--horizon", "10"])
    assert res.exit_code == 0
    store3 = CalibrationStore(db)
    overridden_table = store3.load_table()
    horizon10 = [
        b
        for b in overridden_table.buckets
        if b.score_name == "pullback_risk" and b.horizon_days == 10
    ]
    assert horizon10 and horizon10[0].n_observations == 50


# -----------------------------------------------------------------------------
# status
# -----------------------------------------------------------------------------


def test_status_on_empty_store_prints_zero_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _enable_calibration(monkeypatch, tmp_path / "cal.db")
    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["calibrate", "status"])
    assert res.exit_code == 0, res.output
    assert "0 total" in res.output
    assert "table not built yet" in res.output


def test_status_reports_observation_and_bucket_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "cal.db"
    _enable_calibration(monkeypatch, db)
    from src.intelligence.calibration import CalibrationStore

    store = CalibrationStore(db)
    _seed_settled(
        store,
        score_name="pullback_risk",
        score_value=75.0,
        n=50,
        hits=35,
        outcome_name="return_negative",
    )
    # Also one unsettled observation to confirm the split.
    store.record_observation(
        symbol="UNSETTLED",
        score_name="pullback_risk",
        score_value=50.0,
        observed_at=datetime.now(UTC),
        horizon_days=5,
        outcome_name="return_negative",
        starting_price=100.0,
    )
    store.close()

    from src.main import cli

    runner = CliRunner()
    # Build first so the table has populated buckets.
    runner.invoke(cli, ["calibrate", "build"])
    res = runner.invoke(cli, ["calibrate", "status"])
    assert res.exit_code == 0, res.output
    assert "51 total" in res.output
    assert "50 settled" in res.output
    assert "1 pending" in res.output
    # Output mentions the materialized bucket for the populated pairing.
    assert "pullback_risk" in res.output
    assert "return_negative" in res.output


# -----------------------------------------------------------------------------
# cleanup
# -----------------------------------------------------------------------------


def test_cleanup_removes_old_unsettled_observations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Old unsettled observations are dropped; settled and fresh are kept."""
    db = tmp_path / "cal.db"
    _enable_calibration(monkeypatch, db)
    from src.intelligence.calibration import CalibrationStore

    store = CalibrationStore(db)
    long_ago = datetime.now(UTC) - timedelta(days=60)
    recent = datetime.now(UTC) - timedelta(days=2)

    # Stuck unsettled (60 days old) — should be removed.
    stuck_oid = store.record_observation(
        symbol="STUCK",
        score_name="pullback_risk",
        score_value=50.0,
        observed_at=long_ago,
        horizon_days=5,
        outcome_name="return_negative",
        starting_price=100.0,
    )
    # Settled long ago — should NOT be removed (calibration data).
    settled_oid = store.record_observation(
        symbol="SETTLED",
        score_name="pullback_risk",
        score_value=75.0,
        observed_at=long_ago,
        horizon_days=5,
        outcome_name="return_negative",
        starting_price=100.0,
    )
    store.record_outcome(settled_oid, True)
    # Recent unsettled — should NOT be removed (still maturing).
    recent_oid = store.record_observation(
        symbol="RECENT",
        score_name="pullback_risk",
        score_value=60.0,
        observed_at=recent,
        horizon_days=5,
        outcome_name="return_negative",
        starting_price=100.0,
    )
    store.close()

    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["calibrate", "cleanup", "--older-than", "30"])
    assert res.exit_code == 0, res.output
    assert "Removed 1" in res.output

    store2 = CalibrationStore(db)
    remaining_ids = {o.id for o in store2.all_observations()}
    assert stuck_oid not in remaining_ids
    assert settled_oid in remaining_ids
    assert recent_oid in remaining_ids


def test_cleanup_negative_older_than_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _enable_calibration(monkeypatch, tmp_path / "cal.db")
    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["calibrate", "cleanup", "--older-than", "-1"])
    assert res.exit_code == 2
    assert "non-negative" in res.output


def test_cleanup_default_thirty_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No --older-than → uses 30-day default."""
    _enable_calibration(monkeypatch, tmp_path / "cal.db")
    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["calibrate", "cleanup"])
    assert res.exit_code == 0, res.output
    assert "30 day" in res.output


def test_cleanup_zero_older_than_removes_all_unsettled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--older-than 0 is an explicit "drop everything unsettled" command.

    Settled observations are still preserved — cleanup never touches them.
    """
    db = tmp_path / "cal.db"
    _enable_calibration(monkeypatch, db)
    from src.intelligence.calibration import CalibrationStore

    store = CalibrationStore(db)
    store.record_observation(
        symbol="A",
        score_name="pullback_risk",
        score_value=50.0,
        observed_at=datetime.now(UTC),
        horizon_days=5,
        outcome_name="return_negative",
        starting_price=100.0,
    )
    settled_oid = store.record_observation(
        symbol="B",
        score_name="pullback_risk",
        score_value=80.0,
        observed_at=datetime.now(UTC),
        horizon_days=5,
        outcome_name="return_negative",
        starting_price=100.0,
    )
    store.record_outcome(settled_oid, True)
    store.close()

    from src.main import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["calibrate", "cleanup", "--older-than", "0"])
    assert res.exit_code == 0, res.output

    store2 = CalibrationStore(db)
    remaining = store2.all_observations()
    # Only the settled observation survives.
    assert len(remaining) == 1
    assert remaining[0].id == settled_oid
