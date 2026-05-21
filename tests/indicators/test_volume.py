"""Volume z-score tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.volume import VolumeZScore


def test_constant_volume_produces_nan_after_warmup() -> None:
    # std == 0 on a constant series → divide-by-zero → NaN.
    n = 30
    df = pd.DataFrame({"volume": [1_000_000] * n})
    z = VolumeZScore(period=20).compute(df)
    # All values past the warmup window should be NaN (std = 0).
    assert z.tail(5).isna().all()


def test_huge_recent_spike_drives_z_positive() -> None:
    rng = np.random.default_rng(seed=4)
    base = rng.integers(900_000, 1_100_000, size=30).tolist()
    base.append(10_000_000)  # spike
    df = pd.DataFrame({"volume": base})
    z = VolumeZScore(period=20).compute(df)
    assert z.iloc[-1] > 2.0  # at least 2 sigma over baseline


def test_low_recent_volume_drives_z_negative() -> None:
    rng = np.random.default_rng(seed=4)
    base = rng.integers(900_000, 1_100_000, size=30).tolist()
    base.append(10_000)  # collapse
    df = pd.DataFrame({"volume": base})
    z = VolumeZScore(period=20).compute(df)
    assert z.iloc[-1] < -2.0
