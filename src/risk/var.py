"""Value at Risk (historical and parametric)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def historical_var(returns: "pd.Series", confidence: float = 0.95) -> float:
    """Historical VaR as a positive fraction (e.g. 0.025 == 2.5% loss)."""
    if returns.empty:
        return 0.0
    quantile = returns.quantile(1 - confidence)
    return float(-quantile)


def parametric_var(returns: "pd.Series", confidence: float = 0.95) -> float:
    """Gaussian VaR. Faster but assumes normality."""
    from scipy.stats import norm  # local import: scipy is optional

    mu = float(returns.mean())
    sigma = float(returns.std())
    z = float(norm.ppf(1 - confidence))
    return float(-(mu + z * sigma))


def monte_carlo_var(
    returns: "pd.Series",
    *,
    confidence: float = 0.95,
    horizon_days: int = 1,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo VaR via bootstrap resampling of historical returns."""
    if returns.empty:
        return 0.0
    rng = np.random.default_rng(seed)
    samples = rng.choice(returns.to_numpy(), size=(n_paths, horizon_days), replace=True)
    horizon_returns = samples.sum(axis=1)
    return float(-np.quantile(horizon_returns, 1 - confidence))
