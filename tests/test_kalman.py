"""Tests for the Kalman filter module.

We construct synthetic data with a known time-varying hedge ratio and
verify that the filter recovers it. This is the right way to test
statistical procedures: build inputs whose ground truth you control,
and check that the procedure recovers it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pairs.kalman import fit_kalman_pair, adf_pvalue


def _construct_pair_with_drifting_beta(
    n: int = 1000,
    beta_start: float = 0.8,
    beta_end: float = 1.4,
    obs_noise: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """Construct a synthetic pair where beta drifts linearly from start to end.

    log(Y_t) = beta_t * log(X_t) + alpha + noise
    where beta_t goes from beta_start to beta_end linearly.
    log(X_t) is a random walk.
    """
    rng = np.random.default_rng(seed)
    log_x = np.cumsum(rng.standard_normal(n)) * 0.02 + 4.0   # log(price) ~ 4
    beta_t = np.linspace(beta_start, beta_end, n)
    alpha = -1.0
    noise = rng.standard_normal(n) * obs_noise
    log_y = beta_t * log_x + alpha + noise

    # Convert log-prices back to prices for the API.
    return pd.DataFrame(
        {"A": np.exp(log_y), "B": np.exp(log_x)},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def test_kalman_recovers_drifting_beta():
    """The filter should track a linearly-drifting hedge ratio."""
    prices = _construct_pair_with_drifting_beta(
        n=1000, beta_start=0.8, beta_end=1.4, seed=42
    )

    result = fit_kalman_pair(prices, "A", "B")

    # The filter has a warm-up period; check the second half of the series.
    half = len(result.beta_series) // 2
    beta_late = result.beta_series.iloc[half:]

    # By the second half, beta should be roughly between 1.0 and 1.5.
    assert beta_late.mean() > 0.9, (
        f"Filter failed to track upward-drifting beta; "
        f"second-half mean was {beta_late.mean():.3f}"
    )
    assert beta_late.mean() < 1.5

    # And the spread should be approximately stationary.
    spread_late = result.spread.iloc[half:]
    pvalue = adf_pvalue(spread_late)
    assert pvalue < 0.05, (
        f"Spread should be stationary by construction; ADF p={pvalue:.3f}"
    )


def test_kalman_static_pair():
    """A pair with constant beta should produce a roughly constant beta estimate."""
    prices = _construct_pair_with_drifting_beta(
        n=1000, beta_start=1.2, beta_end=1.2, seed=1
    )

    result = fit_kalman_pair(prices, "A", "B")

    # After warm-up, beta should hover near 1.2.
    half = len(result.beta_series) // 2
    beta_late = result.beta_series.iloc[half:]
    assert abs(beta_late.mean() - 1.2) < 0.2, (
        f"Filter should recover beta=1.2 on static pair; got {beta_late.mean():.3f}"
    )


def test_fit_too_few_observations_raises():
    """Pairs with too little data should raise rather than fit poorly."""
    prices = pd.DataFrame(
        {"A": np.exp(np.random.randn(50)), "B": np.exp(np.random.randn(50))},
        index=pd.date_range("2020-01-01", periods=50, freq="B"),
    )
    with pytest.raises(ValueError, match="at least 100"):
        fit_kalman_pair(prices, "A", "B")