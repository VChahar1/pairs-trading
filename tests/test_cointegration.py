"""Tests for the cointegration module.

We use synthetic data with known properties to verify the tests behave
correctly: a cointegrated pair should produce a low p-value, an uncointegrated
pair should not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pairs.cointegration import (
    coint_test,
    all_pairs_coint,
    bonferroni_threshold,
    benjamini_hochberg,
)


def _make_cointegrated_pair(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Construct two random-walk series whose spread is stationary.

    X is a random walk. Y = 1.5 * X + stationary noise.
    The spread Y - 1.5 * X is by construction stationary, so they
    should test as strongly cointegrated.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    # Mean-zero stationary noise (AR(1) with small coefficient).
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + rng.standard_normal()
    y = 1.5 * x + noise
    return pd.DataFrame(
        {"A": y, "B": x},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def _make_independent_random_walks(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Two truly independent random walks. They should NOT cointegrate."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = np.cumsum(rng.standard_normal(n))
    return pd.DataFrame(
        {"A": y, "B": x},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def test_cointegrated_pair_has_low_pvalue():
    prices = _make_cointegrated_pair(n=500, seed=42)
    result = coint_test(prices, "A", "B")
    assert result.p_value < 0.01, (
        f"Expected low p-value for a constructed cointegrated pair, "
        f"got p={result.p_value:.4f}"
    )


def test_independent_random_walks_dont_cointegrate():
    """Two independent random walks should usually fail to cointegrate.

    Because cointegration tests have finite power, a single instance can
    sometimes give a false positive. We use multiple seeds and check that
    the *median* p-value is well above 0.05.
    """
    p_values = []
    for seed in range(20):
        prices = _make_independent_random_walks(n=500, seed=seed)
        p_values.append(coint_test(prices, "A", "B").p_value)
    median_p = float(np.median(p_values))
    assert median_p > 0.1, (
        f"Independent random walks should mostly not cointegrate; "
        f"median p-value across 20 seeds was {median_p:.3f}"
    )


def test_test_all_pairs_returns_sorted_dataframe():
    prices = _make_cointegrated_pair(n=200, seed=0)
    # Add a third uncorrelated column for a three-way test.
    rng = np.random.default_rng(99)
    prices["C"] = np.cumsum(rng.standard_normal(len(prices)))

    df = all_pairs_coint(prices)
    assert len(df) == 3  # C(3, 2) = 3 pairs
    # Sorted ascending by p-value.
    assert df["p_value"].is_monotonic_increasing


def test_bonferroni_threshold():
    assert bonferroni_threshold(100, alpha=0.05) == pytest.approx(0.0005)
    assert bonferroni_threshold(1, alpha=0.05) == pytest.approx(0.05)


def test_benjamini_hochberg_monotonicity():
    """Adjusted p-values must be non-decreasing in the input p-values."""
    p = pd.Series([0.001, 0.01, 0.04, 0.06, 0.5])
    adj = benjamini_hochberg(p)
    # The adjusted series, sorted by input rank, should be non-decreasing.
    assert adj.is_monotonic_increasing


def test_benjamini_hochberg_bounded():
    p = pd.Series([0.001, 0.01, 0.04, 0.06, 0.5])
    adj = benjamini_hochberg(p)
    assert (adj >= 0).all() and (adj <= 1).all()