"""Tests for the data loading and cleaning module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pairs.data import compute_log_returns, check_data_quality


def test_log_returns_basic():
    """log(P_t / P_{t-1}) for known input."""
    prices = pd.DataFrame({
        "A": [100.0, 110.0, 99.0],
        "B": [50.0, 50.5, 51.0],
    }, index=pd.date_range("2024-01-01", periods=3))

    returns = compute_log_returns(prices)

    # First row should be dropped (NaN from .shift(1)).
    assert len(returns) == 2

    # log(110/100) ≈ 0.0953
    assert returns.iloc[0]["A"] == pytest.approx(np.log(1.10), rel=1e-6)
    # log(99/110) ≈ -0.1054
    assert returns.iloc[1]["A"] == pytest.approx(np.log(99 / 110), rel=1e-6)


def test_log_returns_handles_nan():
    """Tickers with missing data shouldn't break the function."""
    prices = pd.DataFrame({
        "A": [100.0, 110.0, 99.0],
        "B": [np.nan, np.nan, 50.0],
    }, index=pd.date_range("2024-01-01", periods=3))

    returns = compute_log_returns(prices)
    assert "A" in returns.columns
    assert "B" in returns.columns


def test_check_data_quality_detects_constant_runs():
    """A ticker with 10+ identical consecutive prices should be flagged."""
    n_days = 30
    prices = pd.DataFrame({
        "GOOD": np.linspace(100, 120, n_days),
        "STALE": [100.0] * 15 + list(np.linspace(101, 110, n_days - 15)),
    }, index=pd.date_range("2024-01-01", periods=n_days))

    summary = check_data_quality(prices)
    suspicious_tickers = [t for t, _ in summary["suspicious_constant_runs"]]
    assert "STALE" in suspicious_tickers
    assert "GOOD" not in suspicious_tickers