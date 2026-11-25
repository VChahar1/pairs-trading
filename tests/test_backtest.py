"Tests for the backtest module."

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pairs.backtest import backtest_pair
from src.pairs.kalman import KalmanResult


def _make_synthetic_kalman_result(
    n_days: int = 500,
    spread_pattern: str = "mean_reverting",
    seed: int = 0,
) -> tuple[KalmanResult, pd.DataFrame]:
    "Build a synthetic KalmanResult + matching prices for backtest testing."
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    if spread_pattern == "mean_reverting":
        base = rng.normal(0, 0.5, n_days)

        for start in range(80, n_days - 10, 40):
            sign = 1.0 if (start // 40) % 2 == 0 else -1.0
            base[start : start + 5] += sign * 5.0
        spread_values = base
        sigma = 1.0
    elif spread_pattern == "flat":
        spread_values = rng.normal(0, 0.1, n_days)
        sigma = 1.0
    else:
        raise ValueError(f"Unknown pattern: {spread_pattern}")

    spread = pd.Series(spread_values, index=dates, name="spread")
    beta = pd.Series(1.0, index=dates, name="beta")
    alpha = pd.Series(0.0, index=dates, name="alpha")

    log_x = np.cumsum(rng.normal(0, 0.01, n_days))
    log_y = log_x + 0.05 * spread_values
    prices = pd.DataFrame(
        {"A": np.exp(log_y), "B": np.exp(log_x)},
        index=dates,
    )

    kr = KalmanResult(
        y="A",
        x="B",
        beta_series=beta,
        alpha_series=alpha,
        spread=spread,
        spread_std=sigma,
        q_estimated=1e-6,
        r_estimated=1e-4,
        converged=True,
        log_likelihood=0.0,
    )
    return kr, prices


def test_backtest_mean_reverting_spread_makes_trades():
    """An oscillating spread should produce multiple round-trip trades."""
    kr, prices = _make_synthetic_kalman_result(
        n_days=500, spread_pattern="mean_reverting", seed=0
    )
    result = backtest_pair(
        kr, prices, entry_sigma=2.0, cost_bps_per_leg=0.0, burn_in_days=50
    )

    assert len(result.trades) >= 3, (
        f"Expected several trades on oscillating spread; got {len(result.trades)}"
    )


def test_backtest_flat_spread_makes_few_trades():
    """A small-amplitude spread should rarely cross 2-sigma thresholds."""
    kr, prices = _make_synthetic_kalman_result(
        n_days=500, spread_pattern="flat", seed=1
    )
    result = backtest_pair(
        kr, prices, entry_sigma=2.0, cost_bps_per_leg=0.0, burn_in_days=50
    )

    assert len(result.trades) <= 30


def test_backtest_costs_reduce_pnl():
    """With positive transaction costs, net P&L should be less than gross."""
    kr, prices = _make_synthetic_kalman_result(
        n_days=500, spread_pattern="mean_reverting", seed=2
    )
    no_cost = backtest_pair(kr, prices, cost_bps_per_leg=0.0, burn_in_days=50)
    with_cost = backtest_pair(kr, prices, cost_bps_per_leg=10.0, burn_in_days=50)

    no_cost_total = sum(t.net_pnl for t in no_cost.trades)
    with_cost_total = sum(t.net_pnl for t in with_cost.trades)

    assert with_cost_total < no_cost_total, (
        f"Cost should reduce net P&L; got no-cost={no_cost_total:.4f}, "
        f"with-cost={with_cost_total:.4f}"
    )


def test_backtest_summary_keys():
    """summary() should return a dict with the expected keys."""
    kr, prices = _make_synthetic_kalman_result(seed=3)
    result = backtest_pair(kr, prices, cost_bps_per_leg=5.0, burn_in_days=50)
    summary = result.summary()
    expected_keys = {
        "pair",
        "n_trades",
        "win_rate",
        "total_net_pnl",
        "sharpe_daily",
        "max_drawdown",
    }
    assert expected_keys.issubset(summary.keys())
