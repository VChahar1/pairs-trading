"""Cointegration testing for candidate pairs.

Implements the Engle-Granger two-step procedure:
  1. OLS regression to find the hedge ratio: Y_t = beta * X_t + alpha + eps_t
  2. Augmented Dickey-Fuller test on the residuals for stationarity

Pairs whose residuals reject the unit-root null are flagged as cointegrated.

We also compute multiple-testing corrections (Bonferroni and Benjamini-Hochberg)
because with 435 possible pairs, ~22 will pass at p<0.05 by chance alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class CointResult:
    """Result of an Engle-Granger cointegration test on one pair.

    Attributes
    ----------
    y, x : str
        Ticker symbols. The convention is Y = beta * X + alpha + spread,
        so the regression is Y regressed on X.
    p_value : float
        p-value from the augmented Dickey-Fuller test on the residuals.
        Lower means stronger evidence of cointegration. Standard threshold
        is 0.05 before multiple-testing correction.
    t_stat : float
        Test statistic from the ADF test. More negative = stronger evidence
        against the unit-root null.
    critical_values : tuple[float, float, float]
        Critical values at the 1%, 5%, and 10% levels for context.
    n_obs : int
        Number of observations used in the test.
    """
    y: str
    x: str
    p_value: float
    t_stat: float
    critical_values: tuple[float, float, float]
    n_obs: int


def coint_test(prices: pd.DataFrame, y_ticker: str, x_ticker: str) -> CointResult:
    """Run the Engle-Granger cointegration test on a single pair.

    Drops dates where either series is missing before testing.
    """
    pair = prices[[y_ticker, x_ticker]].dropna()
    if len(pair) < 30:
        # Too few observations to test meaningfully.
        return CointResult(
            y=y_ticker, x=x_ticker, p_value=np.nan, t_stat=np.nan,
            critical_values=(np.nan, np.nan, np.nan), n_obs=len(pair)
        )

    y = pair[y_ticker].to_numpy()
    x = pair[x_ticker].to_numpy()

    t_stat, p_value, crit = coint(y, x)
    return CointResult(
        y=y_ticker,
        x=x_ticker,
        p_value=float(p_value),
        t_stat=float(t_stat),
        critical_values=(float(crit[0]), float(crit[1]), float(crit[2])),
        n_obs=len(pair),
    )


def all_pairs_coint(
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Run cointegration tests on every unordered pair in `tickers`.

    Returns a DataFrame sorted by ascending p_value (most cointegrated first).
    Each row has columns: y, x, p_value, t_stat, crit_1pct, crit_5pct,
    crit_10pct, n_obs.

    Convention: for a pair (A, B) with A < B alphabetically, we test with
    y=A, x=B. This makes the ordering deterministic and the results
    reproducible. (Cointegration testing is mildly direction-dependent;
    fixing the direction keeps the results clean.)
    """
    if tickers is None:
        tickers = list(prices.columns)
    tickers = sorted(tickers)

    rows = []
    for a, b in combinations(tickers, 2):
        # a < b by construction (combinations preserves input order).
        result = coint_test(prices, y_ticker=a, x_ticker=b)
        rows.append({
            "y": result.y,
            "x": result.x,
            "p_value": result.p_value,
            "t_stat": result.t_stat,
            "crit_1pct": result.critical_values[0],
            "crit_5pct": result.critical_values[1],
            "crit_10pct": result.critical_values[2],
            "n_obs": result.n_obs,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("p_value", ascending=True).reset_index(drop=True)
    return df


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """Bonferroni-corrected significance threshold.

    With n_tests independent tests at family-wise error rate alpha, each
    individual test must have p < alpha / n_tests to be considered significant.
    Very conservative.
    """
    return alpha / n_tests


def benjamini_hochberg(p_values: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Benjamini-Hochberg FDR-adjusted p-values.

    Controls the expected proportion of false discoveries at level alpha,
    which is less conservative than Bonferroni when many true positives exist.

    Returns a Series of adjusted p-values (same index as input). A test is
    significant at FDR alpha if its adjusted p-value < alpha.
    """
    p = p_values.dropna().sort_values()
    n = len(p)
    # BH adjustment: p_adj[i] = min over k>=i of (p[k] * n / (k+1))
    # We compute it via a cumulative minimum from the right.
    ranks = np.arange(1, n + 1)
    adjusted = p.to_numpy() * n / ranks
    # Enforce monotonicity: adjusted p-values can't decrease as rank increases.
    # We do this by taking the running minimum from the right.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = pd.Series(adjusted, index=p.index, name="p_adj_bh")
    # Reattach NaNs for any inputs that were missing.
    return result.reindex(p_values.index)