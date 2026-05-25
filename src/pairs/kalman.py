"""Kalman-filter estimation of time-varying hedge ratios.

For each candidate pair (Y, X), we model:

  State:        [beta_t, alpha_t]  evolves as a random walk
  Observation:  log(Y_t) = beta_t * log(X_t) + alpha_t + noise

This gives us at each time t:
  - The best estimate of the hedge ratio beta_t
  - The implied dynamic spread (the filter's innovation series)
  - The innovation standard deviation, used later for trading thresholds

The dynamic spread is the analog of the static OLS residual but allows
the hedge ratio to drift. If the static OLS spread looks non-stationary
because beta is drifting, the Kalman innovations should look stationary.

We use statsmodels' state-space framework for the implementation. The
process noise (Q) and observation noise (R) are estimated by maximum
likelihood. Some pairs may not yield well-identified estimates; we
flag those.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel


@dataclass
class KalmanResult:
    """Output of a Kalman-filter fit on one pair.

    Attributes
    ----------
    y, x : str
        Ticker symbols. The relationship modeled is log(Y) = beta * log(X) + alpha.
    beta_series : pd.Series
        Time-varying hedge ratio, indexed by date.
    alpha_series : pd.Series
        Time-varying intercept, indexed by date.
    spread : pd.Series
        Dynamic spread (the Kalman filter's one-step-ahead innovations),
        indexed by date. This is the series we'll trade in Day 4.
    spread_std : float
        Estimated standard deviation of the innovations. Used to
        compute trading thresholds (e.g., enter at ±2 sigma).
    q_estimated : float
        Estimated process-noise variance (the random-walk variance of beta).
    r_estimated : float
        Estimated observation-noise variance.
    converged : bool
        Whether the MLE optimizer converged.
    log_likelihood : float
        Log-likelihood of the fitted model. Higher is better fit.
    """
    y: str
    x: str
    beta_series: pd.Series
    alpha_series: pd.Series
    spread: pd.Series
    spread_std: float
    q_estimated: float
    r_estimated: float
    converged: bool
    log_likelihood: float


class _PairsKalman(MLEModel):
    """State-space model: hedge ratio + intercept as random-walk states.

    State vector: [beta_t, alpha_t]
    Observation: log(Y_t) = beta_t * log(X_t) + alpha_t + noise

    The MLE machinery in statsmodels takes care of running the Kalman
    filter forward and computing the log-likelihood. We just have to
    set up the matrices correctly.
    """

    def __init__(self, log_y: np.ndarray, log_x: np.ndarray):
        # Pass observations to the parent. k_states=2 (beta and alpha).
        super().__init__(
            endog=log_y,
            k_states=2,
            initialization="approximate_diffuse",  # flat prior on initial state
        )

        # The observation design matrix is time-varying: [log(X_t), 1]
        # statsmodels expects shape (n_endog, k_states, nobs) for time-varying.
        design = np.zeros((1, 2, len(log_y)))
        design[0, 0, :] = log_x   # multiplies beta
        design[0, 1, :] = 1.0     # multiplies alpha
        self["design"] = design

        # State transition: identity (random walk for both states).
        self["transition"] = np.eye(2)
        self["selection"] = np.eye(2)

        # Observation noise (R) and state noise covariance (Q) are
        # parameters to be estimated. Initialize to placeholders.
        self["obs_cov"]   = np.array([[1.0]])
        self["state_cov"] = np.eye(2)

    @property
    def param_names(self):
        return ["sigma2_obs", "sigma2_beta", "sigma2_alpha"]

    @property
    def start_params(self):
        # Reasonable starting values for the MLE optimization.
        return np.array([1e-4, 1e-6, 1e-6])

    def transform_params(self, unconstrained):
        # All variances must be positive; we optimize over log-space.
        return unconstrained ** 2

    def untransform_params(self, constrained):
        return np.sqrt(constrained)

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        # params are [sigma2_obs, sigma2_beta, sigma2_alpha]
        self["obs_cov", 0, 0] = params[0]
        self["state_cov", 0, 0] = params[1]
        self["state_cov", 1, 1] = params[2]


def fit_kalman_pair(
    prices: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
) -> KalmanResult:
    """Fit a Kalman filter to one pair and return the dynamic spread.

    Uses log prices throughout. Estimates Q and R by maximum likelihood.
    """
    pair = prices[[y_ticker, x_ticker]].dropna()
    if len(pair) < 100:
        raise ValueError(
            f"Pair {y_ticker}/{x_ticker} has only {len(pair)} observations; "
            f"need at least 100 for stable Kalman estimation."
        )

    log_y = np.log(pair[y_ticker].to_numpy())
    log_x = np.log(pair[x_ticker].to_numpy())

    model = _PairsKalman(log_y, log_x)
    fit = model.fit(disp=False, maxiter=200)

    # Extract the smoothed state estimates. We use the FILTERED estimates
    # (one-sided, only past data), not smoothed (uses future data too),
    # because trading decisions can only use past data.
    filtered_state = fit.filtered_state            # shape (2, n)
    beta_series  = pd.Series(filtered_state[0], index=pair.index, name="beta")
    alpha_series = pd.Series(filtered_state[1], index=pair.index, name="alpha")

    # The innovations: y_t - predicted y_t given filter state at t-1.
    # This is the dynamic spread.
    innovations = pd.Series(
        fit.filter_results.forecasts_error[0],
        index=pair.index,
        name="spread",
    )

    WARMUP = 30
    if len(innovations) > WARMUP + 30:
        spread_std = float(innovations.iloc[WARMUP:].std())
    else:
        spread_std = float(innovations.std())

    return KalmanResult(
        y=y_ticker,
        x=x_ticker,
        beta_series=beta_series,
        alpha_series=alpha_series,
        spread=innovations,
        spread_std=spread_std,
        q_estimated=float(fit.params[1]),
        r_estimated=float(fit.params[0]),
        converged=bool(fit.mle_retvals.get("converged", False)),
        log_likelihood=float(fit.llf),
    )


def adf_pvalue(series: pd.Series) -> float:
    """Augmented Dickey-Fuller p-value on a series.

    Used to test whether the dynamic spread is stationary, which is the
    natural follow-up question after fitting the Kalman model.
    """
    from statsmodels.tsa.stattools import adfuller
    series = series.dropna()
    if len(series) < 30:
        return float("nan")
    result = adfuller(series, autolag="AIC")
    return float(result[1])