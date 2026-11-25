"""Kalman-filter estimation of time-varying hedge ratios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel


@dataclass
class KalmanResult:
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
    def __init__(self, log_y: np.ndarray, log_x: np.ndarray):

        super().__init__(
            endog=log_y,
            k_states=2,
            initialization="approximate_diffuse",
        )

        design = np.zeros((1, 2, len(log_y)))
        design[0, 0, :] = log_x
        design[0, 1, :] = 1.0
        self["design"] = design

        self["transition"] = np.eye(2)
        self["selection"] = np.eye(2)

        self["obs_cov"] = np.array([[1.0]])
        self["state_cov"] = np.eye(2)

    @property
    def param_names(self):
        return ["sigma2_obs", "sigma2_beta", "sigma2_alpha"]

    @property
    def start_params(self):

        return np.array([1e-4, 1e-6, 1e-6])

    def transform_params(self, unconstrained):

        return unconstrained**2

    def untransform_params(self, constrained):
        return np.sqrt(constrained)

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        self["obs_cov", 0, 0] = params[0]
        self["state_cov", 0, 0] = params[1]
        self["state_cov", 1, 1] = params[2]


def fit_kalman_pair(
    prices: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
) -> KalmanResult:
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

    filtered_state = fit.filtered_state
    beta_series = pd.Series(filtered_state[0], index=pair.index, name="beta")
    alpha_series = pd.Series(filtered_state[1], index=pair.index, name="alpha")

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
    from statsmodels.tsa.stattools import adfuller

    series = series.dropna()
    if len(series) < 30:
        return float("nan")
    result = adfuller(series, autolag="AIC")
    return float(result[1])
