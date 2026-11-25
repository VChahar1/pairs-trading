# Pairs-trading study

This study screens US equities for cointegration, estimates time-varying hedge ratios with a Kalman filter, and backtests a mean-reversion strategy with transaction costs. The saved experiment has a negative result: **the strategy did not work after costs**.

## Data

`src/pairs/universe.py` defines a fixed selection of US large-cap equities across several industries. The downloader requests daily adjusted Yahoo Finance closes from `2020-01-01` to `2025-01-01`, with the end date exclusive. Downloads and intermediate artifacts are cached locally under `data/`; the notebooks should be run in order so later steps can use those artifacts.

## Method

- Screen candidate pairs with the Engle–Granger cointegration test. Helpers provide Bonferroni and Benjamini–Hochberg multiple-testing adjustments.
- Fit a state-space regression on log prices, with Kalman-filter estimates of the hedge ratio and intercept. Noise parameters are estimated by maximum likelihood.
- Trade spread deviations from a rolling volatility estimate, exit on a zero crossing, and hold the entry hedge ratio for each trade.
- The main saved backtest uses a 252-day burn-in, a 2-sigma entry threshold and 10 basis points per leg on entry and exit. The implementation charges four leg-side costs for a completed round trip.
- Average the candidate pairs' daily normalized P&Ls and calculate an annualized daily-P&L Sharpe ratio. Notebook 04 also varies entry thresholds and costs.

## Saved results

These figures are verified against the saved tables in [notebook 04](notebooks/04_backtest.ipynb). This cleanup did not rerun the model fitting or regenerate the backtest.

| Equal-weighted candidate-pair portfolio | Annualized Sharpe | Cumulative normalized P&L |
|---|---:|---:|
| No transaction costs | −0.412950 | −0.068332 |
| 10 bps per leg per side | **−1.688082** | **−0.263332** |

The notebook's main run includes eight candidate pairs. Its printed summary rounds net Sharpe to **−1.688**, consistent with the sensitivity tables. These P&L units follow the implementation's position normalization; they are not a fully specified investor capital return.

## Conclusion and limitations

The tested configuration loses before costs, and costs make the result worse. Cointegration and a changing hedge ratio do not by themselves establish a tradable edge. A useful next study would separate pair selection, parameter estimation and evaluation in time, and assess whether a forecast advantage can survive realistic execution and costs.

The current experiment has material limits:

- Pair screening and Kalman noise-parameter estimation use the analysis sample. Filtered states are one-sided, but that alone does not make the entire study a walk-forward out-of-sample evaluation.
- A hand-picked surviving-stock universe can introduce selection and survivorship bias.
- Same-close signals and fills, fixed transaction costs and simplified notional accounting omit important execution, borrowing and financing effects.
- Sensitivity tables describe the saved sample; selecting a better setting from them would require a separate evaluation period.
- Downloaded adjusted prices and dependency versions can change. A fresh run can produce different estimates.

## How to run

```bash
git clone https://github.com/VChahar1/pairs-trading.git
cd pairs-trading
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
jupyter notebook notebooks/
```

Run `01_data_exploration`, `02_cointegration_analysis`, `03_kalman_filter` and `04_backtest` in that order. The first step needs Yahoo Finance access. Intermediate artifacts are written under `data/processed/` and should be generated locally.

## Tests and cleanup

`tests/` checks data calculations and quality flags, cointegration helpers, Kalman estimates and backtest behavior. The cleanup removes the unused `notebooks/Untitled.ipynb`, `notebooks/Untitled1.ipynb` and root `test.py`, and repairs stray prose/indentation that prevented the Bonferroni helper from parsing. The correction retains its existing `alpha / n_tests` formula; the trading and estimation logic is unchanged.
