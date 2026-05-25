"""Backtest a pairs-trading strategy on Kalman-filtered spreads.

Trading rule:
  - Enter long-spread (long Y, short beta*X) when spread < -k*sigma
  - Enter short-spread (short Y, long beta*X) when spread > +k*sigma
  - Exit when the spread returns to zero (crosses to opposite side of mean)
  - Position sizing: $1 notional per leg
  - Transaction costs applied on entry and exit, per leg

All signals computed using only past data (Kalman filter uses filtered
estimates, not smoothed). Burn-in period at the start of the series gives
the filter time to stabilize before we count trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .kalman import KalmanResult


@dataclass
class Trade:
    """A single completed round-trip trade."""
    pair: str
    direction: int          # +1 = long-spread, -1 = short-spread
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_spread: float
    exit_spread: float
    entry_beta: float
    n_bars_held: int
    gross_pnl: float        # before costs
    cost: float             # transaction cost (round-trip, both legs)
    net_pnl: float          # gross - cost


@dataclass
class BacktestResult:
    """Aggregate results of a backtest on one pair."""
    pair: str
    trades: list[Trade] = field(default_factory=list)
    daily_pnl: pd.Series | None = None    # mark-to-market daily P&L
    burn_in_days: int = 252

    # Aggregate stats — computed by summary()
    def summary(self) -> dict:
        if not self.trades:
            return {
                "pair": self.pair,
                "n_trades": 0,
                "total_net_pnl": 0.0,
                "win_rate": float("nan"),
                "avg_trade_net": float("nan"),
                "sharpe_daily": float("nan"),
                "max_drawdown": float("nan"),
            }

        pnls = np.array([t.net_pnl for t in self.trades])
        gross = np.array([t.gross_pnl for t in self.trades])
        costs = np.array([t.cost for t in self.trades])

        # Daily-P&L-based Sharpe (annualized).
        if self.daily_pnl is not None and self.daily_pnl.std() > 0:
            sharpe = (self.daily_pnl.mean() / self.daily_pnl.std()) * np.sqrt(252)
        else:
            sharpe = float("nan")

        # Max drawdown on the cumulative P&L curve.
        if self.daily_pnl is not None:
            cum = self.daily_pnl.cumsum()
            running_max = cum.cummax()
            drawdown = cum - running_max
            max_dd = float(drawdown.min())
        else:
            max_dd = float("nan")

        return {
            "pair": self.pair,
            "n_trades": len(self.trades),
            "win_rate": float(np.mean(pnls > 0)),
            "total_gross_pnl": float(gross.sum()),
            "total_costs": float(costs.sum()),
            "total_net_pnl": float(pnls.sum()),
            "avg_trade_net": float(pnls.mean()),
            "sharpe_daily": float(sharpe),
            "max_drawdown": max_dd,
        }


def backtest_pair(
    kalman_result: KalmanResult,
    prices: pd.DataFrame,
    entry_sigma: float = 2.0,
    cost_bps_per_leg: float = 10.0,
    burn_in_days: int = 252,
) -> BacktestResult:
    """Run a backtest on one pair.

    Parameters
    ----------
    kalman_result : KalmanResult
        Output of fit_kalman_pair, providing the dynamic spread and beta series.
    prices : pd.DataFrame
        Full price DataFrame (used to compute leg P&L from price changes).
    entry_sigma : float
        Enter a position when |spread| exceeds this many standard deviations.
    cost_bps_per_leg : float
        Round-trip transaction cost per leg, in basis points. 10 bps = 0.001.
        Total cost per trade is 2 * cost_bps_per_leg * notional (two legs).
    burn_in_days : int
        Number of initial days during which the Kalman filter is warming up;
        no trades are entered during this period.

    Returns
    -------
    BacktestResult
        Trades and daily mark-to-market P&L.
    """
    y, x = kalman_result.y, kalman_result.x
    pair_name = f"{y}/{x}"

    spread = kalman_result.spread.copy()
    beta = kalman_result.beta_series.copy()
    sigma_global = kalman_result.spread_std

    if sigma_global <= 0:
        # Degenerate filter; no trades.
        return BacktestResult(pair=pair_name, trades=[], daily_pnl=None,
                              burn_in_days=burn_in_days)

    # Rolling 60-day std of the spread, computed from past innovations only.
    # The .shift(1) ensures the std at time t uses only innovations through t-1,
    # preventing look-ahead bias. Adapts to changing volatility, which is what
    # production pairs strategies do. We fill the early NaN values (before the
    # rolling window is full) with the post-warmup global sigma from kalman.py.
    rolling_sigma = spread.rolling(window=60, min_periods=30).std().shift(1)
    rolling_sigma = rolling_sigma.fillna(sigma_global)

    # Cost as a fraction. Two legs per trade, entry + exit:
    cost_per_trade = 4.0 * (cost_bps_per_leg / 10000.0)

    # Align price data with the spread series.
    pair_prices = prices[[y, x]].dropna().reindex(spread.index)

    # Trading state machine.
    position = 0           # +1 long-spread, -1 short-spread, 0 flat
    entry_idx = None
    entry_spread_val = 0.0
    entry_beta_val = 0.0
    entry_y_price = 0.0
    entry_x_price = 0.0

    trades: list[Trade] = []
    daily_pnl_records: list[tuple[pd.Timestamp, float]] = []
    realized_pnl_to_date = 0.0

    for i in range(len(spread)):
        if i < burn_in_days:
            daily_pnl_records.append((spread.index[i], 0.0))
            continue

        date = spread.index[i]
        s = spread.iloc[i]
        b = beta.iloc[i]
        sig_t = rolling_sigma.iloc[i]
        entry_threshold = entry_sigma * sig_t
        py = pair_prices[y].iloc[i]
        px = pair_prices[x].iloc[i]

        # Mark-to-market: if we have a position, compute its current P&L
        # versus the entry. We accumulate this each day.
        if position != 0 and entry_idx is not None:
            # Long-spread means we bought $1 of Y and shorted beta_entry * X (worth $beta_entry).
            # MTM relative to entry, normalized to entry notional.
            ret_y = (py / entry_y_price) - 1.0
            ret_x = (px / entry_x_price) - 1.0
            # Net P&L on $1-of-Y, $beta-of-X position (sign = position):
            mtm_pnl = position * (ret_y - entry_beta_val * ret_x)
            # Daily *change* in mark-to-market: difference from prior day.
            # We'll compute this by storing the cumulative mark each day.
            current_mark = realized_pnl_to_date + mtm_pnl
        else:
            current_mark = realized_pnl_to_date

        # Decide what to do at the close of this bar.
        new_position = position

        if position == 0:
            # Look for an entry.
            if s > entry_threshold:
                new_position = -1   # short the spread (Y too expensive)
            elif s < -entry_threshold:
                new_position = +1   # long the spread (Y too cheap)
        else:
            # Look for an exit: spread has crossed zero relative to entry side.
            if position == +1 and s >= 0:
                new_position = 0
            elif position == -1 and s <= 0:
                new_position = 0

        # If position is changing, record the trade.
        if new_position != position:
            if position != 0:
                # Closing a position.
                ret_y = (py / entry_y_price) - 1.0
                ret_x = (px / entry_x_price) - 1.0
                gross = position * (ret_y - entry_beta_val * ret_x)
                net = gross - cost_per_trade
                trades.append(Trade(
                    pair=pair_name,
                    direction=position,
                    entry_date=spread.index[entry_idx],
                    exit_date=date,
                    entry_spread=entry_spread_val,
                    exit_spread=float(s),
                    entry_beta=entry_beta_val,
                    n_bars_held=i - entry_idx,
                    gross_pnl=float(gross),
                    cost=float(cost_per_trade),
                    net_pnl=float(net),
                ))
                realized_pnl_to_date += net
                current_mark = realized_pnl_to_date

            if new_position != 0:
                # Opening a new position.
                entry_idx = i
                entry_spread_val = float(s)
                entry_beta_val = float(b)
                entry_y_price = float(py)
                entry_x_price = float(px)

            position = new_position

        daily_pnl_records.append((date, current_mark))

    # Force-close any open position at the end of the backtest.
    if position != 0 and entry_idx is not None:
        date = spread.index[-1]
        py = pair_prices[y].iloc[-1]
        px = pair_prices[x].iloc[-1]
        ret_y = (py / entry_y_price) - 1.0
        ret_x = (px / entry_x_price) - 1.0
        gross = position * (ret_y - entry_beta_val * ret_x)
        net = gross - cost_per_trade
        trades.append(Trade(
            pair=pair_name,
            direction=position,
            entry_date=spread.index[entry_idx],
            exit_date=date,
            entry_spread=entry_spread_val,
            exit_spread=float(spread.iloc[-1]),
            entry_beta=entry_beta_val,
            n_bars_held=len(spread) - 1 - entry_idx,
            gross_pnl=float(gross),
            cost=float(cost_per_trade),
            net_pnl=float(net),
        ))
        realized_pnl_to_date += net
        # Update the last entry of daily_pnl_records.
        daily_pnl_records[-1] = (date, realized_pnl_to_date)

    # Convert daily P&L records to a Series, then take first differences
    # to get daily returns.
    cum_pnl = pd.Series(
        [r[1] for r in daily_pnl_records],
        index=[r[0] for r in daily_pnl_records],
        name="cum_pnl",
    )
    daily_pnl = cum_pnl.diff().fillna(0.0)
    daily_pnl.name = f"{pair_name}_daily_pnl"

    return BacktestResult(
        pair=pair_name,
        trades=trades,
        daily_pnl=daily_pnl,
        burn_in_days=burn_in_days,
    )