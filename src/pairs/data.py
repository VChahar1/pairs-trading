"""Price data download and caching.

We use yfinance for daily adjusted close prices. Data is cached as Parquet
files locally so we don't re-download on every notebook run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from .universe import UNIVERSE, START_DATE, END_DATE

logger = logging.getLogger(__name__)

# Resolve project root from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def download_prices(
    tickers: list[str] = UNIVERSE,
    start: str = START_DATE,
    end: str = END_DATE,
    use_cache: bool = True,
) -> pd.DataFrame:
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / "prices.parquet"

    if use_cache and cache_path.exists():
        logger.info("Loading prices from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Downloading prices for %d tickers from %s to %s",
                len(tickers), start, end)

    # yfinance returns a multi-level column index when downloading multiple
    # tickers. We want a flat DataFrame of close prices.
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts for splits and dividends
        progress=False,
        group_by="ticker",
    )

    # Extract the 'Close' column from each ticker's sub-frame.
    closes = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.levels[0]})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()

    # Cache.
    closes.to_parquet(cache_path)
    logger.info("Cached prices to %s", cache_path)

    return closes


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    
    return (prices / prices.shift(1)).apply(lambda x: x.dropna()).pipe(
        lambda df: df  # placeholder to keep the chain explicit; computed below
    )


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    
    import numpy as np
    return np.log(prices / prices.shift(1)).dropna(how="all")


def check_data_quality(prices: pd.DataFrame) -> dict:
    
    summary: dict = {}

    # 1. Date range
    summary["start"] = prices.index.min()
    summary["end"] = prices.index.max()
    summary["n_days"] = len(prices)
    summary["n_tickers"] = prices.shape[1]

    # 2. Missing values
    n_missing = prices.isna().sum()
    summary["tickers_with_missing"] = (n_missing > 0).sum()
    summary["max_missing_per_ticker"] = int(n_missing.max())

    # 3. Constant-price segments (likely delisting or data error).
    # We flag any ticker that has 10+ consecutive identical closes.
    suspicious = []
    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) == 0:
            continue
        # Find the longest run of equal consecutive values.
        run_lengths = (s != s.shift()).cumsum().value_counts()
        max_run = int(run_lengths.max()) if len(run_lengths) else 0
        if max_run >= 10:
            suspicious.append((col, max_run))
    summary["suspicious_constant_runs"] = suspicious

    return summary
