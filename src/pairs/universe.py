"""Definition of the trading universe.

We use a hand-picked subset of US large-cap equities chosen to include
multiple pairs that *plausibly* cointegrate (same-industry pairs) along
with stocks that *plausibly* don't (cross-industry). The universe is small
enough to make all-pairs analysis cheap and large enough to find a few
genuine candidates.

This list is fixed at the start of the project; we don't change it after
seeing results, to avoid look-ahead bias.
"""

# Curated universe across several industries. Each comment names the industry.
UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "META", "AMZN",
    # Semiconductors
    "NVDA", "AMD", "INTC", "TSM", "AVGO",
    # Banks
    "JPM", "BAC", "WFC", "C", "GS",
    # Consumer staples
    "KO", "PEP", "PG", "WMT", "COST",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Telecom / media
    "T", "VZ", "DIS", "NFLX",
    # Industrials
    "BA", "CAT", "GE",
]

# Date range: 5 years ending recently.
START_DATE = "2020-01-01"
END_DATE   = "2025-01-01"