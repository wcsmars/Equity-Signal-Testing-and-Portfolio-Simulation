"""Load locally obtained daily market data and define research universes.

CSV files live in data/ relative to this copy of the source tree. Adjusted
prices represent gross returns before the engine's withholding estimate;
the vendor close series is dividend-unadjusted but still split-adjusted.
The fixed stock universe is survivorship-biased and is not point-in-time
index membership. Market data is not distributed with this public copy.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

ETF_UNIVERSE = [
    # broad equity
    "SPY", "QQQ", "IWM", "DIA", "MDY", "EFA", "EEM", "VGK", "EWJ",
    "FXI", "EWY", "EWT", "EWZ", "EWA", "EWC", "EWG", "EWU", "EWH",
    # US sectors / industries
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB",
    "XBI", "SMH", "KRE", "XME", "XOP", "IYR", "VNQ",
    # bonds / credit
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG", "EMB",
    # commodities / FX / alt
    "GLD", "SLV", "GDX", "DBC", "USO", "UNG", "UUP", "FXE", "FXY",
]

STOCK_UNIVERSE = [  # liquid US mega/large caps - SURVIVORSHIP-BIASED, see caveats
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM",
    "V", "UNH", "XOM", "JNJ", "PG", "HD", "MA", "COST", "ABBV", "CVX",
    "MRK", "PEP", "KO", "WMT", "BAC", "AMD", "CRM", "NFLX", "ORCL", "DIS",
    "CSCO", "INTC", "IBM", "QCOM", "TXN", "GE", "CAT", "BA", "GS", "MS",
    "WFC", "C", "T", "VZ", "PFE", "TMO", "ABT", "NKE", "MCD", "SBUX", "LOW",
]

INDEX_UNIVERSE = ["^VIX", "^VIX3M", "^IRX", "^GSPC", "^TNX"]


def load(name: str) -> pd.DataFrame:
    """name: one of adj_close | open | high | low | close | volume | indices"""
    df = pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)
    return df.sort_index()


def load_prices() -> pd.DataFrame:
    """Adjusted closes for ETFs + stocks (gross total return)."""
    return load("adj_close")


def load_indices() -> pd.DataFrame:
    """^VIX, ^VIX3M, ^IRX (13w T-bill yield), ^GSPC, ^TNX."""
    return load("indices")


def dividend_yields() -> pd.DataFrame:
    """Per-ticker dividend yield on each ex-date (0 elsewhere), derived from
    the cached raw closes: total-return factor (adj_close) minus price-return
    factor (close) isolates the payout; splits cancel in the difference.
    This is an approximation inferred from adjusted return differences.
    Requires data/close.csv with matching split adjustment."""
    ac, c = load("adj_close"), load("close")
    dy = (ac.pct_change(fill_method=None)
          - c.reindex(columns=ac.columns).pct_change(fill_method=None))
    dy = dy.clip(lower=0.0)
    return dy.where(dy > 1e-6, 0.0).fillna(0.0)
