"""Load locally obtained daily market data and define research universes.

CSV files live in data/ relative to this copy of the source tree, or in the
directory named by the QCORE_DATA_DIR environment variable. Adjusted
prices represent gross returns before the engine's withholding estimate;
the vendor close series is dividend-unadjusted but still split-adjusted.
The fixed stock universe is survivorship-biased and is not point-in-time
index membership. Market data is not distributed with this public copy.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The cache lives in data/ beside src/ unless the QCORE_DATA_DIR environment
# variable names another directory. The override is opt-in and read once at
# import: it lets a refresh fill a separate cache so the default one, which
# saved results were computed from, is never replaced. It applies to every
# loader call in the process, so load() announces a non-default directory on
# stderr the first time it reads from it.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _data_dir(environ) -> Path:
    override = environ.get("QCORE_DATA_DIR")
    return Path(override).expanduser().resolve() if override else DEFAULT_DATA_DIR


DATA_DIR = _data_dir(os.environ)
_ANNOUNCED: set = set()


def _announce_override() -> None:
    """Say once per directory, on stderr, that the cache being read is not
    the default one. QCORE_DATA_DIR applies to the whole process: left
    exported, it redirects every loader call, so the redirection is shown."""
    if DATA_DIR != DEFAULT_DATA_DIR and DATA_DIR not in _ANNOUNCED:
        _ANNOUNCED.add(DATA_DIR)
        print(f"NOTE: market data is read from {DATA_DIR}, not the default cache "
              f"{DEFAULT_DATA_DIR} (unset QCORE_DATA_DIR to use the default)",
              file=sys.stderr)


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

# Operational quotes for cash parking; excluded from strategy selection universes.
OPERATIONAL_UNIVERSE = ["SGOV"]

INDEX_UNIVERSE = ["^VIX", "^VIX3M", "^IRX", "^GSPC", "^TNX"]

# Smallest adj_close-vs-close return difference dividend_yields() keeps as a
# payout. Vendor adjusted closes carry symmetric rounding noise of up to
# about 2.3e-6 on non-ex-dates, so at 1e-6 the positive half of that noise
# survives as spurious entries (the negative half is clipped). 5e-6 would
# separate noise from the smallest real payouts (about 8e-6), but it moves a
# rounded statistic of saved results, so the value is changed only together
# with a deliberate regeneration of those results.
DIVIDEND_NOISE_FLOOR = 1e-6


def load(name: str, *, strict: bool = True) -> pd.DataFrame:
    """name: one of adj_close | open | high | low | close | volume | indices

    Raises FileNotFoundError naming the missing file and the downloader, and
    ValueError when the first column is not timezone-naive calendar dates.
    strict (the default) also rejects a file with no rows and duplicate or
    unsorted dates; strict=False returns those unchanged for a caller that
    reports them itself, such as a data-quality check.

    high and low are the vendor's session extremes as delivered: they can
    hold a single print far outside the day's open and close that may never
    have traded. qcore.quality.check_range_plausibility lists such bars;
    review them before using high or low as a fill trigger, a stop level or
    a range estimator."""
    _announce_override()
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - build the local cache first: "
                                "python src/download_data.py")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    idx = df.index
    if len(idx) == 0:
        if strict:
            raise ValueError(f"{path.name}: no rows")
        df.index = pd.DatetimeIndex(idx)
        return df
    if (not isinstance(idx, pd.DatetimeIndex) or idx.hasnans or idx.tz is not None
            or (idx != idx.normalize()).any()):
        raise ValueError(f"{path.name}: first column must be timezone-naive calendar dates")
    if strict and not (idx.is_unique and idx.is_monotonic_increasing):
        raise ValueError(f"{path.name}: dates must be unique and increasing")
    return df


def load_prices() -> pd.DataFrame:
    """Adjusted closes for ETFs + stocks (gross total return)."""
    return load("adj_close").drop(columns=OPERATIONAL_UNIVERSE, errors="ignore")


def load_indices() -> pd.DataFrame:
    """^VIX, ^VIX3M, ^IRX (13w T-bill yield), ^GSPC, ^TNX."""
    return load("indices")


def require_listed_closes(closes, consequence: str, *, what: str = "close") -> None:
    """Raise ValueError on a blank close after a ticker's first close.

    closes holds one column per ticker, or is one ticker's Series. Blanks
    before the first close are a late listing and pass. A blank after it is a
    hole in the cache that a rule would absorb without an error, so the
    message names up to ten ticker/date cells and then `consequence`: what the
    calling rule would silently do with the hole. `what` names the kind of
    close that was checked (for example "month-end close")."""
    present = closes.notna().to_numpy()
    hole = ~present & np.maximum.accumulate(present, axis=0)
    if not hole.any():
        return
    if closes.ndim == 1:
        dates = closes.index[hole]
        found = (f"blank {closes.name} {what} after listing on {len(dates)} date(s): "
                 + ", ".join(str(d.date()) for d in dates[:10]))
    else:
        rows, cols = np.nonzero(hole)
        cells = [f"{closes.columns[c]} {closes.index[r].date()}"
                 for r, c in zip(rows[:10], cols[:10])]
        found = f"blank {what} after listing in {len(rows)} cell(s): {', '.join(cells)}"
    raise ValueError(f"{found}. {consequence}; repair the price cache.")


def dividend_yields() -> pd.DataFrame:
    """Per-ticker dividend yield on each ex-date (0 elsewhere, up to the
    noise described below), derived from the cached raw closes: total-return
    factor (adj_close) minus price-return factor (close) isolates the payout;
    splits cancel in the difference.
    Differences at or below DIVIDEND_NOISE_FLOOR are zeroed; vendor rounding
    noise just above it (up to about 2.3e-6) remains as small spurious
    entries, so a nonzero value is not by itself proof of an ex-date.
    This is an approximation inferred from adjusted return differences.
    Requires data/close.csv with matching split adjustment."""
    ac, c = load("adj_close"), load("close")
    for label, frame in (("adj_close", ac), ("close", c)):
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing or not frame.columns.is_unique:
            raise ValueError(f"{label}: dividend inference requires unique, ordered data")
        if np.isinf(frame.to_numpy(dtype=float)).any() or (frame <= 0).to_numpy().any():
            raise ValueError(f"{label}: dividend inference requires finite positive prices")
    if not ac.index.equals(c.index) or not ac.columns.isin(c.columns).all():
        raise ValueError("raw and adjusted closes must align for dividend inference")
    c = c.reindex(columns=ac.columns)
    if (ac.notna() & c.isna()).to_numpy().any():
        raise ValueError("missing raw close would silently omit dividend withholding")
    dy = ac.pct_change(fill_method=None) - c.pct_change(fill_method=None)
    dy = dy.clip(lower=0.0)
    return dy.where(dy > DIVIDEND_NOISE_FLOOR, 0.0).fillna(0.0)
