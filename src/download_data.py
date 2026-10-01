"""Download a local daily market-data cache using yfinance.

Run: python src/download_data.py. Re-running refreshes the cache from 2000
onward and may change previously observed history after vendor revisions.
The public project includes no downloaded prices; users must obtain data
under terms permitting their intended use. Run the data-quality check after
refreshing and before using the cache in research.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qcore.data import DATA_DIR, ETF_UNIVERSE, INDEX_UNIVERSE, OPERATIONAL_UNIVERSE, STOCK_UNIVERSE

START = "2000-01-01"


def download_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    # One request for every panel. Raw and adjusted closes must price each
    # bar from the same response: two separate downloads during a session
    # price the in-progress bar seconds apart, so adj/raw on that bar is
    # not 1 and the data-quality gate FAILs about half the universe.
    raw = yf.download(tickers, start=START, progress=False, auto_adjust=False,
                      group_by="column", threads=True)
    panels = split_adjusted_panels(raw)
    for field, frame in panels.items():
        _validate_download(frame, tickers, field, positive=field != "volume")
    return panels


def split_adjusted_panels(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Adjusted close/open/high/low + volume, and the raw close, from one
    auto_adjust=False response. Same arithmetic as yfinance's auto_adjust:
    ratio = Adj Close / Close scales open/high/low; volume is unchanged.
    The raw (dividend-unadjusted) close isolates each ex-date's dividend
    yield (needed to charge the 30% US withholding on payouts)."""
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex) or raw.columns.nlevels != 2:
        raise ValueError("download needs nonempty (field, ticker) columns")
    required = {"Adj Close", "Close", "Open", "High", "Low", "Volume"}
    if not required.issubset(raw.columns.get_level_values(0)):
        raise ValueError("download is missing required OHLCV fields")
    ratio = raw["Adj Close"] / raw["Close"]
    out = {
        "close": raw["Adj Close"],
        "open": raw["Open"] * ratio,
        "high": raw["High"] * ratio,
        "low": raw["Low"] * ratio,
        "volume": raw["Volume"],
        "raw_close": raw["Close"],
    }
    return {field: df.dropna(how="all") for field, df in out.items()}


def _validate_download(frame: pd.DataFrame, tickers: list[str], label: str,
                       positive: bool = False) -> None:
    if frame.empty or not frame.columns.is_unique or set(frame.columns) != set(tickers):
        raise ValueError(f"{label}: incomplete or empty ticker response")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans or not frame.index.is_unique:
        raise ValueError(f"{label}: invalid/duplicate dates")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label}: unsorted dates")
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any() or (positive and np.any(values <= 0)):
        raise ValueError(f"{label}: invalid numeric values")
    if not frame.notna().any(axis=0).all():
        raise ValueError(f"{label}: at least one ticker has no observations")
    if label == "volume" and np.any(values < 0):
        raise ValueError("volume: negative values")


def _write_cache(frames: dict[str, pd.DataFrame]) -> None:
    """Stage all CSVs before replacement; restore old files if a write fails.

    Each rename is atomic, but concurrent readers must still avoid refreshes:
    a multi-file CSV cache is not a transactional database.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".refresh-", dir=DATA_DIR) as temp:
        stage = Path(temp)
        existing = set()
        for name, frame in frames.items():
            frame.to_csv(stage / f"{name}.csv")
            destination = DATA_DIR / f"{name}.csv"
            if destination.exists():
                shutil.copy2(destination, stage / f"{name}.backup")
                existing.add(name)
        replaced = []
        try:
            for name in frames:
                os.replace(stage / f"{name}.csv", DATA_DIR / f"{name}.csv")
                replaced.append(name)
        except BaseException:
            for name in replaced:
                destination = DATA_DIR / f"{name}.csv"
                if name in existing:
                    os.replace(stage / f"{name}.backup", destination)
                else:
                    destination.unlink()
            raise


def main() -> None:
    tickers = list(dict.fromkeys(ETF_UNIVERSE + STOCK_UNIVERSE + OPERATIONAL_UNIVERSE))
    print(f"downloading {len(tickers)} tickers from {START} ...")
    data = download_batch(tickers)
    # Preserve one calendar across all six fields. All-NaN rows cannot
    # identify an in-progress bar; a live pull can still include today's bar.
    valid = data["close"].index
    frames = {}
    for field, df in data.items():
        name = {"close": "adj_close", "raw_close": "close"}.get(field, field)
        frames[name] = df.reindex(valid)

    print("downloading index series ...")
    idx = yf.download(INDEX_UNIVERSE, start=START, progress=False,
                      auto_adjust=True, group_by="column")["Close"]
    idx = idx.dropna(how="all")
    _validate_download(idx, INDEX_UNIVERSE, "indices")
    frames["indices"] = idx

    # report coverage so strategies know each ticker's live range
    cov = pd.DataFrame({
        "first": data["close"].apply(lambda s: s.first_valid_index()),
        "last": data["close"].apply(lambda s: s.last_valid_index()),
        "rows": data["close"].count(),
    })
    frames["coverage"] = cov
    _write_cache(frames)
    for name, frame in frames.items():
        print(f"  {name}.csv  {frame.shape}")
    print("\ncoverage summary (earliest starters):")
    print(cov.sort_values("first").head(5))
    print("\nlate starters:")
    print(cov.sort_values("first").tail(8))


if __name__ == "__main__":
    main()
