"""Download a local daily market-data cache using yfinance.

Run: python src/download_data.py [--allow-shrink]. Re-running refreshes the
cache from 2000 onward and may change previously observed history after
vendor revisions. The cache is data/ beside src/, or the directory named by
the QCORE_DATA_DIR environment variable, which keeps a new download apart
from a cache that earlier results were computed from. A refresh is refused,
and the cache left untouched, when the response lacks observations the
existing cache holds (a ticker that starts later, stops earlier or has a new
hole; a withdrawn index bar counts too), when an existing file cannot be read
for that comparison, or when the index panel ends more than
INDEX_MAX_LAG_SESSIONS sessions before prices. The error lists what is
missing. --allow-shrink is the way through in each case: use it when the loss
is a genuine vendor correction, to replace a damaged file, or to take the
price refresh while the index series lag.
The public project includes no downloaded prices; users must obtain data
under terms permitting their intended use. Run the data-quality check after
refreshing and before using the cache in research.
"""

import argparse
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
# Price sessions the newest index row may trail the newest price row. A
# longer lag is a partial index response and does not replace the cache
# unless --allow-shrink is passed.
INDEX_MAX_LAG_SESSIONS = 5


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
    # A timezone-aware or intraday index is written as text the loader
    # cannot read back as dates, so it must never reach the cache.
    if frame.index.tz is not None or (frame.index != frame.index.normalize()).any():
        raise ValueError(f"{label}: dates must be timezone-naive calendar days")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label}: unsorted dates")
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any() or (positive and np.any(values <= 0)):
        raise ValueError(f"{label}: invalid numeric values")
    if not frame.notna().any(axis=0).all():
        raise ValueError(f"{label}: at least one ticker has no observations")
    if label == "volume" and np.any(values < 0):
        raise ValueError("volume: negative values")


def _lost_history(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Observations the existing cache holds that the refreshed frames lack.

    Each file about to be replaced is compared cell by cell with its
    replacement, so a ticker that starts later, stops earlier or gains a hole
    is reported even when its row count grows. Changed values on dates both
    generations hold are vendor revisions, not losses, and are not reported.
    An existing file whose dates cannot be read is reported as well: a
    refresh that cannot be compared is not known to be safe. coverage.csv is
    derived from adj_close and is not compared.
    """
    problems = []
    for name, new in frames.items():
        path = DATA_DIR / f"{name}.csv"
        if name == "coverage" or not path.exists():
            continue
        try:
            old = pd.read_csv(path, index_col=0, parse_dates=True)
            if len(old.index) and (not isinstance(old.index, pd.DatetimeIndex)
                                   or old.index.tz is not None):
                raise ValueError("first column is not timezone-naive dates")
        except Exception as error:  # noqa: BLE001 - unreadable is not "nothing lost"
            problems.append(f"{name}.csv: existing file cannot be compared ({error})")
            continue
        common = old.columns.intersection(new.columns)
        held = old[common].notna()
        gone = held & new.reindex(index=old.index, columns=common).isna()
        for ticker in common[gone.any(axis=0).to_numpy()]:
            dates = gone.index[gone[ticker].to_numpy()]
            problems.append(f"{name}.csv {ticker}: {len(dates)} of {int(held[ticker].sum())} "
                            f"cached observations missing "
                            f"({dates.min().date()} .. {dates.max().date()})")
    return problems


def _write_cache(frames: dict[str, pd.DataFrame]) -> None:
    """Stage all CSVs before replacement; restore old files if a write fails.

    Each rename is atomic, but concurrent readers must still avoid refreshes:
    a multi-file CSV cache is not a transactional database. If a restore
    itself fails, the staging folder with the untouched <name>.backup copies
    is kept and named in the error instead of being deleted.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    leftover = sorted(p.name for p in DATA_DIR.glob(".refresh-*"))
    if leftover:
        print(f"WARNING: {DATA_DIR} holds staging folders from an interrupted refresh "
              f"({', '.join(leftover)}); their <name>.backup files are the pre-refresh "
              "originals. Delete the folders once this refresh has completed.")
    stage = Path(tempfile.mkdtemp(prefix=".refresh-", dir=DATA_DIR))
    keep_stage = False
    try:
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
                # recorded before the rename: restoring a file that was not
                # replaced after all is harmless, missing one is not
                replaced.append(name)
                os.replace(stage / f"{name}.csv", DATA_DIR / f"{name}.csv")
        except BaseException as error:
            unrestored = []
            for name in replaced:
                destination = DATA_DIR / f"{name}.csv"
                try:
                    if name in existing:
                        os.replace(stage / f"{name}.backup", destination)
                    else:
                        destination.unlink(missing_ok=True)
                except BaseException:  # noqa: BLE001 - keep restoring the rest
                    unrestored.append(name)
            if unrestored:
                keep_stage = True
                # named from what existed before, not from which backups are
                # left: a restore interrupted after its rename has no backup
                # either, and that file must not be deleted
                added = [name for name in unrestored if name not in existing]
                raise RuntimeError(
                    f"cache refresh failed ({error!r}) and the rollback could not "
                    f"restore {unrestored}: {DATA_DIR} may now mix old and new files. "
                    f"The originals are kept in {stage} as <name>.backup; copy them "
                    "back before using the cache."
                    + (f" {added} did not exist before this refresh and should be "
                       "deleted." if added else "")) from error
            raise
    finally:
        if not keep_stage:
            shutil.rmtree(stage, ignore_errors=True)


def main(allow_shrink: bool = False) -> None:
    tickers = list(dict.fromkeys(ETF_UNIVERSE + STOCK_UNIVERSE + OPERATIONAL_UNIVERSE))
    print(f"downloading {len(tickers)} tickers from {START} into {DATA_DIR} ...")
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
    lag = int((valid > idx.index[-1]).sum())
    if lag > INDEX_MAX_LAG_SESSIONS:
        stale = (f"indices: panel ends {idx.index[-1].date()}, {lag} sessions "
                 f"before prices ({valid[-1].date()})")
        if not allow_shrink:
            raise ValueError(
                f"{stale}; refresh refused, the cache in {DATA_DIR} is unchanged. "
                "Retry later, or rerun with --allow-shrink to accept the short index panel.")
        print(f"WARNING: --allow-shrink: keeping a short index panel ({stale})")
    frames["indices"] = idx

    # coverage.csv: each ticker's first/last valid date and row count. A
    # reference for choosing sample starts; the data-quality gate also compares
    # the cache against it to detect a ticker that lost history.
    cov = pd.DataFrame({
        "first": data["close"].apply(lambda s: s.first_valid_index()),
        "last": data["close"].apply(lambda s: s.last_valid_index()),
        "rows": data["close"].count(),
    })
    frames["coverage"] = cov

    lost = _lost_history(frames)
    if lost:
        shown = "\n  ".join(lost[:20] + ([f"... and {len(lost) - 20} more"] if len(lost) > 20 else []))
        if not allow_shrink:
            raise ValueError(
                f"refresh refused; the cache in {DATA_DIR} is unchanged. The response "
                f"lacks history the cache holds:\n  {shown}\n"
                "Retry later, or rerun with --allow-shrink if the loss is a genuine vendor "
                "correction or the existing file is damaged and should be replaced.")
        print(f"WARNING: --allow-shrink: replacing the cache although history was lost:\n  {shown}")
    _write_cache(frames)
    for name, frame in frames.items():
        print(f"  {name}.csv  {frame.shape}")
    print("\ncoverage summary (earliest starters):")
    print(cov.sort_values("first").head(5))
    print("\nlate starters:")
    print(cov.sort_values("first").tail(8))


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Refresh the local market-data cache.")
    parser.add_argument("--allow-shrink", action="store_true",
                        help="replace the cache even though the response lacks "
                             "observations the existing cache holds, or its index "
                             f"panel ends more than {INDEX_MAX_LAG_SESSIONS} sessions "
                             "before prices")
    main(allow_shrink=parser.parse_args(argv).allow_shrink)


if __name__ == "__main__":
    _cli()
