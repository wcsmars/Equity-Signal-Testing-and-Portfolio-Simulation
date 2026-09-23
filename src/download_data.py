"""Download a local daily market-data cache using yfinance.

Run: python src/download_data.py. Re-running refreshes the cache from 2000
onward and may change previously observed history after vendor revisions.
The public project includes no downloaded prices; users must obtain data
under terms permitting their intended use. Run the data-quality check after
refreshing and before using the cache in research.
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qcore.data import DATA_DIR, ETF_UNIVERSE, INDEX_UNIVERSE, STOCK_UNIVERSE

START = "2000-01-01"


def download_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, start=START, progress=False, auto_adjust=True,
                      group_by="column", threads=True)
    out = {}
    for field in ["Close", "Open", "High", "Low", "Volume"]:
        df = raw[field].copy()
        df = df.dropna(how="all")
        out[field.lower()] = df

    # raw (dividend-unadjusted) closes: adj_close vs close isolates dividend
    # yield per ex-date (needed to charge the 30% US withholding on payouts)
    raw_px = yf.download(tickers, start=START, progress=False, auto_adjust=False,
                         group_by="column", threads=True)
    out["raw_close"] = raw_px["Close"].dropna(how="all")
    return out


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tickers = ETF_UNIVERSE + STOCK_UNIVERSE
    print(f"downloading {len(tickers)} tickers from {START} ...")
    data = download_batch(tickers)
    # drop rows where everything is NaN (half-holidays, in-progress days)
    valid = data["close"].dropna(how="all").index
    for field, df in data.items():
        name = {"close": "adj_close", "raw_close": "close"}.get(field, field)
        df = df.reindex(valid.intersection(df.index))
        df.to_csv(DATA_DIR / f"{name}.csv")
        print(f"  {name}.csv  {df.shape}")

    print("downloading index series ...")
    idx = yf.download(INDEX_UNIVERSE, start=START, progress=False,
                      auto_adjust=True, group_by="column")["Close"]
    idx = idx.dropna(how="all")
    idx.to_csv(DATA_DIR / "indices.csv")
    print(f"  indices.csv  {idx.shape}")

    # report coverage so strategies know each ticker's live range
    cov = pd.DataFrame({
        "first": data["close"].apply(lambda s: s.first_valid_index()),
        "last": data["close"].apply(lambda s: s.last_valid_index()),
        "rows": data["close"].count(),
    })
    cov.to_csv(DATA_DIR / "coverage.csv")
    print("\ncoverage summary (earliest starters):")
    print(cov.sort_values("first").head(5))
    print("\nlate starters:")
    print(cov.sort_values("first").tail(8))


if __name__ == "__main__":
    main()
