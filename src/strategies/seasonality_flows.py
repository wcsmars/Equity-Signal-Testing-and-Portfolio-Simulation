"""Turn-of-month seasonality example for SPY.

Hold SPY through the last four trading days of a month and the first two
of the next, with modeled cash otherwise. Weights anticipate the next
session's calendar window and the engine shifts them by one row. Window dates use approximate NYSE holiday rules independently of the
price sample; future unscheduled closures are not assumed known. Costs assume 2 bps per side.

Default execution prints metrics. --sweep compares window/filter settings,
prints the table and saves it to results/seasonality_flows_variants.csv; a
saved table that differs from the new run is kept and the new one is written
to results/recomputed/ unless --rebase is given. --overnight-note reports a
descriptive gross close-to-open versus open-to-close decomposition. Unknown
flags are rejected. Neither the fixed parameters nor the calendar split
establish an untouched evaluation sample.

Reported statistics begin at a variant's first position, so the 200-day
filter variants are scored over a shorter window than the unfiltered ones.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import OOS_SPLIT, TRADING_DAYS, metrics, run_backtest
from qcore.quality import SPECIAL_CLOSURES, nyse_bdays
from qcore.costs import IBKRHKCostModel
from qcore.data import load, load_prices
from qcore.records import save_csv

N_LAST = 4   # last N trading days of the month
M_FIRST = 2  # first M trading days of the next month
SLIPPAGE_BPS = 2.0


def tom_weights(spy: pd.Series, n_last: int = N_LAST, m_first: int = M_FIRST,
                dma_filter: bool = False) -> pd.DataFrame:
    """Target at each close for the next scheduled NYSE session.

    Compute the complete calendar month independently of the price sample,
    including at the live edge. Future special closures are excluded only
    after they occur, since an unscheduled closure is not known in advance.
    Regular holiday rules are approximate; verify the live exchange calendar.
    """
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0
           for v in (n_last, m_first)):
        raise ValueError("window lengths must be nonnegative integers")
    idx = spy.index
    if not isinstance(idx, pd.DatetimeIndex) or not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError("prices need unique, sorted dates")
    if idx.empty:
        return pd.DataFrame({"SPY": pd.Series(dtype=float)}, index=idx)
    # Include future special-closure weekdays initially. Each decision can
    # remove only closures already observed by that date.
    start = idx[0].to_period("M").start_time
    end = (idx[-1].to_period("M") + 1).end_time.normalize()
    closures = SPECIAL_CLOSURES.sort_values()
    regular = nyse_bdays(start, end).union(
        closures[(closures >= start) & (closures <= end)]
    ).sort_values()
    # The scheduled calendar changes only when a closure becomes known, so
    # decision rows are grouped by the number of closures already observed.
    known = closures.searchsorted(idx, side="right")
    held = np.zeros(len(idx), dtype=bool)
    for k in np.unique(known):
        calendar = regular.difference(closures[:k])
        sessions = pd.Series(1, index=calendar.to_period("M"))
        rank = sessions.groupby(level=0).cumcount().to_numpy()   # 0 = first session of its month
        size = sessions.groupby(level=0).transform("size").to_numpy()
        in_window = (rank < m_first) | (size - rank <= n_last)
        rows = np.flatnonzero(known == k)
        nxt = calendar.searchsorted(idx[rows], side="right")     # next scheduled session
        ok = nxt < len(calendar)
        held[rows[ok]] = in_window[nxt[ok]]
    hold = pd.Series(held, index=idx)

    if dma_filter:
        hold &= spy > spy.rolling(200).mean()
    return pd.DataFrame({"SPY": hold.astype(float)}, index=idx)


def run_best() -> dict:
    spy = load_prices()["SPY"].dropna()
    w = tom_weights(spy)
    res = run_backtest(w, spy.to_frame("SPY"),
                       IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS),
                       name=f"tom_spy_N{N_LAST}_M{M_FIRST}_nofilt")
    return metrics(res)


def _is_sharpe_unrounded(res: dict) -> float:
    """Full-precision in-sample excess-return Sharpe (same definition as
    metrics(), without the 2dp rounding). Sort key only: the variants CSV
    is ordered on this, and rounding leaves ties (e.g. two rows at 0.55)."""
    r = res["returns"].dropna()
    ex = r - res["rf_daily"].reindex(r.index).fillna(0.0)
    ex = ex.loc[: pd.Timestamp(OOS_SPLIT) - pd.Timedelta(1, unit="D")]
    return float(ex.mean() / ex.std() * np.sqrt(TRADING_DAYS))


def sweep() -> pd.DataFrame:
    """All 12 variants (N x M x dma200), rows sorted by IS Sharpe desc."""
    spy = load_prices()["SPY"].dropna()
    rows = []
    for n_last in (3, 4, 5):
        for m_first in (2, 3):
            for dma_filter in (False, True):
                name = f"N{n_last}_M{m_first}_{'dma200' if dma_filter else 'nofilt'}"
                w = tom_weights(spy, n_last=n_last, m_first=m_first,
                                dma_filter=dma_filter)
                res = run_backtest(w, spy.to_frame("SPY"),
                                   IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS),
                                   name=name)
                mm = metrics(res)
                rows.append({
                    "name": name, "N": n_last, "M": m_first,
                    "dma200_filter": dma_filter,
                    "full_sharpe": mm["full"]["sharpe"],
                    "is_sharpe": mm["in_sample"]["sharpe"],
                    "oos_sharpe": mm["out_of_sample"]["sharpe"],
                    "full_cagr": mm["full"]["cagr"],
                    "full_vol": mm["full"]["vol"],
                    "full_maxdd": mm["full"]["maxdd"],
                    "is_cagr": mm["in_sample"]["cagr"],
                    "oos_cagr": mm["out_of_sample"]["cagr"],
                    "oos_maxdd": mm["out_of_sample"]["maxdd"],
                    "ann_turnover": mm["ann_turnover_oneside"],
                    "ann_cost_drag": mm["ann_cost_drag"],
                    "gross_full_sharpe": mm["gross_full"]["sharpe"],
                    "pct_pos_months": mm["pct_positive_months"],
                    "worst_month": mm["worst_month"],
                    "start": mm["start"],
                    "_is_sort": _is_sharpe_unrounded(res),
                })
                print(f"{name:16s} IS {mm['in_sample']['sharpe']:5.2f}  "
                      f"OOS {mm['out_of_sample']['sharpe']:5.2f}  "
                      f"full {mm['full']['sharpe']:5.2f}", flush=True)
    df = pd.DataFrame(rows).sort_values("_is_sort", ascending=False)
    return df.drop(columns="_is_sort")


def overnight_note() -> dict:
    """Overnight vs intraday split measurement (research note, not traded)."""
    px, op = load_prices(), load("open")
    out = {}
    for tkr in ("SPY", "QQQ"):
        c = px[tkr].dropna()
        o = op[tkr].reindex(c.index)
        on = (o / c.shift(1) - 1).dropna()
        iday = (c / o - 1).reindex(on.index)

        def ann(r: pd.Series) -> dict:
            yrs = len(r) / 252
            return {
                "ann_cagr_gross": round(float((1 + r).prod() ** (1 / yrs) - 1), 4),
                "ann_vol": round(float(r.std() * np.sqrt(252)), 4),
                "sharpe_gross": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
                "mean_bps_per_day": round(float(r.mean() * 1e4), 2),
            }

        out[tkr] = {
            "overnight": ann(on),
            "intraday": ann(iday),
            "overnight_tstat": round(float(on.mean() / (on.std() / np.sqrt(len(on)))), 2),
            # 2 sides traded per day at full notional -> breakeven = mean/2
            "breakeven_cost_bps_per_side": round(float(on.mean() * 1e4 / 2), 2),
            "breakeven_bps_per_side_2018plus":
                round(float(on.loc["2018":].mean() * 1e4 / 2), 2),
            "caveat": "Gross descriptive decomposition; execution costs are not deducted.",
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--sweep", action="store_true",
                      help="re-run the 12 tested variants instead of the default one")
    mode.add_argument("--overnight-note", action="store_true",
                      help="print the overnight/intraday decomposition instead")
    ap.add_argument("--rebase", action="store_true",
                    help="with --sweep: replace the saved variants table if this run differs")
    args = ap.parse_args()  # --rebase itself is read by qcore.records

    if args.sweep:
        df = sweep()
        print()
        # saved table is kept if this run differs (see qcore.records)
        save_csv(ROOT / "results" / "seasonality_flows_variants.csv", df, index=False)
        print("best by IS Sharpe:", df.iloc[0]["name"])
    elif args.overnight_note:
        print(json.dumps(overnight_note(), indent=2))
    else:
        print(json.dumps(run_best(), indent=2))


if __name__ == "__main__":
    main()
