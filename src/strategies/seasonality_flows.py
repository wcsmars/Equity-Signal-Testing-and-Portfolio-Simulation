"""Turn-of-month seasonality on SPY.

Hypothesis
    Money reaches the equity market on a calendar. Salaries and pension
    contributions are invested around the month-end, and funds rebalance and
    tidy positions for month-end reporting. These buyers act on the date,
    not the price, so the sessions around the turn of the month have carried
    more than their share of the equity return. Holding SPY only on those
    sessions collects the price pressure that the scheduled buyers pay.

Rule
    Hold SPY with 100% of capital on the last N=4 trading days of each month
    and the first M=2 of the next. Hold cash, earning the engine's
    Treasury-bill proxy (13-week bill yield less 10 bps), on all other days.
    No trend filter.
    The window is calendar information: the exchange calendar is published
    in advance, so whether the next session lies in the window is known at
    today's close. The weight at a close is 1 exactly when the next
    scheduled session is in the window, and the engine applies it to that
    session's return. Window dates use approximate NYSE holiday rules
    independently of the price sample; an unscheduled closure is not assumed
    known before it happens.
    One entry and one exit a month; the position is held on 28.6% of
    sessions.
    Costs are modelling assumptions: the commissions and fees of
    qcore.costs plus 2 bps of slippage per side.

Variants tried
    12: N {3, 4, 5} x M {2, 3} x 200-day moving-average filter {off, on}.
    --sweep re-runs all of them.

Selection
    In-sample Sharpe (data before 2018-01-01): N=4, M=2, no filter, at 0.58;
    N=4, M=3 is next at 0.56. The 200-day filter lowered the in-sample
    Sharpe of every N=4 and N=5 combination. It raised it for N=3 (0.55 and
    0.46 against 0.42 and 0.40) without reaching the unfiltered N=4, M=2.
    Reported statistics begin at a variant's first position, so the filter
    variants are scored from 2002 and the unfiltered ones from 2000-01-03.
    Out-of-sample results (2018-01-01 onwards) are reported, not selected
    on. Neither the fixed parameters nor the calendar split establish an
    untouched evaluation sample.

Result
    This code on data ending 2026-07-01, net of the modelled costs, Sharpe
    ratios in excess of the cash rate, from 2000-01-03:
    Sharpe 0.58 full sample / 0.58 in sample / 0.58 out of sample (0.52 at
    double slippage); CAGR 7.35%, volatility 10.08%, maximum drawdown
    -13.47%. Turnover is 24.1x a year (12 buys and 12 sells of the whole
    portfolio) and costs take 0.59% a year. Interest on the idle cash
    contributes 1.27% a year.
    SPY held throughout, same engine and period: Sharpe 0.39 full sample /
    0.26 in sample / 0.66 out of sample, CAGR 7.70%, maximum drawdown
    -55.58%.

Verdict
    Kept, as one of the four strategies of the combined portfolio
    (src/ensemble.py; with tsmom_trend, xsec_etf_mom and mean_reversion).

Secondary measurement, rejected (--overnight-note)
    Close-to-open against open-to-close returns of SPY and QQQ from 2000 to
    2026-07-01, before costs. Overnight: SPY 2.94 bps a day (Sharpe 0.66,
    t = 3.40), QQQ 4.66 bps a day (Sharpe 0.82, t = 4.24). Intraday: SPY
    0.94 bps a day (Sharpe 0.15), QQQ 0.10 bps (Sharpe 0.01).
    Harvesting the overnight return takes two trades a day at full size, so
    the breakeven cost per side is half the mean return: SPY 1.47 bps (2.09
    from 2018), QQQ 2.33 bps (2.66 from 2018). The 2 bps of slippage per
    side assumed here, plus commission, is above the SPY breakeven and about
    equal to QQQ's, before closing- and opening-auction slippage and
    withholding on dividends, which accrue overnight. The effect is present
    before costs and not tradable at these costs: rejected, and never part
    of the strategy above.

Default execution prints metrics. --sweep prints the variants table and
saves it to results/seasonality_flows_variants.csv; a saved table that
differs from the new run is kept and the new one is written to
results/recomputed/ unless --rebase is given. Unknown flags are rejected.

A blank SPY close after its first close raises an error naming the date.
Dropping the row instead would merge two sessions into one return and
stretch each 200-day average that spans the gap, without notice.
--overnight-note applies the same check to SPY and QQQ.
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
from qcore.data import load, load_prices, require_listed_closes
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


def _listed_closes(closes: pd.Series) -> pd.Series:
    """Closes from the first one on; raises on a blank close after it
    (see module docstring) instead of dropping the row."""
    require_listed_closes(closes, "Dropping the row would silently merge two sessions into one return")
    return closes.dropna()


def run_best() -> dict:
    spy = _listed_closes(load_prices()["SPY"])
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
    spy = _listed_closes(load_prices()["SPY"])
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
        c = _listed_closes(px[tkr])
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
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n")[0])
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
    try:
        main()
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
