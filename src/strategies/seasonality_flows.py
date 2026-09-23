"""Turn-of-month seasonality example for SPY.

Hold SPY through the last four trading days of a month and the first two
of the next, with modeled cash otherwise. Weights anticipate the next
session's calendar window and the engine shifts them by one row. Historical
window dates come from the cached calendar; incomplete trailing months are
excluded using qcore.calendar's heuristic. Costs assume 2 bps per side.

Default execution prints metrics. --sweep compares window/filter settings
and writes a variants CSV; --overnight-note reports a descriptive gross
close-to-open versus open-to-close decomposition. Neither the fixed
parameters nor the calendar split establish an untouched evaluation sample.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import OOS_SPLIT, TRADING_DAYS, metrics, run_backtest
from qcore.calendar import in_complete_month
from qcore.costs import IBKRHKCostModel
from qcore.data import load, load_prices

N_LAST = 4   # last N trading days of the month
M_FIRST = 2  # first M trading days of the next month
SLIPPAGE_BPS = 2.0


def tom_weights(spy: pd.Series, n_last: int = N_LAST, m_first: int = M_FIRST,
                dma_filter: bool = False) -> pd.DataFrame:
    """Target weights decided at each close (engine applies the 1-day lag).

    Live edge: 'days left in the month' is calendar information, exact for
    completed months but UNKNOWABLE from a data file whose last month is
    partial (the file's final row always looks like a month-end). The
    last-N leg is therefore masked off in an incomplete trailing month
    (qcore.calendar) - a live operator entering the month-end window must
    confirm the window against the published NYSE calendar. The first-M
    leg is backward-looking and always live-safe."""
    idx = spy.index
    months = idx.to_period("M")
    ones = pd.Series(1, index=idx)
    rank_from_start = ones.groupby(months).cumcount() + 1   # 1 = first td of month
    days_left = ones.groupby(months).transform("size") - rank_from_start  # 0 = last td

    complete = in_complete_month(idx)
    in_window = ((days_left <= n_last - 1) & complete) | (rank_from_start <= m_first)
    # hold on day d+1 iff d+1 is in the window: decide at close d
    hold = in_window.shift(-1, fill_value=False).astype(bool)
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
    ex = ex.loc[: pd.Timestamp(OOS_SPLIT) - pd.Timedelta(days=1)]
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


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        df = sweep()
        out = ROOT / "results" / "seasonality_flows_variants.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nsaved {out}")
        print("best by IS Sharpe:", df.iloc[0]["name"])
    elif "--overnight-note" in sys.argv:
        print(json.dumps(overnight_note(), indent=2))
    else:
        print(json.dumps(run_best(), indent=2))
