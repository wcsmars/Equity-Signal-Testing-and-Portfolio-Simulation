"""Selection sweep for the stock momentum example (12-1 momentum, mega caps).

This is the grid behind the parameters retained in
src/strategies/xsec_stock_mom.py (K=5, lam=0.25).

Variants (2 free parameters, 12 variants):
  K (top names held)        in {5, 8, 10, 15}
  lam (short-term reversal) in {0.0, 0.25, 0.5}
      composite score = z(mom_12_1) - lam * z(ret_1m), cross-sectional z each month

Rule:
  - At each month-end close, rank STOCK_UNIVERSE by composite score.
  - Hold top K, inverse-vol weights (63d trailing daily vol), fully invested.
  - Weights placed on month-end date; the engine fills at that close and the
    new weights earn the next close-to-close return (one-row lag).
  - Slippage 5 bps/side, fixed per-share commissions. Long-only.
The weights come from strategies.xsec_stock_mom.build_weights, the strategy
module's own builder, so the grid and the module cannot drift apart.

Selection: best IN-SAMPLE Sharpe (before 2018) only.

Benchmark printed for context: an equal-weight basket of the same universe,
ordered at month-end closes only and left to drift in between, over the
winner's own window, with the winner's CAGR gap and active Sharpe against
it. A basket re-targeted every day would pay for daily micro-orders at the
commission cap and flatter the comparison.

Input: the price cache built by python src/download_data.py.
Run (from anywhere): python scripts/xsec_stock_mom_sweep.py [--rebase]
Output: the variants table, the in-sample winner's metrics, the benchmark
comparison, results/xsec_stock_mom_variants.csv (one row per variant) and
results/xsec_stock_mom.json (the winner's parameters and metrics). Both
files are saved records: a run whose output differs leaves them in place
and writes results/recomputed/<name> instead (qcore.records); --rebase
replaces them. Each run is named as the strategy module names its own run
(strategies.xsec_stock_mom.run_name), so the JSON declares a name that
src/ensemble.py can match.

Exit status: 0 on completion; 1, with the loader's one line, when the price
cache is missing; 2 for an unknown or shortened flag (nothing is run).

Limits: the fixed universe is today's large caps projected backwards, so it
is survivorship-biased: absolute results are an upper bound, and even the
gap to the equal-weight basket is measured inside that biased universe.
Neither figure is an unbiased investable result.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import TRADING_DAYS, drift_weights, metrics, run_backtest
from qcore.calendar import confirmed_month_ends
from qcore.costs import IBKRHKCostModel
from qcore.data import STOCK_UNIVERSE, load_prices
from qcore.records import save_csv, save_json
from strategies.xsec_stock_mom import build_weights, run_name


def ew_benchmark(px: pd.DataFrame, cm, start=None) -> dict:
    """Equal-weight basket of every name with a price, ordered at each
    confirmed month-end close (from `start` on) and left to drift in between,
    so it pays for twelve rebalances a year, not for daily re-targeting."""
    valid = px.notna()
    targets = valid.div(valid.sum(axis=1), axis=0).loc[confirmed_month_ends(px.index)]
    if start is not None:
        targets = targets.loc[start:]
    if targets.empty:
        raise ValueError("no month-end decision for the equal-weight benchmark")
    return run_backtest(drift_weights(targets, px), px, cm, name="EW_universe")


def active_stats(strategy: pd.Series, benchmark: pd.Series) -> dict:
    """CAGR gap and annualized Sharpe of the daily return difference over the
    dates both series cover."""
    idx = strategy.index.intersection(benchmark.index)
    if len(idx) < 2:
        raise ValueError("strategy and benchmark share fewer than two dates")
    s, b = strategy.loc[idx], benchmark.loc[idx]
    years = len(idx) / TRADING_DAYS
    diff = s - b
    sd = float(diff.std())
    return {
        "start": str(idx[0].date()), "end": str(idx[-1].date()),
        "cagr_gap": round(float((1 + s).prod() ** (1 / years) - (1 + b).prod() ** (1 / years)), 4),
        "active_sharpe": round(float(diff.mean() / sd * TRADING_DAYS ** 0.5), 2) if sd > 0 else float("nan"),
    }


def main():
    px = load_prices()[STOCK_UNIVERSE]
    cm = IBKRHKCostModel(slippage_bps=5.0)

    rows, results = [], {}
    for K in [5, 8, 10, 15]:
        for lam in [0.0, 0.25, 0.5]:
            name = f"K{K}_lam{lam:g}"  # short label, the CSV's name column
            w = build_weights(px, K, lam)
            res = run_backtest(w, px, cm, name=run_name(K, lam))
            m = metrics(res)
            results[name] = (m, {"K": K, "lam": lam}, res["returns"])
            rows.append({
                "name": name, "K": K, "lam": lam,
                "full_sharpe": m["full"]["sharpe"],
                "is_sharpe": m["in_sample"]["sharpe"],
                "oos_sharpe": m["out_of_sample"]["sharpe"],
                "full_cagr": m["full"]["cagr"], "full_vol": m["full"]["vol"],
                "full_maxdd": m["full"]["maxdd"],
                "oos_cagr": m["out_of_sample"]["cagr"],
                "oos_maxdd": m["out_of_sample"]["maxdd"],
                "ann_turnover": m["ann_turnover_oneside"],
                "ann_cost_drag": m["ann_cost_drag"],
                "pct_pos_months": m["pct_positive_months"],
                "worst_month": m["worst_month"],
            })

    df = pd.DataFrame(rows)
    save_csv(ROOT / "results" / "xsec_stock_mom_variants.csv", df, index=False)
    print(df.to_string(index=False))

    # select on IN-SAMPLE Sharpe only
    best_row = df.loc[df["is_sharpe"].idxmax()]
    best_name = best_row["name"]
    best_m, best_p, best_r = results[best_name]
    print("\nBEST BY IS SHARPE:", best_name)
    print(json.dumps(best_m, indent=2))

    out = {"params": best_p, "metrics": best_m}
    save_json(ROOT / "results" / "xsec_stock_mom.json", out)

    # benchmark for context: equal-weight universe, month-end orders only,
    # over the winner's own window
    res_b = ew_benchmark(px, cm, start=best_r.index[0])
    bench = metrics(res_b)
    act = active_stats(best_r, res_b["returns"])
    print(f"\nEW universe benchmark (month-end orders only, from {bench['start']}): "
          "full Sharpe", bench["full"]["sharpe"],
          "IS", bench["in_sample"]["sharpe"], "OOS", bench["out_of_sample"]["sharpe"],
          "CAGR", bench["full"]["cagr"], "cost drag", bench["ann_cost_drag"])
    print(f"{best_name} vs benchmark, {act['start']}..{act['end']}: "
          "CAGR gap", act["cagr_gap"], "active Sharpe", act["active_sharpe"])


def _parse_args(argv=None):
    # allow_abbrev=False: a shortened flag such as --reb is refused, not accepted
    # by the parser and then missed by the exact-name checks that act on it
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--rebase", action="store_true",
                        help="replace the stored variants log and JSON when this run differs from them")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _parse_args()
    try:
        main()
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
