"""Selection sweep for the mean_reversion sleeve (RSI-2 dip buying).

This is the grid that chose the parameters retained in
src/strategies/mean_reversion.py (entry 5, exit 70, 10-day limit).

Rule (long-only dip buying on liquid equity ETFs):
  Universe: SPY, QQQ, DIA, IWM, MDY + 9 SPDR sectors (14 ETFs).
  At each close t, per ETF:
    ENTER if flat and RSI(2) < entry_th and close > SMA(200).
    EXIT  if holding and (RSI(2) > exit_th or held >= max_hold days).
  Weight per open position: min(w_max, 1/n_open)  (w_max fixed at 0.20,
  total long exposure capped at 1.0, remainder in cash).

Free parameters (3, 12 variants total):
  entry_th in {5, 10, 15} x exit_th in {60, 70} x max_hold in {5, 10}.

Selection: best IN-SAMPLE (before 2018) Sharpe only. Slippage 3 bps/side for
all variants. The Sharpe compared is the engine's 2-decimal figure, as in
the saved grid; rows that tie at 2 decimals keep grid order, and a tie for
first place is printed instead of being settled silently.

The weights come from strategies.mean_reversion.build_weights, the sleeve's
own state machine (not a copy), so the grid and the sleeve cannot drift
apart.

Input: the price cache built by python src/download_data.py.
Run (from the repository root): python research/mean_reversion_sweep.py [--rebase]
Output: one line per variant, the sweep table, the best variant by
in-sample Sharpe, its sensitivity to slippage from 3 to 30 bps per side, and
results/mean_reversion_variants.csv (12 rows, sorted by in-sample Sharpe).
That file is the sleeve's selection record and is counted by
scripts/trial_registry.py, so it is written through qcore.records: a run
that differs from the saved file keeps it and goes to results/recomputed/
(--rebase replaces the record).

Exit status: 0 on completion; 1, with the loader's one line, when the price
cache is missing; 2 for an unknown or shortened flag (nothing is run).

Limits: twelve variants are ranked on a rounded in-sample Sharpe, so
neighbouring rows are not statistically distinguishable; the 2018 split has
been consulted by later research choices and is not an untouched holdout;
execution is assumed at the decision close.
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import metrics, run_backtest, sweep_table
from qcore.costs import IBKRHKCostModel
from qcore.data import load_prices
from qcore.records import save_csv
# the production rule itself (the state machine computes its own Wilder RSI), never a copy
from strategies.mean_reversion import build_weights

UNIVERSE = ["SPY", "QQQ", "DIA", "IWM", "MDY",
            "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]
W_MAX = 0.20
SLIPPAGE_BPS = 3.0


def main():
    px = load_prices()[UNIVERSE]

    cm = IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS)
    rows, results = [], []
    for entry_th, exit_th, max_hold in product([5, 10, 15], [60, 70], [5, 10]):
        name = f"e{entry_th}_x{exit_th}_h{max_hold}"
        w = build_weights(px, entry_th, exit_th, max_hold, W_MAX)
        res = run_backtest(w, px, cm, name=name)
        m = metrics(res)
        results.append(res)
        rows.append({
            "name": name, "entry_th": entry_th, "exit_th": exit_th,
            "max_hold": max_hold, "w_max": W_MAX,
            "full_sharpe": m["full"]["sharpe"], "is_sharpe": m["in_sample"]["sharpe"],
            "oos_sharpe": m["out_of_sample"]["sharpe"],
            "full_cagr": m["full"]["cagr"], "full_vol": m["full"]["vol"],
            "full_maxdd": m["full"]["maxdd"],
            "oos_cagr": m["out_of_sample"]["cagr"], "oos_maxdd": m["out_of_sample"]["maxdd"],
            "gross_sharpe": m["gross_full"]["sharpe"],
            "ann_turnover": m["ann_turnover_oneside"], "ann_cost_drag": m["ann_cost_drag"],
            "avg_exposure": round(float(w.sum(axis=1).mean()), 3),
        })
        print(name, "IS", m["in_sample"]["sharpe"], "OOS", m["out_of_sample"]["sharpe"],
              "full", m["full"]["sharpe"], "turn", m["ann_turnover_oneside"])

    # stable: rows that tie at the engine's 2 decimals keep grid order
    df = pd.DataFrame(rows).sort_values("is_sharpe", ascending=False, kind="stable")
    save_csv(ROOT / "results" / "mean_reversion_variants.csv", df, index=False)
    print("\n", sweep_table(results).round(3).to_string())
    best = df.iloc[0]
    print("\nBEST BY IS SHARPE:", best["name"])
    tied = df.loc[df.is_sharpe == best.is_sharpe, "name"].tolist()
    if len(tied) > 1:
        print(f"NOTE: {len(tied)} variants tie for the best IS Sharpe at 2 decimals "
              f"({best.is_sharpe}): {', '.join(tied)}. Grid order put {best['name']} "
              "first; that is not a selection - compare them unrounded before "
              "relying on it.")

    # breakeven slippage for the best variant (full-sample net CAGR = 0)
    w = build_weights(px, int(best.entry_th), int(best.exit_th), int(best.max_hold), W_MAX)
    for slip in [3, 5, 8, 10, 15, 20, 30]:
        res = run_backtest(w, px, IBKRHKCostModel(slippage_bps=slip), name=f"slip{slip}")
        m = metrics(res)
        print(f"slippage {slip:>2} bps -> full Sharpe {m['full']['sharpe']:.2f}, "
              f"full CAGR {m['full']['cagr']:.4f}, OOS Sharpe {m['out_of_sample']['sharpe']:.2f}")


def _parse_args(argv=None):
    # allow_abbrev=False: a shortened flag such as --reb is refused, not accepted
    # by the parser and then missed by the exact-name check that acts on it
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--rebase", action="store_true",
                        help="replace results/mean_reversion_variants.csv when this run differs from it")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _parse_args()
    try:
        main()
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
