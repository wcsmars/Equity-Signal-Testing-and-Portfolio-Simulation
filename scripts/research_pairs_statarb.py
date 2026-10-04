"""Selection sweep for the ETF pairs example (z-score spread reversion).

This is the grid behind the window lengths in
src/strategies/pairs_statarb.py. That module keeps H=90, Z=60, the
in-sample winner under an earlier version of the engine. On the current
engine and a cache ending 2026-07-01 the grid ranks H=120, Z=20 first
in-sample (Sharpe 0.51 against 0.48), and all nine variants have a negative
out-of-sample Sharpe: the grid documents an idea that did not survive, not
a traded rule.

Variant grid (2 free parameters, 9 variants):
  H (rolling OLS hedge-ratio window, log prices): {60, 90, 120}
  Z (z-score window of the spread):               {20, 40, 60}
Fixed: entry |z| > 2.0, exit |z| < 0.5 or a 20-day timeout.

Per pair: beta_t = rolling-H OLS slope of log(A) on log(B).
z_t uses the CURRENT beta applied to the trailing-Z window of both legs
(consistent-beta z-score), so the spread definition is coherent at each date.
Entry: z < -2 -> long A / short B (spread cheap); z > +2 -> short A / long B.
Weights frozen at entry (beta_e): |w_A|+|w_B| = G, w_B = -sign*beta_e*G/(1+beta_e).
The frozen quantity is the portfolio WEIGHT, not the share count: the engine
re-trades both legs to target on every holding day, which overstates the
cost drag relative to holding shares (see src/strategies/pairs_statarb.py).
Re-entry is blocked after any exit until |z| < entry once.

Combo: equal weight across the pairs whose STANDALONE in-sample net Sharpe
is positive, scaled so that the maximum gross exposure is 1.5. A borrow
charge of 1% a year on short notional is added on top.
Selection: best combo IN-SAMPLE (before 2018) Sharpe only.

The pair builder, the borrow charge, the pair list and the slippage are
imported from src/strategies/pairs_statarb.py, so the grid and the strategy
module cannot drift apart.

Input: the price cache built by python src/download_data.py.
Run (from anywhere): python scripts/research_pairs_statarb.py [--rebase]
Output: the variants table, the best variant with its per-pair statistics,
and results/pairs_statarb_variants.csv (one row per variant). That file is
a saved selection log, counted by scripts/trial_registry.py: a run whose
table differs leaves it in place and writes
results/recomputed/pairs_statarb_variants.csv instead (qcore.records);
--rebase replaces the log.

Exit status: 0 on completion; 1, with the loader's one line, when the price
cache is missing; 2 for an unknown or shortened flag (nothing is run).

Limits: nine variants are compared on in-sample Sharpe, and the pairs
themselves are screened on the same in-sample window, so the best row is
selected twice over. Borrow availability, variable lending fees and margin
calls are not modeled, and signals assume execution at the decision close.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import metrics, run_backtest
from qcore.costs import IBKRHKCostModel
from qcore.records import save_csv
from strategies.pairs_statarb import PAIRS, SLIPPAGE_BPS, apply_borrow, pair_weights


def main():
    from qcore.data import load_prices
    px = load_prices()
    cm = IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS)

    rows = []
    detail = {}
    for H in (60, 90, 120):
        for Z in (20, 40, 60):
            vname = f"H{H}_Z{Z}"
            pair_w, pair_is = {}, {}
            pair_stats = []
            for a, b in PAIRS:
                w = pair_weights(px, a, b, H, Z, gross=1.0)
                res = apply_borrow(run_backtest(w, px, cm, name=f"{a}/{b}"), w)
                m = metrics(res)
                pair_w[(a, b)] = w
                pair_is[(a, b)] = m["in_sample"]["sharpe"]
                pair_stats.append({
                    "pair": f"{a}/{b}",
                    "IS_Sharpe": m["in_sample"]["sharpe"],
                    "OOS_Sharpe": m["out_of_sample"]["sharpe"],
                    "Full_Sharpe": m["full"]["sharpe"],
                    "CAGR": m["full"]["cagr"], "MaxDD": m["full"]["maxdd"],
                    "Turnover": m["ann_turnover_oneside"],
                    "CostDrag": m["ann_cost_drag"],
                })
            passing = [p for p in pair_w if pair_is[p] is not None
                       and np.isfinite(pair_is[p]) and pair_is[p] > 0]
            if not passing:
                rows.append({"variant": vname, "H": H, "Z": Z, "n_pass": 0})
                continue
            scale = 1.5 / len(passing)  # max gross 1.5 when all pairs active
            cols = sorted({t for p in passing for t in p})
            combo = pd.DataFrame(0.0, index=px.index, columns=cols)
            for p in passing:
                combo = combo.add(
                    pair_w[p].reindex(px.index).fillna(0.0) * scale, fill_value=0.0)
            res = apply_borrow(run_backtest(combo, px, cm, name=vname), combo)
            m = metrics(res)
            rows.append({
                "variant": vname, "H": H, "Z": Z, "n_pass": len(passing),
                "pairs_passing": ";".join(f"{a}/{b}" for a, b in passing),
                "full_sharpe": m["full"]["sharpe"],
                "is_sharpe": m["in_sample"]["sharpe"],
                "oos_sharpe": m["out_of_sample"]["sharpe"],
                "full_cagr": m["full"]["cagr"], "full_vol": m["full"]["vol"],
                "full_maxdd": m["full"]["maxdd"],
                "ann_turnover": m["ann_turnover_oneside"],
                "ann_cost_drag": m["ann_cost_drag"],
                "borrow_drag_ann": res["borrow_drag_ann"],
            })
            detail[vname] = (pd.DataFrame(pair_stats), m)
            print(vname, "IS", m["in_sample"]["sharpe"], "OOS",
                  m["out_of_sample"]["sharpe"], "n_pass", len(passing))

    var_df = pd.DataFrame(rows)
    save_csv(ROOT / "results" / "pairs_statarb_variants.csv", var_df, index=False)
    if "is_sharpe" not in var_df:
        print("\nNo pair combination passed the in-sample gate; no winner selected.")
        return
    ok = var_df.dropna(subset=["is_sharpe"])
    if ok.empty:
        print("\nNo pair combination has a finite in-sample Sharpe; no winner selected.")
        return
    best = ok.loc[ok["is_sharpe"].idxmax()]
    print("\nBEST BY IS SHARPE:", best["variant"])
    print(var_df.to_string())
    print("\nPer-pair table (best variant):")
    print(detail[best["variant"]][0].to_string())
    print("\nBest combo metrics:")
    print(json.dumps(detail[best["variant"]][1], indent=2))


def _parse_args(argv=None):
    # allow_abbrev=False: a shortened flag such as --reb is refused, not accepted
    # by the parser and then missed by the exact-name checks that act on it
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--rebase", action="store_true",
                        help="replace results/pairs_statarb_variants.csv when this run differs from it")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _parse_args()
    try:
        main()
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
