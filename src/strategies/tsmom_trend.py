"""Monthly multi-asset trend following (time-series momentum) on 14 ETFs.

Hypothesis
    Trends in broad asset classes persist for months. News is absorbed
    gradually, and large investors adjust slowly or trade against the move
    for reasons other than expected return: rebalancing to fixed weights,
    hedging, taking profits. Holding what has been rising and stepping aside
    from what has been falling collects from them, at the price of whipsaw
    losses when a trend reverses.

Rule
    Universe: SPY QQQ IWM EFA EEM FXI EWJ (equities), TLT IEF LQD (bonds),
    GLD SLV DBC (commodities), VNQ (real estate).
    Decisions are made at the close of the last trading day of each month,
    from data up to that close.
    Eligibility: 13 month-end closes and a 60-day volatility estimate. ETFs
    listed after 2000 join as they qualify.
    Signal per asset, in {0, 0.5, 1}: the average of
      1 if the month-end close is above the mean of the last 10 month-end
        closes, else 0, and
      1 if 12-1 momentum is positive (close one month ago / close twelve
        months ago - 1 > 0), else 0.
    Weight: signal x inverse-volatility share. The share is (1/vol) divided
    by the sum of 1/vol over ALL eligible assets, vol being the 60-day
    standard deviation of daily returns. Exposure therefore reaches 1.0 only
    when every eligible asset is fully on; it averaged 0.66.
    The remainder stays in cash earning the engine's Treasury-bill proxy
    (13-week bill yield less 10 bps). No shorts, no leverage.
    Holdings drift between month-ends; orders are placed only at the monthly
    decision. The engine assumes execution at the decision close.
    Costs are modelling assumptions: the commissions and fees of
    qcore.costs plus 3 bps of slippage per side.

Variants tried
    12: signal {10-month average, 12-1 momentum, blend of the two} x
    weighting {equal, inverse volatility} x remainder {cash, SHY}.
    --sweep re-runs all of them.

Selection
    In-sample Sharpe (data before 2018-01-01) chose the blend signal with
    inverse-volatility weights. For the remainder the same rule preferred
    SHY in all six signal/weighting pairs (0.82 against 0.77 for the chosen
    pair). The default nevertheless holds cash: from 2018 the order reverses
    in all six pairs (0.42 with cash against 0.34 with SHY), and over the
    full period the two are level (0.64 and 0.65).
    The cash remainder replaced SHY after post-2018 results had been
    examined. It is a post-test revision, not a choice supported by the
    pre-2018 selection rule, and the 2018+ segment is therefore not an
    untouched holdout for this specification. --sweep retains both.

Result
    This code on data ending 2026-07-01, net of the modelled costs, Sharpe
    ratios in excess of the cash rate, from the first position on
    2001-02-28:
    Sharpe 0.64 full sample / 0.77 in sample / 0.42 out of sample (0.41 at
    double slippage); CAGR 5.58%, volatility 6.25%, maximum drawdown
    -14.67%. Turnover is 2.0x a year (buys plus sells) and costs take 0.18%
    a year.

Verdict
    Kept, as one of the four strategies of the combined portfolio
    (src/ensemble.py; with xsec_etf_mom, seasonality_flows and
    mean_reversion).

Run the module to print metrics; --sweep prints the 12 signal, weighting
and remainder combinations and saves them to
results/tsmom_trend_variants.csv. A saved table that differs from the new
run is kept and the new one is written to results/recomputed/ unless
--rebase is given. Unknown flags are rejected.

A blank close after an asset's first close raises an error naming the ticker
and date. Without that check the asset would become ineligible and every
other inverse-volatility share would rise without notice.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import drift_weights, metrics, run_backtest  # noqa: E402
from qcore.calendar import confirmed_month_ends  # noqa: E402
from qcore.costs import IBKRHKCostModel  # noqa: E402
from qcore.data import load_prices, require_listed_closes  # noqa: E402
from qcore.records import save_csv  # noqa: E402

RISK = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "EWJ",
        "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ"]
CASH = "SHY"
SLIPPAGE_BPS = 3.0

BEST = {"signal": "blend", "weighting": "iv", "sleeve": "cash"}
# Cash is a post-test revision; the sweep retains the earlier SHY variant.


def build_signals(px: pd.DataFrame):
    """Month-end signals/eligibility. Uses only data up to each decision close.
    Month-ends are calendar-confirmed (qcore.calendar): a mid-month final data
    row is NOT a decision date, so live runs emit no phantom rebalance.
    Fails closed on a blank close after listing (see module docstring)."""
    mp = px.loc[confirmed_month_ends(px.index), RISK]       # month-end closes
    if len(mp):  # every close up to the last decision feeds a signal or a vol window
        require_listed_closes(px.loc[:mp.index[-1], RISK],
                              "The rule would silently drop the asset and raise every other share")
    ma10 = (mp > mp.rolling(10).mean()).astype(float)       # 10m MA filter
    mom121 = ((mp.shift(1) / mp.shift(12) - 1) > 0).astype(float)  # 12-1 mom
    vol_me = (px[RISK].pct_change(fill_method=None)
              .rolling(60).std().loc[mp.index])             # 60d vol at decision date
    elig = mp.rolling(13).count().eq(13) & vol_me.notna() & vol_me.gt(0)
    shy_ok = px[CASH].loc[mp.index].notna()
    sigs = {"ma10": ma10, "mom121": mom121, "blend": 0.5 * (ma10 + mom121)}
    return sigs, elig, vol_me, shy_ok


def month_end_weights(sig, elig, vol_me, shy_ok, weighting: str, sleeve: str):
    if weighting not in {"ew", "iv"} or sleeve not in {"cash", "shy"}:
        raise ValueError("weighting must be ew/iv and sleeve must be cash/shy")
    elig = elig & vol_me.gt(0) & np.isfinite(vol_me)
    s = sig * elig
    if weighting == "ew":
        n = elig.sum(axis=1)
        w = s.div(n.replace(0, np.nan), axis=0)
    else:  # inverse-vol shares normalized across ALL eligible assets
        inv = (1.0 / vol_me).where(elig)
        w = s * inv.div(inv.sum(axis=1), axis=0)
    w = w.fillna(0.0)
    w[CASH] = 0.0
    if sleeve == "shy":
        listed = shy_ok.to_numpy(dtype=bool)
        blank = ~listed & np.maximum.accumulate(listed)
        if blank.any():  # would silently park the residual in cash instead
            raise ValueError(f"blank {CASH} month-end close after listing: "
                             f"{', '.join(str(t.date()) for t in shy_ok.index[blank][:10])}")
        resid = (1.0 - w[RISK].sum(axis=1)).clip(lower=0.0)
        w[CASH] = resid.where(shy_ok, 0.0)
    return w


def to_daily_drift(w_me: pd.DataFrame, px: pd.DataFrame,
                   cash_rate: float | pd.Series | None = None) -> pd.DataFrame:
    """Use the engine's drift convention, including idle-cash accrual.

    Prices are aligned by ticker, so caller column order cannot change the
    portfolio. Pass the same cash_rate used by run_backtest.
    """
    return drift_weights(w_me, px, cash_rate=cash_rate)


def run_variant(px, sigs, elig, vol_me, shy_ok, signal, weighting, sleeve):
    name = f"{signal}_{weighting}_{sleeve}"
    w_me = month_end_weights(sigs[signal], elig, vol_me, shy_ok, weighting, sleeve)
    w = to_daily_drift(w_me, px)
    res = run_backtest(w, px, IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS), name=name)
    return metrics(res)


def sweep(px, sigs, elig, vol_me, shy_ok):
    rows = []
    for signal in sigs:
        for weighting in ["ew", "iv"]:
            for sleeve in ["cash", "shy"]:
                m = run_variant(px, sigs, elig, vol_me, shy_ok, signal, weighting, sleeve)
                rows.append({
                    "variant": m["name"], "signal": signal, "weighting": weighting,
                    "sleeve": sleeve,
                    "full_sharpe": m["full"]["sharpe"], "is_sharpe": m["in_sample"]["sharpe"],
                    "oos_sharpe": m["out_of_sample"]["sharpe"],
                    "full_cagr": m["full"]["cagr"], "full_vol": m["full"]["vol"],
                    "full_maxdd": m["full"]["maxdd"],
                    "is_cagr": m["in_sample"]["cagr"], "is_maxdd": m["in_sample"]["maxdd"],
                    "oos_cagr": m["out_of_sample"]["cagr"], "oos_maxdd": m["out_of_sample"]["maxdd"],
                    "ann_turnover": m["ann_turnover_oneside"], "ann_cost_drag": m["ann_cost_drag"],
                    "pct_pos_months": m["pct_positive_months"], "worst_month": m["worst_month"],
                    "start": m["start"], "end": m["end"],
                })
    df = pd.DataFrame(rows)
    # saved table is kept if this run differs (see qcore.records)
    save_csv(ROOT / "results" / "tsmom_trend_variants.csv", df, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", action="store_true",
                    help="re-run the 12 tested variants instead of the default one")
    ap.add_argument("--rebase", action="store_true",
                    help="with --sweep: replace the saved variants table if this run differs")
    args = ap.parse_args()  # --rebase itself is read by qcore.records

    px = load_prices()[RISK + [CASH]]
    sigs, elig, vol_me, shy_ok = build_signals(px)

    if args.sweep:
        df = sweep(px, sigs, elig, vol_me, shy_ok)
        print(df.to_string(index=False))
        return

    m = run_variant(px, sigs, elig, vol_me, shy_ok, **BEST)
    out = {"params": BEST, "slippage_bps": SLIPPAGE_BPS, "universe": RISK,
           "off_sleeve": CASH if BEST["sleeve"] == "shy" else "T-bill cash (^IRX - 10bp)",
           "metrics": m}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
