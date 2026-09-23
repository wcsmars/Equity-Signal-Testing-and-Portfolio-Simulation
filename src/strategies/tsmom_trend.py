"""Monthly multi-asset time-series momentum research example.

Blend a 10-month moving-average filter with positive 12-1 momentum.
Inverse 60-day volatility shares are normalized over all eligible ETFs;
the signal scales each share and residual weight remains in modeled cash.
Assets require 13 month-end observations and available volatility estimates.
Holdings drift between monthly decisions; the model assumes execution at
the decision close and charges 3 bps per-side slippage.

The cash residual replaced SHY after post-2018 results had been examined.
It is a historical post-test revision, not a choice supported solely by
the pre-2018 selection rule. The 2018+ segment is therefore not an untouched
holdout for this specification. --sweep retains both cash and SHY variants.

Run the module to print metrics; --sweep writes the 12 signal, weighting
and residual-allocation combinations to results/tsmom_trend_variants.csv.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import metrics, run_backtest  # noqa: E402
from qcore.calendar import confirmed_month_ends  # noqa: E402
from qcore.costs import IBKRHKCostModel  # noqa: E402
from qcore.data import load_prices  # noqa: E402

RISK = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "EWJ",
        "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ"]
CASH = "SHY"
SLIPPAGE_BPS = 3.0

BEST = {"signal": "blend", "weighting": "iv", "sleeve": "cash"}
# Cash is a post-test revision; the sweep retains the earlier SHY variant.


def build_signals(px: pd.DataFrame):
    """Month-end signals/eligibility. Uses only data up to each decision close.
    Month-ends are calendar-confirmed (qcore.calendar): a mid-month final data
    row is NOT a decision date, so live runs emit no phantom rebalance."""
    mp = px.loc[confirmed_month_ends(px.index), RISK]       # month-end closes
    ma10 = (mp > mp.rolling(10).mean()).astype(float)       # 10m MA filter
    mom121 = ((mp.shift(1) / mp.shift(12) - 1) > 0).astype(float)  # 12-1 mom
    vol_me = (px[RISK].pct_change(fill_method=None)
              .rolling(60).std().loc[mp.index])             # 60d vol at decision date
    elig = mp.rolling(13).count().eq(13) & vol_me.notna()
    shy_ok = px[CASH].loc[mp.index].notna()
    sigs = {"ma10": ma10, "mom121": mom121, "blend": 0.5 * (ma10 + mom121)}
    return sigs, elig, vol_me, shy_ok


def month_end_weights(sig, elig, vol_me, shy_ok, weighting: str, sleeve: str):
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
        resid = (1.0 - w[RISK].sum(axis=1)).clip(lower=0.0)
        w[CASH] = resid.where(shy_ok, 0.0)
    return w


def to_daily_drift(w_me: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Expand month-end targets to daily rows that drift with prices, so the
    engine charges turnover only at the monthly rebalance (true buy-and-hold
    between month-ends). The drift model holds the cash remainder constant,
    ignoring the engine's T-bill credit on it - sub-bp/yr turnover effect."""
    pxf = px.ffill()
    P, W = pxf.values, w_me.reindex(px.index).values
    out = np.zeros_like(P)
    cur = None
    p0 = None
    cash = 0.0
    for i in range(len(P)):
        if not np.isnan(W[i]).all():                 # rebalance day: set target
            cur = np.nan_to_num(W[i])
            p0 = P[i].copy()
            cash = 1.0 - cur.sum()
            out[i] = cur
        elif cur is not None:                        # drift day
            g = np.where(np.isnan(P[i] / p0), 1.0, P[i] / p0)
            vals = cur * g
            out[i] = vals / (vals.sum() + cash)
    return pd.DataFrame(out, index=px.index, columns=w_me.columns)


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
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results" / "tsmom_trend_variants.csv", index=False)
    return df


def main():
    px = load_prices()[RISK + [CASH]]
    sigs, elig, vol_me, shy_ok = build_signals(px)

    if "--sweep" in sys.argv:
        df = sweep(px, sigs, elig, vol_me, shy_ok)
        print(df.to_string(index=False))
        return

    m = run_variant(px, sigs, elig, vol_me, shy_ok, **BEST)
    out = {"params": BEST, "slippage_bps": SLIPPAGE_BPS, "universe": RISK,
           "off_sleeve": CASH, "metrics": m}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
