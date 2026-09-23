"""Monthly cross-sectional ETF momentum rotation example.

Rank eligible equity ETFs by mean 3-, 6- and 12-month adjusted returns.
Hold three equal-weight positions, retain incumbents through rank nine,
and fill vacancies from the highest ranks. A SPY 10-month moving-average
filter sends the portfolio to IEF when disabled; modeled cash covers periods
before IEF is available. Weights drift between month-end decisions. Costs
assume 3 bps per side and execution at the decision close.

Default execution prints metrics for the buffered, drifting rule. --sweep
retains an earlier unbuffered, daily-retargeted parameter grid; it does not
reconstruct selection of every parameter in the default specification.
The fixed 2018 split should not be interpreted as an untouched holdout.
"""

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
from qcore.data import load_prices  # noqa: E402

EQ_UNIVERSE = [
    # US broad + style/size
    "SPY", "QQQ", "IWM", "DIA", "MDY",
    # international / country
    "EFA", "EEM", "VGK", "EWJ", "FXI", "EWY", "EWT", "EWZ",
    "EWA", "EWC", "EWG", "EWU", "EWH",
    # US sectors / industries
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB",
    "XBI", "SMH", "KRE", "XME", "XOP", "IYR", "VNQ",
]
DEFENSIVE = "IEF"
LOOKBACKS = (3, 6, 12)
SMA_MONTHS = 10           # breaker: SPY vs 10-month SMA (fixed, not swept)
MIN_HISTORY_MONTHS = 14   # common start across variants (skip needs 13)
SLIPPAGE_BPS = 3.0        # sector/country ETF class

BEST_PARAMS = {"k": 3, "skip": False, "breaker": True, "buffer": 9, "drift": True}
# The sweep below retains the earlier unbuffered, daily-retargeted rule.


def month_end_closes(px: pd.DataFrame) -> pd.DataFrame:
    """Month-end closes under the engine's same-close execution assumption.
    Calendar-confirmed (qcore.calendar): a mid-month final data row is not
    a month-end, so live runs emit no phantom rebalance decision."""
    return px.loc[confirmed_month_ends(px.index)]


def build_targets(px: pd.DataFrame, k: int, skip: bool, breaker: bool,
                  buffer: int) -> pd.DataFrame:
    """Monthly decision rows (post-trade target weights at month-end closes)."""
    cols = EQ_UNIVERSE + [DEFENSIVE]
    m = month_end_closes(px[cols])

    parts = []
    for lb in LOOKBACKS:
        if skip:
            parts.append(m[EQ_UNIVERSE].shift(1) / m[EQ_UNIVERSE].shift(1 + lb) - 1.0)
        else:
            parts.append(m[EQ_UNIVERSE] / m[EQ_UNIVERSE].shift(lb) - 1.0)
    # eligible only if every lookback exists -> mean over aligned frames
    score = (parts[0] + parts[1] + parts[2]) / 3.0

    spy = m["SPY"]
    risk_off = spy < spy.rolling(SMA_MONTHS).mean()

    w = pd.DataFrame(0.0, index=m.index, columns=cols)
    held: list = []
    for i, t in enumerate(m.index):
        if i < MIN_HISTORY_MONTHS:
            continue  # common warm-up so all variants start together
        if breaker and bool(risk_off.loc[t]):
            held = []  # book liquidated into IEF: hysteresis memory gone
            if not np.isnan(m.loc[t, DEFENSIVE]):
                w.loc[t, DEFENSIVE] = 1.0
            # else: stay in cash (IEF not yet listed)
            continue
        s = score.loc[t].dropna()
        if len(s) < k:
            held = []
            continue
        rank = s.rank(ascending=False, method="first")
        keep = [c for c in held if c in rank.index and rank[c] <= buffer]
        order = s.sort_values(ascending=False, kind="stable").index
        held = keep + [c for c in order if c not in keep][: k - len(keep)]
        w.loc[t, held] = 1.0 / k
    return w


def build_weights(px: pd.DataFrame, k: int, skip: bool, breaker: bool,
                  buffer: int | None = None, drift: bool = False) -> pd.DataFrame:
    """Daily weights. buffer=None -> plain top-K (B=K). drift=False -> hold
    the monthly target constant, so the engine charges daily re-targeting
    (legacy accounting, kept for the --sweep selection record); drift=True ->
    weights drift intramonth, orders only at month-end."""
    monthly = build_targets(px, k, skip, breaker,
                            buffer if buffer is not None else k)
    if drift:
        return drift_weights(monthly, px[monthly.columns])
    return monthly.reindex(px.index).ffill().fillna(0.0)


def run_variant(px: pd.DataFrame, k: int, skip: bool, breaker: bool,
                buffer: int | None = None, drift: bool = False) -> dict:
    name = f"b3612{'skip' if skip else ''}_K{k}_brk{'On' if breaker else 'Off'}"
    if buffer:
        name += f"_B{buffer}"
    if drift:
        name += "_drift"
    w = build_weights(px, k=k, skip=skip, breaker=breaker, buffer=buffer, drift=drift)
    return run_backtest(w, px[w.columns], IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS), name=name)


def sweep() -> pd.DataFrame:
    """Legacy 12-variant blend/K/breaker selection sweep (B=K, daily
    re-target accounting) — regenerates results/xsec_etf_mom_variants.csv."""
    px = load_prices()
    rows = []
    for skip in (False, True):
        for k in (3, 4, 5):
            for breaker in (True, False):
                res = run_variant(px, k=k, skip=skip, breaker=breaker)
                mm = metrics(res)
                rows.append({
                    "name": mm["name"], "k": k, "skip": skip, "breaker": breaker,
                    "start": mm["start"], "end": mm["end"],
                    "full_sharpe": mm["full"]["sharpe"],
                    "is_sharpe": mm["in_sample"]["sharpe"],
                    "oos_sharpe": mm["out_of_sample"]["sharpe"],
                    "full_cagr": mm["full"]["cagr"], "full_vol": mm["full"]["vol"],
                    "full_maxdd": mm["full"]["maxdd"],
                    "oos_cagr": mm["out_of_sample"]["cagr"],
                    "oos_maxdd": mm["out_of_sample"]["maxdd"],
                    "ann_turnover": mm["ann_turnover_oneside"],
                    "ann_cost_drag": mm["ann_cost_drag"],
                    "pct_pos_months": mm["pct_positive_months"],
                    "worst_month": mm["worst_month"],
                })
                print(f"{mm['name']:24s} IS {mm['in_sample']['sharpe']:5.2f}  "
                      f"OOS {mm['out_of_sample']['sharpe']:5.2f}  "
                      f"full {mm['full']['sharpe']:5.2f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    if "--sweep" in sys.argv:
        df = sweep()
        out = ROOT / "results" / "xsec_etf_mom_variants.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nsaved {out}")
        best = df.sort_values("is_sharpe", ascending=False).iloc[0]
        print("best by IS Sharpe:", best["name"])
        return

    px = load_prices()
    res = run_variant(px, **BEST_PARAMS)
    m = metrics(res)
    m["params"] = BEST_PARAMS
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
