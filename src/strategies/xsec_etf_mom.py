"""Monthly cross-sectional momentum rotation across 34 equity ETFs.

Hypothesis
    Relative strength among country, sector and style funds tends to carry
    over to the following months: capital moves between market segments
    slowly, and investors who rebalance to benchmark weights or take profits
    on winners sell into strength. Holding the leaders collects from those
    sellers. The holdings are concentrated equity risk, so a market-wide
    breaker moves the portfolio to Treasuries when SPY is in a downtrend.

Rule
    Universe: the 34 equity ETFs of EQ_UNIVERSE below (US broad and size
    indices, countries and regions, US sectors and industries). Defensive
    asset: IEF, or cash before IEF has prices (first close 2002-07-30).
    At each month-end close:
      1. Score each ETF by the mean of its 3-, 6- and 12-month returns from
         dividend-adjusted month-end closes. An ETF is eligible only if all
         three exist.
      2. Breaker: if SPY's month-end close is below the mean of its last 10
         month-end closes, hold 100% IEF and drop the current holdings.
      3. Otherwise hold K=3 ETFs at equal weight. A holding is kept while it
         ranks 9th or better (sell buffer B=9); vacancies are filled from
         the top of the ranking.
      4. Orders are placed only at month-end; weights drift in between.
    Long-only and fully invested, except in breaker months before IEF
    exists. The engine assumes execution at the decision close.
    Costs are modelling assumptions: the commissions and fees of
    qcore.costs plus 3 bps of slippage per side.

Variants tried
    First sweep, 12 (re-run by --sweep): score {3/6/12 blend, the same blend
    skipping the latest month} x K {3, 4, 5} x breaker {on, off}, with B = K
    and the monthly target re-traded daily.
    Second sweep, 18 (research/rank_hysteresis_sweep.py): buffer B {3, 4, 5,
    6, 9, 12} x execution {daily re-trade to target, month-end orders only,
    orders only when the held names change}. build_weights(buffer=...,
    drift=...) covers the first two execution modes.

Selection
    In-sample Sharpe (data before 2018-01-01). Out-of-sample results
    (2018-01-01 onwards) are reported, not selected on.
    First sweep: blend without skip, K=3, breaker on, at 0.67 (K=5 0.66,
    K=4 0.65). The breaker-off variants did better from 2018 (0.55 to 0.66
    against 0.30 to 0.39) and worse before it (0.25 to 0.48 against 0.57 to
    0.67); the choice rests on the earlier period.
    Second sweep: B=9 has the highest in-sample Sharpe, under daily
    re-trading and under month-end orders alike (0.74 with month-end orders;
    B=12 0.73, B=3 0.70). It was adopted, with month-end orders, as a cost
    decision:
    against the first-sweep winner, costs fall from 0.91% to 0.26% a year
    and turnover from 10.4x to 5.7x. The Sharpe differences between buffers
    are small and are not claimed as an edge.
    The buffer was adopted after post-2018 results had been examined, so the
    fixed 2018 split is not an untouched holdout for the default rule.
    --sweep keeps the first sweep's unbuffered, daily-retargeted rule; it
    does not reconstruct the selection of every default parameter.

Result
    This code on data ending 2026-07-01, net of the modelled costs, Sharpe
    ratios in excess of the cash rate, from the first position on
    2002-03-28:
    Sharpe 0.65 full sample / 0.74 in sample / 0.49 out of sample (0.48 at
    double slippage); CAGR 13.74%, volatility 20.86%, maximum drawdown
    -27.85%. Turnover is 5.7x a year (buys plus sells) and costs take 0.26%
    a year.
    Reported statistics begin at the first position, not at the first
    decision. Every variant decides from the 15th month-end (2001-03-30),
    but with the breaker on the rule is risk-off then and, IEF not yet
    having prices, sits in cash until 2002-03-28. The engine trims those
    days, so breaker-on and breaker-off variants are scored over different
    windows. Counting the trimmed year at zero excess return gives 0.63
    full sample / 0.72 in sample.

Verdict
    Kept, as one of the four strategies of the combined portfolio
    (src/ensemble.py; with tsmom_trend, seasonality_flows and
    mean_reversion). What the second sweep bought is lower trading cost, not
    a better signal.

Default execution prints metrics for the buffered, drifting rule. --sweep
prints the first-sweep grid and saves it to
results/xsec_etf_mom_variants.csv; a saved table that differs from the new
run is kept and the new one is written to results/recomputed/ unless
--rebase is given. Unknown flags are rejected.

A blank month-end close after a ticker's first close raises an error naming
the ticker and date. Without that check a blank SPY close would read as
risk-on, a blank member close would drop the name from the ranking, and a
blank IEF close would leave the portfolio in cash.
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
MIN_HISTORY_MONTHS = 14   # first decision row for every variant (skip needs 13)
SLIPPAGE_BPS = 3.0        # sector/country ETF class

BEST_PARAMS = {"k": 3, "skip": False, "breaker": True, "buffer": 9, "drift": True}
# k, skip and breaker: best in-sample Sharpe of the first sweep (unrounded
# 0.674 for K=3 against 0.661 for K=5). buffer and drift: second sweep, a
# cost decision (see the module docstring). The sweep below retains the
# first sweep's unbuffered, daily-retargeted rule.


def month_end_closes(px: pd.DataFrame) -> pd.DataFrame:
    """Month-end closes under the engine's same-close execution assumption.
    Calendar-confirmed (qcore.calendar): a mid-month final data row is not
    a month-end, so live runs emit no phantom rebalance decision."""
    return px.loc[confirmed_month_ends(px.index)]


def build_targets(px: pd.DataFrame, k: int, skip: bool, breaker: bool,
                  buffer: int) -> pd.DataFrame:
    """Monthly decision rows (post-trade target weights at month-end closes).
    Fails closed on a blank decision close after listing (module docstring)."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(EQ_UNIVERSE):
        raise ValueError("k must be an integer within the equity universe")
    if isinstance(buffer, bool) or not isinstance(buffer, (int, np.integer)) or buffer < k:
        raise ValueError("buffer must be an integer >= k")
    cols = EQ_UNIVERSE + [DEFENSIVE]
    m = month_end_closes(px[cols])
    require_listed_closes(m, "The rule would silently change the breaker or the ranking",
                          what="month-end close")

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
            continue  # warm-up: same first decision month for every variant
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
    re-target accounting) — the rule behind results/xsec_etf_mom_variants.csv."""
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
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", action="store_true",
                    help="re-run the legacy 12-variant grid instead of the default rule")
    ap.add_argument("--rebase", action="store_true",
                    help="with --sweep: replace the saved variants table if this run differs")
    args = ap.parse_args()  # --rebase itself is read by qcore.records

    if args.sweep:
        df = sweep()
        print()
        # saved table is kept if this run differs (see qcore.records)
        save_csv(ROOT / "results" / "xsec_etf_mom_variants.csv", df, index=False)
        best = df.sort_values("is_sharpe", ascending=False).iloc[0]
        print("best by IS Sharpe:", best["name"])
        return

    px = load_prices()
    res = run_variant(px, **BEST_PARAMS)
    m = metrics(res)
    m["params"] = BEST_PARAMS
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
