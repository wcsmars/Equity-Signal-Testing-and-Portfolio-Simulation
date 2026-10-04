"""VIX term-structure regime switch, QQQ or IEF. Rejected: no timing value.

Hypothesis
    When one-month implied volatility (VIX) rises above three-month implied
    volatility (VIX3M), option buyers are paying more for immediate
    protection than for later protection, which has coincided with equity
    stress. When the curve slopes upward, markets are calm and the equity
    premium is earned. Moving from equities to Treasuries when the curve
    inverts should avoid part of the drawdowns; the other side would be the
    investors who hold equities through the stress.

Rule
    Signal: the 5-day average of VIX/VIX3M, lagged one further session
    because index closes are published after the equity close.
      below 0.95      100% QQQ
      0.95 to 1.05    50% QQQ, 50% IEF
      above 1.05      100% IEF
    Evaluated daily. The engine assumes execution at the decision close.
    The sample starts on 2006-07-24, once VIX3M history allows a signal.
    Costs are modelling assumptions: the commissions and fees of
    qcore.costs plus 2 bps of slippage per side.
    The rule held 100% QQQ on 73.6% of days, the mix on 22.1% and 100% IEF
    on 4.2%; its average QQQ weight was 0.85.

Variants tried
    12 (--sweep). Nine term-structure variants: one setting changed at a
    time from a SPY/IEF base (lower threshold 0.90 or 0.95, upper threshold
    1.00 or 1.05, smoothing 1, 5 or 10 days, cash in place of IEF), then
    three on QQQ. Three VIX-spike variants: long SPY for five sessions after
    VIX closes above k times its 20-day mean, k = 1.25, 1.30, 1.40, with the
    same one-session index lag.

Selection
    In-sample Sharpe (data before 2018-01-01): QQQ/IEF, 5-day average,
    0.95/1.05, at 0.79; the next two are 0.78 and 0.77. Out-of-sample
    results (2018-01-01 onwards) are reported, not selected on. The spike
    variants score 0.20 or less in sample and 0.17 or less out of sample
    and were dropped.

Result
    This code on data ending 2026-07-01, net of the modelled costs, Sharpe
    ratios in excess of the cash rate, from 2006-07-24:
    Sharpe 0.70 full sample / 0.79 in sample / 0.62 out of sample;
    CAGR 11.95%, volatility 15.79%, maximum drawdown -40.95%. Turnover is
    12.0x a year (buys plus sells) and costs take 0.41% a year.
    Same engine, costs and period, Sharpe full / in / out of sample:
      QQQ held throughout                        0.75 / 0.71 / 0.80
      85% QQQ and 15% IEF, rebalanced monthly
        (the rule's average weights)             0.77 / 0.77 / 0.78
      the rule without the publication lag       0.81 / 0.91 / 0.72
    From 2018 the rule's maximum drawdown is -40.95% and QQQ's is -35.24%.

Verdict
    Rejected; not part of the combined portfolio. In sample the rule beat
    holding QQQ (0.79 against 0.71). Out of sample it lost to QQQ (0.62
    against 0.80), with a deeper drawdown, and to a fixed mix with its own
    average weights (0.78). What it earned is equity exposure, not timing.
    An earlier version used the same-day index close, which is published
    after the trade would have been placed. The one-session lag that
    corrects this takes 0.11 off the full-sample Sharpe.

Default execution prints metrics. --sweep runs the 12 variants and saves
the table to results/vol_regime_variants.csv; a saved table that differs is
kept and the new one goes to results/recomputed/ unless --rebase is passed.

A session without a VIX or VIX3M print invalidates the averages containing
it; the last decided weights are held over those sessions. Up to
MAX_MISSING_PRINTS consecutive missing prints are held without a message
(scripts/data_quality.py is what reports index holes); more raise an error.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import metrics, run_backtest
from qcore.costs import IBKRHKCostModel
from qcore.data import load_indices, load_prices
from qcore.records import save_csv

# --- recorded variant: best in-sample Sharpe (before 2018) of the 12 swept ---
RISK_ASSET = "QQQ"
DEFENSIVE_ASSET = "IEF"
SMOOTH_DAYS = 5
LO_THRESHOLD = 0.95   # below -> full risk-on
HI_THRESHOLD = 1.05   # above -> full risk-off
SLIPPAGE_BPS = 2.0

# --- sweep grid (results/vol_regime_variants.csv, legacy row order) ---
# 9 term-structure variants: (risk, defensive, smooth, lo, hi) — the SPY/IEF
# baseline, one-at-a-time perturbations, then the QQQ branch.
# defensive "CASH" = no defensive leg; the idle remainder earns T-bills.
TS_VARIANTS = [
    ("SPY", "IEF", 5, 0.95, 1.00),
    ("SPY", "IEF", 5, 0.90, 1.00),
    ("SPY", "IEF", 5, 0.95, 1.05),
    ("SPY", "IEF", 10, 0.95, 1.00),
    ("SPY", "IEF", 1, 0.95, 1.00),
    ("SPY", "CASH", 5, 0.95, 1.00),
    ("QQQ", "IEF", 5, 0.95, 1.00),
    ("QQQ", "IEF", 10, 0.95, 1.00),
    ("QQQ", "IEF", 5, 0.95, 1.05),
]
SPIKE_KS = (1.25, 1.3, 1.4)  # trigger: VIX close > k * its 20d rolling mean
SPIKE_MA_DAYS = 20
SPIKE_HOLD_DAYS = 5

# A session without a VIX/VIX3M print has no ratio, which invalidates every
# SMOOTH_DAYS average containing it; the last decided weights are held over
# those sessions. Up to this many consecutive missing prints are held without
# a message; more is refused.
MAX_MISSING_PRINTS = 5


def build_weights(prices: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    ratio = (indices["^VIX"] / indices["^VIX3M"]).reindex(prices.index)
    # Use a prior-session index observation to avoid depending on an
    # official index close published after the assumed equity-close fill.
    smoothed = ratio.rolling(SMOOTH_DAYS).mean().shift(1).dropna()

    w_risk = pd.Series(0.5, index=smoothed.index)
    w_risk[smoothed < LO_THRESHOLD] = 1.0
    w_risk[smoothed > HI_THRESHOLD] = 0.0

    weights = pd.DataFrame(
        {RISK_ASSET: w_risk, DEFENSIVE_ASSET: 1.0 - w_risk}
    )
    # The signal dates are already on the price calendar. This fill only
    # covers sessions whose average is invalid because an index print inside
    # the window is missing: the last decided weights are held (past
    # information only, no lookahead). A run of missing prints longer than
    # MAX_MISSING_PRINTS (a holed or stale index file) is refused instead of
    # passing off an old regime as current; a shorter run is held without a
    # message, so a sweep is not interrupted once per variant.
    weights = (
        weights.reindex(prices.index.union(weights.index))
        .ffill()
        .reindex(prices.index)
        .dropna()
    )
    held = ~weights.index.isin(smoothed.index)
    if held.any():
        missing = ratio.loc[ratio.first_valid_index():].isna()
        longest = int(missing.groupby((~missing).cumsum()).sum().max())
        if longest > MAX_MISSING_PRINTS:
            latest = weights.index[held][-1].date()
            raise ValueError(
                f"VIX/VIX3M ratio missing for {longest} consecutive sessions "
                f"(limit {MAX_MISSING_PRINTS}; signal last unavailable on {latest}): "
                "refresh or repair the index data"
            )
    return weights


def _ts_weights(prices: pd.DataFrame, indices: pd.DataFrame, risk: str,
                defensive: str, smooth: int, lo: float, hi: float) -> pd.DataFrame:
    """build_weights() with the module constants temporarily patched, so the
    sweep exercises the default code path. defensive='CASH'
    drops the defensive column: the un-deployed remainder is idle cash,
    which the engine credits at ^IRX minus haircut."""
    global RISK_ASSET, DEFENSIVE_ASSET, SMOOTH_DAYS, LO_THRESHOLD, HI_THRESHOLD
    saved = (RISK_ASSET, DEFENSIVE_ASSET, SMOOTH_DAYS, LO_THRESHOLD, HI_THRESHOLD)
    RISK_ASSET, DEFENSIVE_ASSET = risk, defensive
    SMOOTH_DAYS, LO_THRESHOLD, HI_THRESHOLD = smooth, lo, hi
    try:
        weights = build_weights(prices, indices)
    finally:
        RISK_ASSET, DEFENSIVE_ASSET, SMOOTH_DAYS, LO_THRESHOLD, HI_THRESHOLD = saved
    if defensive == "CASH":
        weights = weights.drop(columns=["CASH"])
    return weights


def _spike_weights(prices: pd.DataFrame, indices: pd.DataFrame, k: float) -> pd.DataFrame:
    """VIX-spike example using the previous session's official close.

    The 4:15pm VIX print is unavailable for a same-day equity close fill.
    Historical sweep logs predating this lag used a different timing model.
    """
    vix = indices["^VIX"].reindex(prices.index)
    trigger = (vix > k * vix.rolling(SPIKE_MA_DAYS).mean()).shift(1, fill_value=False)
    in_position = trigger.astype(float).rolling(SPIKE_HOLD_DAYS, min_periods=1).max()
    weights = pd.DataFrame({"SPY": in_position})
    # align signal dates to the price calendar without lookahead
    weights = (
        weights.reindex(prices.index.union(weights.index))
        .ffill()
        .reindex(prices.index)
        .dropna()
    )
    return weights


def _sweep_row(res: dict, family: str, risk: str, defensive: str,
               smooth: float, p1: float, p2: float) -> dict:
    mm = metrics(res)
    return {
        "name": mm["name"], "family": family, "risk": risk,
        "defensive": defensive, "smooth": smooth, "p1": p1, "p2": p2,
        "start": mm["start"], "end": mm["end"],
        "full_sharpe": mm["full"]["sharpe"],
        "is_sharpe": mm["in_sample"]["sharpe"],
        "oos_sharpe": mm["out_of_sample"]["sharpe"],
        "full_cagr": mm["full"]["cagr"], "full_vol": mm["full"]["vol"],
        "full_maxdd": mm["full"]["maxdd"],
        "ann_turnover": mm["ann_turnover_oneside"],
        "ann_cost_drag": mm["ann_cost_drag"],
        "pct_pos_months": mm["pct_positive_months"],
        "worst_month": mm["worst_month"],
    }


def sweep() -> pd.DataFrame:
    """All 12 variants (9 term-structure + 3 spike-reversion), in the legacy
    row order of results/vol_regime_variants.csv (not sorted by Sharpe)."""
    prices, indices = load_prices(), load_indices()
    rows = []
    for risk, defensive, smooth, lo, hi in TS_VARIANTS:
        name = f"TS_{risk}_{defensive}_s{smooth}_lo{lo}_hi{hi}"
        w = _ts_weights(prices, indices, risk, defensive, smooth, lo, hi)
        res = run_backtest(w, prices, IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS),
                           name=name)
        rows.append(_sweep_row(res, "term_structure", risk, defensive,
                               smooth, lo, hi))
        print(f"{name:28s} IS {rows[-1]['is_sharpe']:5.2f}  "
              f"OOS {rows[-1]['oos_sharpe']:5.2f}  "
              f"full {rows[-1]['full_sharpe']:5.2f}", flush=True)
    for k in SPIKE_KS:
        name = f"SPIKE_SPY_k{k}_h{SPIKE_HOLD_DAYS}"
        w = _spike_weights(prices, indices, k)
        res = run_backtest(w, prices, IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS),
                           name=name)
        rows.append(_sweep_row(res, "spike_reversion", "SPY", "CASH",
                               float("nan"), k, float(SPIKE_HOLD_DAYS)))
        print(f"{name:28s} IS {rows[-1]['is_sharpe']:5.2f}  "
              f"OOS {rows[-1]['oos_sharpe']:5.2f}  "
              f"full {rows[-1]['full_sharpe']:5.2f}", flush=True)
    return pd.DataFrame(rows)


def main() -> dict:
    prices = load_prices()
    indices = load_indices()
    weights = build_weights(prices, indices)
    result = run_backtest(
        weights,
        prices,
        IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS),
        name="vol_regime_TS_QQQ_IEF_s5_lo0.95_hi1.05",
    )
    return metrics(result)


def _parse_args(argv=None):
    # allow_abbrev=False: a shortened flag such as --reb is refused, not accepted
    # by the parser and then missed by the exact-name checks that act on it
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--sweep", action="store_true",
                        help="run all 12 variants and save results/vol_regime_variants.csv")
    parser.add_argument("--rebase", action="store_true",
                        help="with --sweep: replace the saved variants file when this run differs from it")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    try:
        if args.sweep:
            df = sweep()
            print()
            save_csv(ROOT / "results" / "vol_regime_variants.csv", df, index=False)
        else:
            print(json.dumps(main(), indent=2))
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
