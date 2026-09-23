"""Volatility term-structure allocation research example.

A five-day average of VIX/VIX3M, lagged one additional session for index
publication timing, assigns QQQ weight 1 below 0.95, 0.5 between 0.95 and
1.05, and 0 above 1.05; IEF receives the remainder. Costs assume 2 bps
per-side slippage. The engine credits returns after the decision close.

Default execution prints metrics. --sweep also includes older VIX-spike
experiments using same-day index closes without the extra publication lag;
those rows do not satisfy the same equity-close information constraint and
must not be treated as an executable comparison. Results are exploratory.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import metrics, run_backtest
from qcore.costs import IBKRHKCostModel
from qcore.data import load_indices, load_prices

# --- retained example parameters ---
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


def build_weights(prices: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    ratio = (indices["^VIX"] / indices["^VIX3M"]).dropna()
    # Use a prior-session index observation to avoid depending on an
    # official index close published after the assumed equity-close fill.
    smoothed = ratio.rolling(SMOOTH_DAYS).mean().shift(1).dropna()

    w_risk = pd.Series(0.5, index=smoothed.index)
    w_risk[smoothed < LO_THRESHOLD] = 1.0
    w_risk[smoothed > HI_THRESHOLD] = 0.0

    weights = pd.DataFrame(
        {RISK_ASSET: w_risk, DEFENSIVE_ASSET: 1.0 - w_risk}
    )
    # align signal dates to the price calendar without lookahead
    weights = (
        weights.reindex(prices.index.union(weights.index))
        .ffill()
        .reindex(prices.index)
        .dropna()
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
    """VIX-spike mean reversion: 100% SPY for the 5 days after a close where
    VIX > k * its 20-day rolling mean; cash otherwise.

    This historical experiment uses the same-day VIX close without the
    additional publication lag in build_weights(). It therefore uses
    information unavailable at the assumed equity-close fill time."""
    vix = indices["^VIX"].dropna()
    trigger = vix > k * vix.rolling(SPIKE_MA_DAYS).mean()
    in_position = trigger.astype(float).rolling(SPIKE_HOLD_DAYS).max()
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


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        df = sweep()
        out = ROOT / "results" / "vol_regime_variants.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nsaved {out}")
    else:
        print(json.dumps(main(), indent=2))
