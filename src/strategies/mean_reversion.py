"""Short-horizon dip buying on a fixed universe of liquid US equity ETFs.

Hypothesis
    A sharp fall over a few days in a broad or sector ETF that is still in a
    long-term uptrend is more often caused by investors who have to sell at
    once (risk limits, redemptions, stop-losses) than by news that changes
    its value. Buying from them supplies liquidity, and the expected payment
    is a partial rebound within days. The 200-day filter keeps the rule out
    of downtrends, where a fall is more likely to carry information.

Rule
    Universe (fixed, 14 ETFs): SPY QQQ DIA IWM MDY and the sector funds
    XLK XLF XLE XLV XLI XLP XLY XLU XLB.
    At each close, per ETF, from dividend-adjusted closes:
      enter long if flat, Wilder RSI(2) < 5 and close > 200-day average;
      exit if RSI(2) > 70 or after 10 trading days.
    Each open position receives min(0.20, 1/n_open) of capital, so total
    exposure never exceeds 1.0. The remainder is cash earning the engine's
    Treasury-bill proxy (13-week bill yield less 10 bps). Long-only, no
    leverage.
    Weights decided at a close earn the return from that close to the next:
    the engine assumes execution at the decision close.
    Costs are modelling assumptions: the commissions and fees of
    qcore.costs plus 3 bps of slippage per side.

Variants tried
    12: entry threshold {5, 10, 15} x exit threshold {60, 70} x maximum
    holding period {5, 10} days. The 20% position cap was fixed.
    research/mean_reversion_sweep.py runs the grid through build_weights.

Selection
    The variant with the highest in-sample Sharpe ratio (data before
    2018-01-01): entry 5, exit 70, hold 10, at 0.69. The next best is 0.58.
    Out-of-sample results (2018-01-01 onwards) are reported, not selected on.

Result
    This code on data ending 2026-07-01, net of the modelled costs, Sharpe
    ratios in excess of the cash rate, from the first position on
    2000-11-10:
    Sharpe 0.50 full sample / 0.69 in sample / 0.23 out of sample;
    CAGR 4.97%, volatility 6.81%, maximum drawdown -14.42%.
    Before costs the full-sample Sharpe is 0.66. Turnover is 20.3x a year
    (buys plus sells) and costs take 1.05% a year; at double slippage the
    out-of-sample Sharpe falls to 0.14. About one sixth of capital is
    invested on average.

Verdict
    Kept, as one of the four strategies of the combined portfolio
    (src/ensemble.py; with tsmom_trend, xsec_etf_mom and seasonality_flows).
    It is the weakest of the four out of sample (0.23, against 0.42 to 0.58
    for the others) and the most sensitive to trading costs, and is carried
    with that caveat.

Run this module to print the metrics for the local cache; it takes no
options and rejects unknown flags.

A blank close after a ticker's first close raises an error naming the ticker
and date. Without that check the 200-day average would be undefined, so the
ticker could not be entered for the next 200 sessions and the other positions
would be sized as if nothing had happened. Every close is a decision close,
the final row included.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import metrics, run_backtest
from qcore.costs import IBKRHKCostModel
from qcore.data import load_prices, require_listed_closes

UNIVERSE = ["SPY", "QQQ", "DIA", "IWM", "MDY",
            "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]

BEST_PARAMS = {"entry_th": 5, "exit_th": 70, "max_hold": 10, "w_max": 0.20}
SLIPPAGE_BPS = 3.0


def rsi(px: pd.DataFrame, period: int = 2) -> pd.DataFrame:
    """Wilder RSI computed with rolling data only (no lookahead)."""
    if isinstance(period, bool) or not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError("period must be a positive integer")
    delta = px.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    def wilder(values):
        out = np.full(values.shape, np.nan, dtype=float)
        for col in range(values.shape[1]):
            count, total, average = 0, 0.0, np.nan
            for row, value in enumerate(values.iloc[:, col].to_numpy()):
                if not np.isfinite(value):
                    count, total, average = 0, 0.0, np.nan
                    continue
                if count < period:
                    count += 1
                    total += value
                    if count == period:
                        average = total / period
                else:
                    average = ((period - 1) * average + value) / period
                out[row, col] = average
        return pd.DataFrame(out, index=values.index, columns=values.columns)
    ag, al = wilder(gain), wilder(loss)
    out = 100.0 - 100.0 / (1.0 + ag / al)
    out = out.where(al > 0, 100.0).where((ag != 0) | (al != 0), 50.0)
    return out.where(ag.notna() & al.notna())


def build_weights(px: pd.DataFrame, entry_th: float, exit_th: float,
                  max_hold: int, w_max: float) -> pd.DataFrame:
    """Per-ticker entry/exit state machine -> capped equal weights.
    Fails closed on a blank close after listing (see module docstring)."""
    if not (0 <= entry_th < exit_th <= 100) or not (0 < w_max <= 1):
        raise ValueError("require 0 <= entry_th < exit_th <= 100 and 0 < w_max <= 1")
    if isinstance(max_hold, bool) or not isinstance(max_hold, (int, np.integer)) or max_hold < 1:
        raise ValueError("max_hold must be a positive integer")
    require_listed_closes(px, "The rule would silently stop entering the ticker for 200 sessions")
    rsi2 = rsi(px, 2)
    sma200 = px.rolling(200, min_periods=200).mean()
    pos = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for c in px.columns:
        r, p, s = rsi2[c].values, px[c].values, sma200[c].values
        out = np.zeros(len(p))
        holding, held = False, 0
        for i in range(len(p)):
            if holding:
                held += 1
                if (not np.isnan(r[i]) and r[i] > exit_th) or held >= max_hold:
                    holding, held = False, 0
            elif not (np.isnan(r[i]) or np.isnan(s[i]) or np.isnan(p[i])):
                if r[i] < entry_th and p[i] > s[i]:
                    holding, held = True, 0
            out[i] = 1.0 if holding else 0.0
        pos[c] = out
    n_open = pos.sum(axis=1)
    scale = np.minimum(w_max, 1.0 / n_open.replace(0.0, np.nan))
    return pos.mul(scale.fillna(0.0), axis=0)


def main() -> dict:
    px = load_prices()[UNIVERSE]
    w = build_weights(px, BEST_PARAMS["entry_th"], BEST_PARAMS["exit_th"],
                      BEST_PARAMS["max_hold"], BEST_PARAMS["w_max"])
    res = run_backtest(w, px, IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS),
                       name="mean_reversion_e5_x70_h10")
    m = metrics(res)
    m["params"] = BEST_PARAMS | {"slippage_bps": SLIPPAGE_BPS, "universe": UNIVERSE}
    return m


if __name__ == "__main__":
    argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n")[0]).parse_args()
    try:
        print(json.dumps(main(), indent=2))
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
