"""Short-horizon dip buying on a fixed liquid US equity ETF universe.

Enter long when RSI(2) < 5 and close > SMA(200); exit when RSI(2) > 70 or
after 10 trading days. Each open position receives min(0.20, 1/n_open),
with remaining capital in modeled cash. Signals use adjusted closes and
assume execution at that close; the engine credits subsequent returns and
charges fixed commissions plus 3 bps per side in assumed slippage.

The retained parameters are research examples, not a validated investment
recommendation. Run this module to print metrics for the local cache.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import metrics, run_backtest
from qcore.costs import IBKRHKCostModel
from qcore.data import load_prices

UNIVERSE = ["SPY", "QQQ", "DIA", "IWM", "MDY",
            "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]

BEST_PARAMS = {"entry_th": 5, "exit_th": 70, "max_hold": 10, "w_max": 0.20}
SLIPPAGE_BPS = 3.0


def rsi(px: pd.DataFrame, period: int = 2) -> pd.DataFrame:
    """Wilder RSI computed with rolling data only (no lookahead)."""
    delta = px.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    out = 100.0 - 100.0 / (1.0 + ag / al)
    return out.where(al > 0, 100.0).where(ag.notna() & al.notna())


def build_weights(px: pd.DataFrame, entry_th: float, exit_th: float,
                  max_hold: int, w_max: float) -> pd.DataFrame:
    """Per-ticker entry/exit state machine -> capped equal weights."""
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
    print(json.dumps(main(), indent=2))
