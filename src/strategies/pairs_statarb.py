"""ETF pairs spread-reversion example with rolling hedge ratios.

A trailing 90-day log-price regression defines the hedge ratio; a 60-day
consistent-beta spread z-score triggers entries outside +/-2. Entry leg
weights remain fixed until |z| < 0.5 or a 20-day holding limit. Re-entry
requires the spread to return inside the entry band. Included pairs are
screened by standalone pre-2018 net Sharpe and share a 1.5 gross budget.

Costs include 3 bps per-side slippage and a fixed annual stock-borrow proxy.
Borrow availability, variable lending fees and margin calls are not modeled.
Signals assume same-close execution. This is an exploratory example; run
the module to calculate metrics on the locally obtained cache.
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

PAIRS = [
    ("GLD", "GDX"), ("XLE", "XOP"), ("EWA", "EWC"), ("SPY", "MDY"),
    ("QQQ", "XLK"), ("KRE", "XLF"), ("IEF", "TLT"),
]
H, Z = 90, 60                    # hedge and spread lookback lengths
ENTRY, EXIT_Z, TIMEOUT = 2.0, 0.5, 20
GROSS_CAP, BORROW_RATE, SLIPPAGE_BPS = 1.5, 0.01, 3.0


def pair_weights(px: pd.DataFrame, a: str, b: str, gross: float = 1.0) -> pd.DataFrame:
    """Daily target weights for one pair, unit gross while in a trade."""
    sub = np.log(px[[a, b]].dropna())
    la, lb = sub[a], sub[b]
    beta = la.rolling(H).cov(lb) / lb.rolling(H).var()
    m_a, m_b = la.rolling(Z).mean(), lb.rolling(Z).mean()
    v_a, v_b = la.rolling(Z).var(), lb.rolling(Z).var()
    c_ab = la.rolling(Z).cov(lb)
    sd = np.sqrt((v_a + beta**2 * v_b - 2 * beta * c_ab).clip(lower=0))
    z = ((la - beta * lb - (m_a - beta * m_b)) / sd.replace(0, np.nan)).values
    bet = beta.values

    n = len(sub)
    w_a, w_b = np.zeros(n), np.zeros(n)
    sign, days, blocked = 0, 0, False
    cw_a = cw_b = 0.0
    for t in range(n):
        zt, bt = z[t], bet[t]
        if not np.isfinite(zt) or not np.isfinite(bt):
            if sign != 0:
                days += 1
                if days >= TIMEOUT:
                    sign, blocked = 0, True
            w_a[t], w_b[t] = (cw_a, cw_b) if sign != 0 else (0.0, 0.0)
            continue
        if blocked and abs(zt) < ENTRY:
            blocked = False
        if sign == 0:
            if not blocked and bt > 0 and abs(zt) > ENTRY:
                sign, days = (-1 if zt > 0 else 1), 0
                cw_a = sign * gross / (1.0 + bt)
                cw_b = -sign * bt * gross / (1.0 + bt)
        else:
            days += 1
            if abs(zt) < EXIT_Z or days >= TIMEOUT:
                sign, blocked = 0, True
        w_a[t], w_b[t] = (cw_a, cw_b) if sign != 0 else (0.0, 0.0)
    return pd.DataFrame({a: w_a, b: w_b}, index=sub.index)


def apply_borrow(result: dict, weights: pd.DataFrame) -> dict:
    """1%/yr borrow on short notional (lagged weights = in force)."""
    short = weights.clip(upper=0).abs().sum(axis=1).shift(1).fillna(0.0)
    drag = (BORROW_RATE / 252.0) * short.reindex(result["returns"].index).fillna(0.0)
    result = dict(result)
    result["returns"] = result["returns"] - drag
    result["equity"] = (1.0 + result["returns"]).cumprod()
    result["borrow_drag_ann"] = round(float(drag.mean() * 252), 5)
    return result


def main():
    px = load_prices()
    cm = IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS)

    # standalone per-pair runs -> in-sample gate (IS net Sharpe > 0)
    pair_w, per_pair = {}, []
    for a, b in PAIRS:
        w = pair_weights(px, a, b, gross=1.0)
        m = metrics(apply_borrow(run_backtest(w, px, cm, name=f"{a}/{b}"), w))
        pair_w[(a, b)] = w
        per_pair.append({"pair": f"{a}/{b}",
                         "IS_Sharpe": m["in_sample"]["sharpe"],
                         "OOS_Sharpe": m["out_of_sample"]["sharpe"],
                         "Full_Sharpe": m["full"]["sharpe"]})
    per_pair = pd.DataFrame(per_pair)
    passing = [tuple(p.split("/")) for p, s in
               zip(per_pair["pair"], per_pair["IS_Sharpe"]) if s > 0]

    scale = GROSS_CAP / len(passing)
    cols = sorted({t for p in passing for t in p})
    combo = pd.DataFrame(0.0, index=px.index, columns=cols)
    for p in passing:
        combo = combo.add(pair_w[p].reindex(px.index).fillna(0.0) * scale,
                          fill_value=0.0)

    res = apply_borrow(run_backtest(combo, px, cm, name="pairs_statarb_H90_Z60"),
                       combo)
    m = metrics(res)
    m["params"] = {"H": H, "Z": Z, "entry": ENTRY, "exit": EXIT_Z,
                   "timeout_days": TIMEOUT, "gross_cap": GROSS_CAP,
                   "borrow_rate": BORROW_RATE, "slippage_bps": SLIPPAGE_BPS,
                   "pairs_passing_in_sample": ["/".join(p) for p in passing]}
    m["borrow_drag_ann"] = res["borrow_drag_ann"]

    print("Per-pair standalone (unit gross, net of all costs):", file=sys.stderr)
    print(per_pair.to_string(index=False), file=sys.stderr)
    print(json.dumps(m, indent=2))

    output = ROOT / "results" / "pairs_statarb.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
