"""ETF pairs spread-reversion example with rolling hedge ratios.

A trailing 90-day log-price regression defines the hedge ratio; a 60-day
consistent-beta spread z-score triggers entries outside +/-2. Entry leg
weights remain fixed until |z| < 0.5 or a 20-day holding limit. Re-entry
requires the spread to return inside the entry band. Included pairs are
screened by standalone pre-2018 net Sharpe and share a 1.5 gross budget.

The fixed quantity is the portfolio weight, not the share count, so the
engine trades both legs back to target on every holding day. A desk holding
shares would order only at entry and exit; the reported cost drag is
therefore on the high side (qcore.backtest.drift_weights is the
share-holding expansion used by the monthly strategies).

Costs include 3 bps per-side slippage and a fixed annual stock-borrow proxy.
Borrow availability, variable lending fees and margin calls are not modeled.
Signals assume same-close execution. This is an exploratory example; run
the module to calculate metrics on the locally obtained cache and save them
to results/pairs_statarb.json. A saved file that differs from the new run
is kept and the new output goes to results/recomputed/ instead; pass
--rebase to replace the saved file.
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import MAX_ABS_WEIGHT, metrics, run_backtest
from qcore.costs import IBKRHKCostModel
from qcore.data import load_prices
from qcore.records import save_json

PAIRS = [
    ("GLD", "GDX"), ("XLE", "XOP"), ("EWA", "EWC"), ("SPY", "MDY"),
    ("QQQ", "XLK"), ("KRE", "XLF"), ("IEF", "TLT"),
]
H, Z = 90, 60                    # hedge and spread lookback lengths
ENTRY, EXIT_Z, TIMEOUT = 2.0, 0.5, 20
GROSS_CAP, BORROW_RATE, SLIPPAGE_BPS = 1.5, 0.01, 3.0


def pair_weights(px: pd.DataFrame, a: str, b: str, H: int = H, Z: int = Z,
                 gross: float = 1.0) -> pd.DataFrame:
    """Daily target weights for one pair, unit gross while in a trade.

    H and Z are the hedge-ratio and z-score windows; they default to the
    module's retained example lengths."""
    for window in (H, Z):
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 2:
            raise ValueError("H and Z must be integer windows of at least 2 sessions")
    if not np.isfinite(gross) or gross < 0:
        raise ValueError("gross must be finite and nonnegative")
    sub = px[[a, b]]
    common = sub.notna().all(axis=1)
    if not common.any():
        return pd.DataFrame(0.0, index=px.index, columns=[a, b])
    sub = sub.loc[common[common].index[0]:]
    if sub.isna().to_numpy().any() or not (np.isfinite(sub) & (sub > 0)).to_numpy().all():
        raise ValueError("pair prices must be complete and positive after both assets start")
    sub = np.log(sub)
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
    return pd.DataFrame({a: w_a, b: w_b}, index=sub.index).reindex(px.index, fill_value=0.0)


def apply_borrow(result: dict, weights: pd.DataFrame, rate: float = BORROW_RATE) -> dict:
    """1%/yr borrow on short notional (lagged weights = in force)."""
    # Align before lagging, matching run_backtest's zero-target convention
    # for omitted price-calendar rows.
    aligned = weights.reindex(result["returns"].index).fillna(0.0).clip(-MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)
    short = aligned.clip(upper=0).abs().sum(axis=1).shift(1).fillna(0.0)
    drag = (rate / 252.0) * short.reindex(result["returns"].index).fillna(0.0)
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

    scale = GROSS_CAP / len(passing) if passing else 0.0
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

    # A saved file that differs is kept (see qcore.records). The notice
    # goes to stderr so stdout stays the metrics JSON alone.
    with contextlib.redirect_stdout(sys.stderr):
        save_json(ROOT / "results" / "pairs_statarb.json", m)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebase", action="store_true",
                        help="replace results/pairs_statarb.json when this run differs from it")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _parse_args()
    main()
