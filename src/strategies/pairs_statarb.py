"""ETF pairs spread reversion. Rejected: negative out of sample after costs.

Hypothesis
    Two ETFs that hold closely related assets (gold and gold miners, energy
    and oil producers, 7-10 and 20+ year Treasuries) are tied together by
    common fundamentals. A gap between them is often caused by flow in one
    leg, not by news, and should close. Selling the rich leg and buying the
    cheap one supplies liquidity to whoever pushed the spread, and is paid
    when the gap closes. The risk is that the gap is information.

Rule
    Pairs considered (7): GLD/GDX, XLE/XOP, EWA/EWC, SPY/MDY, QQQ/XLK,
    KRE/XLF, IEF/TLT, from dividend-adjusted closes.
    Hedge ratio: beta = slope of log(A) on log(B) over the trailing H=90
    days.
    Signal: z-score of the spread log(A) - beta*log(B) over the trailing
    Z=60 days, with the current beta applied to the whole window.
    Entry at a close, if beta > 0: z > +2, short A and long B; z < -2, long
    A and short B. Leg weights are fixed at entry: sign*G/(1+beta) for A and
    -sign*beta*G/(1+beta) for B, G being the gross weight of the pair.
    Exit: |z| < 0.5, or 20 trading days. Re-entry requires the spread to
    return inside the entry band first.
    Portfolio: the pairs whose standalone in-sample net Sharpe is positive
    (XLE/XOP, EWA/EWC and QQQ/XLK, 3 of 7) share a gross budget of 1.5
    equally.
    The engine assumes execution at the decision close. Costs are modelling
    assumptions: the commissions and fees of qcore.costs, 3 bps of slippage
    per side, and 1% a year of borrow cost on short positions. Borrow
    availability, variable lending fees and margin calls are not modelled.

    The fixed quantity is the portfolio weight, not the share count, so the
    engine trades both legs back to target on every holding day. A desk
    holding shares would order only at entry and exit; the reported cost is
    therefore on the high side (see Result).

Variants tried
    9: H {60, 90, 120} x Z {20, 40, 60}. Entry, exit and time limit were
    fixed. scripts/research_pairs_statarb.py runs the grid through
    pair_weights.

Selection
    In-sample Sharpe of the combined pairs (data before 2018-01-01).
    H=90, Z=60 was the in-sample best when the study was run and is the
    variant recorded here; on the current engine H=120, Z=20 ranks first
    (0.51 against 0.48). Out-of-sample results (2018-01-01 onwards) are
    reported, not selected on, and here the choice changes nothing: all nine
    variants are negative out of sample.

Result
    This code on data ending 2026-07-01, net of the modelled costs, Sharpe
    ratios in excess of the cash rate, from 2000-06-30:
    Sharpe 0.18 full sample / 0.48 in sample / -0.54 out of sample;
    CAGR 2.37%, volatility 3.90%, maximum drawdown -9.98%.
    Before costs the full-sample Sharpe is 0.59. Turnover is 18.4x a year
    (buys plus sells); trading costs take 1.26% a year and borrow 0.22%.
    Out-of-sample Sharpe across the nine variants: -0.54 to -1.00.
    Holding shares instead of constant weights (entry and exit rows expanded
    with qcore.backtest.drift_weights) gives 0.26 full sample / 0.54 in
    sample / -0.44 out of sample, with trading costs of 0.92% a year, so the
    verdict does not depend on that convention.

Verdict
    Rejected; not part of the combined portfolio. Costs take the full-sample
    Sharpe from 0.59 to 0.18, and from 2018 no variant earns more than cash.
    The file is kept as the record of a negative result.

Run the module to calculate metrics on the locally obtained cache and save
them to results/pairs_statarb.json. A saved file that differs from the new
run is kept and the new output goes to results/recomputed/ instead; pass
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
H, Z = 90, 60                    # hedge and spread lookbacks, recorded variant
ENTRY, EXIT_Z, TIMEOUT = 2.0, 0.5, 20
GROSS_CAP, BORROW_RATE, SLIPPAGE_BPS = 1.5, 0.01, 3.0


def pair_weights(px: pd.DataFrame, a: str, b: str, H: int = H, Z: int = Z,
                 gross: float = 1.0) -> pd.DataFrame:
    """Daily target weights for one pair, unit gross while in a trade.

    H and Z are the hedge-ratio and z-score windows; they default to the
    recorded variant and are arguments so the nine-variant grid can be run
    through this same function."""
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
    # allow_abbrev=False: a shortened flag such as --reb is refused, not accepted
    # by the parser and then missed by the exact-name checks that act on it
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--rebase", action="store_true",
                        help="replace results/pairs_statarb.json when this run differs from it")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _parse_args()
    try:
        main()
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
