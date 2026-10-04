"""Rank hysteresis and execution-mode sweep for the two momentum rotations.

This is the study behind the sell buffer (B=9) and the month-end-only
ordering in src/strategies/xsec_etf_mom.py. It covers xsec_etf_mom and
xsec_stock_mom and separates two cost levers that are easy to confuse:

1. RANK HYSTERESIS (the hypothesis): only sell a holding when it drops
   below rank B > K; keep incumbents ranked <= B, fill the remaining slots
   up to K with the best-ranked outsiders. B = K reproduces the plain top-K
   rule exactly (asserted at startup). Incumbency resets whenever the book
   is liquidated anyway (ETF breaker months, too-few-eligible months):
   those sales are real, and no hysteresis credit survives them.

2. EXECUTION MODE (found while decomposing costs): holding a monthly target
   weight constant on daily rows makes the engine charge a
   rebalance-back-to-target on about four trading days in five. The median
   such day moves about 0.5% of the book across three holdings, orders
   small enough to pay the broker's $1 minimum or the 1%-of-notional cap.
   That accounting choice, not name rotation, is about half of the measured
   cost drag. Daily re-targeting also EARNS a small rebalancing premium
   before costs, so the honest measure of the alternatives is the change in
   NET CAGR, not the change in drag.
   Modes compared, identical decision rules:
     E0 retarget : daily rebalance to the monthly target
     E1 drift    : trade once at month-end, weights drift within the month
     E2 lazy     : trade only when the holding NAMES change; months whose
                   holdings are unchanged place no orders at all (weights
                   keep drifting; on any swap, rebalance fully to target)
   E1/E2 daily weights replicate the engine's own drift formula
   (w_lag*(1+r)/(1+gross), gross including the T-bill cash credit), so
   turnover within the month is exactly zero by construction (asserted).

Sweep (per sleeve: 6/5 buffers x 3 modes, 33 rows, all logged; the ONLY
selected parameter is the buffer, by IN-SAMPLE Sharpe before 2018, because
the execution mode is cost accounting, not signal):
  xsec_etf_mom   (K=3, blend 3/6/12, breaker on):  B in {3, 4, 5, 6, 9, 12}
  xsec_stock_mom (K=5, lam=0.25):                  B in {5, 7, 10, 15, 20}
  Every variant is also re-run at 2x slippage (stress columns in the CSV).

What a run on the current engine shows (cache ending 2026-07-01):
  - ETF rotation, plain top-3 with daily re-targeting: in-sample Sharpe
    0.67, cost drag 0.91% a year. The same rule ordered at month-ends only
    costs 0.41% a year. With B=9 and month-end orders the in-sample Sharpe
    is 0.74 (out of sample 0.49 against 0.39), the drag 0.26% a year, net
    CAGR 1.6 points a year higher, and the average holding period 4.7
    months against 2.5. B=9 is adopted as a COST decision, not as a
    Sharpe claim: the neighbouring buffer B=12 scores 0.73 in-sample, and
    the in-sample Sharpe is not monotone in B (B=5 scores 0.66).
  - Stock rotation: in-sample selection keeps B=5, so hysteresis is NOT
    adopted there. The out-of-sample Sharpe rises with B, but "hold the
    losers longer" is mechanically flattered in a universe made of today's
    winners, so that pattern is not evidence.
  - E2 lazy adds almost nothing over E1 (0.24% against 0.26% drag for the
    ETF rotation at B=9) and can leave the book unrebalanced for more than
    a year (17 month-ends in a row without an order), so E1 is the mode
    used.
  - The 2x slippage stress changes no selection.

Input: the price cache built by python src/download_data.py.
Run (from the repository root): python research/rank_hysteresis_sweep.py [--rebase]
Output: one line per variant, the chosen buffer per sleeve and mode,
results/rank_hysteresis_variants.csv (the 33 rows) and
results/rank_hysteresis.json (baseline, chosen rows and their differences).
Both are written through qcore.records: an existing record that differs is
kept and this run's output goes to results/recomputed/; pass --rebase to
replace the record. scripts/trial_registry.py counts the ETF rows as part
of the search behind the deployed ETF rule.

Exit status: 0 on completion; 1, with the loader's one line, when the price
cache is missing; 2 for an unknown or shortened flag (nothing is run). A
failed reproduction check stops the run with an AssertionError before
anything is written.

Limits: the buffer is chosen on in-sample Sharpe from a handful of
correlated rows; the stock universe is survivorship-biased; execution is
assumed at the month-end close; and the figures above depend on the data
vintage, so a fresh download will not reproduce them to the last digit.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "strategies"))

from qcore.backtest import cash_daily_return, metrics, run_backtest  # noqa: E402
from qcore.costs import IBKRHKCostModel  # noqa: E402
from qcore.data import STOCK_UNIVERSE, load_prices  # noqa: E402
from qcore.records import save_csv, save_json  # noqa: E402

import xsec_etf_mom as etf  # noqa: E402  (reuse universe/constants/baseline)
import xsec_stock_mom as stk  # noqa: E402

ETF_BUFFERS = (3, 4, 5, 6, 9, 12)
STK_BUFFERS = (5, 7, 10, 15, 20)
MODES = ("retarget", "drift", "lazy")
# signal params only — production BEST_PARAMS also carries the adopted
# buffer/drift keys, which this study sweeps explicitly
ETF_LEGACY = {p: etf.BEST_PARAMS[p] for p in ("k", "skip", "breaker")}


def pick_holdings(s: pd.Series, held: list, k: int, buffer: int) -> list:
    """Hysteresis selection: keep incumbents ranked <= buffer, fill to k from
    the top. s: scores of ELIGIBLE names only (higher = better)."""
    rank = s.rank(ascending=False, method="first")
    keep = [c for c in held if c in rank.index and rank[c] <= buffer]
    order = s.sort_values(ascending=False, kind="stable").index
    fill = [c for c in order if c not in keep][: k - len(keep)]
    return keep + fill


def etf_monthly(px: pd.DataFrame, k: int, skip: bool, breaker: bool,
                buffer: int) -> pd.DataFrame:
    """xsec_etf_mom decision rows (month-end dates) with a sell buffer."""
    cols = etf.EQ_UNIVERSE + [etf.DEFENSIVE]
    m = etf.month_end_closes(px[cols])

    parts = []
    for lb in etf.LOOKBACKS:
        if skip:
            parts.append(m[etf.EQ_UNIVERSE].shift(1) / m[etf.EQ_UNIVERSE].shift(1 + lb) - 1.0)
        else:
            parts.append(m[etf.EQ_UNIVERSE] / m[etf.EQ_UNIVERSE].shift(lb) - 1.0)
    score = (parts[0] + parts[1] + parts[2]) / 3.0

    spy = m["SPY"]
    risk_off = spy < spy.rolling(etf.SMA_MONTHS).mean()

    w = pd.DataFrame(0.0, index=m.index, columns=cols)
    held: list = []
    for i, t in enumerate(m.index):
        if i < etf.MIN_HISTORY_MONTHS:
            continue
        if breaker and bool(risk_off.loc[t]):
            held = []  # book liquidated into IEF: incumbency gone
            if not np.isnan(m.loc[t, etf.DEFENSIVE]):
                w.loc[t, etf.DEFENSIVE] = 1.0
            continue
        s = score.loc[t].dropna()
        if len(s) < k:
            held = []
            continue
        held = pick_holdings(s, held, k, buffer)
        w.loc[t, held] = 1.0 / k
    return w


def stock_monthly(px: pd.DataFrame, K: int, lam: float, buffer: int) -> pd.DataFrame:
    """xsec_stock_mom decision rows (month-end dates) with a sell buffer."""
    from qcore.calendar import confirmed_month_ends  # match production live-edge guard
    me_dates = confirmed_month_ends(px.index)
    pm = px.loc[me_dates]

    mom = pm.shift(1) / pm.shift(12) - 1.0
    ret1m = pm / pm.shift(1) - 1.0
    vol = px.pct_change(fill_method=None).rolling(stk.VOL_WIN).std().loc[me_dates]

    elig = mom.notna() & ret1m.notna() & vol.notna() & (vol > 0)

    def zscore(df):
        mu = df.mean(axis=1)
        sd = df.std(axis=1)
        return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

    score = zscore(mom.where(elig)) - lam * zscore(ret1m.where(elig))

    w = pd.DataFrame(0.0, index=pm.index, columns=pm.columns)
    held: list = []
    for dt in pm.index:
        s = score.loc[dt].dropna()
        if len(s) < stk.MIN_NAMES:
            held = []  # book liquidated to cash: incumbency gone
            continue
        held = pick_holdings(s, held, K, buffer)
        iv = 1.0 / vol.loc[dt, held]
        w.loc[dt, held] = (iv / iv.sum()).values
    return w


def expand_retarget(monthly: pd.DataFrame, px_index: pd.Index) -> pd.DataFrame:
    """E0: hold the monthly target constant -> engine rebalances daily."""
    return monthly.reindex(px_index).ffill().fillna(0.0)


def expand_drift(monthly: pd.DataFrame, px: pd.DataFrame,
                 lazy: bool = False) -> pd.DataFrame:
    """E1/E2: between decisions, daily rows follow the engine's own drift
    formula so |w - w_drift| = 0 on non-decision days (no intramonth trades).
    lazy=True additionally skips the month-end trade when the target's name
    set equals the drifted name set (no reweighting orders)."""
    cols = list(monthly.columns)
    rets = px[cols].pct_change(fill_method=None).fillna(0.0).to_numpy()
    rf = cash_daily_return(px.index).to_numpy()
    targets = {t: monthly.loc[t].to_numpy(dtype=float) for t in monthly.index}

    out = np.zeros((len(px.index), len(cols)))
    w = np.zeros(len(cols))
    for i, t in enumerate(px.index):
        if i > 0:
            r = np.nan_to_num(rets[i])
            cash_w = max(0.0, 1.0 - w.clip(min=0.0).sum())
            gross = float(w @ r) + cash_w * rf[i]
            if 1.0 + gross != 0.0:
                w = w * (1.0 + r) / (1.0 + gross)
        tgt = targets.get(t)
        if tgt is not None:
            same = (set(np.nonzero(tgt > 1e-12)[0]) ==
                    set(np.nonzero(w > 1e-12)[0]))
            if not (lazy and same):
                w = tgt.copy()
        out[i] = w
    return pd.DataFrame(out, index=px.index, columns=cols)


def check_no_intramonth_trades(res: dict, monthly_index: pd.Index) -> None:
    off = res["turnover"].drop(monthly_index, errors="ignore")
    assert off.abs().max() < 1e-8, "drift mode produced intramonth trades"


def swap_stats(monthly: pd.DataFrame, mom_cols: list) -> dict:
    """Name-level entry rate and average holding period (months) from the
    monthly decision rows of the momentum columns (defensive excluded).
    Holdings sets are identical across execution modes by construction."""
    hold = monthly[mom_cols].gt(1e-12)
    live = hold.any(axis=1)
    if not live.any():
        return {"entries_per_year": np.nan, "avg_hold_months": np.nan}
    hold = hold.loc[live.idxmax():]
    entries = int((hold & ~hold.shift(1, fill_value=False)).sum().sum())
    position_months = int(hold.sum().sum())
    years = len(hold) / 12.0
    return {
        "entries_per_year": round(entries / years, 2),
        "avg_hold_months": round(position_months / max(entries, 1), 2),
    }


def reproduce_baselines(px: pd.DataFrame) -> None:
    """The study must tie out against production in both directions:
    B=K/E0 == the legacy top-K rule, and (since the buffer was adopted)
    the study's B=9 drift expansion == the ADOPTED production weights."""
    base = etf.build_weights(px, **ETF_LEGACY)  # buffer=None -> top-K, ffill
    mine = expand_retarget(etf_monthly(px, buffer=ETF_LEGACY["k"],
                                       **ETF_LEGACY), px.index)
    pd.testing.assert_frame_equal(base, mine)

    prod = etf.build_weights(px, **etf.BEST_PARAMS)  # adopted: B=9 + drift
    mine = expand_drift(etf_monthly(px, buffer=etf.BEST_PARAMS["buffer"],
                                    **ETF_LEGACY), px)
    pd.testing.assert_frame_equal(prod, mine)

    spx = px[STOCK_UNIVERSE]
    base = stk.build_weights(spx)
    mine = expand_retarget(stock_monthly(spx, K=stk.K, lam=stk.LAM, buffer=stk.K),
                           spx.index)
    pd.testing.assert_frame_equal(base, mine)
    print("reproduction check: B=K/E0 == legacy rule; etf B=9 drift == adopted "
          "production weights; stock B=K/E0 == production")


def run_one(sleeve: str, monthly: pd.DataFrame, buffer: int, mode: str,
            px: pd.DataFrame, slip_mult: float = 1.0) -> dict:
    slip = (etf.SLIPPAGE_BPS if sleeve == "etf" else stk.SLIPPAGE_BPS) * slip_mult
    if mode == "retarget":
        w = expand_retarget(monthly, px.index)
    else:
        w = expand_drift(monthly, px, lazy=(mode == "lazy"))
    res = run_backtest(w, px[w.columns], IBKRHKCostModel(slippage_bps=slip),
                       name=f"{sleeve}_B{buffer}_{mode}")
    if mode != "retarget":
        check_no_intramonth_trades(res, monthly.index)
    return metrics(res)


def main() -> None:
    px = load_prices()
    reproduce_baselines(px)

    rows = []
    for sleeve, buffers, k in (("etf", ETF_BUFFERS, etf.BEST_PARAMS["k"]),
                               ("stock", STK_BUFFERS, stk.K)):
        for b in buffers:
            if sleeve == "etf":
                monthly = etf_monthly(px, buffer=b, **ETF_LEGACY)
                mom_cols, frame = etf.EQ_UNIVERSE, px
            else:
                frame = px[STOCK_UNIVERSE]
                monthly = stock_monthly(frame, K=stk.K, lam=stk.LAM, buffer=b)
                mom_cols = STOCK_UNIVERSE
            swaps = swap_stats(monthly, mom_cols)
            for mode in MODES:
                mm = run_one(sleeve, monthly, b, mode, frame)
                st = run_one(sleeve, monthly, b, mode, frame, slip_mult=2.0)
                rows.append({
                    "sleeve": sleeve, "k": k, "buffer": b, "mode": mode,
                    "name": mm["name"], "start": mm["start"], "end": mm["end"],
                    "is_sharpe": mm["in_sample"]["sharpe"],
                    "oos_sharpe": mm["out_of_sample"]["sharpe"],
                    "full_sharpe": mm["full"]["sharpe"],
                    "full_cagr": mm["full"]["cagr"], "full_vol": mm["full"]["vol"],
                    "full_maxdd": mm["full"]["maxdd"],
                    "gross_full_sharpe": mm["gross_full"]["sharpe"],
                    "gross_full_cagr": mm["gross_full"]["cagr"],
                    "ann_turnover": mm["ann_turnover_oneside"],
                    "ann_cost_drag": mm["ann_cost_drag"],
                    "entries_per_year": swaps["entries_per_year"],
                    "avg_hold_months": swaps["avg_hold_months"],
                    "is_sharpe_2x": st["in_sample"]["sharpe"],
                    "oos_sharpe_2x": st["out_of_sample"]["sharpe"],
                    "full_sharpe_2x": st["full"]["sharpe"],
                    "ann_cost_drag_2x": st["ann_cost_drag"],
                })
                r = rows[-1]
                print(f"{r['name']:22s} IS {r['is_sharpe']:5.2f}  OOS {r['oos_sharpe']:5.2f}  "
                      f"full {r['full_sharpe']:5.2f}  gross {r['gross_full_sharpe']:5.2f}  "
                      f"to {r['ann_turnover']:5.1f}  drag {r['ann_cost_drag']*100:4.2f}%  "
                      f"hold {r['avg_hold_months']:4.1f}m", flush=True)

    df = pd.DataFrame(rows)
    # preserved record: an existing, different file is kept and this run's
    # output goes to results/recomputed/ unless --rebase is passed
    save_csv(ROOT / "results" / "rank_hysteresis_variants.csv", df, index=False)

    # selection: buffer by IS Sharpe within each mode (ties -> larger buffer:
    # rounded Sharpes tie often and lower turnover is the point of the study)
    summary = {}
    for sleeve in ("etf", "stock"):
        d = df[df.sleeve == sleeve]
        e0_base = d[(d.buffer == d.k) & (d["mode"] == "retarget")].iloc[0]
        block = {"baseline_E0": e0_base.drop(["sleeve"]).to_dict()}
        for mode in MODES:
            dm = d[d["mode"] == mode]
            chosen = dm.sort_values(["is_sharpe", "buffer"], ascending=False).iloc[0]
            block[f"chosen_{mode}"] = chosen.drop(["sleeve"]).to_dict()
            block[f"delta_{mode}_vs_E0base"] = {
                "ann_cost_drag": round(float(chosen.ann_cost_drag - e0_base.ann_cost_drag), 4),
                "ann_turnover": round(float(chosen.ann_turnover - e0_base.ann_turnover), 1),
                "gross_full_cagr": round(float(chosen.gross_full_cagr - e0_base.gross_full_cagr), 4),
                "full_cagr": round(float(chosen.full_cagr - e0_base.full_cagr), 4),
                "full_sharpe": round(float(chosen.full_sharpe - e0_base.full_sharpe), 2),
                "oos_sharpe": round(float(chosen.oos_sharpe - e0_base.oos_sharpe), 2),
            }
            print(f"{sleeve}/{mode}: B={int(chosen.buffer)} "
                  f"IS {chosen.is_sharpe:.2f} OOS {chosen.oos_sharpe:.2f} "
                  f"drag {chosen.ann_cost_drag*100:.2f}% "
                  f"(E0 base: IS {e0_base.is_sharpe:.2f} OOS {e0_base.oos_sharpe:.2f} "
                  f"drag {e0_base.ann_cost_drag*100:.2f}%)")
        summary[sleeve] = block

    save_json(ROOT / "results" / "rank_hysteresis.json", summary, indent=2, default=str)


def _parse_args(argv=None):
    # allow_abbrev=False: a shortened flag such as --reb is refused, not accepted
    # by the parser and then missed by the exact-name check that acts on it
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--rebase", action="store_true",
                        help="replace the saved variants log and summary when this run differs from them")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _parse_args()
    try:
        main()
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
