"""Exploratory volatility-targeting and concentration variants.

Reuse tsmom_trend's signals and compare normalization across all eligible
assets versus active assets. A trailing 60-day proposed-portfolio volatility
estimate scales risky exposure toward 10%, subject to decision-time gross
and per-name caps. SHY holds positive residual capital. Leveraged positions
pay a modeled prior ^IRX rate plus 150 bps; actual broker financing, margin
requirements and forced liquidation are not implemented. Gross exposure
can drift above its decision-time cap between rebalances.

The sweep includes unscaled, capped-volatility and fixed 1.64x variants.
Scaled variants in an 8.5%-11.5% pre-2018 volatility band are ranked by
in-sample Sharpe, then drawdown. This is a retained exploratory rule, not
an independently verified preregistration; the fixed-leverage comparison
and historical specification changes followed examination of test results.
There is no untouched holdout claim. Under common binding constraints,
volatility scaling cancels the two normalization schemes' scalar difference.

Run the module to write computed CSV/JSON results; --quiet hides the table.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "strategies"))

from qcore.backtest import TRADING_DAYS, metrics, run_backtest  # noqa: E402
from qcore.costs import IBKRHKCostModel  # noqa: E402
from qcore.data import load_indices, load_prices  # noqa: E402
from tsmom_trend import (  # noqa: E402
    CASH, RISK, SLIPPAGE_BPS, build_signals, month_end_weights, to_daily_drift,
)

TARGET_VOL = 0.10
VOL_LOOKBACK = 60          # days; matches the sleeve's per-asset estimator
MARGIN_SPREAD_BPS = 150.0  # IBKR Pro Tiered USD, first 100k: BM + 1.5%
ENGINE_PER_ASSET_CAP = 1.5  # run_backtest clips weights to +/-1.5
KLEV = round(TARGET_VOL / 0.061, 2)  # 1.64x: target / baseline IS vol

# In-sample volatility selection band; see the exploratory caveat above
VOL_BAND = (0.085, 0.115)


def conc_weights(sigs_one, elig, vol_me, shy_ok):
    """Inverse-vol shares renormalized over signal-ON assets only.

    The book is fully deployed (gross risky = 1.0) whenever at least one
    signal is on; an asset at half-signal (0.5) gets half the raw share of
    an equal-vol asset at full signal before renormalization. All-off
    months park in SHY exactly like the dilute sleeve.
    """
    s = (sigs_one * elig).fillna(0.0)
    inv = (1.0 / vol_me).where(elig)
    raw = s * inv
    w = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    w[CASH] = 0.0
    resid = (1.0 - w[RISK].sum(axis=1)).clip(lower=0.0)
    w[CASH] = resid.where(shy_ok, 0.0)
    return w


def scale_to_target(w_me, px, shy_ok, target, gross_cap):
    """Vol-target the month-end RISKY weights; residual to SHY, deficit
    borrowed. Predicted vol uses only returns up to the decision close."""
    rets = px[RISK].pct_change(fill_method=None)
    out = w_me.copy()
    for dt in out.index:
        wr = out.loc[dt, RISK]
        g = float(wr.sum())
        if g <= 1e-9:
            continue
        window = rets.loc[:dt].tail(VOL_LOOKBACK).fillna(0.0)
        pred = float((window @ wr).std() * np.sqrt(TRADING_DAYS))
        if not np.isfinite(pred) or pred <= 0:
            continue
        k = min(target / pred, gross_cap / g, ENGINE_PER_ASSET_CAP / float(wr.max()))
        out.loc[dt, RISK] = wr * k
    resid = 1.0 - out[RISK].sum(axis=1)
    out[CASH] = resid.clip(lower=0.0).where(shy_ok, 0.0)
    return out


def scale_constant(w_me, shy_ok, k):
    out = w_me.copy()
    out[RISK] = out[RISK] * k
    resid = 1.0 - out[RISK].sum(axis=1)
    out[CASH] = resid.clip(lower=0.0).where(shy_ok, 0.0)
    return out


def charge_margin(result, w_daily):
    """Subtract daily margin interest on the borrowed fraction of NAV.

    Borrowed fraction in force during day t = gross long weight decided at
    close t-1 (the daily-drift row, lagged like the engine lags weights)
    minus 1, floored at 0. Rate in force at t = ^IRX observed at the
    previous close (no lookahead, mirrors the engine's cash credit) plus
    the 150bp spread; ^IRX floored at 0.
    """
    idx = result["returns"].index
    lev = (w_daily.shift(1).clip(lower=0.0).sum(axis=1) - 1.0).clip(lower=0.0)
    lev = lev.reindex(idx).fillna(0.0)
    irx = load_indices()["^IRX"].dropna()
    ann = (irx.reindex(irx.index.union(idx)).ffill().reindex(idx)
           .clip(lower=0.0) + MARGIN_SPREAD_BPS / 100.0)
    daily_rate = ((1.0 + ann / 100.0) ** (1.0 / TRADING_DAYS) - 1.0).shift(1).fillna(0.0)
    drag = (lev * daily_rate).fillna(0.0)
    for key in ("returns", "gross_returns"):
        result[key] = result[key] - drag
    result["equity"] = (1.0 + result["returns"]).cumprod()
    result["margin_drag"] = drag
    result["leverage"] = lev
    return result


def run_variant(px, sigs, elig, vol_me, shy_ok, scheme, scaling,
                slippage_bps=SLIPPAGE_BPS):
    name = f"{scheme}_{scaling}"
    base = (month_end_weights(sigs["blend"], elig, vol_me, shy_ok, "iv", "shy")
            if scheme == "dilute"
            else conc_weights(sigs["blend"], elig, vol_me, shy_ok))
    if scaling == "none":
        w_me = base
    elif scaling.startswith("vt10_g"):
        cap = float(scaling.split("_g")[1]) / 100.0
        w_me = scale_to_target(base, px, shy_ok, TARGET_VOL, cap)
    elif scaling == "klev164":
        w_me = scale_constant(base, shy_ok, KLEV)
    else:
        raise ValueError(scaling)

    w = to_daily_drift(w_me, px)
    res = run_backtest(w, px, IBKRHKCostModel(slippage_bps=slippage_bps), name=name)
    res = charge_margin(res, w)
    m = metrics(res)
    m["avg_gross_risky"] = round(float(
        w[RISK].clip(lower=0.0).sum(axis=1).loc[res["returns"].index].mean()), 3)
    m["ann_margin_drag"] = round(float(res["margin_drag"].mean() * TRADING_DAYS), 4)
    m["pct_days_levered"] = round(float((res["leverage"] > 1e-9).mean()), 3)
    m["max_gross"] = round(float(1.0 + res["leverage"].max()), 2)
    return m, res


VARIANTS = [("dilute", "none"),          # baseline reproduction check
            ("dilute", "vt10_g100"), ("dilute", "vt10_g150"), ("dilute", "vt10_g200"),
            ("dilute", "klev164"),
            ("conc", "none"),
            ("conc", "vt10_g100"), ("conc", "vt10_g150"), ("conc", "vt10_g200")]


def sweep(px, sigs, elig, vol_me, shy_ok):
    rows, runs = [], {}
    for scheme, scaling in VARIANTS:
        m, res = run_variant(px, sigs, elig, vol_me, shy_ok, scheme, scaling)
        runs[m["name"]] = (m, res)
        rows.append({
            "variant": m["name"], "scheme": scheme, "scaling": scaling,
            "full_sharpe": m["full"]["sharpe"], "is_sharpe": m["in_sample"]["sharpe"],
            "oos_sharpe": m["out_of_sample"]["sharpe"],
            "full_cagr": m["full"]["cagr"], "full_vol": m["full"]["vol"],
            "full_maxdd": m["full"]["maxdd"],
            "is_vol": m["in_sample"]["vol"], "is_maxdd": m["in_sample"]["maxdd"],
            "oos_cagr": m["out_of_sample"]["cagr"], "oos_vol": m["out_of_sample"]["vol"],
            "oos_maxdd": m["out_of_sample"]["maxdd"],
            "avg_gross_risky": m["avg_gross_risky"],
            "pct_days_levered": m["pct_days_levered"], "max_gross": m["max_gross"],
            "ann_margin_drag": m["ann_margin_drag"],
            "ann_turnover": m["ann_turnover_oneside"], "ann_cost_drag": m["ann_cost_drag"],
            "pct_pos_months": m["pct_positive_months"], "worst_month": m["worst_month"],
            "start": m["start"], "end": m["end"],
        })
    df = pd.DataFrame(rows)
    # conc_vt10_gX == dilute_vt10_gX by the scaling identity (see docstring)
    df["duplicate_of"] = df.variant.where(
        df.variant.str.startswith("conc_vt10_"), ""
    ).str.replace("conc_", "dilute_", regex=False)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results" / "tsmom_voltarget_variants.csv", index=False)
    return df, runs


def main():
    px = load_prices()[RISK + [CASH]]
    sigs, elig, vol_me, shy_ok = build_signals(px)
    df, runs = sweep(px, sigs, elig, vol_me, shy_ok)
    if "--quiet" not in sys.argv:
        print(df.drop(columns=["start", "end"]).to_string(index=False))

    # Selection: max IS Sharpe inside the IS-vol band
    cand = df[(df.scaling != "none") & df.is_vol.between(*VOL_BAND)]
    if cand.empty:
        sys.exit("no variant landed in the configured IS-vol band -- "
                 "report the sweep, declare no winner")
    cand = cand.sort_values(["is_sharpe", "is_maxdd"], ascending=[False, False])
    winner = cand.iloc[0]["variant"]
    sharpe_max = df.loc[df.is_sharpe.idxmax(), "variant"]

    # 2x slippage stress on the winner only
    scheme, scaling = winner.split("_", 1)
    m2, _ = run_variant(px, sigs, elig, vol_me, shy_ok, scheme, scaling,
                        slippage_bps=2 * SLIPPAGE_BPS)

    # daily-return correlation with the baseline sleeve
    base_r = runs["dilute_none"][1]["returns"]
    win_r = runs[winner][1]["returns"]
    corr = float(base_r.corr(win_r))

    m, _ = runs[winner]
    out = {
        "study": True,
        "question": ("Scale the trend sleeve to ~10% vol -- "
                     "lever via margin vs concentrate the sleeve mix?"),
        "selected_by": ("Exploratory rule: max IS Sharpe among scaled variants "
                        f"with IS vol in {list(VOL_BAND)}"),
        "winner_by_rule": winner,
        "is_sharpe_max_variant": sharpe_max,
        "target_vol": TARGET_VOL, "vol_lookback_days": VOL_LOOKBACK,
        "margin_model": f"^IRX(prev close, floor 0) + {MARGIN_SPREAD_BPS:.0f}bp, daily",
        "klev": KLEV, "slippage_bps": SLIPPAGE_BPS,
        "universe": RISK, "off_sleeve": CASH,
        "winner_metrics": m,
        "stress_2x_slippage_on_winner": {
            k: m2[k] for k in ["full", "in_sample", "out_of_sample", "ann_cost_drag"]},
        "winner_corr_with_baseline_daily": round(corr, 3),
        "baseline_metrics": runs["dilute_none"][0],
        "klev164_metrics": runs["dilute_klev164"][0],
        "conc_metrics": runs["conc_none"][0],
        "limitations": [
            "Exploratory specification; the 2018+ segment is not an untouched holdout.",
            "Financing is a fixed-spread proxy, not a historical broker-rate schedule.",
            "Decision-time gross caps do not enforce daily broker margin constraints.",
            "Selection depends on the data snapshot, costs and rounded in-sample metrics.",
        ],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "tsmom_voltarget.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nselected variant by configured rule: {winner}"
          f"   (unconstrained IS-Sharpe max: {sharpe_max})")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
