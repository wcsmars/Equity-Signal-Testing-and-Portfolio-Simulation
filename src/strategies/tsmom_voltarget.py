"""Volatility-target and concentration study on the trend rule. Rejected.

Question
    tsmom_trend runs at about 6% volatility because, on average, only two
    thirds of capital is in risky assets. Can it be brought to about 10%
    without losing risk-adjusted return? Two routes were compared:
      A. Leverage: scale the existing mix above 1.0 gross and borrow.
      B. Concentration: normalise the inverse-volatility shares over the
         assets whose signal is on, not over all eligible assets, so the
         portfolio is fully invested without borrowing.

Hypothesis
    Scaling exposure to hold predicted volatility constant keeps the Sharpe
    ratio of the unscaled rule, less the financing cost, and concentration
    does the same without financing. This is a sizing study on an existing
    signal, so there is no separate argument about who takes the other side.

Rule
    Signals, eligibility, month-end decisions and drift between decisions
    are imported from tsmom_trend. The baseline here keeps SHY for the
    remainder.
    Volatility target: at each month-end the predicted volatility is the
    standard deviation of the proposed risky portfolio's daily returns over
    the trailing 60 days (weights held fixed across the window), annualised,
    from data up to the decision close. The scale factor is
      k = min(10% / predicted, gross cap / gross, 1.5 / largest weight).
    A positive remainder goes to SHY. A negative remainder is borrowed:
    interest is (gross long weight - 1) x (the previous close of the 13-week
    bill yield + 150 bps), charged daily in this module because the engine
    does not model margin. The spread is a modelling assumption. Margin
    requirements and forced liquidation are not implemented, and gross
    exposure can drift above its decision-time cap between rebalances.
    Constant leverage: 1.64x, which is 10% divided by the unscaled rule's
    in-sample volatility of 6.1%.
    Trading costs are modelling assumptions: the commissions and fees of
    qcore.costs plus 3 bps of slippage per side.

Variants tried
    9 runs: mix {existing, concentrated} x scaling {none, 10% target with
    gross cap 1.0, 1.5 or 2.0}, plus constant 1.64x on the existing mix. The
    unscaled existing mix is the baseline. Under a volatility target the two
    mixes are identical: both are proportional to signal/vol and differ only
    by a normalising constant, which the scale factor cancels. The three
    duplicate runs stay in the table, marked in its duplicate_of column,
    which leaves six distinct portfolios.

Selection
    Among scaled variants whose in-sample volatility (data before
    2018-01-01) lies between 8.5% and 11.5%: the highest in-sample Sharpe at
    two decimals, ties broken by the shallower in-sample drawdown. This is a
    retained exploratory rule, not an independently verified
    pre-registration; the constant-leverage comparison and earlier
    specification changes followed examination of test results. There is no
    untouched holdout claim.

Result
    This code on data ending 2026-07-01, net of the modelled costs and
    financing, Sharpe ratios in excess of the cash rate, from 2001-02-28.
    Columns: Sharpe full sample / in sample / out of sample, volatility,
    maximum drawdown.
      unscaled (baseline)       0.65 / 0.82 / 0.34     6.34%   -14.45%
      10% target, cap 1.0       0.53 / 0.70 / 0.26     8.55%   -18.35%
      10% target, cap 1.5       0.53 / 0.72 / 0.23    10.58%   -26.97%
      10% target, cap 2.0       0.54 / 0.76 / 0.22    11.49%   -35.17%
      constant 1.64x            0.62 / 0.76 / 0.37    10.29%   -23.47%
      concentrated, unscaled    0.44 / 0.58 / 0.19    10.52%   -26.29%
    The rule selects constant 1.64x: it ties the cap-2.0 variant at 0.76 in
    sample and has the shallower in-sample drawdown (-12.59% against
    -19.28%). At double slippage it scores 0.61 / 0.75 / 0.36. Its daily
    returns correlate 0.997 with the baseline's.

Verdict
    Rejected: no variant was adopted and the trend rule stays unscaled.
    - No scaled or concentrated variant matches the unscaled rule in sample
      (0.82), and the selected one is a tie decided on drawdown.
    - The volatility target adds leverage at the wrong time. Its scale
      factor is negatively correlated with the rule's own exposure (-0.57
      over the full sample and -0.70 from 2018, at cap 2.0), so it levers
      most when few assets are trending. It reached the 2.0 cap at 27 of 292
      decisions, and gross exposure drifted to 2.38x on 2020-03-19, the
      trough of the -35.17% drawdown of a portfolio aimed at 10% volatility.
    - Concentration discards the information in the exposure level: 0.58 in
      sample against 0.82.
    - Constant leverage is the baseline scaled up. It pays 0.78% a year in
      financing and deepens the drawdown from -14.45% to -23.47%. Its
      out-of-sample 0.37 against 0.34 was seen after the fact and is not
      evidence for it.
    If more trend exposure is wanted, giving the unscaled rule a larger
    share of the combined portfolio provides it without borrowing.

Run the module to write computed CSV/JSON results; --quiet hides the table.
A saved result file that differs from the new run is kept and the new output
goes to results/recomputed/ instead; pass --rebase to replace the saved files.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import TRADING_DAYS, MAX_ABS_WEIGHT, _resolve_cash_rate, metrics, run_backtest  # noqa: E402
from qcore.costs import IBKRHKCostModel  # noqa: E402
from qcore.data import load_indices, load_prices  # noqa: E402
from qcore.records import save_csv, save_json  # noqa: E402
from strategies.tsmom_trend import (  # noqa: E402
    CASH, RISK, SLIPPAGE_BPS, build_signals, month_end_weights, to_daily_drift,
)

TARGET_VOL = 0.10
VOL_LOOKBACK = 60          # days; matches the sleeve's per-asset estimator
MARGIN_SPREAD_BPS = 150.0  # modelling assumption: bill yield + 1.5% to borrow
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
    effective = w_daily.clip(-MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)
    lev = (effective.shift(1).clip(lower=0.0).sum(axis=1) - 1.0).clip(lower=0.0)
    lev = lev.reindex(idx).fillna(0.0)
    irx = load_indices()["^IRX"].dropna()
    ann = irx.clip(lower=0.0) + MARGIN_SPREAD_BPS / 100.0
    daily_rate = _resolve_cash_rate(ann, idx)
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
    save_csv(ROOT / "results" / "tsmom_voltarget_variants.csv", df, index=False)
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
    save_json(ROOT / "results" / "tsmom_voltarget.json", out)
    print(f"\nselected variant by configured rule: {winner}"
          f"   (unconstrained IS-Sharpe max: {sharpe_max})")
    print(json.dumps(m, indent=2))


def _parse_args(argv=None):
    # allow_abbrev=False: a shortened flag such as --reb is refused, not accepted
    # by the parser and then missed by the exact-name checks that act on it
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--quiet", action="store_true", help="do not print the sweep table")
    parser.add_argument("--rebase", action="store_true",
                        help="replace the saved variants CSV and JSON when this run differs from them")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _parse_args()
    try:
        main()
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
