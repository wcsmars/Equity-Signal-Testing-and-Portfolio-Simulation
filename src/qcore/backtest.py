"""Vectorized daily backtests with explicit research assumptions.

Weights decided using close t earn the close(t) to close(t+1) return.
This assumes execution at close t. The one-row weight lag does not make
final-close signals executable before a market-on-close order deadline;
actual execution needs separately available signals or a later fill model.

Turnover incurs modeled commissions, regulatory fees and slippage. The
cached dividend-unadjusted close supplies share-count estimates; yfinance
still adjusts that series for splits, so it is not historical tape price.
Missing close data falls back to adjusted prices. Missing returns are filled
with zero, so the data-quality gate is necessary before interpretation.

Idle long cash earns the prior ^IRX observation minus a fixed haircut.
Short proceeds earn no rebate. Margin interest and stock borrow are not
charged here; strategies using leverage or shorts must model them separately.
Dividend withholding uses inferred payouts and fixed Treasury exemptions,
which are approximations rather than fund/year-specific tax accounting.

Metrics include a fixed 2018 split and excess-return Sharpe. The split alone
does not establish untouched evaluation data or validate parameter selection.
"""

import numpy as np
import pandas as pd

from .costs import IBKRHKCostModel, QII_EXEMPT_TREASURY, US_DIV_WITHHOLDING

OOS_SPLIT = "2018-01-01"
TRADING_DAYS = 252
TBILL_HAIRCUT_BPS = 10.0  # fixed annual cash-return haircut assumption

_IRX_CACHE: list[pd.Series] = []
_DIVYIELD_CACHE: list[pd.DataFrame] = []
_RAWCLOSE_CACHE: list[pd.DataFrame] = []


def _raw_close() -> pd.DataFrame:
    """Raw (unadjusted) closes from the data cache, cached."""
    if not _RAWCLOSE_CACHE:
        try:
            from .data import load
            _RAWCLOSE_CACHE.append(load("close"))
        except Exception:  # no close.csv: use adjusted-price share-count estimates
            import warnings
            warnings.warn("close.csv unavailable - per-share commissions use "
                          "adjusted closes, overstating costs early in the "
                          "sample (rerun src/download_data.py)")
            _RAWCLOSE_CACHE.append(pd.DataFrame())
    return _RAWCLOSE_CACHE[0]


def _dividend_yields() -> pd.DataFrame:
    """Per-ticker ex-date dividend yields from the data cache, cached."""
    if not _DIVYIELD_CACHE:
        try:
            from .data import dividend_yields
            _DIVYIELD_CACHE.append(dividend_yields())
        except Exception:  # no close.csv: no withholding charged, but say so
            import warnings
            warnings.warn("close.csv unavailable - dividend withholding NOT "
                          "charged (rerun src/download_data.py)")
            _DIVYIELD_CACHE.append(pd.DataFrame())
    return _DIVYIELD_CACHE[0]


def _irx_series() -> pd.Series:
    """Raw ^IRX (13-week T-bill discount yield, annualized %), cached."""
    if not _IRX_CACHE:
        try:
            from .data import load_indices
            _IRX_CACHE.append(load_indices()["^IRX"].dropna())
        except Exception:  # no data file: legacy 0% cash, but say so
            import warnings
            warnings.warn("indices.csv / ^IRX unavailable - crediting 0% on cash")
            _IRX_CACHE.append(pd.Series(dtype=float))
    return _IRX_CACHE[0]


def _annual_pct_to_daily(ann_pct: pd.Series) -> pd.Series:
    return (1.0 + ann_pct / 100.0) ** (1.0 / TRADING_DAYS) - 1.0


def cash_daily_return(index: pd.Index) -> pd.Series:
    """Daily return credited to idle cash on each date of `index`: the
    ^IRX yield in force AT that date = observed at the previous date's
    close, net of TBILL_HAIRCUT_BPS, compounded to daily. This transforms
    a quoted discount yield as if it were an annual effective rate; it
    is an approximate cash proxy rather than a bond valuation model."""
    irx = _irx_series()
    if irx.empty:
        return pd.Series(0.0, index=index)
    ann = (irx.reindex(irx.index.union(index)).ffill().reindex(index)
           .sub(TBILL_HAIRCUT_BPS / 100.0).clip(lower=0.0))
    return _annual_pct_to_daily(ann).shift(1).fillna(0.0)


def _resolve_cash_rate(cash_rate, index: pd.Index) -> pd.Series:
    """None -> ^IRX (haircut). Scalar -> flat annual %. Series -> annual %
    by date, observed at each close (lagged here, like the weights)."""
    if cash_rate is None:
        return cash_daily_return(index)
    if isinstance(cash_rate, pd.Series):
        ann = cash_rate.reindex(cash_rate.index.union(index)).ffill().reindex(index)
        return _annual_pct_to_daily(ann).shift(1).fillna(0.0)
    return pd.Series(_annual_pct_to_daily(pd.Series(float(cash_rate), index=index)),
                     index=index)


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_model: IBKRHKCostModel | None = None,
    name: str = "strategy",
    cash_rate: float | pd.Series | None = None,
    withholding: float | None = None,
) -> dict:
    """weights: target portfolio weights decided at close of each date
    (rows: dates, cols: tickers; rows need not sum to 1 - the long
    remainder is cash earning cash_rate). prices: adjusted close, superset
    of weight columns/dates. cash_rate: None -> ^IRX minus haircut
    (default); scalar -> flat annual % (0.0 = legacy zero-cash engine);
    Series -> annual % observed at each date's close. withholding: None ->
    US_DIV_WITHHOLDING (0.30) charged on each long position's ex-date
    dividend yield, pure-Treasury ETFs exempt (QII); 0.0 disables.

    Returns dict with 'returns' (net daily), 'equity', 'gross_returns',
    'turnover' (daily one-side), 'costs' (daily, as return drag),
    'cash_returns' (daily credit), 'rf_daily' (cash rate in force),
    'withholding' (daily dividend-tax drag).
    """
    cm = cost_model or IBKRHKCostModel()
    px = prices.reindex(columns=weights.columns)
    w = weights.reindex(px.index).fillna(0.0).clip(-1.5, 1.5)
    rets = px.pct_change(fill_method=None).fillna(0.0)

    # weights in force during day t's return = decided at close t-1
    w_lag = w.shift(1).fillna(0.0)
    rf_daily = _resolve_cash_rate(cash_rate, px.index)
    # idle cash = un-deployed long capital; shorts post proceeds as
    # collateral (no retail rebate), leverage>1 is not charged here
    cash_w = (1.0 - w_lag.clip(lower=0.0).sum(axis=1)).clip(lower=0.0)
    cash_ret = cash_w * rf_daily
    gross = (w_lag * rets).sum(axis=1) + cash_ret

    # 30% US dividend withholding: adj_close returns embed GROSS payouts;
    # an ex-date's dividend goes to whoever held INTO day t, i.e. w_lag.
    # Shorts owe payments-in-lieu at the gross rate - no tax relief either
    # way, so only long weights are charged.
    rate = US_DIV_WITHHOLDING if withholding is None else float(withholding)
    withheld = pd.Series(0.0, index=px.index)
    if rate > 0.0:
        dy = _dividend_yields()
        if not dy.empty:
            cols = [c for c in px.columns if c not in QII_EXEMPT_TREASURY]
            dy = dy.reindex(index=px.index, columns=cols).fillna(0.0)
            withheld = rate * (w_lag[cols].clip(lower=0.0) * dy).sum(axis=1)

    # drifted weights just before rebalancing at close t
    denom = (1.0 + gross).replace(0.0, np.nan)
    w_drift = (w_lag * (1.0 + rets)).div(denom, axis=0).fillna(0.0)
    trade = (w - w_drift).abs()  # per-asset one-side turnover at close t

    # per-asset cost in bps, converted to portfolio return drag; share
    # share-count estimates use the dividend-unadjusted close instead of the
    # dividend-adjusted close, with a per-column fallback to adjusted prices
    cost_bps = pd.DataFrame(0.0, index=trade.index, columns=trade.columns)
    px_filled = px.ffill()
    raw = _raw_close()
    px_trade = px_filled if raw.empty else (
        raw.reindex(index=px.index, columns=px.columns).ffill().fillna(px_filled))
    for col in trade.columns:
        notional = trade[col] * cm.capital
        active = notional > 1e-9
        if not active.any():
            continue
        p = px_trade.loc[active, col]
        n = notional[active]
        shares = n / p
        commission = np.minimum(
            np.maximum(cm.min_commission, cm.commission_per_share * shares),
            cm.max_commission_pct * n,
        )
        regulatory = 0.5 * (cm.sec_fee_rate * n + np.minimum(cm.finra_taf_per_share * shares, 8.30))
        cost_bps.loc[active, col] = (commission + regulatory) / n * 1e4 + cm.slippage_bps

    daily_cost = (trade * cost_bps).sum(axis=1) / 1e4  # return drag, charged at close t
    net = gross - daily_cost - withheld
    # trim leading dead period before first position (no pure-cash prelude),
    # keeping the entry day itself so the first trade's cost is charged
    live = w_lag.abs().sum(axis=1) > 0
    if live.any():
        start = px.index[max(int(np.argmax(live.values)) - 1, 0)]
        net, gross = net.loc[start:], gross.loc[start:]
        daily_cost, trade = daily_cost.loc[start:], trade.loc[start:]
        cash_ret, rf_daily = cash_ret.loc[start:], rf_daily.loc[start:]
        withheld = withheld.loc[start:]

    return {
        "name": name,
        "returns": net,
        "gross_returns": gross,
        "equity": (1.0 + net).cumprod(),
        "turnover": trade.sum(axis=1),
        "costs": daily_cost,
        "cash_returns": cash_ret,
        "rf_daily": rf_daily,
        "withholding": withheld,
    }


def drift_weights(targets: pd.DataFrame, prices: pd.DataFrame,
                  cash_rate: float | pd.Series | None = None) -> pd.DataFrame:
    """Expand decision-date target rows into daily weights that DRIFT with
    returns between decisions, i.e. orders are placed only on decision dates.

    Why this exists: ffilling targets to daily rows makes run_backtest
    charge a rebalance-back-to-target on every multi-asset day (it trades
    |w - w_drift| daily) — mostly $1-minimum micro-orders a live monthly
    trader would not place between scheduled decisions.

    Replicates run_backtest's own drift formula — w_lag*(1+r)/(1+gross),
    gross including the cash credit on idle long capital — so feeding the
    result back into run_backtest yields exactly zero turnover on
    non-decision dates. `targets`: post-trade weights at each decision
    close (rows must be dates present in prices.index). Pass the same
    cash_rate you pass to run_backtest.
    """
    cols = list(targets.columns)
    if not targets.index.isin(prices.index).all():
        raise ValueError("drift_weights: target dates missing from prices index")
    px = prices[cols]
    rets = px.pct_change(fill_method=None).fillna(0.0).to_numpy()
    rf = _resolve_cash_rate(cash_rate, px.index).to_numpy()
    tgt = {t: targets.loc[t].to_numpy(dtype=float) for t in targets.index}

    out = np.zeros((len(px.index), len(cols)))
    w = np.zeros(len(cols))
    for i, t in enumerate(px.index):
        if i > 0:
            r = rets[i]
            cash_w = max(0.0, 1.0 - w.clip(min=0.0).sum())
            gross = float(w @ r) + cash_w * rf[i]
            if 1.0 + gross != 0.0:
                w = w * (1.0 + r) / (1.0 + gross)
        if t in tgt:
            w = tgt[t].copy()
        out[i] = w
    return pd.DataFrame(out, index=px.index, columns=cols)


def _stats(r: pd.Series, rf: pd.Series) -> dict:
    """cagr/vol/maxdd from TOTAL returns r; Sharpe from EXCESS returns r - rf."""
    r = r.dropna()
    if len(r) < 60 or r.std() == 0:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    ex = r - rf.reindex(r.index).fillna(0.0)
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    dd = (eq / eq.cummax() - 1).min()
    sharpe = 0.0 if ex.std() == 0 else float(ex.mean() / ex.std() * np.sqrt(TRADING_DAYS))
    return {
        "cagr": round(float(cagr), 4),
        "vol": round(float(vol), 4),
        "sharpe": round(sharpe, 2),
        "maxdd": round(float(dd), 4),
    }


def metrics(result: dict, oos_split: str = OOS_SPLIT) -> dict:
    """Full + in-sample (< split) + out-of-sample (>= split) statistics.
    Sharpe is excess over the cash rate credited in the backtest (result's
    'rf_daily' when present, else the default ^IRX series)."""
    r = result["returns"].dropna()
    g = result["gross_returns"].dropna()
    rf = result.get("rf_daily")
    rf = rf.reindex(r.index).fillna(0.0) if rf is not None else cash_daily_return(r.index)
    split = pd.Timestamp(oos_split) - pd.Timedelta(days=1)
    monthly = (1 + r).resample("ME").prod() - 1
    out = {
        "name": result["name"],
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "full": _stats(r, rf),
        "in_sample": _stats(r.loc[:split], rf),
        "out_of_sample": _stats(r.loc[oos_split:], rf),
        "gross_full": _stats(g, rf),
        "ann_turnover_oneside": round(float(result["turnover"].mean() * TRADING_DAYS), 1),
        "ann_cost_drag": round(float(result["costs"].mean() * TRADING_DAYS), 4),
        "pct_positive_months": round(float((monthly > 0).mean()), 3),
        "worst_month": round(float(monthly.min()), 4),
    }
    if "cash_returns" in result:
        out["ann_cash_income"] = round(float(result["cash_returns"].mean() * TRADING_DAYS), 4)
    if "withholding" in result:
        out["ann_withholding_drag"] = round(float(result["withholding"].mean() * TRADING_DAYS), 4)
    return out


def sweep_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        m = metrics(res)
        rows.append({
            "name": m["name"],
            "CAGR": m["full"]["cagr"], "Vol": m["full"]["vol"],
            "Sharpe": m["full"]["sharpe"], "MaxDD": m["full"]["maxdd"],
            "IS_Sharpe": m["in_sample"]["sharpe"], "OOS_Sharpe": m["out_of_sample"]["sharpe"],
            "Turnover": m["ann_turnover_oneside"], "CostDrag": m["ann_cost_drag"],
        })
    return pd.DataFrame(rows).set_index("name")
