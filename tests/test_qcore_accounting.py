"""Engine accounting pinned to hand-computed values (synthetic prices only).

Every expected number below is worked out by hand in the comment next to it:
trade costs (commission, sell-side regulatory fees, slippage, raw-close share
counts), dividend withholding (rate, timing, exemption list, long-only), the
cash leg, the in-sample / out-of-sample split, metric annualisation, the
drawdown base and the turnover / cost / monthly summaries.
"""

import numpy as np
import pandas as pd
import pytest

from qcore import backtest as bt
from qcore import data as qdata
from qcore.costs import IBKRHKCostModel, QII_EXEMPT_TREASURY, US_DIV_WITHHOLDING

K = 10  # row of the first trade in most tests


@pytest.fixture(autouse=True)
def no_data_cache(monkeypatch):
    """Nothing here may read a downloaded cache: raw closes, inferred
    dividends and the T-bill series are empty unless a test plants its own."""
    monkeypatch.setattr(bt, "_raw_close", lambda: pd.DataFrame())
    monkeypatch.setattr(bt, "_dividend_yields", lambda: pd.DataFrame())
    monkeypatch.setattr(bt, "_irx_series", lambda: pd.Series(dtype=float))


def _flat(n=30, price=50.0, cols=("ZZA", "ZZB", "SHY")):
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({c: price for c in cols}, index=dates)


def _entry(px, weight=1.0, col="ZZA"):
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w.iloc[K:, px.columns.get_loc(col)] = weight
    return w


def _no_fees(**kw):
    return IBKRHKCostModel(slippage_bps=0.0, sec_fee_rate=0.0, finra_taf_per_share=0.0, **kw)


# ---------------------------------------------------------------- trade costs
def test_engine_charges_commission_regulatory_fees_and_slippage_by_hand():
    px = _flat()
    cm = IBKRHKCostModel()  # $100k account, 3 bps slippage, default SEC/TAF rates
    res = bt.run_backtest(_entry(px), px, cm, cash_rate=0.0, withholding=0.0)
    # $100k at $50 = 2,000 shares: commission 2,000 x $0.005 = $10.00;
    # SEC fee $100k x 0.0000278 = $2.78; TAF 2,000 x $0.000166 = $0.332;
    # the two sell-only fees are halved across both sides: $1.556.
    # ($10 + $1.556) / $100k = 1.1556 bps, plus 3 bps slippage = 4.1556 bps.
    assert res["costs"].loc[px.index[K]] == pytest.approx(4.1556e-4, rel=1e-12)
    assert res["costs"].drop(px.index[K]).abs().max() == 0.0
    assert res["turnover"].loc[px.index[K]] == pytest.approx(1.0)
    # the public helper and the engine's inline arithmetic must agree
    assert cm.cost_bps_per_side(price=50.0, trade_notional=100_000.0) == pytest.approx(4.1556, rel=1e-12)


def test_slippage_is_charged_once_per_side_on_top_of_commission():
    px = _flat()
    cost = {}
    for bps in (0.0, 3.0, 6.0):
        cm = IBKRHKCostModel(slippage_bps=bps, sec_fee_rate=0.0, finra_taf_per_share=0.0)
        cost[bps] = bt.run_backtest(_entry(px), px, cm, cash_rate=0.0, withholding=0.0)["costs"].loc[px.index[K]]
        # $10 commission on $100k = 1 bp, plus the slippage itself
        assert cm.cost_bps_per_side(50.0, 100_000.0) == pytest.approx(1.0 + bps, rel=1e-12)
        assert cost[bps] == pytest.approx((1.0 + bps) / 1e4, rel=1e-12)


def test_regulatory_fees_are_halved_and_the_taf_is_capped():
    px = _flat(price=0.50)
    cm = IBKRHKCostModel(slippage_bps=0.0)
    res = bt.run_backtest(_entry(px), px, cm, cash_rate=0.0, withholding=0.0)
    # 200,000 shares at $0.50: commission $1,000 (also the 1% cap);
    # SEC $2.78; TAF 200,000 x $0.000166 = $33.20, capped at $8.30;
    # ($1,000 + 0.5 x ($2.78 + $8.30)) / $100k = 100.554 bps
    assert res["costs"].loc[px.index[K]] == pytest.approx(0.0100554, rel=1e-12)
    assert cm.cost_bps_per_side(0.50, 100_000.0) == pytest.approx(100.554, rel=1e-12)
    # below the cap the TAF is proportional to the share count
    sec_taf_only = IBKRHKCostModel(slippage_bps=0.0, commission_per_share=0.0, min_commission=0.0)
    # 10,000 shares at $10: 0.5 x ($2.78 + $1.66) = $2.22 on $100k = 0.222 bps
    assert sec_taf_only.cost_bps_per_side(10.0, 100_000.0) == pytest.approx(0.222, rel=1e-12)
    px = _flat(price=10.0)
    res = bt.run_backtest(_entry(px), px, sec_taf_only, cash_rate=0.0, withholding=0.0)
    assert res["costs"].loc[px.index[K]] == pytest.approx(0.222e-4, rel=1e-12)


def test_share_counts_use_the_raw_close_with_a_per_column_fallback(monkeypatch):
    px = _flat()
    raw = (px * 2.0)[["ZZA"]]  # ZZA really traded at $100; no raw series for ZZB
    monkeypatch.setattr(bt, "_raw_close", lambda: raw)
    # 1,000 shares at the raw $100 -> $5.00, not 2,000 at the adjusted $50
    res = bt.run_backtest(_entry(px), px, _no_fees(), cash_rate=0.0, withholding=0.0)
    assert res["costs"].loc[px.index[K]] == pytest.approx(5.0 / 100_000.0, rel=1e-12)
    # ZZB has no raw close: its share count falls back to the adjusted $50
    res = bt.run_backtest(_entry(px, col="ZZB"), px, _no_fees(), cash_rate=0.0, withholding=0.0)
    assert res["costs"].loc[px.index[K]] == pytest.approx(10.0 / 100_000.0, rel=1e-12)


def test_costs_scale_with_account_capital():
    px = _flat()
    res = bt.run_backtest(_entry(px), px, _no_fees(capital=10_000.0), cash_rate=0.0, withholding=0.0)
    # $10k at $50 = 200 shares -> $1.00 (the per-share rate and the minimum agree)
    assert res["costs"].loc[px.index[K]] == pytest.approx(1.0 / 10_000.0, rel=1e-12)
    res = bt.run_backtest(_entry(px), px, _no_fees(capital=1_000_000.0), cash_rate=0.0, withholding=0.0)
    # $1m at $50 = 20,000 shares -> $100 = 1 bp
    assert res["costs"].loc[px.index[K]] == pytest.approx(1e-4, rel=1e-12)


def test_turnover_counts_the_buy_leg_and_the_sell_leg():
    px = _flat()
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w.iloc[K:K + 5, 0] = 1.0   # cash -> ZZA: one leg
    w.iloc[K + 5:, 1] = 1.0    # ZZA -> ZZB: a sell leg and a buy leg
    res = bt.run_backtest(w, px, IBKRHKCostModel(), cash_rate=0.0, withholding=0.0)
    assert res["turnover"].loc[px.index[K]] == pytest.approx(1.0)
    assert res["turnover"].loc[px.index[K + 5]] == pytest.approx(2.0)
    assert res["turnover"].drop(px.index[[K, K + 5]]).abs().max() == 0.0
    # each leg pays the 4.1556 bps worked out above
    assert res["costs"].loc[px.index[K + 5]] == pytest.approx(2 * 4.1556e-4, rel=1e-12)
    # 3.0 of traded weight over the 20 live days, annualised: 3 / 20 x 252 = 37.8
    assert len(res["turnover"]) == 20
    assert bt.metrics(res)["ann_turnover_oneside"] == 37.8


def test_net_is_gross_minus_costs_minus_withholding(monkeypatch):
    px = _flat()
    px["ZZA"] = 50.0 * (1.0 + 0.01 * np.sin(np.arange(len(px))))
    dy = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    dy.loc[px.index[K + 3], "ZZA"] = 0.01
    monkeypatch.setattr(bt, "_dividend_yields", lambda: dy)
    res = bt.run_backtest(_entry(px), px, cash_rate=0.0)
    assert res["costs"].sum() > 0 and res["withholding"].sum() == pytest.approx(0.003)
    pd.testing.assert_series_equal(res["returns"], res["gross_returns"] - res["costs"] - res["withholding"])
    pd.testing.assert_series_equal(res["equity"], (1.0 + res["returns"]).cumprod())
    assert (res["equity"] < (1.0 + res["gross_returns"]).cumprod()).all()


def test_weights_beyond_one_and_a_half_are_clipped():
    assert bt.MAX_ABS_WEIGHT == 1.5
    dates = pd.bdate_range("2021-01-04", periods=3)
    px = pd.DataFrame({"X": [100.0, 100.0, 110.0]}, index=dates)
    w = pd.DataFrame({"X": [2.0, 2.0, 2.0]}, index=dates)
    with pytest.warns(UserWarning, match="clipped"):
        res = bt.run_backtest(w, px, cash_rate=0.0, withholding=0.0)
    assert res["gross_returns"].iloc[2] == pytest.approx(0.15)  # 1.5 x 10%, not 2 x 10%


# ---------------------------------------------------------------- withholding
def test_withholding_rate_timing_exemption_and_long_only(monkeypatch):
    assert US_DIV_WITHHOLDING == 0.30
    assert QII_EXEMPT_TREASURY == {"SHY", "IEF", "TLT", "TIP", "SGOV", "BIL", "GOVT"}
    px = _flat()
    ex_date = px.index[K + 3]
    dy = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    dy.loc[ex_date] = 0.02  # every ticker goes ex 2% on the same date
    monkeypatch.setattr(bt, "_dividend_yields", lambda: dy)

    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w.iloc[K:, 0] = 0.5    # ZZA long: taxed
    w.iloc[K:, 1] = -0.2   # ZZB short: pays in lieu, no tax relief
    w.iloc[K:, 2] = 0.3    # SHY: on the exemption list
    res = bt.run_backtest(w, px, cash_rate=0.0)  # withholding=None -> 30%
    # 30% x 0.5 weight x 2% yield = 0.3%; nothing for the short or for SHY
    assert res["withholding"].loc[ex_date] == pytest.approx(0.003, rel=1e-12)
    assert res["withholding"].drop(ex_date).abs().max() == 0.0
    untaxed = bt.run_backtest(w, px, cash_rate=0.0, withholding=0.0)
    assert untaxed["withholding"].abs().max() == 0.0
    assert (untaxed["returns"] - res["returns"]).loc[ex_date] == pytest.approx(0.003, rel=1e-9)
    assert bt.run_backtest(w, px, cash_rate=0.0, withholding=0.15)["withholding"].loc[ex_date] == \
        pytest.approx(0.0015, rel=1e-12)

    # the payout belongs to whoever held INTO the ex-date: weights set at the
    # ex-date close are irrelevant, weights set the close before are charged
    resized = w.copy()
    resized.loc[ex_date:, "ZZA"] = 0.1
    assert bt.run_backtest(resized, px, cash_rate=0.0)["withholding"].loc[ex_date] == \
        pytest.approx(0.003, rel=1e-12)
    late = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    late.loc[ex_date:, "ZZA"] = 0.5      # bought at the ex-date close
    assert bt.run_backtest(late, px, cash_rate=0.0)["withholding"].abs().max() == 0.0
    early = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    early.iloc[K:K + 3, 0] = 0.5         # sold at the ex-date close
    assert bt.run_backtest(early, px, cash_rate=0.0)["withholding"].loc[ex_date] == \
        pytest.approx(0.003, rel=1e-12)


def test_dividend_yield_is_adjusted_return_minus_raw_return(monkeypatch):
    dates = pd.bdate_range("2021-01-04", periods=5)
    # X: flat, goes ex a $1 dividend (100 -> 99), flat, then halves on news
    # with no payout. Y never pays.
    raw = pd.DataFrame({"X": [100.0, 100.0, 99.0, 99.0, 49.5], "Y": [10.0] * 5}, index=dates)
    total_return = pd.Series([0.0, 0.0, 0.0, 0.0, -0.5], index=dates)  # (99 + 1) / 100 - 1 = 0 on the ex-date
    adj = raw.copy()
    adj["X"] = 80.0 * (1.0 + total_return).cumprod()
    monkeypatch.setattr(qdata, "load", lambda name: adj if name == "adj_close" else raw)
    dy = qdata.dividend_yields()
    # adjusted return 0% minus raw return -1% = 1% payout
    assert dy.loc[dates[2], "X"] == pytest.approx(0.01, rel=1e-9)
    assert dy["X"].drop(dates[2]).abs().max() == 0.0  # incl. the -50% day
    assert dy["Y"].abs().max() == 0.0


# ------------------------------------------------------------------ cash leg
def test_idle_long_capital_earns_cash_and_short_proceeds_do_not():
    px = _flat()
    rf = 1.05 ** (1 / 252) - 1
    part = bt.run_backtest(_entry(px, 0.4), px, cash_rate=5.0, withholding=0.0)
    assert part["cash_returns"].iloc[-1] == pytest.approx(0.6 * rf, rel=1e-12)
    assert part["rf_daily"].iloc[-1] == pytest.approx(rf, rel=1e-12)
    short = bt.run_backtest(_entry(px, -0.5), px, cash_rate=5.0, withholding=0.0)
    assert short["cash_returns"].iloc[-1] == pytest.approx(1.0 * rf, rel=1e-12)  # not 1.5 x rf
    w = _entry(px, 0.6)
    w.iloc[K:, 1] = -0.6
    hedged = bt.run_backtest(w, px, cash_rate=5.0, withholding=0.0)
    assert hedged["cash_returns"].iloc[-1] == pytest.approx(0.4 * rf, rel=1e-12)  # not 1.0 x rf
    levered = bt.run_backtest(_entry(px, 1.2), px, cash_rate=5.0, withholding=0.0)
    assert levered["cash_returns"].iloc[-1] == 0.0  # no margin interest is modelled here
    # the credit is part of the gross return (flat prices earn nothing else)
    assert part["gross_returns"].iloc[-1] == pytest.approx(0.6 * rf, rel=1e-12)


def test_cash_credit_is_lagged_net_of_the_haircut_and_floored_at_zero(monkeypatch):
    dates = pd.bdate_range("2021-01-04", periods=4)
    monkeypatch.setattr(bt, "_irx_series", lambda: pd.Series([0.05, 0.05, 0.30, 0.30], index=dates))
    credit = bt.cash_daily_return(dates)
    assert credit.iloc[0] == 0.0                                   # no prior observation
    assert credit.iloc[1] == 0.0 and credit.iloc[2] == 0.0         # 5 bp yield < 10 bp haircut
    assert credit.iloc[3] == pytest.approx(1.002 ** (1 / 252) - 1, rel=1e-12)  # 30 bp - 10 bp


# ------------------------------------------------------------------- metrics
def _result(returns, gross=None, turnover=0.0, costs=0.0, rf=0.0):
    idx = returns.index
    return {"name": "t", "returns": returns,
            "gross_returns": returns if gross is None else gross,
            "turnover": pd.Series(turnover, index=idx), "costs": pd.Series(costs, index=idx),
            "rf_daily": pd.Series(rf, index=idx)}


def _two_point_year():
    """252 days alternating +2% / -1%: mean 0.5%, deviations of +/-1.5%."""
    idx = pd.bdate_range("2021-01-04", periods=252)
    return pd.Series(np.where(np.arange(252) % 2 == 0, 0.02, -0.01), index=idx)


def test_volatility_sharpe_and_cagr_annualisation_by_hand():
    r = _two_point_year()
    rf = pd.Series(0.001, index=r.index)
    s = bt._stats(r, rf)
    # sample variance = 252 x 0.015^2 / 251, so the daily sd is 0.015 x sqrt(252/251)
    # vol    = 0.015 x sqrt(252/251) x sqrt(252)              = 0.015 x 252 / sqrt(251) = 0.2386
    # Sharpe = (0.005 - 0.001) / (0.015 x sqrt(252/251)) x sqrt(252) = (4/15) x sqrt(251) = 4.22
    # CAGR   = (1.02 x 0.99)^126 - 1 over exactly one 252-day year
    assert s["vol"] == round(0.015 * 252 / 251 ** 0.5, 4) == 0.2386
    assert s["sharpe"] == round(4 / 15 * 251 ** 0.5, 2) == 4.22
    assert s["cagr"] == round(1.0098 ** 126 - 1, 4)
    assert s["maxdd"] == -0.01  # every down day gives back 1% from a fresh peak
    # two years of the same pattern: same annual figures, not doubled or halved
    r2 = pd.Series(np.tile(r.to_numpy(), 2), index=pd.bdate_range("2021-01-04", periods=504))
    s2 = bt._stats(r2, pd.Series(0.001, index=r2.index))
    assert s2["cagr"] == s["cagr"]
    assert s2["vol"] == round(0.015 * (504 / 503) ** 0.5 * 252 ** 0.5, 4)
    # fewer than 60 observations: no statistics at all
    assert all(np.isnan(v) for v in bt._stats(r.iloc[:59], rf).values())
    assert not np.isnan(bt._stats(r.iloc[:60], rf)["sharpe"])


def test_metrics_uses_the_cash_rate_credited_in_the_backtest():
    r = _two_point_year()
    m = bt.metrics(_result(r, rf=0.001))
    assert m["full"]["sharpe"] == 4.22           # excess over the result's own rf_daily
    assert bt.metrics(_result(r))["full"]["sharpe"] == round(1 / 3 * 251 ** 0.5, 2) == 5.28
    assert m["start"] == "2021-01-04" and m["end"] == str(r.index[-1].date())


def test_drawdown_is_measured_from_the_running_peak_and_from_initial_capital():
    idx = pd.bdate_range("2024-01-02", periods=90)
    r = pd.Series(0.0, index=idx)
    r.iloc[0], r.iloc[1] = 0.10, -0.10   # 1.10 -> 0.99: -10% from the peak, only -1% from the start
    assert bt._stats(r, r * 0)["maxdd"] == pytest.approx(-0.10)
    r = pd.Series(0.0, index=idx)
    r.iloc[0] = -0.10                    # an opening loss counts against the initial 1.0
    assert bt._stats(r, r * 0)["maxdd"] == pytest.approx(-0.10)


def test_in_sample_and_out_of_sample_are_disjoint_at_the_split():
    assert bt.OOS_SPLIT == "2018-01-01"
    idx = pd.bdate_range("2017-01-02", "2018-12-31")
    r = pd.Series(np.where(idx < "2018-01-01", 0.001, -0.002), index=idx)
    r += np.where(np.arange(len(idx)) % 2, 1e-4, -1e-4)  # nonzero variance
    zero = r * 0
    m = bt.metrics(_result(r))
    is_r, oos_r = r.loc[:"2017-12-31"], r.loc["2018-01-01":]
    assert len(is_r) == 260 and len(oos_r) == 261 and len(is_r) + len(oos_r) == len(r)
    assert m["in_sample"] == bt._stats(is_r, zero)
    assert m["out_of_sample"] == bt._stats(oos_r, zero)
    assert m["full"] == bt._stats(r, zero)
    # hand check on the sign and size: about +0.1%/day before, -0.2%/day after
    assert m["in_sample"]["cagr"] == pytest.approx(1.001 ** 252 - 1, abs=2e-3)
    assert m["out_of_sample"]["cagr"] == pytest.approx(0.998 ** 252 - 1, abs=2e-3)
    assert m["in_sample"]["maxdd"] > -0.001 and m["out_of_sample"]["maxdd"] < -0.35
    # a split that IS a trading day belongs to the out-of-sample segment only
    split = "2018-03-01"
    assert pd.Timestamp(split) in r.index
    m2 = bt.metrics(_result(r), oos_split=split)
    assert m2["in_sample"] == bt._stats(r.loc[:"2018-02-28"], zero)
    assert m2["out_of_sample"] == bt._stats(r.loc[split:], zero)
    assert m2["in_sample"] != bt._stats(r.loc[:split], zero)
    assert m2["in_sample"] != m["in_sample"] and m2["out_of_sample"] != m["out_of_sample"]


def test_turnover_cost_cash_and_withholding_summaries_are_annualised_means():
    r = _two_point_year()
    res = _result(r, gross=r + 0.0005, turnover=0.1, costs=0.0002)
    res["cash_returns"] = pd.Series(0.0001, index=r.index)
    res["withholding"] = pd.Series(0.00005, index=r.index)
    m = bt.metrics(res)
    assert m["ann_turnover_oneside"] == 25.2      # 0.1 x 252
    assert m["ann_cost_drag"] == 0.0504           # 0.0002 x 252
    assert m["ann_cash_income"] == 0.0252         # 0.0001 x 252
    assert m["ann_withholding_drag"] == 0.0126    # 0.00005 x 252
    # gross_full is computed from the gross series: (1.0205 x 0.9905)^126 - 1
    assert m["gross_full"]["cagr"] == round((1.0205 * 0.9905) ** 126 - 1, 4)
    assert m["gross_full"]["cagr"] > m["full"]["cagr"] == round(1.0098 ** 126 - 1, 4)
    table = bt.sweep_table([res])
    assert table.loc["t", "Turnover"] == 25.2 and table.loc["t", "CostDrag"] == 0.0504
    assert table.loc["t", "Sharpe"] == m["full"]["sharpe"] and table.loc["t", "CAGR"] == m["full"]["cagr"]


def test_monthly_summaries_by_hand_including_a_month_with_no_rows():
    idx = pd.bdate_range("2021-01-01", "2021-03-31")
    assert [int((idx.month == k).sum()) for k in (1, 2, 3)] == [21, 20, 23]
    r = pd.Series(np.select([idx.month == 1, idx.month == 2], [0.001, -0.003], 0.0005), index=idx)
    m = bt.metrics(_result(r))
    # months: 1.001^21 - 1 > 0, 0.997^20 - 1 = -5.83%, 1.0005^23 - 1 > 0
    assert m["worst_month"] == round(0.997 ** 20 - 1, 4) == -0.0583
    assert m["pct_positive_months"] == 0.667
    # February absent altogether: it counts as a flat (not positive) month
    gap = bt.metrics(_result(r[idx.month != 2]))
    assert gap["worst_month"] == 0.0
    assert gap["pct_positive_months"] == 0.667
    down = bt.metrics(_result(-r.abs()))
    assert down["pct_positive_months"] == 0.0 and down["worst_month"] == -0.0583


# ------------------------------------------------------- engine as a whole
def test_run_backtest_does_not_change_when_future_rows_are_removed(monkeypatch):
    n = 120
    dates = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(4)
    px = pd.DataFrame({t: 50.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for t in ("ZZA", "ZZB", "TLT")},
                      index=dates)
    dy = pd.DataFrame(0.0, index=dates, columns=px.columns)
    dy.iloc[::17] = 0.005
    monkeypatch.setattr(bt, "_dividend_yields", lambda: dy)
    monkeypatch.setattr(bt, "_raw_close", lambda: px * 1.5)
    w = pd.DataFrame(rng.uniform(-0.4, 0.6, px.shape), index=dates, columns=px.columns)
    rate = pd.Series(rng.uniform(0.5, 5.0, n), index=dates)
    full = bt.run_backtest(w, px, cash_rate=rate)
    assert full["withholding"].sum() > 0 and full["cash_returns"].sum() > 0
    for cut in (37, 80, 119):
        part = bt.run_backtest(w.iloc[:cut], px.iloc[:cut], cash_rate=rate.iloc[:cut])
        for key in ("returns", "gross_returns", "costs", "withholding", "turnover", "cash_returns", "rf_daily"):
            pd.testing.assert_series_equal(part[key], full[key].iloc[:cut], check_exact=True)


# --------------------------------------------------- one guard per failure
def _wiggly(n=30):
    px = _flat(n)
    px["ZZA"] = 50.0 * (1.0 + 0.01 * np.sin(np.arange(n)))
    return px


@pytest.mark.parametrize("case,message", [
    ("exit_on_missing_close", "missing price for a held asset"),
    ("enter_on_missing_final_close", "missing execution price"),
    ("negative_prices_while_held", "must be positive"),
    ("weight_date_not_in_prices", "weight dates missing"),
    ("weight_column_not_in_prices", "weight columns missing"),
    ("withholding_above_one", "withholding must be between 0 and 1"),
    ("withholding_negative", "withholding must be between 0 and 1"),
    ("wiped_out_by_costs", "exhausted after costs"),
    ("wiped_out_before_costs", "leveraged bankruptcy"),
])
def test_each_engine_guard_reports_its_own_failure(case, message):
    px = _wiggly()
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    kwargs = dict(cash_rate=0.0, withholding=0.0)
    cm = IBKRHKCostModel()
    if case == "exit_on_missing_close":
        w.iloc[5:10, 0] = 1.0             # flat again at close 10 ...
        px.iloc[10, 0] = np.nan           # ... which never printed
    elif case == "enter_on_missing_final_close":
        w.iloc[-1, 0] = 1.0
        px.iloc[-1, 0] = np.nan
    elif case == "negative_prices_while_held":
        w.iloc[5:15, 0] = 1.0
        px.iloc[7:9, 0] = [-50.0, -55.0]  # would otherwise book a +10% "return"
    elif case == "weight_date_not_in_prices":
        w.loc[px.index[-1] + pd.Timedelta(7, unit="D")] = 0.5
    elif case == "weight_column_not_in_prices":
        w["NOPE"] = 0.0
    elif case == "withholding_above_one":
        kwargs["withholding"] = 1.5
    elif case == "withholding_negative":
        kwargs["withholding"] = -0.1
    elif case == "wiped_out_by_costs":
        w.iloc[5:, 0] = 1.0
        cm = IBKRHKCostModel(slippage_bps=10_000.0)
    else:
        w.iloc[5:, 0] = 1.5
        px.iloc[8:, 0] = 5.0              # -90% on 1.5x leverage
    with pytest.raises(ValueError, match=message):
        bt.run_backtest(w, px, cm, **kwargs)
