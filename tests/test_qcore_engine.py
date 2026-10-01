"""QCore engine tests on synthetic prices (no data cache needed).

Synthetic tickers and a stubbed raw-close loader keep every test independent
of the downloaded cache; cash_rate=0.0 and withholding=0.0 switch off the
^IRX credit and inferred-dividend withholding, which need that cache.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from qcore import backtest as bt
from qcore.calendar import confirmed_month_ends
from qcore.costs import IBKRHKCostModel
from qcore.quality import nyse_bdays


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setattr(bt, "_raw_close", lambda: pd.DataFrame())


def _prices(n=30, seed=0):
    dates = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {t: 50.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for t in ("ZZA", "ZZB", "ZZC")},
        index=dates,
    )


def _run(w, px, **kw):
    return bt.run_backtest(w, px, cash_rate=0.0, withholding=0.0, **kw)


def test_weights_decided_at_close_t_earn_the_next_return():
    px = _prices()
    k = 10
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w.iloc[k, 0] = 1.0  # decided at close k, flat again at close k+1
    res = _run(w, px, cost_model=IBKRHKCostModel(slippage_bps=0.0))
    r = px["ZZA"].pct_change()
    assert res["gross_returns"].loc[px.index[k + 1]] == pytest.approx(r.iloc[k + 1])
    assert res["gross_returns"].drop(px.index[k + 1]).abs().max() == 0.0
    # costs are charged on the entry close k and the exit close k+1 only
    charged = res["costs"][res["costs"] > 0].index
    assert list(charged) == [px.index[k], px.index[k + 1]]


def test_per_trade_cost_minimum_rate_and_cap():
    cm = IBKRHKCostModel(slippage_bps=0.0, sec_fee_rate=0.0, finra_taf_per_share=0.0)
    # $1 minimum: 10 shares at $50 would owe $0.05
    assert cm.cost_bps_per_side(price=50.0, trade_notional=500.0) == pytest.approx(1.0 / 500.0 * 1e4)
    # per-share rate: 2,000 shares at $50
    assert cm.cost_bps_per_side(price=50.0, trade_notional=100_000.0) == pytest.approx(0.005 * 2000 / 100_000 * 1e4)
    # the 1% cap binds below $0.50, where $0.005/share exceeds 1% of price
    assert cm.cost_bps_per_side(price=0.10, trade_notional=10_000.0) == pytest.approx(100.0)
    assert cm.cost_bps_per_side(price=50.0, trade_notional=0.0) == 0.0


def test_engine_charges_commission_minimum_and_cap():
    # run_backtest prices commissions inline (not via cost_bps_per_side), so
    # pin its $1 minimum and 1%-of-notional cap directly; capital $100k
    cm = IBKRHKCostModel(slippage_bps=0.0, sec_fee_rate=0.0, finra_taf_per_share=0.0)
    k = 10

    def entry_cost(scale, weight):
        px = _prices() * scale
        w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        w.iloc[k, 0] = weight
        return _run(w, px, cost_model=cm)["costs"].loc[px.index[k]]

    # $500 at ~$50: 10 shares owe $0.05, the $1 minimum applies (20 bps)
    assert entry_cost(1.0, 0.005) == pytest.approx(0.005 * 1.0 / 500.0)
    # $100k at ~$0.10: $0.005/share would be ~5% of notional, capped at 1%
    assert entry_cost(0.002, 1.0) == pytest.approx(0.01)


def test_drift_weights_trade_only_on_decision_dates():
    px = _prices()
    decisions = px.index[[5, 15, 25]]
    targets = pd.DataFrame(
        [[0.5, 0.3, 0.2], [0.2, 0.2, 0.6], [0.4, 0.4, 0.0]], index=decisions, columns=px.columns
    )
    w = bt.drift_weights(targets, px, cash_rate=0.0)
    res = _run(w, px)
    off_decision = res["turnover"].drop(decisions, errors="ignore")
    assert off_decision.abs().max() < 1e-12
    assert (res["turnover"].loc[decisions] > 0).all()


def test_drift_weights_treat_nan_targets_as_zero():
    px = _prices()
    decisions = px.index[[5, 15]]
    targets = pd.DataFrame([[0.5, np.nan, 0.5], [0.3, 0.3, 0.4]], index=decisions, columns=px.columns)
    w = bt.drift_weights(targets, px, cash_rate=0.0)
    assert w.notna().all().all()
    assert w.loc[decisions[0], "ZZB"] == 0.0
    assert w.loc[decisions[0]:].sum(axis=1).gt(0.9).all()


def test_weights_beyond_the_clip_warn_and_are_capped():
    px = _prices()
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w.iloc[10:, 0] = 2.0
    with pytest.warns(UserWarning, match="clipped"):
        res = _run(w, px)
    r = px["ZZA"].pct_change()
    assert res["gross_returns"].loc[px.index[12]] == pytest.approx(bt.MAX_ABS_WEIGHT * r.iloc[12])


def _nyse_index(end):
    return nyse_bdays("2024-01-02", end)


def test_month_end_before_good_friday_is_kept_with_warning():
    idx = _nyse_index("2024-03-28")  # Good Friday 2024-03-29 closes the month
    with pytest.warns(UserWarning, match="2024-03-29"):
        me = confirmed_month_ends(idx)
    assert me[-1] == pd.Timestamp("2024-03-28")


def test_ordinary_month_end_is_kept_silently_and_partial_month_dropped():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert confirmed_month_ends(_nyse_index("2024-04-30"))[-1] == pd.Timestamp("2024-04-30")
        assert confirmed_month_ends(_nyse_index("2024-04-15"))[-1] == pd.Timestamp("2024-03-28")


@pytest.mark.parametrize("corruption", ["missing_held", "missing_entry", "zero", "infinite", "unsorted", "duplicate"])
def test_invalid_market_inputs_do_not_manufacture_returns(corruption):
    px = _prices()
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w.iloc[5:15, 0] = 1.0
    if corruption == "missing_held":
        px.iloc[7, 0] = np.nan
    elif corruption == "missing_entry":
        px.iloc[5, 0] = np.nan
    elif corruption == "zero":
        px.iloc[7, 0] = 0.0
    elif corruption == "infinite":
        w.iloc[7, 0] = np.inf
    elif corruption == "unsorted":
        px = px.iloc[::-1]
    else:
        px = pd.concat([px, px.iloc[[-1]]])
    with pytest.raises(ValueError):
        _run(w, px)


def test_preinception_missing_prices_are_safe_while_unheld():
    px = _prices()
    px.iloc[:10, 0] = np.nan
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w.iloc[12:, 0] = 1.0
    assert np.isfinite(_run(w, px)["returns"]).all()


def test_prior_cash_rate_is_available_on_first_sliced_date(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=4)
    annual = pd.Series([5.0, 6.0, 70.0], index=dates[:3])
    expected = (1.06 ** (1 / 252)) - 1
    rate = bt._resolve_cash_rate(annual, dates[2:])
    assert rate.iloc[0] == pytest.approx(expected)
    assert rate.iloc[1] == pytest.approx(1.70 ** (1 / 252) - 1)
    monkeypatch.setattr(bt, "_irx_series", lambda: annual)
    assert bt.cash_daily_return(dates[2:]).iloc[0] == pytest.approx(1.059 ** (1 / 252) - 1)


def test_cash_only_metrics_are_defined():
    px = _prices(n=90)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    result = bt.run_backtest(weights, px, cash_rate=5.0, withholding=0.0)
    stats = bt.metrics(result)["full"]
    assert stats == {"cagr": 0.05, "vol": 0.0, "sharpe": 0.0, "maxdd": 0.0}


def test_max_drawdown_includes_initial_capital():
    r = pd.Series(0.0, index=pd.bdate_range("2024-01-02", periods=90))
    r.iloc[0] = -0.10
    assert bt._stats(r, r * 0)["maxdd"] == pytest.approx(-0.10)


def test_final_day_entry_retains_its_cost_without_cash_prelude():
    px = _prices()
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w.iloc[-1, 0] = 1.0
    result = _run(w, px)
    assert result["returns"].index.tolist() == [px.index[-1]]
    assert result["returns"].iloc[0] < 0


@pytest.mark.parametrize("kwargs", [{"capital": 0}, {"slippage_bps": -1}, {"sec_fee_rate": np.nan}])
def test_invalid_cost_model_is_rejected(kwargs):
    with pytest.raises(ValueError):
        IBKRHKCostModel(**kwargs)


def test_exhausted_leveraged_book_fails_instead_of_resurrecting():
    dates = pd.bdate_range("2024-01-02", periods=3)
    px = pd.DataFrame({"X": [100.0, 10.0, 20.0]}, index=dates)
    w = pd.DataFrame({"X": [1.5, 1.5, 1.5]}, index=dates)
    with pytest.raises(ValueError, match="exhausted"):
        _run(w, px)


def test_drift_includes_cash_and_aligns_price_columns():
    from strategies.tsmom_trend import to_daily_drift
    px = _prices()
    targets = pd.DataFrame([[0.2, 0.3]], index=px.index[[0]], columns=["ZZB", "ZZA"])
    daily = to_daily_drift(targets, px, cash_rate=5.0)
    result = bt.run_backtest(daily, px, cash_rate=5.0, withholding=0.0)
    assert result["turnover"].iloc[1:].abs().max() < 1e-12


def test_turn_of_month_is_prefix_invariant_and_final_row_is_actionable():
    from strategies.seasonality_flows import tom_weights
    idx = nyse_bdays("2024-01-02", "2024-04-30")
    spy = pd.Series(100.0, index=idx)
    full = tom_weights(spy)
    for date in ("2024-01-25", "2024-01-31", "2024-03-27", "2024-03-28"):
        truncated = tom_weights(spy.loc[:date])
        pd.testing.assert_frame_equal(truncated, full.loc[:date])
    assert full.loc["2024-03-28", "SPY"] == 1.0  # next session April 1
    assert full.loc["2024-01-25", "SPY"] == 1.0  # Jan 26 is 4th-last session
    sliced = tom_weights(spy.loc["2024-01-15":])
    pd.testing.assert_frame_equal(sliced, full.loc["2024-01-15":])


def test_vix_spike_cannot_use_same_day_post_equity_close_print():
    from strategies.vol_regime import _spike_weights
    idx = nyse_bdays("2024-01-02", "2024-03-01")
    prices = pd.DataFrame({"SPY": 100.0}, index=idx)
    indices = pd.DataFrame({"^VIX": 10.0}, index=idx)
    indices.iloc[25, 0] = 100.0
    weights = _spike_weights(prices, indices, k=1.3)
    assert weights.iloc[25, 0] == 0.0
    assert weights.iloc[26:31, 0].eq(1.0).all()
    assert weights.iloc[31, 0] == 0.0


def test_pair_builder_rejects_internal_gap_instead_of_dropping_session():
    from strategies.pairs_statarb import pair_weights
    px = _prices(n=150)
    px.iloc[120, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        pair_weights(px, "ZZA", "ZZB")


def test_flat_price_rsi_is_neutral():
    from strategies.mean_reversion import rsi
    px = pd.DataFrame({"X": [100.0] * 6})
    assert rsi(px).iloc[2:, 0].eq(50.0).all()


def test_statistics_handle_cash_and_reject_invalid_trials():
    from qcore.stats import block_bootstrap_sharpe_ci, expected_max_sharpe_daily, deflated_sharpe
    cash = pd.Series(np.zeros(90))
    result = block_bootstrap_sharpe_ci(cash, n_boot=20)
    assert result["sharpe_ann"] == 0.0 and result["ci"] == [0.0, 0.0]
    assert expected_max_sharpe_daily(1, 1.0) == 0.0
    assert expected_max_sharpe_daily(20, 0.0) == 0.0
    with pytest.raises(ValueError):
        expected_max_sharpe_daily(0, 1.0)
    with pytest.raises(ValueError):
        expected_max_sharpe_daily(10, -1.0)
    with pytest.raises(ValueError):
        block_bootstrap_sharpe_ci(cash, block=91)
    # np.array input must not hit an ambiguous-truth-value error.
    assert deflated_sharpe(cash, 2, trial_sharpes_ann=np.array([0.1, 0.2]))["T"] == 90


def test_wilder_rsi_uses_mean_of_initial_changes():
    from strategies.mean_reversion import rsi
    px = pd.DataFrame({"X": [100.0, 102.0, 101.0, 105.0, 104.0]})
    result = rsi(px, period=3)["X"]
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(100 * 2.0 / (2.0 + 1.0 / 3.0))
    assert result.iloc[4] == pytest.approx(100 * (4.0 / 3.0) / (4.0 / 3.0 + 5.0 / 9.0))


def test_margin_charges_prior_rate_on_first_return_slice(monkeypatch):
    from strategies import tsmom_voltarget as vt
    dates = pd.bdate_range("2024-01-02", periods=4)
    weights = pd.DataFrame({"X": 1.5}, index=dates)
    raw = pd.DataFrame({"^IRX": 5.0}, index=dates)
    monkeypatch.setattr(vt, "load_indices", lambda: raw)
    r = pd.Series(0.0, index=dates[2:])
    result = vt.charge_margin({"returns": r.copy(), "gross_returns": r.copy()}, weights)
    assert result["margin_drag"].iloc[0] == pytest.approx(0.5 * (1.065 ** (1 / 252) - 1))


def test_missing_historical_month_end_does_not_move_rebalance_earlier():
    idx = nyse_bdays("2024-01-02", "2024-02-15").difference(pd.DatetimeIndex(["2024-01-31"]))
    with pytest.raises(ValueError, match="historical month-end"):
        confirmed_month_ends(idx)


def test_strategy_weights_do_not_change_when_future_prices_are_removed(monkeypatch):
    from strategies import mean_reversion, pairs_statarb, seasonality_flows, tsmom_trend
    from strategies import vol_regime, xsec_etf_mom, xsec_stock_mom, tsmom_voltarget
    monkeypatch.setattr(bt, "_irx_series", lambda: pd.Series(dtype=float))
    idx = nyse_bdays("2020-01-02", "2022-12-30")
    tickers = sorted(set(tsmom_trend.RISK + [tsmom_trend.CASH]
                         + xsec_etf_mom.EQ_UNIVERSE + [xsec_etf_mom.DEFENSIVE]
                         + [f"STOCK{k}" for k in range(25)]))
    rng = np.random.default_rng(46)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, (len(idx), len(tickers))), axis=0)),
                      index=idx, columns=tickers)
    indices = pd.DataFrame({"^VIX": rng.uniform(10, 30, len(idx)), "^VIX3M": 20.0}, index=idx)

    def trend(p):
        sig, eligible, vol, shy = tsmom_trend.build_signals(p)
        monthly = tsmom_trend.month_end_weights(sig["blend"], eligible, vol, shy, "iv", "cash")
        return tsmom_trend.to_daily_drift(monthly, p, cash_rate=0.0)

    def voltarget(p):
        sig, eligible, vol, shy = tsmom_trend.build_signals(p)
        monthly = tsmom_trend.month_end_weights(sig["blend"], eligible, vol, shy, "iv", "shy")
        scaled = tsmom_voltarget.scale_to_target(monthly, p, shy, 0.10, 1.5)
        return tsmom_trend.to_daily_drift(scaled, p, cash_rate=0.0)

    builders = [
        lambda p: mean_reversion.build_weights(p[["SPY", "QQQ"]], **mean_reversion.BEST_PARAMS),
        lambda p: pairs_statarb.pair_weights(p, "SPY", "QQQ"),
        lambda p: seasonality_flows.tom_weights(p["SPY"]),
        trend,
        voltarget,
        lambda p: vol_regime.build_weights(p, indices.loc[p.index]),
        lambda p: xsec_etf_mom.build_weights(p, **xsec_etf_mom.BEST_PARAMS),
        lambda p: xsec_stock_mom.build_weights(p[[f"STOCK{k}" for k in range(25)]]),
    ]
    for build in builders:
        full = build(px)
        for cutoff in ("2021-08-17", "2022-11-30"):
            truncated = build(px.loc[:cutoff])
            pd.testing.assert_frame_equal(truncated, full.loc[:cutoff])


@pytest.mark.parametrize("bad_final", ["2024-03-29", "2024-03-30", "2024-03-31"])
def test_off_calendar_final_row_cannot_become_a_month_end(bad_final):
    idx = _nyse_index("2024-03-28").union(pd.DatetimeIndex([bad_final]))
    ends = confirmed_month_ends(idx)
    assert pd.Timestamp(bad_final) not in ends
    assert ends[-1] == pd.Timestamp("2024-02-29")


def test_an_entire_missing_month_cannot_compress_monthly_lookbacks():
    idx = nyse_bdays("2024-01-02", "2024-04-15")
    idx = idx[idx.month != 2]
    with pytest.raises(ValueError, match="2024-02"):
        confirmed_month_ends(idx)


@pytest.mark.parametrize("corruption", ["unsorted", "missing_date", "missing_value", "infinite"])
def test_bad_raw_cache_cannot_silently_omit_dividend_withholding(monkeypatch, corruption):
    from qcore import data
    adjusted = _prices()
    raw = adjusted.copy()
    if corruption == "unsorted":
        raw = raw.iloc[::-1]
    elif corruption == "missing_date":
        raw = raw.iloc[:-1]
    elif corruption == "missing_value":
        raw.iloc[10, 0] = np.nan
    else:
        raw.iloc[10, 0] = np.inf
    monkeypatch.setattr(data, "load", lambda name: adjusted if name == "adj_close" else raw)
    with pytest.raises(ValueError):
        data.dividend_yields()
