"""Tests for the transaction cost models (alpha_lab.backtest.costs).

Trade panels are built by hand — no engine import. The hand-built markets use
constant (or two-value alternating) prices and volumes so every rolling
statistic has an exact closed form.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.costs import COST_MODELS, FixedBps, RealisticCost, Zero, from_config
from alpha_lab.config.schema import CostConfig
from alpha_lab.core.errors import ConfigError
from alpha_lab.core.interfaces import ZeroCost
from alpha_lab.core.types import MarketData
from alpha_lab.data.synthetic import make_market

NAV = 1_000_000.0


# --------------------------------------------------------------------------
# hand-built panels
# --------------------------------------------------------------------------

def _flat_market(prices: dict, volumes: dict, n_days: int = 70, with_unadjusted: bool = True) -> MarketData:
    """Constant prices/volumes: rolling means are the constants, vol is 0."""
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    close = pd.DataFrame(prices, index=dates, dtype=float)
    volume = pd.DataFrame(volumes, index=dates, dtype=float)
    fields = {"volume": volume}
    if with_unadjusted:
        fields["unadjusted_close"] = close.copy()
    return MarketData.from_frames(close, **fields)


def _alternating_market(lo: float = 100.0, hi: float = 104.0, volume: float = 1_000_000.0,
                        n_days: int = 80) -> MarketData:
    """One asset whose price alternates lo/hi: returns take exactly two values,
    so trailing vol and ADV over even-length windows are exact by hand."""
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    close = pd.DataFrame({"AAA": [lo if i % 2 == 0 else hi for i in range(n_days)]}, index=dates)
    vol = pd.DataFrame({"AAA": volume}, index=dates, dtype=float)
    return MarketData.from_frames(close, volume=vol, unadjusted_close=close.copy())


def _zero_trades(data: MarketData) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=data.dates, columns=data.tickers)


def _alt_vol(lo: float, hi: float, window: int = 10) -> float:
    """Sample std of an even-length window of alternating two-value returns."""
    r_up = hi / lo - 1.0
    r_dn = lo / hi - 1.0
    vals = [r_dn, r_up] * (window // 2)
    m = sum(vals) / window
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (window - 1))


# --------------------------------------------------------------------------
# registry / config bridge
# --------------------------------------------------------------------------

def test_registry_has_all_models():
    for name in ("zero", "fixed_bps", "realistic"):
        assert name in COST_MODELS


def test_from_config_dispatch():
    zero = from_config(CostConfig(model="zero"))
    assert isinstance(zero, Zero) and isinstance(zero, ZeroCost)

    fixed = from_config(CostConfig(model="fixed_bps", fixed_bps=12.0))
    assert isinstance(fixed, FixedBps)
    assert fixed.bps == 12.0

    real = from_config(
        CostConfig(
            model="realistic",
            commission_per_share=0.01,
            half_spread_bps=1.5,
            impact_coeff=0.2,
            adv_window=30,
            vol_window=40,
        )
    )
    assert isinstance(real, RealisticCost)
    assert real.commission_per_share == 0.01
    assert real.half_spread_bps == 1.5
    assert real.impact_coeff == 0.2
    assert real.adv_window == 30
    assert real.vol_window == 40


def test_from_config_unknown_model():
    with pytest.raises(ConfigError):
        from_config(CostConfig(model="bogus"))
    with pytest.raises(ConfigError):
        from_config({"model": "zero"})  # not a CostConfig


def test_bad_params_raise_config_error():
    with pytest.raises(ConfigError):
        FixedBps(bps=-1.0)
    with pytest.raises(ConfigError):
        RealisticCost(commission_per_share=-0.01)
    with pytest.raises(ConfigError):
        RealisticCost(adv_window=3)
    with pytest.raises(ConfigError):
        RealisticCost(vol_window=5)


# --------------------------------------------------------------------------
# closed-form values
# --------------------------------------------------------------------------

def test_zero_and_fixed_bps_exact():
    data = _flat_market({"AAA": 100.0, "BBB": 50.0}, {"AAA": 1e6, "BBB": 2e6})
    trades = _zero_trades(data)
    t = data.dates[65]
    trades.loc[t, "AAA"] = 0.10
    trades.loc[t, "BBB"] = -0.05

    zero = Zero().cost(trades, data, NAV)
    assert (zero == 0.0).all()

    fixed = FixedBps(bps=10.0).cost(trades, data, NAV)
    assert fixed.loc[t] == (abs(0.10) + abs(0.05)) * 10.0 * 1e-4
    assert (fixed.drop(t) == 0.0).all()
    assert not fixed.isna().any()


def test_realistic_closed_form():
    data = _flat_market({"AAA": 100.0, "BBB": 50.0}, {"AAA": 1e6, "BBB": 2e6})
    trades = _zero_trades(data)
    t = data.dates[65]
    trades.loc[t, "AAA"] = 0.10
    trades.loc[t, "BBB"] = -0.05

    cost = RealisticCost().cost(trades, data, NAV)

    spread = (abs(0.10) + abs(0.05)) * 2.5 * 1e-4
    shares = 0.10 * NAV / 100.0 + 0.05 * NAV / 50.0
    commission = shares * 0.005 / NAV
    impact = 0.0  # constant prices: trailing vol is exactly zero
    expected = spread + commission + impact

    assert abs(cost.loc[t] - expected) < 1e-12
    # all-zero trade rows cost exactly 0
    assert (cost.drop(t) == 0.0).all()
    assert not cost.isna().any()
    assert (cost >= 0.0).all()


def test_realistic_impact_closed_form():
    lo, hi, volume = 100.0, 104.0, 1_000_000.0
    data = _alternating_market(lo, hi, volume)
    trades = _zero_trades(data)
    t = data.dates[60]
    trades.loc[t, "AAA"] = 0.5

    model = RealisticCost(
        commission_per_share=0.0, half_spread_bps=0.0, impact_coeff=0.1,
        adv_window=20, vol_window=10,
    )
    cost = model.cost(trades, data, NAV)

    vol = _alt_vol(lo, hi, window=10)          # exact trailing vol through t-1
    adv = volume * (lo + hi) / 2.0             # 20-day window: 10 of each price
    part = 0.5 * NAV / adv
    expected = 0.5 * 0.1 * vol * math.sqrt(part)
    assert cost.loc[t] == pytest.approx(expected, rel=1e-9)
    assert (cost.drop(t) == 0.0).all()


def test_early_window_fallbacks():
    # trade before any rolling stat exists: documented conservative defaults
    data = _alternating_market()
    trades = _zero_trades(data)
    t = data.dates[3]
    trades.loc[t, "AAA"] = 0.3

    model = RealisticCost(commission_per_share=0.0, half_spread_bps=0.0)
    cost = model.cost(trades, data, NAV)
    expected = 0.3 * 0.1 * RealisticCost.VOL_FALLBACK * math.sqrt(RealisticCost.PARTICIPATION_FALLBACK)
    assert cost.loc[t] == pytest.approx(expected, rel=1e-12)
    assert not cost.isna().any()


def test_nan_trade_cells_treated_as_no_trade():
    data = _flat_market({"AAA": 100.0, "BBB": 50.0}, {"AAA": 1e6, "BBB": 2e6})
    trades = _zero_trades(data)
    trades.loc[data.dates[30], "AAA"] = np.nan

    for model in (RealisticCost(), FixedBps(), Zero()):
        cost = model.cost(trades, data, NAV)
        assert not cost.isna().any()
        assert (cost == 0.0).all()


def test_nan_price_traded_cell_contributes_zero_commission():
    # delisting artifact: |trade| > 0 where the price has stopped printing
    data = make_market(n_assets=6, n_days=400, seed=5, split_asset=False, universe_churn=True)
    delisted = data.tickers[-2]
    t = data.dates[360]  # exit at int(400 * 0.8) = 320
    assert np.isnan(data.unadjusted_close.loc[t, delisted])

    trades = _zero_trades(data)
    trades.loc[t, delisted] = 0.05
    model = RealisticCost(half_spread_bps=0.0, impact_coeff=0.0)  # isolate commission
    cost = model.cost(trades, data, NAV)
    assert cost.loc[t] == 0.0
    assert not cost.isna().any()


# --------------------------------------------------------------------------
# SPLIT REGRESSION — the critical one
# --------------------------------------------------------------------------

def test_split_regression_commission_uses_unadjusted_close():
    data = make_market(n_assets=6, n_days=400, seed=5, split_asset=True, universe_churn=False)
    # 4:1 split on SYM00 at int(400 * 0.6) = 240: raw price is 4x adjusted before.
    # The trade dated t executes at close t-1 (rule 7), so share math uses the
    # RAW close at t-1.
    t_pre = data.dates[120]
    t_exec = data.dates[119]
    raw = data.unadjusted_close.loc[t_exec, "SYM00"]
    adj = data.close.loc[t_exec, "SYM00"]
    assert raw == pytest.approx(4.0 * adj, rel=1e-12)

    trades = _zero_trades(data)
    trades.loc[t_pre, "SYM00"] = 0.04
    nav = 2_000_000.0
    model = RealisticCost(commission_per_share=0.005, half_spread_bps=0.0, impact_coeff=0.0)
    cost = model.cost(trades, data, nav)

    # correct: shares from the raw close (raw is HIGHER pre-split => FEWER shares)
    expected = (0.04 * nav / raw) * 0.005 / nav
    # wrong: adjusted close claims 4x the shares, hence 4x the commission
    wrong = (0.04 * nav / adj) * 0.005 / nav

    assert cost.loc[t_pre] == pytest.approx(expected, rel=1e-12)
    assert wrong / cost.loc[t_pre] == pytest.approx(4.0, rel=1e-9)

    # post-split the raw and adjusted closes agree, and so do the formulas
    t_post = data.dates[300]
    trades_post = _zero_trades(data)
    trades_post.loc[t_post, "SYM00"] = 0.04
    cost_post = model.cost(trades_post, data, nav)
    raw_exec_post = data.unadjusted_close.loc[data.dates[299], "SYM00"]
    assert raw_exec_post == pytest.approx(data.close.loc[data.dates[299], "SYM00"], rel=1e-12)
    assert cost_post.loc[t_post] == pytest.approx(
        (0.04 * nav / raw_exec_post) * 0.005 / nav, rel=1e-12
    )


def test_split_ex_date_commission_uses_execution_price():
    """Regression: a trade DATED on the split ex-date executed at the pre-split
    raw close (t-1). Using t's post-split raw close would claim split-factor-x
    the shares actually traded and overcharge commission by the same factor."""
    n = 60
    dates = pd.bdate_range("2021-01-04", periods=n)
    adj = pd.DataFrame({"AAA": 100.0}, index=dates)
    raw = pd.DataFrame({"AAA": [400.0] * 30 + [100.0] * 30}, index=dates)  # 4:1 at pos 30
    vol = pd.DataFrame({"AAA": 1e6}, index=dates)
    data = MarketData.from_frames(adj, unadjusted_close=raw, volume=vol)

    trades = pd.DataFrame(0.0, index=dates, columns=["AAA"])
    ex_date = dates[30]
    trades.loc[ex_date, "AAA"] = 0.10
    model = RealisticCost(commission_per_share=0.005, half_spread_bps=0.0, impact_coeff=0.0)
    cost = model.cost(trades, data, NAV)

    # executed at close t-1: 0.10 * NAV / 400 = 250 shares -> $1.25
    assert cost.loc[ex_date] * NAV == pytest.approx(1.25, rel=1e-12)


def test_first_row_trade_has_zero_commission():
    """No t-1 price exists on the first date: commission contributes 0."""
    data = _flat_market({"AAA": 100.0}, {"AAA": 1e6})
    trades = _zero_trades(data)
    trades.loc[data.dates[0], "AAA"] = 0.10
    model = RealisticCost(half_spread_bps=0.0, impact_coeff=0.0)
    cost = model.cost(trades, data, NAV)
    assert cost.loc[data.dates[0]] == 0.0
    assert not cost.isna().any()


# --------------------------------------------------------------------------
# point-in-time discipline
# --------------------------------------------------------------------------

def test_point_in_time_consistency(market):
    rng = np.random.default_rng(0)
    trades = pd.DataFrame(
        rng.normal(0.0, 0.01, (len(market.dates), len(market.tickers))),
        index=market.dates,
        columns=market.tickers,
    )
    model = RealisticCost()
    full = model.cost(trades, market, NAV)
    for i in (150, 400, 700):
        t = market.dates[i]
        truncated = model.cost(trades.loc[:t], market.slice_until(t), NAV)
        # shift(1) discipline: nothing at t depends on data after t
        assert truncated.loc[t] == pytest.approx(full.loc[t], rel=1e-12, abs=1e-15)


# --------------------------------------------------------------------------
# participation cap
# --------------------------------------------------------------------------

def test_participation_cap_engages():
    lo, hi = 100.0, 104.0
    data = _alternating_market(lo, hi, volume=50.0)  # tiny ADV (~5100 dollars)
    trades = _zero_trades(data)
    t = data.dates[60]
    trades.loc[t, "AAA"] = 1.0
    nav = 1e8

    model = RealisticCost(
        commission_per_share=0.0, half_spread_bps=0.0, impact_coeff=0.1,
        adv_window=20, vol_window=10,
    )
    cost = model.cost(trades, data, nav)

    vol = _alt_vol(lo, hi, window=10)
    adv = 50.0 * (lo + hi) / 2.0
    raw_participation = 1.0 * nav / adv
    assert raw_participation > RealisticCost.PARTICIPATION_CAP  # cap really binds

    capped = 1.0 * 0.1 * vol * math.sqrt(RealisticCost.PARTICIPATION_CAP)
    uncapped = 1.0 * 0.1 * vol * math.sqrt(raw_participation)
    assert cost.loc[t] == pytest.approx(capped, rel=1e-9)
    assert cost.loc[t] < uncapped


# --------------------------------------------------------------------------
# unadjusted_close missing: warn once, fall back to adjusted close
# --------------------------------------------------------------------------

def test_missing_unadjusted_close_warns_and_falls_back():
    data = _flat_market({"AAA": 80.0, "BBB": 40.0}, {"AAA": 1e6, "BBB": 1e6},
                        with_unadjusted=False)
    assert data.unadjusted_close is None
    trades = _zero_trades(data)
    t = data.dates[65]
    trades.loc[t, "AAA"] = 0.02

    model = RealisticCost(half_spread_bps=0.0, impact_coeff=0.0)
    with pytest.warns(UserWarning, match="unadjusted_close"):
        cost = model.cost(trades, data, NAV)
    assert cost.loc[t] == pytest.approx((0.02 * NAV / 80.0) * 0.005 / NAV, rel=1e-12)

    # warns ONCE per model instance: a second call is silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.cost(trades, data, NAV)
