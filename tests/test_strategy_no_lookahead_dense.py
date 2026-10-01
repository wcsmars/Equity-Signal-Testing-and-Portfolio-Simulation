"""Dense truncation tests for the four sleeve builders.

tests/test_qcore_engine.py compares full and truncated runs at two cutoffs.
A one-bar lookahead changes only the LAST row of a truncated run, and only
when that bar carries a decision, so two cutoffs almost never see it. Here
every builder is truncated at many consecutive closes, on prices built to
trigger decisions often, and each test first proves that the cutoffs contain
decision closes (entries, exits, filter crossings, breaker flips).
"""

import numpy as np
import pandas as pd
import pytest

from qcore import backtest as bt
from qcore.calendar import confirmed_month_ends
from qcore.quality import nyse_bdays
from strategies import mean_reversion, seasonality_flows, tsmom_trend, xsec_etf_mom

IDX = nyse_bdays("2020-01-02", "2022-12-30")


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setattr(bt, "_raw_close", lambda: pd.DataFrame())
    monkeypatch.setattr(bt, "_irx_series", lambda: pd.Series(dtype=float))


def _panel(tickers, seed, vol=0.012, drift=0.0004, late=()):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, (len(IDX), len(tickers)))
    px = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=IDX, columns=tickers)
    for k, ticker in enumerate(late):            # staggered mid-month listings
        px.loc[: IDX[140 + 45 * k], ticker] = np.nan
    return px


def _assert_prefix_invariant(build, px, cutoffs):
    """build(px up to t) must equal build(px) up to t, at every cutoff t."""
    full = build(px)
    for t in cutoffs:
        pd.testing.assert_frame_equal(build(px.loc[:t]), full.loc[:t], check_exact=True,
                                      check_freq=False, obj=f"weights known at {t.date()}")
    return full


def test_mean_reversion_is_prefix_invariant_at_every_recent_close():
    # an uptrend with large daily swings: RSI(2) dips below 5 above the
    # 200-day average often, and positions open and close inside the window
    px = _panel(["SPY", "QQQ", "DIA", "IWM"], seed=21, vol=0.015, drift=0.0015).iloc[-420:]
    cutoffs = px.index[-200:]
    full = _assert_prefix_invariant(
        lambda p: mean_reversion.build_weights(p, **mean_reversion.BEST_PARAMS), px, cutoffs)
    held = full.loc[cutoffs[0]:].gt(0)
    changes = held.astype(int).diff().iloc[1:]
    assert (changes == 1).sum().sum() >= 10 and (changes == -1).sum().sum() >= 10   # entries, exits
    assert held.sum(axis=1).max() >= 2                                              # sizing path too


def test_turn_of_month_filter_is_prefix_invariant_at_every_recent_close():
    # a slow oscillation crosses its own 200-day average inside the window
    days = IDX[-300:]
    spy = pd.DataFrame({"SPY": 100.0 + 5.0 * np.sin(np.arange(300) / 8.0)}, index=days)
    cutoffs = days[-90:]
    full = _assert_prefix_invariant(
        lambda p: seasonality_flows.tom_weights(p["SPY"], dma_filter=True), spy, cutoffs)
    above = (spy["SPY"] > spy["SPY"].rolling(200).mean()).loc[cutoffs]
    assert (above != above.shift()).iloc[1:].sum() >= 3                 # the filter flips
    plain = seasonality_flows.tom_weights(spy["SPY"]).loc[cutoffs, "SPY"]
    filtered = full.loc[cutoffs, "SPY"]
    assert (plain > filtered).any() and filtered.any()                  # it blocks some window days, not all


def test_monthly_builders_are_prefix_invariant_around_every_month_end():
    tickers = sorted(set(tsmom_trend.RISK + [tsmom_trend.CASH] + xsec_etf_mom.EQ_UNIVERSE
                         + [xsec_etf_mom.DEFENSIVE]))
    px = _panel(tickers, seed=4, drift=0.0001, late=("GLD", "XBI", "SHY"))
    month_ends = confirmed_month_ends(px.index)
    # each month-end close and the closes either side of it, for the last 18 months
    cutoffs = sorted({px.index[px.index.get_loc(t) + k] for t in month_ends[-19:-1] for k in (-1, 0, 1)})
    trend_px = px[tsmom_trend.RISK + [tsmom_trend.CASH]]

    def trend(sleeve, daily=False):
        def build(p):
            sig, elig, vol, shy = tsmom_trend.build_signals(p[trend_px.columns])
            monthly = tsmom_trend.month_end_weights(sig["blend"], elig, vol, shy, "iv", sleeve)
            return tsmom_trend.to_daily_drift(monthly, p[trend_px.columns], cash_rate=0.0) if daily else monthly
        return build

    def rotation_targets(p):
        params = {k: v for k, v in xsec_etf_mom.BEST_PARAMS.items() if k != "drift"}
        return xsec_etf_mom.build_targets(p, **params)

    # the monthly decision frames at every cutoff ...
    trend_full = _assert_prefix_invariant(trend("cash"), px, cutoffs)
    _assert_prefix_invariant(trend("shy"), px, cutoffs)
    targets = _assert_prefix_invariant(rotation_targets, px, cutoffs)
    # ... and their daily expansions around three of the month-ends
    some = cutoffs[9:12] + cutoffs[30:33] + cutoffs[-3:]
    _assert_prefix_invariant(trend("cash", daily=True), px, some)
    _assert_prefix_invariant(lambda p: xsec_etf_mom.build_weights(p, **xsec_etf_mom.BEST_PARAMS), px, some)

    window = month_ends[-19:-1]
    # the trend book is rebalanced, and a late listing becomes eligible, inside the window
    sig, elig, *_ = tsmom_trend.build_signals(trend_px)
    assert (sig["blend"].loc[window].diff().abs().sum(axis=1) > 0).sum() >= 10
    assert trend_full.loc[window].gt(0).any(axis=1).all()
    assert not elig.loc[window[0], "GLD"] and elig.loc[window[-1], "GLD"]
    # the breaker flips inside the window, or its timing would be untested
    spy = px.loc[month_ends, "SPY"]
    risk_off = (spy < spy.rolling(xsec_etf_mom.SMA_MONTHS).mean()).loc[window]
    assert (risk_off != risk_off.shift()).iloc[1:].sum() >= 3
    assert (targets.loc[window, xsec_etf_mom.DEFENSIVE] == 1.0).equals(risk_off)
    # and the rotation changes its book on risk-on months
    risk_on = targets.loc[window].drop(columns=xsec_etf_mom.DEFENSIVE).loc[~risk_off.to_numpy()]
    assert (risk_on.diff().abs().sum(axis=1) > 0).sum() >= 3
