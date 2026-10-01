"""Rules, timing and saved-result handling of the four exploratory strategies.

pairs_statarb, xsec_stock_mom, tsmom_voltarget and vol_regime are pinned on
small constructed inputs (no data cache needed): what each documented rule
does, that no builder reads a later row, that a missing index print cannot
pass silently, and that running a module never replaces a saved result file
that differs.
"""

import ast
import json
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qcore import backtest as bt
from qcore.costs import IBKRHKCostModel
from qcore.quality import nyse_bdays

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ROOT / "src" / "strategies"
MODULES = ("pairs_statarb", "tsmom_voltarget", "vol_regime", "xsec_stock_mom")


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setattr(bt, "_raw_close", lambda: pd.DataFrame())
    monkeypatch.delenv("QCORE_REBASE", raising=False)


def _run(w, px, **kw):
    return bt.run_backtest(w, px, cash_rate=0.0, withholding=0.0, **kw)


# ---- pairs: direction, hedge ratio, exits, re-entry block, borrow ----------

def _pair_scenario():
    """A = B**0.6 * exp(spread). The spread is a small 30-day sine, plus an
    accelerating widening over rows 120-164 and a sharp three-row
    dislocation the other way at rows 290-292 that half-reverts on row 293."""
    from strategies.pairs_statarb import H, Z
    t = np.arange(320)
    lb = np.log(50.0) + 0.08 * np.sin(2 * np.pi * t / 45)
    spread = 0.001 * np.sin(2 * np.pi * t / 30) + 0.00004 * np.clip(t - 119, 0, 45) ** 2
    spread[290:293] -= 0.02
    spread[293] -= 0.005
    la = 0.6 * lb + spread
    px = pd.DataFrame({"A": np.exp(la), "B": np.exp(lb)}, index=pd.bdate_range("2021-01-04", periods=len(t)))
    # Brute-force reference: window OLS slope, then the z-score of the spread
    # rebuilt with that slope over the trailing Z rows.
    beta, z = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for i in range(H - 1, len(t)):
        beta[i] = np.polyfit(lb[i - H + 1:i + 1], la[i - H + 1:i + 1], 1)[0]
        s = la[i - Z + 1:i + 1] - beta[i] * lb[i - Z + 1:i + 1]
        z[i] = (s[-1] - s.mean()) / s.std(ddof=1)
    return px, beta, z


def test_pair_rule_direction_hedge_ratio_exits_and_reentry_block():
    from strategies.pairs_statarb import ENTRY, EXIT_Z, TIMEOUT, pair_weights
    px, beta, z = _pair_scenario()
    w = pair_weights(px, "A", "B")
    a, b = w["A"].to_numpy(), w["B"].to_numpy()

    # Flat until the first close with z > +2, then short A / long B at the
    # entry-day hedge ratio, unit gross.
    assert np.nanmax(np.abs(z[:124])) < ENTRY < z[124]
    assert not a[:124].any() and not b[:124].any()
    assert a[124] < 0 < b[124]
    assert b[124] / a[124] == pytest.approx(-beta[124])
    assert abs(a[124]) + abs(b[124]) == pytest.approx(1.0)
    # Leg weights are frozen while the trade is open.
    assert (a[124:144] == a[124]).all() and (b[124:144] == b[124]).all()
    # Time stop: z never came back inside the exit band, so the exit on the
    # 20th row after entry is the holding limit.
    assert z[125:145].min() > ENTRY and TIMEOUT == 20
    assert a[144] == 0.0 and b[144] == 0.0
    # No re-entry while z has not printed back inside the entry band.
    assert z[144:167].min() > ENTRY > z[167]
    assert not a[144:290].any() and not b[144:290].any()

    # Later dislocation the other way: long A / short B.
    assert np.nanmax(np.abs(z[167:290])) < ENTRY and z[290] < -ENTRY
    assert a[290] > 0 > b[290]
    assert b[290] / a[290] == pytest.approx(-beta[290])
    # Held through a partial reversion that is still outside the exit band,
    # closed on the first close with |z| < 0.5.
    assert EXIT_Z < abs(z[293]) < 1.5 and abs(z[294]) < EXIT_Z
    assert (a[290:294] == a[290]).all() and (b[290:294] == b[290]).all()
    assert not a[294:].any() and not b[294:].any()

    # The gross argument scales both legs.
    pd.testing.assert_frame_equal(pair_weights(px, "A", "B", gross=0.5), 0.5 * w)


def test_pair_windows_default_to_the_module_lengths_and_are_validated():
    from strategies import pairs_statarb as ps
    px, _, _ = _pair_scenario()
    default = ps.pair_weights(px, "A", "B")
    pd.testing.assert_frame_equal(default, ps.pair_weights(px, "A", "B", ps.H, ps.Z))
    pd.testing.assert_frame_equal(default, ps.pair_weights(px, "A", "B", H=ps.H, Z=ps.Z, gross=1.0))
    # Other window lengths are a different rule, not a silently ignored argument.
    assert not ps.pair_weights(px, "A", "B", H=60, Z=20).equals(default)
    # A gross passed by position lands on H and is refused, as is any
    # window that is not an integer of at least two sessions.
    for bad in (0.5, 1, 0, True, None, 60.0):
        with pytest.raises(ValueError, match="integer windows"):
            ps.pair_weights(px, "A", "B", bad)
        with pytest.raises(ValueError, match="integer windows"):
            ps.pair_weights(px, "A", "B", Z=bad)


def test_borrow_is_charged_on_short_notional_held_into_each_day():
    from strategies.pairs_statarb import BORROW_RATE, apply_borrow
    dates = pd.bdate_range("2024-01-02", periods=6)
    weights = pd.DataFrame({"ZZA": [0.0, 0.0, -0.5, -0.5, -0.5, 0.0],
                            "ZZB": [0.0, 0.0, 0.7, 0.7, 0.7, 0.0]}, index=dates)
    result = {"returns": pd.Series(0.0, index=dates)}
    out = apply_borrow(result, weights)
    daily = 0.5 * BORROW_RATE / 252.0
    # Short opened at close 2 is in force on days 3-5; long notional is free.
    expected = pd.Series([0.0, 0.0, 0.0, daily, daily, daily], index=dates)
    assert BORROW_RATE == 0.01
    pd.testing.assert_series_equal(out["returns"], -expected)
    pd.testing.assert_series_equal(out["equity"], (1.0 - expected).cumprod())
    assert out["borrow_drag_ann"] == round(float(expected.mean() * 252), 5) > 0
    assert result["returns"].eq(0.0).all()  # caller's result is not modified
    long_only = apply_borrow(result, weights.clip(lower=0.0))
    assert long_only["returns"].eq(0.0).all()
    # The rate argument replaces the default and scales the charge.
    doubled = apply_borrow(result, weights, rate=2 * BORROW_RATE)
    pd.testing.assert_series_equal(doubled["returns"], -2 * expected)


def test_pair_weights_are_constant_in_a_trade_so_hold_days_are_charged():
    """The builder freezes WEIGHTS, not share counts, so the engine trades both
    legs back to target on every holding day. Expanding only the entry and
    exit rows with drift_weights removes exactly that turnover."""
    from strategies import pairs_statarb as ps
    idx = pd.bdate_range("2021-01-04", periods=160)
    rng = np.random.default_rng(7)
    b = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    spread = rng.normal(0, 0.002, len(idx))
    spread[60:] += 0.03       # A turns rich at row 60: short A / long B
    spread[110:118] -= 0.04   # A turns cheap at row 110: long A / short B
    px = pd.DataFrame({"ZZA": b * np.exp(spread), "ZZB": b}, index=idx)
    w = ps.pair_weights(px, "ZZA", "ZZB", H=20, Z=10)
    assert w["ZZA"].iloc[60] < 0 < w["ZZB"].iloc[60]
    assert w["ZZB"].iloc[110] < 0 < w["ZZA"].iloc[110]
    in_trade = w.abs().sum(axis=1) > 0
    changed = (w != w.shift(1).fillna(0.0)).any(axis=1)
    hold = in_trade & ~changed
    assert hold.sum() >= 5
    assert np.allclose(w[in_trade].abs().sum(axis=1), 1.0)               # unit gross
    assert (w.loc[in_trade, "ZZA"] * w.loc[in_trade, "ZZB"] < 0).all()  # hedged
    cm = IBKRHKCostModel(slippage_bps=3.0)
    daily = _run(w, px, cost_model=cm)
    on_hold = hold.reindex(daily["costs"].index)
    assert (daily["turnover"][on_hold] > 0).all() and (daily["costs"][on_hold] > 0).all()
    shares_held = _run(bt.drift_weights(w[changed], px, cash_rate=0.0), px, cost_model=cm)
    assert shares_held["turnover"][on_hold].max() < 1e-12
    assert shares_held["costs"][on_hold].eq(0.0).all()
    assert shares_held["costs"].sum() < daily["costs"].sum()


# ---- stock momentum: ranking, skip month, reversal tilt, breadth gate ------

def _stock_panel(bumps=None, late=0):
    """25 names over 14 month-ends. Name k drifts 1 bp/day per rank, so S24
    has the highest 12-1 momentum and S00 the lowest; daily returns
    alternate around the drift with an amplitude that is not ordered like
    the momentum rank. bumps adds a one-day return to named stocks inside
    the final month; late blanks the first six weeks of the first `late`
    names so they lack a 12-month-old month-end price."""
    idx = nyse_bdays("2021-01-04", "2022-02-28")
    names = [f"S{k:02d}" for k in range(25)]
    drift = 0.0001 * np.arange(25)
    amplitude = 0.002 + 0.0005 * (np.arange(25) % 5)
    swing = np.where(np.arange(len(idx)) % 2 == 0, 1.0, -1.0)
    rets = pd.DataFrame(drift + np.outer(swing, amplitude), index=idx, columns=names)
    for name, jump in (bumps or {}).items():
        rets.loc["2022-02-15", name] += jump
    px = 100.0 * (1.0 + rets).cumprod()
    px.iloc[:30, :late] = np.nan
    return px


def test_stock_momentum_holds_top_k_weighted_by_inverse_volatility():
    from strategies.xsec_stock_mom import K, VOL_WIN, build_weights
    px = _stock_panel()
    w = build_weights(px)
    winners = [f"S{k:02d}" for k in range(20, 25)]
    # 12-1 momentum needs 13 month-ends: no position before 2022-01-31.
    assert not w.loc[:"2022-01-28"].to_numpy().any()
    for date in ("2022-01-31", "2022-02-28"):
        row = w.loc[date]
        assert K == 5 and sorted(row[row > 0].index) == winners
        assert row.sum() == pytest.approx(1.0)
        vol = px[winners].pct_change().loc[:date].tail(VOL_WIN).std()
        pd.testing.assert_series_equal(row[winners], (1 / vol) / (1 / vol).sum(), check_names=False)
        # S20 is the calmest of the five and S24 the most volatile.
        assert row[winners].is_monotonic_decreasing and row["S20"] > 1.5 * row["S24"]
    # Month-end weights are carried until the next decision.
    assert (w.loc["2022-02-01":"2022-02-25"] == w.loc["2022-01-31"]).all().all()


def test_stock_momentum_skips_the_latest_month_and_tilts_against_it():
    from strategies.xsec_stock_mom import build_weights
    last = pd.Timestamp("2022-02-28")
    # A mid-ranked name that jumps 60% in the final month has the best
    # 12-month return but an unchanged 12-1 momentum: it is not bought.
    held = build_weights(_stock_panel({"S10": 0.60})).loc[last]
    assert sorted(held[held > 0].index) == [f"S{k:02d}" for k in range(20, 25)]
    # Reversal tilt: the sixth-ranked name fell 3% in the final month and
    # the fifth-ranked name rose 3%, so they swap places.
    held = build_weights(_stock_panel({"S19": -0.03, "S20": 0.03})).loc[last]
    assert held["S19"] > 0 and held["S20"] == 0
    # With the tilt switched off the momentum rank alone decides.
    held = build_weights(_stock_panel({"S19": -0.03, "S20": 0.03}), lam=0.0).loc[last]
    assert held["S20"] > 0 and held["S19"] == 0


def test_stock_momentum_stays_in_cash_below_twenty_eligible_names():
    from strategies.xsec_stock_mom import MIN_NAMES, build_weights
    assert MIN_NAMES == 20
    # Six names lack the January 2021 month-end: 19 eligible on 2022-01-31,
    # all 25 a month later.
    w = build_weights(_stock_panel(late=6))
    assert not w.loc[:"2022-02-25"].to_numpy().any()
    assert w.loc["2022-02-28"].sum() == pytest.approx(1.0)
    # Exactly 20 eligible names is enough.
    w = build_weights(_stock_panel(late=5))
    assert w.loc["2022-01-31"].sum() == pytest.approx(1.0)
    assert w.loc["2022-01-31", [f"S{k:02d}" for k in range(5)]].eq(0.0).all()


def test_stock_momentum_run_is_named_by_run_name(monkeypatch, capsys):
    from strategies import xsec_stock_mom as xs
    assert xs.run_name() == "xsec_stock_mom_K5_lam0.25"
    assert xs.run_name(10, 0.5) == "xsec_stock_mom_K10_lam0.5"
    assert xs.run_name(8, 0.0) == "xsec_stock_mom_K8_lam0"
    px = _stock_panel()
    monkeypatch.setattr(xs, "STOCK_UNIVERSE", list(px.columns))
    monkeypatch.setattr(xs, "load_prices", lambda: px)
    monkeypatch.setattr(xs, "run_backtest", lambda w, p, cm, name: {"name": name})
    monkeypatch.setattr(xs, "metrics", lambda res: {"name": res["name"]})
    assert xs.main() == {"name": xs.run_name()}
    assert json.loads(capsys.readouterr().out) == {"name": "xsec_stock_mom_K5_lam0.25"}


# ---- vol target: scaler, caps, residual, margin lag ------------------------

def test_vol_target_scales_to_target_and_respects_both_caps():
    from strategies import tsmom_voltarget as vt
    risk, cash = vt.RISK, vt.CASH
    dates = pd.bdate_range("2024-01-02", periods=90)
    swing = np.where(np.arange(len(dates)) % 2 == 0, 1.0, -1.0)
    rets = pd.DataFrame(0.0, index=dates, columns=risk + [cash])
    rets[risk[0]] = 0.02 * swing       # about 32% annualised
    rets[risk[1]] = 0.001 * swing      # about 1.6% annualised
    rets[risk[2]] = -0.001 * swing
    px = 100.0 * (1.0 + rets).cumprod()
    decisions = dates[[70, 75, 80, 85]]
    w = pd.DataFrame(0.0, index=decisions, columns=risk + [cash])
    w.loc[decisions[0], [risk[0], cash]] = 0.5            # too volatile: scaled down
    w.loc[decisions[1], [risk[1], risk[2]]] = [0.3, 0.2]  # calm: gross cap binds
    w.loc[decisions[1], cash] = 0.5
    w.loc[decisions[2], risk[1]] = 1.0                    # calm single name: per-asset cap
    w.loc[decisions[3], cash] = 1.0                       # nothing on: untouched
    shy_ok = pd.Series(True, index=decisions)
    out = vt.scale_to_target(w, px, shy_ok, 0.10, 2.0)

    def predicted(row, date):
        window = px[risk].pct_change().loc[:date].tail(vt.VOL_LOOKBACK)
        return float((window @ row[risk]).std() * np.sqrt(252))

    d0, d1, d2, d3 = decisions
    before = predicted(w.loc[d0], d0)
    assert before > 0.10
    assert out.loc[d0, risk[0]] == pytest.approx(0.5 * 0.10 / before)
    assert predicted(out.loc[d0], d0) == pytest.approx(0.10)
    assert out.loc[d0, cash] == pytest.approx(1.0 - out.loc[d0, risk[0]])
    # 0.5 gross with a 2.0 cap: every weight is multiplied by 4, no residual.
    assert predicted(out.loc[d1], d1) < 0.10
    assert out.loc[d1, risk].sum() == pytest.approx(2.0)
    assert out.loc[d1, [risk[1], risk[2]]].tolist() == pytest.approx([1.2, 0.8])
    assert out.loc[d1, cash] == 0.0
    # The engine clips each asset at 1.5, so a single name stops there.
    assert vt.ENGINE_PER_ASSET_CAP == bt.MAX_ABS_WEIGHT
    assert out.loc[d2, risk[1]] == pytest.approx(1.5) and out.loc[d2, cash] == 0.0
    assert out.loc[d3, risk].eq(0.0).all() and out.loc[d3, cash] == 1.0
    # A tighter gross cap binds first.
    tight = vt.scale_to_target(w, px, shy_ok, 0.10, 1.0)
    assert tight.loc[d1, risk].sum() == pytest.approx(1.0)
    assert tight.loc[d2, risk[1]] == pytest.approx(1.0)


def test_constant_leverage_and_concentration_weights():
    from strategies import tsmom_voltarget as vt
    risk, cash = vt.RISK, vt.CASH
    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-28"])
    w = pd.DataFrame(0.0, index=dates, columns=risk + [cash])
    w.loc[dates[0], [risk[0], risk[1], cash]] = [0.2, 0.1, 0.7]
    w.loc[dates[1], [risk[0], risk[1], cash]] = [0.5, 0.3, 0.2]
    w.loc[dates[2], [risk[0], risk[1], cash]] = [0.2, 0.1, 0.7]
    shy_ok = pd.Series([True, True, False], index=dates)
    out = vt.scale_constant(w, shy_ok, 2.0)
    assert out[risk[0]].tolist() == pytest.approx([0.4, 1.0, 0.4])
    assert out[risk[1]].tolist() == pytest.approx([0.2, 0.6, 0.2])
    # Unlevered residual goes to the cash ETF once it exists; a levered book
    # holds none.
    assert out[cash].tolist() == pytest.approx([0.4, 0.0, 0.0])

    signal = pd.DataFrame(0.0, index=dates, columns=risk)
    signal.loc[dates[0], [risk[0], risk[1]]] = [1.0, 0.5]
    signal.loc[dates[2], risk[2]] = 1.0
    eligible = pd.DataFrame(True, index=dates, columns=risk)
    eligible.loc[dates[2], risk[2]] = False
    vol = pd.DataFrame(0.10, index=dates, columns=risk)
    vol[risk[1]] = 0.20
    conc = vt.conc_weights(signal, eligible, vol, pd.Series(True, index=dates))
    # Signal/vol shares 10 : 2.5, renormalised to a fully invested book.
    assert conc.loc[dates[0], [risk[0], risk[1]]].tolist() == pytest.approx([0.8, 0.2])
    assert conc.loc[dates[0], risk].sum() == pytest.approx(1.0) and conc.loc[dates[0], cash] == 0.0
    # No signal on, or the only signal is on an ineligible asset: all cash ETF.
    for date in dates[1:]:
        assert conc.loc[date, risk].eq(0.0).all() and conc.loc[date, cash] == 1.0


def test_margin_follows_leverage_decided_at_the_previous_close(monkeypatch):
    from strategies import tsmom_voltarget as vt
    dates = pd.bdate_range("2024-01-02", periods=8)
    weights = pd.DataFrame({"X": [1.0] * 4 + [1.5] * 3 + [1.0], "Y": -0.5}, index=dates)
    monkeypatch.setattr(vt, "load_indices", lambda: pd.DataFrame({"^IRX": 5.0}, index=dates))
    r = pd.Series(0.0, index=dates)
    result = vt.charge_margin({"returns": r.copy(), "gross_returns": r.copy()}, weights)
    daily = 0.5 * (1.065 ** (1 / 252) - 1)
    # Leverage set at close 4 is financed on days 5-7; the short leg does
    # not offset the borrowed long.
    expected = pd.Series([0.0] * 5 + [daily] * 3, index=dates)
    pd.testing.assert_series_equal(result["margin_drag"], expected)
    pd.testing.assert_series_equal(result["returns"], -expected)
    pd.testing.assert_series_equal(result["gross_returns"], -expected)
    pd.testing.assert_series_equal(result["equity"], (1.0 - expected).cumprod())
    assert result["leverage"].tolist() == pytest.approx([0.0] * 5 + [0.5] * 3)


def test_voltarget_uses_the_trend_module_itself_not_a_second_copy():
    from strategies import tsmom_trend
    from strategies import tsmom_voltarget as vt
    for name in ("build_signals", "month_end_weights", "to_daily_drift"):
        assert getattr(vt, name) is getattr(tsmom_trend, name)
    assert vt.RISK is tsmom_trend.RISK
    assert vt.build_signals.__module__ == "strategies.tsmom_trend"


# ---- vol regime: regime map, publication lag, missing index prints ---------

def test_vix_term_structure_regimes_use_the_prior_session_average():
    from strategies import vol_regime
    idx = nyse_bdays("2024-01-02", "2024-03-01")
    prices = pd.DataFrame({"QQQ": 100.0, "IEF": 100.0}, index=idx)
    indices = pd.DataFrame({"^VIX": 18.0, "^VIX3M": 20.0}, index=idx)
    indices.iloc[20:, 0] = 24.0  # ratio steps from 0.90 to 1.20 on row 20
    w = vol_regime.build_weights(prices, indices)
    assert w.index[0] == idx[5]  # five-session average, lagged one session
    qqq, ief = w["QQQ"].reindex(idx), w["IEF"].reindex(idx)
    # Row 20 still averages rows 15-19 (0.90): contango, fully in QQQ.
    assert qqq.iloc[5:21].eq(1.0).all() and ief.iloc[5:21].eq(0.0).all()
    # Averages 0.96 and 1.02: transition, half each.
    assert qqq.iloc[21:23].eq(0.5).all() and ief.iloc[21:23].eq(0.5).all()
    # Averages 1.08 and above: inversion, fully in IEF.
    assert qqq.iloc[23:].eq(0.0).all() and ief.iloc[23:].eq(1.0).all()


def test_vix_term_structure_cannot_use_same_day_post_equity_close_print():
    from strategies import vol_regime as vr
    idx = nyse_bdays("2024-01-02", "2024-03-01")
    prices = pd.DataFrame({vr.RISK_ASSET: 100.0, vr.DEFENSIVE_ASSET: 100.0}, index=idx)
    # Ratio 0.9 (below the lower threshold) except one inverted print at d.
    indices = pd.DataFrame({"^VIX": 9.0, "^VIX3M": 10.0}, index=idx)
    d, n = 25, vr.SMOOTH_DAYS
    indices.iloc[d, 0] = 100.0
    weights = vr.build_weights(prices, indices)
    risk, defensive = weights[vr.RISK_ASSET], weights[vr.DEFENSIVE_ASSET]
    # The first decision needs n prints through the PREVIOUS session.
    assert weights.index[0] == idx[n]
    # The print published after the close of session d is not known at that
    # close ...
    assert risk.loc[idx[n]:idx[d]].eq(1.0).all()
    assert defensive.loc[idx[n]:idx[d]].eq(0.0).all()
    # ... it moves the allocation from d+1, for as long as it stays in the mean.
    assert risk.loc[idx[d + 1]:idx[d + n]].eq(0.0).all()
    assert defensive.loc[idx[d + 1]:idx[d + n]].eq(1.0).all()
    assert risk.loc[idx[d + n + 1]:].eq(1.0).all()
    # Changing only the day-d print leaves every row through d unchanged.
    calm = vr.build_weights(prices, indices.assign(**{"^VIX": 9.0}))
    pd.testing.assert_frame_equal(weights.loc[:idx[d]], calm.loc[:idx[d]])
    # The sweep path shares the lag at every smoothing length, including 1.
    one = vr._ts_weights(prices, indices, "QQQ", "IEF", 1, 0.95, 1.00)
    assert one.index[0] == idx[1]
    assert one.loc[idx[d], "QQQ"] == 1.0
    assert one.loc[idx[d + 1], "QQQ"] == 0.0
    assert one.loc[idx[d + 2], "QQQ"] == 1.0


def test_vol_regime_reports_a_held_regime_and_refuses_a_stale_index():
    from strategies import vol_regime as vr
    n, limit = vr.SMOOTH_DAYS, vr.MAX_MISSING_PRINTS
    idx = nyse_bdays("2024-01-02", "2024-04-30")
    prices = pd.DataFrame({vr.RISK_ASSET: 100.0, vr.DEFENSIVE_ASSET: 100.0}, index=idx)
    indices = pd.DataFrame({"^VIX": 18.0, "^VIX3M": 20.0}, index=idx)
    indices.iloc[40:, 0] = 22.0  # ratio 0.90 -> 1.10 at row 40
    risk = vr.RISK_ASSET

    # Complete input: every session from the first average on has a signal,
    # and nothing is reported.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full = vr.build_weights(prices, indices)
    assert full.index.equals(idx[n:])
    assert full.loc[idx[40], risk] == 1.0 and full.loc[idx[40 + n], risk] == 0.0

    # One missing print: the n averages containing it are invalid, the last
    # decided weights are held over those sessions, and the hold is reported.
    holed = indices.copy()
    holed.iloc[40, 1] = np.nan
    with pytest.warns(UserWarning, match=f"unavailable on {n} session"):
        held = vr.build_weights(prices, holed)
    assert held.index.equals(full.index)
    assert held.loc[idx[41]:idx[40 + n], risk].eq(1.0).all()
    assert not full.loc[idx[41]:idx[40 + n], risk].eq(1.0).all()
    pd.testing.assert_frame_equal(held.loc[:idx[40]], full.loc[:idx[40]])
    pd.testing.assert_frame_equal(held.loc[idx[41 + n]:], full.loc[idx[41 + n]:])

    # Prints missing before the series starts only delay the first decision.
    late = indices.copy()
    late.iloc[:10, 1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert vr.build_weights(prices, late).index.equals(idx[10 + n:])

    # The tolerated run of missing prints is bounded.
    gap = indices.copy()
    gap.iloc[50:50 + limit, 0] = np.nan
    with pytest.warns(UserWarning, match="previous regime is held"):
        assert vr.build_weights(prices, gap).index.equals(full.index)
    gap.iloc[50 + limit, 0] = np.nan
    with pytest.raises(ValueError, match=f"{limit + 1} consecutive sessions"):
        vr.build_weights(prices, gap)

    # An index file that ends before the prices: a short lag is held and
    # reported, a long one is refused instead of passing as a current signal.
    with pytest.warns(UserWarning, match=f"latest {idx[-1].date()}"):
        short = vr.build_weights(prices, indices.iloc[:-3])
    assert short.index.equals(full.index) and short.loc[idx[-1], risk] == 0.0
    with pytest.raises(ValueError, match="refresh or repair the index data"):
        vr.build_weights(prices, indices.iloc[:60])
    # The sweep path goes through the same check.
    with pytest.raises(ValueError, match="consecutive sessions"):
        vr._ts_weights(prices, indices.iloc[:60], "SPY", "CASH", 10, 0.95, 1.00)


# ---- no builder reads a later row -----------------------------------------

def test_pair_and_vol_target_builders_do_not_read_later_rows(monkeypatch):
    """Truncating the input must not change any earlier weight. A peek of a
    few rows shows only where the truncation lands on a row that matters:
    for the pair rule a row where the position changes, for the vol target
    a decision whose scale is set by predicted volatility, not by a cap."""
    from strategies import pairs_statarb, tsmom_trend, tsmom_voltarget
    monkeypatch.setattr(bt, "_irx_series", lambda: pd.Series(dtype=float))
    idx = nyse_bdays("2020-01-02", "2022-12-30")
    tickers = sorted(tsmom_trend.RISK + [tsmom_trend.CASH])
    rng = np.random.default_rng(46)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, (len(idx), len(tickers))), axis=0)),
                      index=idx, columns=tickers)

    def pair(p):
        return pairs_statarb.pair_weights(p, "SPY", "QQQ")

    full = pair(px)
    changed = full.index[full.diff().abs().sum(axis=1) > 0]
    assert len(changed) >= 6
    for cutoff in changed[-12:]:
        pd.testing.assert_frame_equal(pair(px.loc[:cutoff]), full.loc[:cutoff])

    def voltarget(p, target):
        sig, eligible, vol, shy = tsmom_trend.build_signals(p)
        monthly = tsmom_trend.month_end_weights(sig["blend"], eligible, vol, shy, "iv", "shy")
        scaled = tsmom_voltarget.scale_to_target(monthly, p, shy, target, 1.5)
        return tsmom_trend.to_daily_drift(scaled, p, cash_rate=0.0)

    # At a 3% target the predicted-volatility term sets the scale on every
    # funded decision of this panel (neither cap binds), so a peek inside
    # the estimator would move the compared weights.
    risk = tsmom_trend.RISK
    sig, eligible, vol, shy = tsmom_trend.build_signals(px)
    unscaled = tsmom_trend.month_end_weights(sig["blend"], eligible, vol, shy, "iv", "shy")
    scaled = tsmom_voltarget.scale_to_target(unscaled, px, shy, 0.03, 1.5)[risk]
    funded = unscaled[risk].sum(axis=1) > 1e-9
    decisions = list(funded.index[funded][-3:])
    assert len(decisions) == 3
    assert (scaled[funded].sum(axis=1) < 1.5 - 1e-6).all()
    assert (scaled[funded].max(axis=1) < tsmom_voltarget.ENGINE_PER_ASSET_CAP - 1e-6).all()
    assert not np.isclose(scaled[funded].sum(axis=1), unscaled.loc[funded, risk].sum(axis=1)).any()
    full = voltarget(px, 0.03)
    for cutoff in decisions:
        pd.testing.assert_frame_equal(voltarget(px.loc[:cutoff], 0.03), full.loc[:cutoff])


# ---- running a module: arguments and saved result files --------------------

@pytest.mark.parametrize("name", MODULES)
@pytest.mark.parametrize("argv,code", [(["--help"], 0), (["--sweeep"], 2), (["extra"], 2)])
def test_entry_points_reject_unknown_arguments_before_any_backtest(monkeypatch, capsys, name, argv, code):
    import qcore.data

    def refuse(*args, **kwargs):
        raise AssertionError("a backtest was started")

    monkeypatch.setattr(qcore.data, "load_prices", refuse)
    monkeypatch.setattr(qcore.data, "load_indices", refuse)
    monkeypatch.setattr(sys, "path", list(sys.path))
    script = STRATEGIES / f"{name}.py"
    monkeypatch.setattr(sys, "argv", [str(script)] + argv)
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(script), run_name="__main__")
    assert stop.value.code == code
    shown = capsys.readouterr()
    assert "usage:" in (shown.out if code == 0 else shown.err)


def _direct_writes(path):
    """Calls in a source file that write a file without going through
    qcore.records (which never replaces a differing saved file silently)."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        constants = [a.value for a in list(node.args) + [k.value for k in node.keywords]
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if name in {"write_text", "write_bytes", "dump", "to_json", "to_parquet", "to_pickle"}:
            found.append((name, node.lineno))
        elif name == "to_csv" and (node.args or any(k.arg == "path_or_buf" for k in node.keywords)):
            found.append((name, node.lineno))
        elif name == "open" and any(set(c) <= set("rwaxbt+") and set(c) & set("wax+") for c in constants):
            found.append((name, node.lineno))
    return found


@pytest.mark.parametrize("name", MODULES)
def test_strategy_modules_save_results_only_through_the_record_helpers(name):
    assert _direct_writes(STRATEGIES / f"{name}.py") == []


def test_direct_write_scan_recognises_the_usual_ways_to_write(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text(
        "import json\n"
        "def run(df, path, out):\n"
        "    df.to_csv(path, index=False)\n"
        "    with open(path, 'w') as f:\n"
        "        json.dump(out, f, indent=2)\n"
        "    with path.open('a') as f:\n"
        "        pass\n"
        "    path.write_text(json.dumps(out))\n"
        "    text = df.to_csv(index=False)\n"
        "    with open(path) as f:\n"
        "        return f.read(), text\n"
    )
    assert [name for name, _ in sorted(_direct_writes(source), key=lambda hit: hit[1])] == [
        "to_csv", "open", "dump", "open", "write_text"]


def test_pairs_run_never_replaces_a_saved_result_that_differs(monkeypatch, tmp_path, capsys):
    from strategies import pairs_statarb as ps
    idx = pd.bdate_range("2024-01-02", periods=5)
    px = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    level = {"sharpe": 0.5}
    monkeypatch.setattr(ps, "ROOT", tmp_path)
    monkeypatch.setattr(ps, "PAIRS", [("A", "B")])
    monkeypatch.setattr(ps, "load_prices", lambda: px)
    monkeypatch.setattr(ps, "pair_weights",
                        lambda p, a, b, gross=1.0: pd.DataFrame(0.0, index=p.index, columns=[a, b]))
    monkeypatch.setattr(ps, "run_backtest",
                        lambda w, p, cm, name: {"name": name, "returns": pd.Series(0.0, index=p.index)})
    monkeypatch.setattr(ps, "metrics", lambda res: {
        "name": res["name"], "full": {"sharpe": level["sharpe"]},
        "in_sample": {"sharpe": level["sharpe"]}, "out_of_sample": {"sharpe": level["sharpe"]}})
    monkeypatch.setattr(sys, "argv", ["pairs_statarb.py"])
    saved = tmp_path / "results" / "pairs_statarb.json"
    recomputed = tmp_path / "results" / "recomputed" / "pairs_statarb.json"

    ps.main()
    first = capsys.readouterr()
    printed = json.loads(first.out)  # stdout is the metrics JSON alone
    assert printed["name"] == "pairs_statarb_H90_Z60"
    assert printed["params"]["pairs_passing_in_sample"] == ["A/B"]
    assert saved.read_text() == json.dumps(printed, indent=2)
    assert "saved" in first.err

    ps.main()  # an identical run changes nothing
    assert saved.read_text() == json.dumps(printed, indent=2) and not recomputed.exists()
    assert "unchanged" in capsys.readouterr().err

    level["sharpe"] = 0.9  # a run that differs: the saved file stays
    ps.main()
    second = capsys.readouterr()
    assert json.loads(saved.read_text()) == printed
    assert json.loads(recomputed.read_text()) == json.loads(second.out) != printed
    assert "kept existing record" in second.err and "--rebase" in second.err

    monkeypatch.setattr(sys, "argv", ["pairs_statarb.py", "--rebase"])
    ps.main()  # replacing it is an explicit request
    assert json.loads(saved.read_text()) == json.loads(capsys.readouterr().out) != printed


def test_voltarget_run_never_replaces_saved_results_that_differ(monkeypatch, tmp_path, capsys):
    from strategies import tsmom_voltarget as vt
    idx = pd.bdate_range("2024-01-02", periods=5)
    level = {"sharpe": 0.5}

    def run_variant(px, sigs, elig, vol_me, shy_ok, scheme, scaling, slippage_bps=vt.SLIPPAGE_BPS):
        block = {"sharpe": level["sharpe"], "cagr": 0.05, "vol": 0.10, "maxdd": -0.1}
        m = {"name": f"{scheme}_{scaling}", "start": "2024-01-02", "end": "2024-01-08",
             "full": dict(block), "in_sample": dict(block), "out_of_sample": dict(block),
             "avg_gross_risky": 1.0, "pct_days_levered": 0.0, "max_gross": 1.0,
             "ann_margin_drag": 0.0, "ann_turnover_oneside": 1.0, "ann_cost_drag": 0.001,
             "pct_positive_months": 0.6, "worst_month": -0.05}
        return m, {"returns": pd.Series([0.0, 0.01, -0.01, 0.02, 0.0], index=idx)}

    monkeypatch.setattr(vt, "ROOT", tmp_path)
    monkeypatch.setattr(vt, "load_prices", lambda: pd.DataFrame(100.0, index=idx, columns=vt.RISK + [vt.CASH]))
    monkeypatch.setattr(vt, "build_signals", lambda px: ({}, None, None, None))
    monkeypatch.setattr(vt, "run_variant", run_variant)
    monkeypatch.setattr(sys, "argv", ["tsmom_voltarget.py", "--quiet"])
    table = tmp_path / "results" / "tsmom_voltarget_variants.csv"
    summary = tmp_path / "results" / "tsmom_voltarget.json"
    recomputed = tmp_path / "results" / "recomputed"

    vt.main()
    capsys.readouterr()
    first_table, first_summary = table.read_text(), summary.read_text()
    frame = pd.read_csv(table)
    assert list(frame["variant"]) == [f"{scheme}_{scaling}" for scheme, scaling in vt.VARIANTS]
    assert "duplicate_of" in frame.columns and not first_table.startswith(",")  # no index column
    assert first_summary == json.dumps(json.loads(first_summary), indent=2)
    assert json.loads(first_summary)["study"] is True

    level["sharpe"] = 0.9
    vt.main()
    shown = capsys.readouterr().out
    assert table.read_text() == first_table and summary.read_text() == first_summary
    assert pd.read_csv(recomputed / table.name)["full_sharpe"].eq(0.9).all()
    assert json.loads((recomputed / summary.name).read_text())["winner_metrics"]["full"]["sharpe"] == 0.9
    assert shown.count("kept existing record") == 2

    monkeypatch.setattr(sys, "argv", ["tsmom_voltarget.py", "--quiet", "--rebase"])
    vt.main()
    assert pd.read_csv(table)["full_sharpe"].eq(0.9).all()
    assert json.loads(summary.read_text())["winner_metrics"]["full"]["sharpe"] == 0.9
