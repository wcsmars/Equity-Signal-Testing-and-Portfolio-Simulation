"""Trading-rule tests for the four strategy sleeves on synthetic prices.

tests/test_qcore_engine.py checks that the builders do not look ahead; these
tests check WHAT each builder holds. Every case uses a price path whose
correct answer is known by construction, or recomputes the documented rule
independently (a scalar reference), so a changed lookback, skip month,
volatility window, eligibility rule, normalisation, rank buffer, breaker,
threshold or calendar window fails here. No data cache is needed.
"""

import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from qcore import backtest as bt
from qcore.calendar import confirmed_month_ends
from qcore.quality import SPECIAL_CLOSURES, nyse_bdays
from strategies import mean_reversion, seasonality_flows, tsmom_trend, xsec_etf_mom


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setattr(bt, "_raw_close", lambda: pd.DataFrame())
    monkeypatch.setattr(bt, "_irx_series", lambda: pd.Series(dtype=float))


# ------------------------------------------------------------------ helpers
def _monthly_path(idx: pd.DatetimeIndex, monthly_growth, wiggle: float = 0.002) -> pd.Series:
    """Daily closes that compound `monthly_growth[m]` over calendar month m
    (constant daily log drift inside the month) plus a +/- `wiggle` daily
    alternation, so every 60-day volatility is positive and known."""
    months = idx.to_period("M")
    uniq = months.unique()
    growth = dict(zip(uniq, np.broadcast_to(monthly_growth, len(uniq))))
    log_r = np.empty(len(idx))
    for m in uniq:
        mask = months == m
        log_r[mask] = np.log1p(growth[m]) / mask.sum()
    alt = np.where(np.arange(len(idx)) % 2 == 0, wiggle, -wiggle)
    return pd.Series(100.0 * np.exp(np.cumsum(log_r + alt)), index=idx)


def _random_panel(tickers, seed, start, end, late=()):
    """Random walks with per-ticker drift and volatility; tickers in `late`
    start at staggered mid-month dates (blank before their first close)."""
    idx = nyse_bdays(start, end)
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.004, 0.02, len(tickers))
    drift = rng.normal(0.0002, 0.0004, len(tickers))
    steps = rng.normal(0.0, 1.0, (len(idx), len(tickers))) * vols + drift
    px = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=tickers)
    for k, ticker in enumerate(late):
        px.loc[: idx[150 + 137 * k], ticker] = np.nan
    return px


IDX = nyse_bdays("2021-01-04", "2022-12-30")
N_MONTHS = len(IDX.to_period("M").unique())


# ============================================================== tsmom_trend
def _trend_panel(growth_by_ticker: dict, wiggle_by_ticker: dict | None = None) -> pd.DataFrame:
    wiggle_by_ticker = wiggle_by_ticker or {}
    return pd.DataFrame({
        t: _monthly_path(IDX, growth_by_ticker.get(t, 0.01), wiggle_by_ticker.get(t, 0.002))
        for t in tsmom_trend.RISK + [tsmom_trend.CASH]})


def _trend_weights(px, weighting="iv", sleeve="cash", signal="blend"):
    sig, elig, vol, shy_ok = tsmom_trend.build_signals(px)
    return tsmom_trend.month_end_weights(sig[signal], elig, vol, shy_ok, weighting, sleeve)


def test_trend_specification_is_pinned():
    assert tsmom_trend.BEST == {"signal": "blend", "weighting": "iv", "sleeve": "cash"}
    assert tsmom_trend.SLIPPAGE_BPS == 3.0
    assert tsmom_trend.RISK == ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "EWJ",
                                "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ"]
    assert tsmom_trend.CASH == "SHY"


def test_trend_signal_is_long_uptrends_and_flat_downtrends():
    px = _trend_panel({"SPY": 0.02, "QQQ": -0.02})
    sig, elig, vol, shy_ok = tsmom_trend.build_signals(px)
    last = sig["blend"].index[-1]
    assert (sig["ma10"].loc[last, "SPY"], sig["mom121"].loc[last, "SPY"]) == (1.0, 1.0)
    assert (sig["ma10"].loc[last, "QQQ"], sig["mom121"].loc[last, "QQQ"]) == (0.0, 0.0)
    assert sig["blend"].loc[last, "SPY"] == 1.0 and sig["blend"].loc[last, "QQQ"] == 0.0
    w = tsmom_trend.month_end_weights(sig["blend"], elig, vol, shy_ok, "iv", "cash")
    assert w.loc[last, "SPY"] > 0.0 and w.loc[last, "QQQ"] == 0.0
    assert w.loc[last, tsmom_trend.CASH] == 0.0
    # one of 14 equal-volatility assets is off: 13/14 invested, the rest is cash
    assert w.loc[last].sum() == pytest.approx(13 / 14, rel=2e-3)


def test_trend_signal_needs_a_strict_uptrend():
    px = _trend_panel({})
    px["SPY"] = 100.0                      # exactly at its average, exactly zero momentum
    sig, elig, *_ = tsmom_trend.build_signals(px)
    last = sig["blend"].index[-1]
    assert (sig["ma10"].loc[last, "SPY"], sig["mom121"].loc[last, "SPY"]) == (0.0, 0.0)
    assert not elig.loc[last, "SPY"] and elig.loc[last, "QQQ"]      # zero volatility: not eligible


def test_trend_momentum_skips_the_most_recent_month():
    # flat, then +20% in the final month only: the 12-1 window (t-12 .. t-1)
    # is flat, so the momentum leg stays OFF while the moving-average leg
    # turns ON -> blend exactly 0.5
    late_pop = np.r_[np.zeros(N_MONTHS - 1), 0.20]
    # +3%/month until a -15% final month: 12-1 momentum ON, moving average OFF
    late_drop = np.r_[np.full(N_MONTHS - 1, 0.03), -0.15]
    sig, *_ = tsmom_trend.build_signals(_trend_panel({"SPY": late_pop, "QQQ": late_drop}))
    last = sig["blend"].index[-1]
    assert (sig["ma10"].loc[last, "SPY"], sig["mom121"].loc[last, "SPY"]) == (1.0, 0.0)
    assert (sig["ma10"].loc[last, "QQQ"], sig["mom121"].loc[last, "QQQ"]) == (0.0, 1.0)
    assert sig["blend"].loc[last, "SPY"] == 0.5 and sig["blend"].loc[last, "QQQ"] == 0.5


def test_trend_momentum_window_starts_twelve_month_ends_back():
    # one +30% month, 12 month-ends before the decision, then a slow decline:
    # close(t-1)/close(t-12) < 1 (the jump is outside the window) while
    # close(t-1)/close(t-13) > 1 (a 13-month window would include it)
    growth = np.full(N_MONTHS, -0.005)
    growth[N_MONTHS - 13] = 0.30
    px = _trend_panel({"SPY": growth})
    mp = px.loc[confirmed_month_ends(px.index), "SPY"]
    assert mp.iloc[-2] / mp.iloc[-13] < 1.0 < mp.iloc[-2] / mp.iloc[-14]
    sig, *_ = tsmom_trend.build_signals(px)
    assert sig["mom121"].loc[mp.index[-1], "SPY"] == 0.0


def test_trend_moving_average_filter_uses_ten_month_ends():
    # -4%/month, then three +3% months: above the 5-month average of
    # month-end closes but still below the 10-month one
    px = _trend_panel({"SPY": np.r_[np.full(N_MONTHS - 3, -0.04), np.full(3, 0.03)]})
    mp = px.loc[confirmed_month_ends(px.index), "SPY"]
    assert mp.iloc[-5:].mean() < mp.iloc[-1] < mp.iloc[-10:].mean()
    sig, *_ = tsmom_trend.build_signals(px)
    last = sig["blend"].index[-1]
    assert sig["ma10"].loc[last, "SPY"] == 0.0 and sig["blend"].loc[last, "SPY"] == 0.0
    # +3%/month, then one -12% month: below the 10-month average, above the 12-month one
    px = _trend_panel({"SPY": np.r_[np.full(N_MONTHS - 1, 0.03), -0.12]})
    mp = px.loc[confirmed_month_ends(px.index), "SPY"]
    assert mp.iloc[-12:].mean() < mp.iloc[-1] < mp.iloc[-10:].mean()
    assert tsmom_trend.build_signals(px)[0]["ma10"].loc[last, "SPY"] == 0.0


def test_trend_weights_are_inverse_volatility_shares_over_all_eligible_assets():
    px = _trend_panel({}, {"SPY": 0.004})   # SPY twice as volatile as the rest
    sig, elig, vol, shy_ok = tsmom_trend.build_signals(px)
    last = sig["blend"].index[-1]
    assert vol.loc[last, "SPY"] == pytest.approx(2.0 * vol.loc[last, "QQQ"], rel=1e-3)
    w = tsmom_trend.month_end_weights(sig["blend"], elig, vol, shy_ok, "iv", "cash")
    assert w.loc[last, "QQQ"] == pytest.approx(2.0 * w.loc[last, "SPY"], rel=1e-3)
    assert w.loc[last, tsmom_trend.RISK].sum() == pytest.approx(1.0)   # every asset fully on
    # a switched-off asset's share goes to cash, not to the remaining assets
    off = sig["blend"].copy()
    off.loc[last, "QQQ"] = 0.0
    w_off = tsmom_trend.month_end_weights(off, elig, vol, shy_ok, "iv", "cash")
    assert w_off.loc[last, "SPY"] == pytest.approx(w.loc[last, "SPY"])
    assert w_off.loc[last, tsmom_trend.RISK].sum() == pytest.approx(1.0 - w.loc[last, "QQQ"])
    assert w_off.loc[last, tsmom_trend.CASH] == 0.0
    # the SHY off-sleeve variant parks that residual in SHY instead
    w_shy = tsmom_trend.month_end_weights(off, elig, vol, shy_ok, "iv", "shy")
    assert w_shy.loc[last, tsmom_trend.CASH] == pytest.approx(w.loc[last, "QQQ"])
    with pytest.raises(ValueError, match="weighting must be ew/iv"):
        tsmom_trend.month_end_weights(off, elig, vol, shy_ok, "risk_parity", "cash")
    with pytest.raises(ValueError, match="sleeve must be cash/shy"):
        tsmom_trend.month_end_weights(off, elig, vol, shy_ok, "iv", "bills")


def test_trend_volatility_is_the_sixty_day_standard_deviation():
    px = _trend_panel({})
    r = px["SPY"].pct_change(fill_method=None)
    _, _, vol, _ = tsmom_trend.build_signals(px)
    last = vol.index[-1]
    assert vol.loc[last, "SPY"] == pytest.approx(r.loc[:last].iloc[-60:].std(), rel=1e-12)
    assert vol.loc[last, "SPY"] != pytest.approx(r.loc[:last].iloc[-20:].std(), rel=1e-6)


def test_trend_needs_thirteen_month_ends_of_history():
    px = _trend_panel({})
    px.loc[px.index < "2022-03-01", "GLD"] = np.nan     # 10 month-ends by 2022-12
    sig, elig, vol, shy_ok = tsmom_trend.build_signals(px)
    last = elig.index[-1]
    assert not elig.loc[last, "GLD"] and elig.loc[last, "SPY"]
    w = tsmom_trend.month_end_weights(sig["blend"], elig, vol, shy_ok, "iv", "cash")
    assert w.loc[last, "GLD"] == 0.0
    assert w.loc[last, "SPY"] == pytest.approx(1 / 13, rel=2e-3)   # 13 eligible, equal volatility
    # the first 12 month-ends have no eligible asset; the 13th has all but GLD
    assert not elig.iloc[:12].to_numpy().any() and elig.iloc[12].drop("GLD").all()


def _trend_reference(px: pd.DataFrame) -> pd.DataFrame:
    """The documented rule, one asset and one month at a time."""
    me = confirmed_month_ends(px.index)
    rets = px[tsmom_trend.RISK].pct_change(fill_method=None)
    out = pd.DataFrame(0.0, index=me, columns=tsmom_trend.RISK)
    for j, t in enumerate(me):
        inv, sig = {}, {}
        for a in tsmom_trend.RISK:
            closes = px.loc[me[: j + 1], a]
            if j < 12 or closes.iloc[-13:].isna().any():
                continue
            window = rets.loc[:t, a].iloc[-60:]
            if len(window) < 60 or window.isna().any() or not window.std() > 0:
                continue
            s_ma = float(closes.iloc[-1] > closes.iloc[-10:].mean())
            s_mom = float(closes.iloc[-2] / closes.iloc[-13] - 1 > 0)
            inv[a], sig[a] = 1.0 / window.std(), 0.5 * (s_ma + s_mom)
        for a in inv:
            out.loc[t, a] = sig[a] * inv[a] / sum(inv.values())
    return out


def test_trend_weights_match_a_scalar_reference_with_late_listings():
    px = _random_panel(tsmom_trend.RISK + [tsmom_trend.CASH], 7, "2017-01-03", "2022-12-30",
                       late=("GLD", "SLV", "DBC", "SHY"))
    sig, elig, vol, shy_ok = tsmom_trend.build_signals(px)
    cash = tsmom_trend.month_end_weights(sig["blend"], elig, vol, shy_ok, "iv", "cash")
    pd.testing.assert_frame_equal(cash[tsmom_trend.RISK], _trend_reference(px),
                                  check_exact=False, rtol=0, atol=1e-12, check_freq=False)
    live = (sig["blend"] * elig).to_numpy()
    assert {0.0, 0.5, 1.0} == set(np.unique(live))                 # all three states occur
    assert (cash[tsmom_trend.CASH] == 0).all() and cash.sum(axis=1).max() <= 1 + 1e-12
    assert cash.sum(axis=1).iloc[13:].min() < 0.9                  # residual really stays in cash
    first = elig["GLD"].idxmax()                                   # late listing enters at its 13th month-end
    assert px.loc[confirmed_month_ends(px.index), "GLD"].loc[:first].notna().sum() == 13
    shy = tsmom_trend.month_end_weights(sig["blend"], elig, vol, shy_ok, "iv", "shy")
    listed = shy_ok & elig.any(axis=1)
    assert np.allclose(shy.loc[listed].sum(axis=1), 1.0)
    assert (~shy_ok).any() and (shy.loc[~shy_ok, tsmom_trend.CASH] == 0).all()   # no SHY before it exists
    ew = tsmom_trend.month_end_weights(sig["blend"], elig, vol, shy_ok, "ew", "cash")
    n = elig.sum(axis=1)
    expected = (sig["blend"] * elig).div(n.where(n > 0), axis=0).fillna(0.0)
    pd.testing.assert_frame_equal(ew[tsmom_trend.RISK], expected, check_freq=False)


def test_trend_fails_closed_on_a_blank_close_after_listing():
    px = _trend_panel({})
    month_ends = confirmed_month_ends(px.index)
    mid_month = px.index[-30]
    assert mid_month not in month_ends
    for day in (month_ends[-3], mid_month):           # a decision close, a volatility-window close
        bad = px.copy()
        bad.loc[day, "EEM"] = np.nan
        with pytest.raises(ValueError, match=f"blank close after listing in 1 cell.*EEM {day.date()}"):
            tsmom_trend.build_signals(bad)
    # blanks before the first close are a late listing, not a gap
    late = px.copy()
    late.loc[: px.index[100], "GLD"] = np.nan
    assert not tsmom_trend.build_signals(late)[1]["GLD"].iloc[:12].any()
    # a blank after the last decision date has not fed any decision yet
    partial = px.loc[:"2022-12-15"].copy()
    clean = _trend_weights(partial)
    partial.loc["2022-12-14", "EEM"] = np.nan
    pd.testing.assert_frame_equal(_trend_weights(partial), clean)


def test_trend_shy_sleeve_fails_closed_on_a_blank_shy_decision_close():
    px = _trend_panel({"SPY": -0.02})                  # SPY off: a residual exists
    day = confirmed_month_ends(px.index)[-2]
    px.loc[day, tsmom_trend.CASH] = np.nan
    sig, elig, vol, shy_ok = tsmom_trend.build_signals(px)
    with pytest.raises(ValueError, match=f"blank SHY month-end close after listing: {day.date()}"):
        tsmom_trend.month_end_weights(sig["blend"], elig, vol, shy_ok, "iv", "shy")
    cash = tsmom_trend.month_end_weights(sig["blend"], elig, vol, shy_ok, "iv", "cash")
    assert (cash[tsmom_trend.CASH] == 0).all()         # the cash sleeve never needs SHY


# ============================================================= xsec_etf_mom
XSEC_COLS = xsec_etf_mom.EQ_UNIVERSE + [xsec_etf_mom.DEFENSIVE]


def _xsec_panel(late=()) -> pd.DataFrame:
    return _random_panel(XSEC_COLS, 11, "2015-01-02", "2022-12-30", late=late)


def _blended_score(px: pd.DataFrame, skip: bool = False) -> pd.DataFrame:
    """Docstring rule: mean of the 3-, 6- and 12-month returns from month-end
    closes, measured to the previous month-end for the skip blend."""
    m = px.loc[confirmed_month_ends(px.index), xsec_etf_mom.EQ_UNIVERSE]
    base = m.shift(1) if skip else m
    return sum(base / m.shift(lb + int(skip)) - 1.0 for lb in (3, 6, 12)) / 3.0


def _risk_off(px: pd.DataFrame) -> pd.Series:
    spy = px.loc[confirmed_month_ends(px.index), "SPY"]
    return spy < spy.rolling(10).mean()


def test_xsec_specification_is_pinned():
    assert xsec_etf_mom.BEST_PARAMS == {"k": 3, "skip": False, "breaker": True, "buffer": 9, "drift": True}
    assert xsec_etf_mom.LOOKBACKS == (3, 6, 12) and xsec_etf_mom.SMA_MONTHS == 10
    assert xsec_etf_mom.MIN_HISTORY_MONTHS == 14 and xsec_etf_mom.DEFENSIVE == "IEF"
    assert len(xsec_etf_mom.EQ_UNIVERSE) == 34 == len(set(xsec_etf_mom.EQ_UNIVERSE))
    assert xsec_etf_mom.SLIPPAGE_BPS == 3.0


def test_xsec_plain_top_k_holds_the_highest_blended_momentum():
    px = _xsec_panel()
    score = _blended_score(px)
    w = xsec_etf_mom.build_targets(px, k=3, skip=False, breaker=False, buffer=3)
    for t in w.index[14:]:
        held = set(w.columns[w.loc[t] > 0])
        assert held == set(score.loc[t].nlargest(3).index)
        assert w.loc[t, list(held)].tolist() == pytest.approx([1 / 3] * 3)
    assert len(w.index[14:]) > 60
    assert not w.iloc[:14].to_numpy().any()          # warm-up: no positions
    assert w.iloc[14].sum() == pytest.approx(1.0)    # first decision is the 15th month-end


@pytest.mark.parametrize("k,buffer", [(3, 9), (3, 3), (4, 6)])
def test_xsec_obeys_rank_buffer_breaker_and_eligibility(k, buffer):
    px = _xsec_panel(late=("XBI", "KRE", "XME", "XOP"))
    target = xsec_etf_mom.build_targets(px, k=k, skip=False, breaker=True, buffer=buffer)
    score, risk_off = _blended_score(px), _risk_off(px)
    live = risk_off.iloc[xsec_etf_mom.MIN_HISTORY_MONTHS:]
    assert live.any() and (~live).any()
    previous, kept_beyond_k, dropped, resets = set(), 0, 0, 0
    for i, t in enumerate(target.index):
        row = target.loc[t]
        held = set(row.index[row > 0])
        if i < xsec_etf_mom.MIN_HISTORY_MONTHS:
            assert not held
            continue
        if risk_off.loc[t]:
            assert held == {xsec_etf_mom.DEFENSIVE} and row[xsec_etf_mom.DEFENSIVE] == 1.0
            resets += bool(previous)
            previous = set()                                   # memory resets with the book
            continue
        s = score.loc[t].dropna()
        rank = s.rank(ascending=False, method="first")
        assert len(held) == k and np.allclose(row[list(held)], 1.0 / k)
        assert held <= set(s.index)                            # only names with all three lookbacks
        must_keep = {c for c in previous if c in rank.index and rank[c] <= buffer}
        assert must_keep <= held
        assert not (previous - must_keep) & held               # anything outside the buffer is sold
        best_unheld = [c for c in rank.sort_values().index if c not in must_keep][: k - len(must_keep)]
        assert held - must_keep == set(best_unheld)
        kept_beyond_k += sum(rank[c] > k for c in must_keep)
        dropped += len(previous - must_keep)
        previous = held
    # both branches must be exercised or the loop proves nothing
    assert dropped > 0 and resets > 0
    assert (kept_beyond_k > 0) == (buffer > k)


def test_xsec_rank_hysteresis_trades_less_than_plain_top_k():
    px = _xsec_panel()
    score = _blended_score(px)
    k, buffer = 3, 9
    w = xsec_etf_mom.build_targets(px, k=k, skip=False, breaker=False, buffer=buffer)
    prev: set = set()
    kept_outside_top_k = sold = 0
    for t in w.index[14:]:
        rank = score.loc[t].rank(ascending=False, method="first")
        held = set(w.columns[w.loc[t] > 0])
        for name in prev:                          # keep iff still ranked <= buffer
            assert (name in held) == (rank[name] <= buffer), (t, name, rank[name])
            kept_outside_top_k += name in held and rank[name] > k
            sold += name not in held
        prev = held
    assert kept_outside_top_k > 10 and sold > 10
    plain = xsec_etf_mom.build_targets(px, k=k, skip=False, breaker=False, buffer=k)
    assert w.diff().abs().sum().sum() < 0.8 * plain.diff().abs().sum().sum()


def test_xsec_breaker_moves_the_whole_book_to_the_defensive_asset_and_resets_memory():
    px = _xsec_panel()
    risk_off, score = _risk_off(px), _blended_score(px)
    w = xsec_etf_mom.build_targets(px, k=3, skip=False, breaker=True, buffer=9)
    live = w.index[14:]
    for t in live:
        if risk_off.loc[t]:
            assert w.loc[t, "IEF"] == 1.0 and w.loc[t].sum() == 1.0
        else:
            assert w.loc[t, "IEF"] == 0.0 and (w.loc[t] > 0).sum() == 3
    # first month back on after a breaker month starts from a clean top-3
    back_on = [t for before, t in zip(live[:-1], live[1:]) if risk_off.loc[before] and not risk_off.loc[t]]
    assert back_on
    for t in back_on:
        assert set(w.columns[w.loc[t] > 0]) == set(score.loc[t].nlargest(3).index)
    # with the breaker off the defensive asset is never held
    off = xsec_etf_mom.build_targets(px, k=3, skip=False, breaker=False, buffer=9)
    assert (off["IEF"] == 0).all() and np.allclose(off.iloc[14:].sum(axis=1), 1.0)
    # before the defensive asset exists the breaker parks in cash
    no_ief = px.copy()
    no_ief["IEF"] = np.nan
    w2 = xsec_etf_mom.build_targets(no_ief, k=3, skip=False, breaker=True, buffer=9)
    months_off = risk_off.loc[live][risk_off.loc[live]].index
    assert len(months_off) and not w2.loc[months_off].to_numpy().any()


def test_xsec_breaker_compares_spy_with_its_ten_month_average():
    px = _xsec_panel()
    months = len(px.index.to_period("M").unique())

    def decide(rising_months):
        # SPY sits at 130, falls to 100 and then gains 0.2% a month for the
        # last `rising_months` month-ends
        growth = np.r_[0.30, np.zeros(months - rising_months - 1), 100.2 / 130.0 - 1.0,
                       np.full(rising_months - 1, 0.002)]
        panel = px.copy()
        panel["SPY"] = _monthly_path(px.index, growth, wiggle=0.0)
        spy = panel.loc[confirmed_month_ends(px.index), "SPY"]
        target = xsec_etf_mom.build_targets(panel, k=3, skip=False, breaker=True, buffer=9).iloc[-1]
        return spy, target

    # ten rising month-ends: above the 10-month average (risk-on), though
    # still below the 11- and 12-month ones, which include the old high
    spy, target = decide(10)
    assert spy.iloc[-10:].mean() < spy.iloc[-1] < min(spy.iloc[-11:].mean(), spy.iloc[-12:].mean())
    assert target["IEF"] == 0.0 and (target > 0).sum() == 3
    # nine rising month-ends: the old high is still inside the 10-month
    # average (risk-off), though above the 9-month one
    spy, target = decide(9)
    assert spy.iloc[-9:].mean() < spy.iloc[-1] < spy.iloc[-10:].mean()
    assert target["IEF"] == 1.0 and target.sum() == 1.0
    # a close exactly AT its average is not below it: the book stays in equities
    flat = px.copy()
    flat["SPY"] = 100.0
    target = xsec_etf_mom.build_targets(flat, k=3, skip=False, breaker=True, buffer=9).iloc[-1]
    assert target["IEF"] == 0.0 and (target > 0).sum() == 3


def test_xsec_skip_blend_measures_returns_to_the_previous_month_end():
    px = _xsec_panel()
    score = _blended_score(px, skip=True)
    w = xsec_etf_mom.build_targets(px, k=3, skip=True, breaker=False, buffer=3)
    plain = _blended_score(px)
    differs = 0
    for t in w.index[14:]:
        held = set(w.columns[w.loc[t] > 0])
        assert held == set(score.loc[t].nlargest(3).index)
        differs += held != set(plain.loc[t].nlargest(3).index)
    assert differs > 5
    # only the decision close changes: the skip blend must not see it
    last = w.index[-1]
    bumped = px.copy()
    bumped.loc[last, "XLE"] *= 3.0
    pd.testing.assert_frame_equal(
        w, xsec_etf_mom.build_targets(bumped, k=3, skip=True, breaker=False, buffer=3))
    assert xsec_etf_mom.build_targets(bumped, k=3, skip=False, breaker=False, buffer=3).loc[last, "XLE"] > 0


def test_xsec_stays_in_cash_with_fewer_than_k_eligible_names():
    px = _xsec_panel()
    young = [c for c in xsec_etf_mom.EQ_UNIVERSE if c not in ("SPY", "QQQ")]
    px.loc[px.index < "2022-06-01", young] = np.nan       # 7 month-ends by the end: no 12-month return
    w = xsec_etf_mom.build_targets(px, k=3, skip=False, breaker=False, buffer=9)
    assert not w.to_numpy().any()                          # two eligible names cannot fill three slots
    two = xsec_etf_mom.build_targets(px, k=2, skip=False, breaker=False, buffer=2)
    assert (two.iloc[14:][["SPY", "QQQ"]] == 0.5).all().all() and two.iloc[14:].sum(axis=1).eq(1.0).all()


def test_xsec_default_weights_trade_only_at_month_ends():
    px = _xsec_panel()
    w = xsec_etf_mom.build_weights(px, **xsec_etf_mom.BEST_PARAMS)
    res = bt.run_backtest(w, px[w.columns], cash_rate=0.0, withholding=0.0)
    month_ends = confirmed_month_ends(px.index)
    turnover = res["turnover"]
    assert turnover.drop(month_ends, errors="ignore").abs().max() < 1e-12
    assert turnover.loc[turnover.index.isin(month_ends)].sum() > 5
    # buffer=None means plain top-K; drift=False holds the target constant
    held = xsec_etf_mom.build_weights(px, k=3, skip=False, breaker=True)
    monthly = xsec_etf_mom.build_targets(px, k=3, skip=False, breaker=True, buffer=3)
    pd.testing.assert_frame_equal(held, monthly.reindex(px.index).ffill().fillna(0.0))


@pytest.mark.parametrize("bad", [{"k": 0}, {"k": 2.0}, {"k": True}, {"k": 35},
                                 {"buffer": 2}, {"buffer": 9.5}, {"buffer": True}])
def test_xsec_rejects_invalid_parameters(bad):
    px = _xsec_panel().loc[:"2016-12-30"]
    params = {"k": 3, "skip": False, "breaker": True, "buffer": 9} | bad
    with pytest.raises(ValueError, match="must be an integer"):
        xsec_etf_mom.build_targets(px, **params)


@pytest.mark.parametrize("ticker", ["SPY", "XLE", "IEF"])
def test_xsec_fails_closed_on_a_blank_decision_close(ticker):
    px = _xsec_panel(late=("XBI", "KRE"))             # late listings alone do not raise
    clean = xsec_etf_mom.build_targets(px, k=3, skip=False, breaker=True, buffer=9)
    day = clean.index[40]
    bad = px.copy()
    bad.loc[day, ticker] = np.nan                     # breaker input, a member, the defensive asset
    with pytest.raises(ValueError, match=f"blank month-end close after listing in 1 cell.*{ticker} {day.date()}"):
        xsec_etf_mom.build_targets(bad, k=3, skip=False, breaker=True, buffer=9)
    with pytest.raises(ValueError, match=f"{ticker} {day.date()}"):
        xsec_etf_mom.build_weights(bad, **xsec_etf_mom.BEST_PARAMS)
    # a blank on a non-decision close does not enter the monthly rule
    mid = px.copy()
    mid.loc[px.index[px.index.get_loc(day) - 7], ticker] = np.nan
    pd.testing.assert_frame_equal(
        xsec_etf_mom.build_targets(mid, k=3, skip=False, breaker=True, buffer=9), clean)


# ======================================================== seasonality_flows
def _held_sessions(weights: pd.DataFrame) -> list[str]:
    """Sessions on which the position is in force (the engine lags one row)."""
    lagged = weights["SPY"].shift(1).fillna(0.0)
    return [str(d.date()) for d in lagged.index[lagged == 1.0]]


def _tom_reference(idx: pd.DatetimeIndex, n_last: int, m_first: int) -> pd.Series:
    """The rule one close at a time: the calendar as scheduled on that date
    (closures not yet observed still count as sessions), the next scheduled
    session, and its position inside its own month."""
    start = idx[0].to_period("M").start_time
    end = (idx[-1].to_period("M") + 1).end_time.normalize()
    scheduled = nyse_bdays(start, end).union(
        SPECIAL_CLOSURES[(SPECIAL_CLOSURES >= start) & (SPECIAL_CLOSURES <= end)]).sort_values()
    out = []
    for date in idx:
        calendar = scheduled.difference(SPECIAL_CLOSURES[SPECIAL_CLOSURES <= date])
        nxt = calendar[calendar > date][0]
        month = calendar[calendar.to_period("M") == nxt.to_period("M")]
        rank = month.get_loc(nxt)
        out.append(float(rank < m_first or len(month) - rank <= n_last))
    return pd.Series(out, index=idx, name="SPY")


def test_turn_of_month_window_is_the_last_four_and_first_two_sessions():
    assert (seasonality_flows.N_LAST, seasonality_flows.M_FIRST, seasonality_flows.SLIPPAGE_BPS) == (4, 2, 2.0)
    idx = nyse_bdays("2024-01-02", "2024-06-28")
    w = seasonality_flows.tom_weights(pd.Series(100.0, index=idx))
    held = _held_sessions(w)
    assert [d for d in held if "2024-04-10" < d < "2024-05-15"] == [
        "2024-04-25", "2024-04-26", "2024-04-29", "2024-04-30",   # last 4 of April
        "2024-05-01", "2024-05-02",                               # first 2 of May
    ]
    # March 2024 ends on Thursday 28th (Good Friday closed): holiday-aware
    assert [d for d in held if "2024-03-10" < d < "2024-04-15"] == [
        "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-04-01", "2024-04-02",
    ]
    # the decision row is the close BEFORE each held session
    assert w.loc["2024-04-24", "SPY"] == 1.0 and w.loc["2024-04-23", "SPY"] == 0.0
    assert w.loc["2024-05-01", "SPY"] == 1.0 and w.loc["2024-05-02", "SPY"] == 0.0
    assert set(np.unique(w["SPY"])) == {0.0, 1.0}
    # every session of the half-year, against a count inside each month
    in_force = w["SPY"].shift(1).iloc[1:].astype(bool)
    month = idx.to_period("M")
    k = pd.Series(1, index=idx).groupby(month).cumcount()
    n = pd.Series(1, index=idx).groupby(month).transform("size")
    pd.testing.assert_series_equal(in_force, ((k < 2) | (n - k <= 4)).iloc[1:], check_names=False)
    # the final row targets the first session after the sample (July 1st)
    assert w["SPY"].iloc[-1] == 1.0


def test_turn_of_month_does_not_anticipate_an_unscheduled_closure():
    # The exchange closed on 2012-10-29/30. Until it happened both were
    # scheduled sessions, so the last-four window opened on Oct 26, not Oct 24.
    idx = nyse_bdays("2012-09-04", "2012-11-30")
    w = seasonality_flows.tom_weights(pd.Series(100.0, index=idx))["SPY"]
    assert w.loc["2012-10-23":"2012-10-24"].eq(0.0).all()
    assert w.loc["2012-10-25":"2012-10-26"].eq(1.0).all()
    assert w.loc["2012-10-31"] == 1.0 and w.loc["2012-11-01"] == 1.0 and w.loc["2012-11-02"] == 0.0


@pytest.mark.parametrize("start,end", [
    ("2001-08-01", "2001-10-31"), ("2004-05-03", "2004-07-30"), ("2006-12-01", "2007-02-28"),
    ("2012-09-04", "2012-11-30"), ("2018-11-01", "2019-01-31"), ("2024-12-02", "2025-02-28"),
])
def test_turn_of_month_matches_a_close_by_close_reference_around_special_closures(start, end):
    idx = nyse_bdays(start, end)
    flat = pd.Series(100.0, index=idx)
    for n_last, m_first in ((4, 2), (5, 3), (3, 0), (0, 2), (0, 0)):
        got = seasonality_flows.tom_weights(flat, n_last=n_last, m_first=m_first)
        assert list(got.columns) == ["SPY"] and got["SPY"].dtype == float
        pd.testing.assert_series_equal(got["SPY"], _tom_reference(idx, n_last, m_first))
    # the answer at a close does not depend on where the sample starts or ends
    full = seasonality_flows.tom_weights(flat)
    for cut in range(5, len(idx), 7):
        pd.testing.assert_frame_equal(seasonality_flows.tom_weights(flat.iloc[:cut]), full.iloc[:cut])
        pd.testing.assert_frame_equal(seasonality_flows.tom_weights(flat.iloc[cut:]), full.iloc[cut:])


def test_turn_of_month_parameters_and_trend_filter():
    idx = nyse_bdays("2023-01-03", "2024-06-28")
    flat = pd.Series(100.0, index=idx)
    base = _held_sessions(seasonality_flows.tom_weights(flat, n_last=4, m_first=2))
    wider = _held_sessions(seasonality_flows.tom_weights(flat, n_last=5, m_first=3))
    per_month = pd.Series(pd.to_datetime(base)).dt.to_period("M").value_counts().sort_index()
    assert (per_month.iloc[1:-1] == 6).all()                 # 4 + 2 sessions every month
    assert set(base) < set(wider) and len(wider) - len(base) >= 2 * 15
    rising = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)
    falling = pd.Series(np.linspace(200.0, 100.0, len(idx)), index=idx)
    up = seasonality_flows.tom_weights(rising, dma_filter=True)
    assert not seasonality_flows.tom_weights(falling, dma_filter=True)["SPY"].any()
    assert not up["SPY"].iloc[:199].any()                    # filter undefined: flat
    pd.testing.assert_frame_equal(up.iloc[199:], seasonality_flows.tom_weights(rising).iloc[199:])
    assert up["SPY"].iloc[199:].any()
    # the filter is the 200-session average: a price above its 100-session
    # mean but below its 200-session mean stays out
    v_shape = pd.Series(np.r_[np.linspace(200.0, 100.0, len(idx) - 60), np.linspace(100.0, 112.0, 60)], index=idx)
    assert v_shape.iloc[-1] > v_shape.iloc[-100:].mean() and v_shape.iloc[-1] < v_shape.iloc[-200:].mean()
    assert seasonality_flows.tom_weights(v_shape)["SPY"].iloc[-1] == 1.0
    assert seasonality_flows.tom_weights(v_shape, dma_filter=True)["SPY"].iloc[-1] == 0.0


def test_turn_of_month_trend_filter_uses_only_the_decision_close():
    idx = nyse_bdays("2023-01-03", "2024-03-28")
    spy = pd.Series(100.0, index=idx)
    spy.iloc[-1] = 50.0                         # final close breaks the 200-day average
    w = seasonality_flows.tom_weights(spy, dma_filter=True)["SPY"]
    assert seasonality_flows.tom_weights(spy)["SPY"].iloc[-1] == 1.0   # next session is April 1st
    assert w.iloc[-1] == 0.0
    assert w.loc["2024-03-27"] == 0.0           # a flat price is not ABOVE its average
    rising = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    rising.iloc[-1] = 50.0                      # tomorrow's break must not reach today's row
    assert seasonality_flows.tom_weights(rising, dma_filter=True)["SPY"].loc["2024-03-27"] == 1.0


@pytest.mark.parametrize("bad", [{"n_last": -1}, {"m_first": 2.0}, {"n_last": True}])
def test_turn_of_month_rejects_invalid_windows(bad):
    idx = nyse_bdays("2024-01-02", "2024-02-29")
    with pytest.raises(ValueError, match="nonnegative integers"):
        seasonality_flows.tom_weights(pd.Series(100.0, index=idx), **bad)
    with pytest.raises(ValueError, match="unique, sorted dates"):
        seasonality_flows.tom_weights(pd.Series(100.0, index=idx[::-1]))
    assert seasonality_flows.tom_weights(pd.Series(100.0, index=idx[:0])).empty


# =========================================================== mean_reversion
def _dip_path(trend: float, n_before=230, after=()):
    """A steady trend, then two sharp down days (RSI(2) -> ~0), then `after`."""
    r = np.r_[np.full(n_before, trend), [-0.03, -0.03], list(after)]
    idx = pd.bdate_range("2020-01-02", periods=len(r))
    return pd.DataFrame({"SPY": 100.0 * np.cumprod(1.0 + r)}, index=idx), n_before + 1


def _injected_rsi(monkeypatch, rsi_events, px_events=(), cols="ABCDEFG", rows=230):
    """Flat prices above their 200-day average from row 199 on, with RSI(2)
    replaced by a frame that is 50 except at the listed (row, column) cells."""
    idx = pd.bdate_range("2020-01-01", periods=rows)
    px = pd.DataFrame(100.0, index=idx, columns=list(cols))
    px.iloc[:100] = 90.0
    for row, col, value in px_events:
        px.iloc[row:, px.columns.get_loc(col)] = value
    rsi = pd.DataFrame(50.0, index=idx, columns=px.columns)
    for row, col, value in rsi_events:
        rsi.iloc[row, rsi.columns.get_loc(col)] = value

    def fake_rsi(prices, period):
        assert period == 2 and prices is px
        return rsi

    monkeypatch.setattr(mean_reversion, "rsi", fake_rsi)
    return px


def test_mean_reversion_specification_is_pinned():
    assert mean_reversion.BEST_PARAMS == {"entry_th": 5, "exit_th": 70, "max_hold": 10, "w_max": 0.20}
    assert mean_reversion.SLIPPAGE_BPS == 3.0
    assert mean_reversion.UNIVERSE == ["SPY", "QQQ", "DIA", "IWM", "MDY", "XLK", "XLF", "XLE",
                                       "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]


def test_mean_reversion_buys_oversold_dips_in_uptrends_only():
    params = mean_reversion.BEST_PARAMS
    up, entry = _dip_path(0.004, after=[-0.001] * 3)
    w = mean_reversion.build_weights(up, **params)
    assert mean_reversion.rsi(up, 2)["SPY"].iloc[entry] < 5
    assert mean_reversion.rsi(up, 14)["SPY"].iloc[entry] > 5          # only the 2-day RSI is oversold
    assert up["SPY"].iloc[entry] > up["SPY"].rolling(200).mean().iloc[entry]
    assert not w["SPY"].iloc[:entry].any()                    # overbought uptrend: no entry
    assert w["SPY"].iloc[entry] == pytest.approx(0.20)        # entered at the dip close
    down, entry_d = _dip_path(-0.002, after=[-0.001] * 3)     # same dip below the 200-day average
    assert mean_reversion.rsi(down, 2)["SPY"].iloc[entry_d] < 5
    assert not mean_reversion.build_weights(down, **params)["SPY"].any()


def test_mean_reversion_exits_on_rsi_recovery_or_after_max_hold():
    params = mean_reversion.BEST_PARAMS
    # strong rebound the day after entry: RSI(2) > 70 -> exit at that close
    rebound, entry = _dip_path(0.004, after=[0.10, 0.001, 0.001])
    w = mean_reversion.build_weights(rebound, **params)
    assert mean_reversion.rsi(rebound, 2)["SPY"].iloc[entry + 1] > 70
    assert w["SPY"].iloc[entry] > 0 and not w["SPY"].iloc[entry + 1:].any()
    # no recovery: a slow bleed keeps RSI(2) at 0, so the time stop fires
    bleed, entry = _dip_path(0.004, after=[-0.0005] * 15)
    w = mean_reversion.build_weights(bleed, **params)
    assert (mean_reversion.rsi(bleed, 2)["SPY"].iloc[entry:] < 70).all()
    held = w["SPY"].to_numpy() > 0
    # held for exactly max_hold=10 closes, flat at the 11th (it may re-enter
    # afterwards: the dip condition still holds)
    assert held[entry:entry + 10].all() and not held[entry + 10]
    short = mean_reversion.build_weights(bleed, **{**params, "max_hold": 5})["SPY"].to_numpy() > 0
    assert short[entry:entry + 5].all() and not short[entry + 5]


def test_mean_reversion_state_machine_with_an_injected_rsi(monkeypatch):
    px = _injected_rsi(monkeypatch,
                       rsi_events=[(205, "A", 1.0), (208, "A", 80.0),      # RSI exit
                                   (205, "B", 1.0),                        # time stop
                                   (205, "C", 1.0),                        # below the 200-day average
                                   (205, "D", 5.0),                        # not < 5
                                   (150, "E", 1.0),                        # 200-day average not available
                                   (205, "F", 1.0), (207, "F", 70.0),      # 70 is not > 70
                                   (205, "G", 4.9)],                       # just oversold enough
                       px_events=[(205, "C", 80.0)])
    w = mean_reversion.build_weights(px, **mean_reversion.BEST_PARAMS)
    held = w.gt(0)
    assert list(np.flatnonzero(held["A"])) == [205, 206, 207]
    for name in ("B", "F", "G"):
        assert list(np.flatnonzero(held[name])) == list(range(205, 215))    # 10 return days
    assert not held[["C", "D", "E"]].any().any()
    assert np.allclose(w.iloc[205][["A", "B", "F", "G"]], 0.20)            # w_max binds with 4 open
    assert w.iloc[208][["B", "F", "G"]].eq(0.20).all() and w.iloc[208]["A"] == 0.0


def test_mean_reversion_weights_are_capped_and_never_levered(monkeypatch):
    params = mean_reversion.BEST_PARAMS
    one, entry = _dip_path(0.004, after=[-0.0005] * 3)
    for n_open, expected in ((1, 0.20), (3, 0.20), (5, 0.20), (8, 1 / 8), (14, 1 / 14)):
        cols = {f"T{k:02d}": one["SPY"] * (1 + k / 100) for k in range(n_open)}
        # the rest of the universe never dips
        cols.update({f"U{k:02d}": pd.Series(100 * 1.004 ** np.arange(len(one)), index=one.index)
                     for k in range(14 - n_open)})
        w = mean_reversion.build_weights(pd.DataFrame(cols), **params)
        row = w.iloc[entry]
        assert (row > 0).sum() == n_open
        assert row[row > 0].tolist() == pytest.approx([expected] * n_open)
        assert row.sum() <= 1.0 + 1e-12
    px = _injected_rsi(monkeypatch, rsi_events=[(205, c, 1.0) for c in "ABCDEFG"])
    w = mean_reversion.build_weights(px, **params)
    assert np.allclose(w.iloc[205], 1 / 7) and w.sum(axis=1).max() <= 1.0 + 1e-12


@pytest.mark.parametrize("bad", [{"entry_th": 70}, {"exit_th": 101}, {"entry_th": -1}, {"w_max": 0.0},
                                 {"w_max": 1.5}, {"max_hold": 0}, {"max_hold": 2.5}, {"max_hold": True}])
def test_mean_reversion_rejects_invalid_parameters(bad):
    px, _ = _dip_path(0.004)
    with pytest.raises(ValueError, match="require 0 <= entry_th|max_hold must be a positive integer"):
        mean_reversion.build_weights(px, **{**mean_reversion.BEST_PARAMS, **bad})


# ============================================================ command line
def _cli_prices() -> pd.DataFrame:
    tickers = sorted(set(tsmom_trend.RISK + [tsmom_trend.CASH] + XSEC_COLS))
    return _random_panel(tickers, 5, "2016-07-01", "2018-06-29")


SWEEPS = [(tsmom_trend, "tsmom_trend_variants.csv", b"variant,signal,weighting,sleeve,full_sharpe,"),
          (xsec_etf_mom, "xsec_etf_mom_variants.csv", b"name,k,skip,breaker,start,end,full_sharpe,"),
          (seasonality_flows, "seasonality_flows_variants.csv", b"name,N,M,dma200_filter,full_sharpe,")]


@pytest.mark.parametrize("module,filename,header", SWEEPS, ids=[s[1].split("_variants")[0] for s in SWEEPS])
def test_sweep_never_replaces_a_differing_saved_table_without_rebase(
        module, filename, header, monkeypatch, tmp_path, capsys):
    px = _cli_prices()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_prices", lambda: px)
    monkeypatch.setattr(module, "run_backtest",
                        lambda *a, **k: bt.run_backtest(*a, cash_rate=0.0, withholding=0.0, **k))
    monkeypatch.delenv("QCORE_REBASE", raising=False)
    monkeypatch.setattr(sys, "argv", [filename, "--sweep"])
    saved = tmp_path / "results" / filename
    recomputed = tmp_path / "results" / "recomputed" / filename

    module.main()                                   # no table yet: written, 12 variants, no index column
    first = saved.read_bytes()
    assert first.startswith(header) and first.count(b"\n") == 13
    assert not recomputed.exists()

    saved.write_bytes(b"an,earlier,table\n")
    module.main()                                   # differing run: the saved table is kept
    assert saved.read_bytes() == b"an,earlier,table\n"
    assert recomputed.read_bytes() == first
    assert "kept existing record" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", [filename, "--sweep", "--rebase"])
    module.main()                                   # explicit replacement
    assert saved.read_bytes() == first


@pytest.mark.parametrize("module", [tsmom_trend, xsec_etf_mom, seasonality_flows, mean_reversion],
                         ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_command_line_rejects_unknown_flags_before_running_anything(module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [module.__file__, "--sweeep"])
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(module.__file__, run_name="__main__")
    assert stop.value.code == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments: --sweeep" in captured.err and captured.out == ""
