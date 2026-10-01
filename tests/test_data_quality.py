"""Synthetic-corruption tests for qcore.quality.

Strategy: build a CLEAN synthetic OHLCV bundle (GBM paths, consistent
adjusted OHLC, real dividend/split adjustment mechanics), assert it produces
zero FAIL/WARN findings (false-positive guard), then inject one known defect
per test into a fresh copy and assert the matching check - and only a
sensible set of checks - fires. A DQ framework that cries wolf gets ignored;
one that misses a planted defect is worse than none.

Run: python3 tests/test_data_quality.py   (also pytest-compatible)
"""

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import qcore.data as qdata  # noqa: E402
import qcore.quality as quality  # noqa: E402
from qcore.quality import (  # noqa: E402
    DQReport, Finding, PriceBundle, apply_known_events, check_fundamentals,
    nyse_bdays, run_all)

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
UNIVERSE = set(TICKERS)
SPLIT_TICKER, SPLIT_POS, SPLIT_RATIO = "GGG", 380, 4.0
LATE_TICKER, LATE_POS = "HHH", 120
DIV_TICKERS, DIV_YIELD = ["AAA", "BBB", "CCC", "DDD"], 0.005


def build_clean() -> PriceBundle:
    idx = nyse_bdays("2020-01-02", "2022-12-30")
    n = len(idx)
    adj, raw, op, hi, lo, vol = {}, {}, {}, {}, {}, {}
    # shared market factor so cross-sectional checks (lead_lag, idio
    # extreme returns) see a realistic correlation structure
    mkt = np.clip(np.random.default_rng(7).normal(0.0, 0.010, n), -0.03, 0.03)
    for k, t in enumerate(TICKERS):
        rng = np.random.default_rng(1000 + k)
        r = 0.9 * mkt + np.clip(rng.normal(0.0003, 0.008, n), -0.04, 0.04)
        a = 100.0 * np.cumprod(1.0 + r)
        # cumulative adjustment factor: dividends step it up at each
        # ex-date going forward; a split multiplies it by the ratio
        f = np.ones(n)
        if t in DIV_TICKERS:
            for ex in range(63, n, 63):
                f[:ex] *= (1.0 - DIV_YIELD)
        if t == SPLIT_TICKER:
            f[:SPLIT_POS] /= SPLIT_RATIO
        rw = a / f
        u = rng.uniform(-0.004, 0.004, n)
        o = a * (1.0 + u)
        h = np.maximum(o, a) * (1.0 + rng.uniform(0.0, 0.004, n))
        l = np.minimum(o, a) * (1.0 - rng.uniform(0.0, 0.004, n))
        v = rng.integers(100_000, 5_000_000, n).astype(float)
        if t == LATE_TICKER:
            a, rw, o, h, l, v = (np.where(np.arange(n) < LATE_POS, np.nan, x)
                                 for x in (a, rw, o, h, l, v))
        adj[t], raw[t], op[t], hi[t], lo[t], vol[t] = a, rw, o, h, l, v
    mk = lambda d: pd.DataFrame(d, index=idx)  # noqa: E731
    # raw_split_adjusted=False: this synthetic vendor's raw close is the
    # actual tape price (GGG's split jumps in it), unlike the yfinance cache
    return PriceBundle(open=mk(op), high=mk(hi), low=mk(lo),
                       adj_close=mk(adj), close=mk(raw), volume=mk(vol),
                       raw_split_adjusted=False)


def copy_bundle(b: PriceBundle) -> PriceBundle:
    return PriceBundle(**{f: getattr(b, f).copy() for f in PriceBundle.FIELDS},
                       raw_split_adjusted=b.raw_split_adjusted)


def run(b: PriceBundle) -> list:
    rep = run_all(b, universe=UNIVERSE)
    crashed = [f for f in rep.findings if "crashed" in f.detail]
    assert not crashed, f"check crashed: {[f.detail for f in crashed]}"
    return rep.findings


def grab(findings, check, severity=None, ticker=None):
    return [f for f in findings if f.check == check
            and (severity is None or f.severity == severity)
            and (ticker is None or f.ticker == ticker)]


PRICE_FIELDS = ["open", "high", "low", "adj_close", "close"]


def scale_prices(b: PriceBundle, ticker: str, rows, mult: float) -> None:
    """Multiply one ticker's five price panels over `rows` (position/slice)."""
    for f in PRICE_FIELDS:
        df = getattr(b, f)
        df.iloc[rows, df.columns.get_loc(ticker)] *= mult


def blank(b: PriceBundle, ticker: str, rows, fields=PriceBundle.FIELDS) -> None:
    for f in fields:
        df = getattr(b, f)
        df.iloc[rows, df.columns.get_loc(ticker)] = np.nan


def day(b: PriceBundle, pos: int) -> str:
    return str(b.adj_close.index[pos].date())


CLEAN = build_clean()


# ------------------------------------------------------------ the two guards
def test_clean_bundle_has_no_fail_or_warn():
    findings = run(CLEAN)
    bad = [f for f in findings if f.severity in ("FAIL", "WARN")]
    assert not bad, "clean data flagged: " + \
        "; ".join(f"{f.check}/{f.ticker}/{f.date}: {f.detail}" for f in bad[:8])


def test_clean_bundle_reports_expected_info():
    findings = run(CLEAN)
    assert grab(findings, "split_adjustment", "INFO", SPLIT_TICKER), \
        "handled 4:1 split should be counted as INFO"
    assert grab(findings, "delisting", "INFO", LATE_TICKER), \
        "late inception should be counted as INFO"


# --------------------------------------------------------------- injections
def test_internal_missing_prices():
    b = copy_bundle(CLEAN)
    rows = slice(300, 305)
    for f in ["open", "high", "low", "adj_close", "close"]:
        getattr(b, f).iloc[rows, b.adj_close.columns.get_loc("AAA")] = np.nan
    hits = grab(run(b), "missing_prices", ticker="AAA")
    fields = {f.detail.split(":")[0] for f in hits}
    assert {"open", "high", "low", "adj_close", "close"} <= fields, fields
    assert all(f.severity == "FAIL" for f in hits), \
        "a 5-day contiguous feed outage is FAIL regardless of ticker age"


def test_missing_raw_close_only_is_caught():
    b = copy_bundle(CLEAN)
    b.close.iloc[320:327, b.close.columns.get_loc("BBB")] = np.nan
    hits = grab(run(b), "missing_prices", "FAIL", "BBB")
    assert any(f.detail.startswith("close:") for f in hits), \
        "a hole only in close.csv silently breaks dividend_yields()"


def test_missing_high_only_is_caught():
    b = copy_bundle(CLEAN)
    b.high.iloc[330:337, b.high.columns.get_loc("CCC")] = np.nan
    hits = grab(run(b), "missing_prices", "FAIL", "CCC")
    assert any(f.detail.startswith("high:") for f in hits), hits


def test_missed_split_fails():
    b = copy_bundle(CLEAN)
    c = b.adj_close.columns.get_loc("BBB")
    for f in ["open", "high", "low", "adj_close", "close"]:
        getattr(b, f).iloc[400:, c] *= 0.5  # provider halves price, no factor
    hits = grab(run(b), "split_adjustment", "FAIL", "BBB")
    assert any("NOT adjusted" in f.detail for f in hits), hits


def test_phantom_adjustment_fails():
    b = copy_bundle(CLEAN)
    b.adj_close.iloc[500:, b.adj_close.columns.get_loc("CCC")] *= 0.5
    hits = grab(run(b), "split_adjustment", "FAIL", "CCC")
    assert any("phantom" in f.detail for f in hits), hits


def test_negative_dividend_factor_fails():
    b = copy_bundle(CLEAN)
    b.adj_close.iloc[350:, b.adj_close.columns.get_loc("DDD")] *= 0.97
    hits = grab(run(b), "adjustment_factor", "FAIL", "DDD")
    assert any("negative-dividend" in f.detail for f in hits), hits


def test_duplicate_date_row_fails():
    b = copy_bundle(CLEAN)
    for f in PriceBundle.FIELDS:
        df = getattr(b, f)
        setattr(b, f, pd.concat([df, df.iloc[[100]]]).sort_index())
    assert grab(run(b), "calendar_alignment", "FAIL")


def test_stale_price_run_fails():
    b = copy_bundle(CLEAN)
    c = b.close.columns.get_loc("EEE")
    pinned = b.close.iloc[200, c]
    b.close.iloc[200:212, c] = pinned
    b.adj_close.iloc[200:212, b.adj_close.columns.get_loc("EEE")] = pinned
    hits = grab(run(b), "stale_prices", "FAIL", "EEE")
    assert any("pinned" in f.detail for f in hits), hits


def test_frozen_full_row_warns():
    b = copy_bundle(CLEAN)
    for f in PriceBundle.FIELDS:
        df = getattr(b, f)
        c = df.columns.get_loc("FFF")
        df.iloc[600:603, c] = df.iloc[599, c]
    hits = grab(run(b), "duplicate_rows", "WARN", "FFF")
    assert any("repeated 4 consecutive days" in f.detail for f in hits), hits


def test_zero_volume_with_price_move():
    b = copy_bundle(CLEAN)
    b.volume.iloc[450:453, b.volume.columns.get_loc("AAA")] = 0.0
    hits = grab(run(b), "zero_volume", ticker="AAA")
    assert any(f.severity == "WARN" and "price moved" in f.detail
               for f in hits), hits


def test_one_day_spike_that_reverts_fails():
    b = copy_bundle(CLEAN)
    for f in ["open", "high", "low", "adj_close", "close"]:
        df = getattr(b, f)
        df.iloc[700, df.columns.get_loc("BBB")] *= 1.8
    findings = run(b)
    hits = grab(findings, "extreme_returns", "FAIL", "BBB")
    assert any("bad print" in f.detail for f in hits), hits
    assert not grab(findings, "split_adjustment", "FAIL", "BBB"), \
        "a non-split-shaped spike must not be misread as a missed split"


def test_persistent_jump_warns_not_fails():
    b = copy_bundle(CLEAN)
    for f in ["open", "high", "low", "adj_close", "close"]:
        df = getattr(b, f)
        df.iloc[710:, df.columns.get_loc("CCC")] *= 1.6  # sticks forever
    hits = grab(run(b), "extreme_returns", ticker="CCC")
    assert hits and all(f.severity != "FAIL" for f in hits), \
        "a repricing that persists is WARN (verify), never auto-FAIL"


def test_silent_delisting_fails():
    b = copy_bundle(CLEAN)
    n = len(b.adj_close)
    for f in PriceBundle.FIELDS:
        getattr(b, f).iloc[n - 30:, b.adj_close.columns.get_loc("CCC")] = np.nan
    hits = grab(run(b), "delisting", "FAIL", "CCC")
    assert any("delisted or feed broke" in f.detail for f in hits), hits


def test_dropped_trading_day_fails():
    b = copy_bundle(CLEAN)
    victim = b.adj_close.index[150]
    for f in PriceBundle.FIELDS:
        setattr(b, f, getattr(b, f).drop(victim))
    hits = grab(run(b), "calendar_gaps", "FAIL")
    assert any(str(victim.date()) == f.date for f in hits), hits


def test_weekend_row_fails():
    b = copy_bundle(CLEAN)
    sat = b.adj_close.index[100] + pd.Timedelta(5 - b.adj_close.index[100].dayofweek, unit="D")
    assert sat.dayofweek == 5
    for f in PriceBundle.FIELDS:
        df = getattr(b, f)
        row = df.iloc[[100]].copy()
        row.index = [sat]
        setattr(b, f, pd.concat([df, row]).sort_index())
    hits = grab(run(b), "calendar_alignment", "FAIL")
    assert any("weekend" in f.detail for f in hits), hits


def test_column_missing_from_one_file_fails():
    b = copy_bundle(CLEAN)
    b.volume = b.volume.drop(columns=["DDD"])
    hits = grab(run(b), "symbol_mapping", "FAIL", "DDD")
    assert any("missing from volume.csv" in f.detail for f in hits), hits


def test_all_nan_column_fails():
    b = copy_bundle(CLEAN)
    b.adj_close["HHH"] = np.nan
    hits = grab(run(b), "symbol_mapping", "FAIL", "HHH")
    assert any("no data" in f.detail for f in hits), hits


def test_universe_mismatch():
    findings = run_all(copy_bundle(CLEAN),
                       universe=UNIVERSE | {"ZZZ"}).findings
    assert grab(findings, "symbol_mapping", "FAIL", "ZZZ")
    findings = run_all(copy_bundle(CLEAN),
                       universe=UNIVERSE - {"AAA"}).findings
    assert grab(findings, "symbol_mapping", "WARN", "AAA")


def test_ohlc_violation_flagged():
    b = copy_bundle(CLEAN)
    c = b.high.columns.get_loc("EEE")
    b.high.iloc[320, c] = b.low.iloc[320, c] * 0.5  # high below low
    hits = grab(run(b), "ohlc_consistency", ticker="EEE")
    assert any(f.severity == "FAIL" for f in hits), hits


# ------------------------------------------- premise / factor-ledger checks
def test_factor_rescale_fails_when_raw_split_adjusted():
    b = copy_bundle(CLEAN)
    b.raw_split_adjusted = True  # yfinance-cache premise
    b.adj_close.iloc[450:, b.adj_close.columns.get_loc("AAA")] *= 2.0
    hits = grab(run(b), "adjustment_factor", ticker="AAA")
    assert any(f.severity == "FAIL" and "final bar" in f.detail
               for f in hits), "factor must be anchored at 1.0 on last bar"
    assert any(f.severity == "WARN" and "files disagree" in f.detail
               for f in hits), \
        "split-shaped factor step must not be exempt under the premise"


def test_subtolerance_factor_drip_fails():
    b = copy_bundle(CLEAN)
    c = b.adj_close.columns.get_loc("EEE")
    n = len(b.adj_close)
    drip = np.full(n, 1.0 + 8e-6)  # each day inside FACTOR_NOISE_TOL,
    b.adj_close.iloc[:, c] *= np.cumprod(drip)  # ~0.6% compounded
    hits = grab(run(b), "adjustment_factor", "FAIL", "EEE")
    assert any("unexplained" in f.detail for f in hits), \
        "a sub-tolerance drip must be caught by the conservation ledger"


def test_missing_dividends_on_expected_payer_fails():
    findings = run_all(copy_bundle(CLEAN), universe=UNIVERSE,
                       expect_dividends={"EEE", "AAA"}).findings
    assert grab(findings, "dividend_presence", "FAIL", "EEE"), \
        "EEE never distributes: total-return data degraded silently"
    assert not grab(findings, "dividend_presence", ticker="AAA"), \
        "AAA pays quarterly - must not be flagged"


def test_volume_unit_break_fails():
    b = copy_bundle(CLEAN)
    b.volume.iloc[300:, b.volume.columns.get_loc("FFF")] *= 1000.0
    hits = grab(run(b), "volume_scale", "FAIL", "FFF")
    assert any("stepped" in f.detail for f in hits), hits


def _assert_shift_fails_lead_lag(shift: int, day: str) -> None:
    b = copy_bundle(CLEAN)
    for f in PriceBundle.FIELDS:
        df = getattr(b, f)
        df["AAA"] = df["AAA"].shift(shift)
    hits = grab(run(b), "lead_lag", "FAIL", "AAA")
    assert any("shifted" in f.detail and day in f.detail for f in hits), \
        "a whole-history one-day shift is look-ahead poison"


def test_shifted_series_fails_lead_lag():
    _assert_shift_fails_lead_lag(1, "previous")  # stale by one day


def test_future_shifted_series_fails_lead_lag():
    _assert_shift_fails_lead_lag(-1, "next")  # leaks tomorrow


def test_interpolated_segment_warns():
    b = copy_bundle(CLEAN)
    c = b.adj_close.columns.get_loc("EEE")
    lo_v, hi_v = b.adj_close.iloc[500, c], b.adj_close.iloc[521, c]
    ramp = np.linspace(lo_v, hi_v, 22)
    b.adj_close.iloc[500:522, c] = ramp
    b.close.iloc[500:522, b.close.columns.get_loc("EEE")] = ramp
    hits = grab(run(b), "stale_prices", "WARN", "EEE")
    assert any("smoothed/interpolated" in f.detail for f in hits), \
        "vol collapse must catch interpolation that never repeats a price"


def test_sub_gate_glitch_attributed_to_print_day():
    # EEE (no dividends/splits) is re-levered to ~2.5x its daily moves, so
    # a -25% print sits below BOTH gates (30% idio; 15x trailing vol) while
    # the +33% bounce trips the idio gate: only the previous-day attribution
    # branch can put the FAIL on the print day.
    b = copy_bundle(CLEAN)
    t = "EEE"
    r = b.adj_close[t].pct_change().fillna(0.0)
    scale = (1.0 + 2.5 * r).cumprod() / (1.0 + r).cumprod()
    for f in ["open", "high", "low", "adj_close", "close"]:
        getattr(b, f)[t] *= scale
    d_pos = 600
    for f in ["open", "high", "low", "adj_close", "close"]:
        df = getattr(b, f)
        df.iloc[d_pos, df.columns.get_loc(t)] *= 0.75  # -25%: below gate
    glitch_day = str(b.adj_close.index[d_pos].date())
    hits = grab(run(b), "extreme_returns", "FAIL", t)
    assert any(f.date == glitch_day and "fully reversed by the next day's" in f.detail
               for f in hits), \
        "the FAIL must land on the glitch day, not the bounce day"


def test_premise_mode_missed_split_in_close_fails():
    # yfinance premise: close.csv is split-adjusted at source, so GGG's
    # tape-price split jump means close.csv missed a split adj_close applied
    b = copy_bundle(CLEAN)
    b.raw_split_adjusted = True
    hits = grab(run(b), "split_adjustment", "FAIL", SPLIT_TICKER)
    assert any("missed a split" in f.detail for f in hits), hits


def test_clean_bundle_premise_mode_has_no_fail_or_warn():
    b = copy_bundle(CLEAN)
    b.raw_split_adjusted = True
    c = b.close.columns.get_loc(SPLIT_TICKER)
    b.close.iloc[:SPLIT_POS, c] /= SPLIT_RATIO  # split-adjust the raw close
    bad = [f for f in run(b) if f.severity in ("FAIL", "WARN")]
    assert not bad, bad


# ------------------------------------------- carried rows / the live edge
def test_single_reserved_record_warns():
    b = copy_bundle(CLEAN)
    for f in PriceBundle.FIELDS:
        df = getattr(b, f)
        c = df.columns.get_loc("FFF")
        df.iloc[600, c] = df.iloc[599, c]
    findings = run(b)
    hits = grab(findings, "duplicate_rows", "WARN", "FFF")
    assert any("repeated 2 consecutive days" in f.detail for f in hits), \
        "one exact repeat of all five fields is a re-served record"
    assert not grab(findings, "carried_rows"), "one ticker is not the board"


def test_market_wide_carried_row_fails():
    for pos in (600, -1):
        b = copy_bundle(CLEAN)
        for f in PriceBundle.FIELDS:
            df = getattr(b, f)
            df.iloc[pos] = df.iloc[pos - 1].to_numpy()
        hits = grab(run(b), "carried_rows")
        assert [(f.severity, f.date) for f in hits] == [("FAIL", day(b, pos))], hits
        assert ("latest bar" in hits[0].detail) == (pos == -1), hits


def test_carried_prices_with_zero_volume_still_fail():
    # the placeholder bar: flat at the previous close, no volume
    b = copy_bundle(CLEAN)
    for f in ["open", "high", "low", "adj_close"]:
        getattr(b, f).iloc[-1] = b.adj_close.iloc[-2].to_numpy()
    b.close.iloc[-1] = b.close.iloc[-2].to_numpy()
    b.volume.iloc[-1] = 0.0
    findings = run(b)
    assert grab(findings, "carried_rows", "FAIL"), findings
    assert not grab(findings, "duplicate_rows"), \
        "zero volume: not a re-served record, only the board-wide check sees it"


def test_carried_row_needs_a_material_share_of_the_board():
    def carried(n_tickers):
        b = copy_bundle(CLEAN)
        cols = [b.close.columns.get_loc(t) for t in TICKERS[:n_tickers]]
        b.close.iloc[300, cols] = b.close.iloc[299, cols].to_numpy()
        return grab(run(b), "carried_rows")
    assert not carried(4), "below CARRIED_ROW_MIN_TICKERS unchanged closes"
    assert carried(quality.CARRIED_ROW_MIN_TICKERS)


def test_ticker_missing_latest_bars_warns_within_grace():
    for k in (1, quality.DELIST_GRACE_DAYS, quality.DELIST_GRACE_DAYS + 1):
        b = copy_bundle(CLEAN)
        blank(b, "CCC", slice(len(b.adj_close) - k, None))
        findings = run(b)
        hits = grab(findings, "delisting", ticker="CCC")
        if k <= quality.DELIST_GRACE_DAYS:
            assert [f.severity for f in hits] == ["WARN"], (k, hits)
            assert f"latest {k} row(s)" in hits[0].detail, hits
            assert hits[0].date == day(b, -k - 1), "keyed on the last valid bar"
            assert not grab(findings, "delisting", "FAIL"), (k, findings)
        else:
            assert [f.severity for f in hits] == ["FAIL"], (k, hits)


def test_adj_close_only_missing_on_latest_bar_warns():
    b = copy_bundle(CLEAN)
    b.adj_close.iloc[-1, b.adj_close.columns.get_loc("DDD")] = np.nan
    assert grab(run(b), "delisting", "WARN", "DDD"), \
        "the other five panels cannot vouch for a missing adjusted close"


def test_partial_final_row_fails():
    def final_row_missing(tickers):
        b = copy_bundle(CLEAN)
        for t in tickers:
            blank(b, t, -1)
        return grab(run(b), "delisting", "FAIL")
    assert not final_row_missing(["AAA"]), "one late print is a WARN"
    hits = final_row_missing(["AAA", "BBB"])
    assert len(hits) == 1 and hits[0].ticker == "", hits
    assert "2 of 8 live tickers" in hits[0].detail, hits
    assert hits[0].date == day(CLEAN, -1), hits


def test_empty_row_fails_like_a_dropped_day():
    for pos in (150, -1):
        b = copy_bundle(CLEAN)
        for f in PriceBundle.FIELDS:
            getattr(b, f).iloc[pos] = np.nan
        hits = grab(run(b), "calendar_gaps", "FAIL")
        assert any(f.date == day(b, pos) and "empty trading day" in f.detail
                   for f in hits), hits


# ------------------------------------------------- extreme returns: edges
def _gspc(idx, shocks=None):
    r = np.clip(np.random.default_rng(11).normal(0.0, 0.004, len(idx)),
                -0.01, 0.01)
    for pos, shock in (shocks or {}).items():
        r[pos] = shock
    return pd.DataFrame({"^GSPC": 3000.0 * np.cumprod(1.0 + r)}, index=idx)


def _scale_row(b: PriceBundle, pos: int, mult: float) -> None:
    for f in PRICE_FIELDS:
        getattr(b, f).iloc[pos] *= mult


def test_confirmed_sub_threshold_outlier_stays_info():
    b = copy_bundle(CLEAN)
    scale_prices(b, "BBB", slice(710, None), 0.75)
    hits = [f for f in grab(run(b), "extreme_returns", ticker="BBB")
            if f.date == day(b, 710)]
    assert [f.severity for f in hits] == ["INFO"], hits
    assert "persisted next day" in hits[0].detail, hits


def test_latest_bar_extreme_print_warns_as_unconfirmed():
    b = copy_bundle(CLEAN)
    scale_prices(b, "BBB", -1, 0.75)
    hits = [f for f in grab(run(b), "extreme_returns", ticker="BBB")
            if f.date == day(b, -1)]
    assert [f.severity for f in hits] == ["WARN"], \
        "nothing has confirmed or reversed a print on the latest bar"
    assert "latest bar" in hits[0].detail, hits
    assert "persisted" not in hits[0].detail, "there is no next day yet"


def test_latest_bar_market_wide_move_needs_the_index_to_stay_info():
    def last_bar(indices):
        b = copy_bundle(CLEAN)
        _scale_row(b, -1, 0.75)
        b.indices = indices(b.adj_close.index)
        hits = [f for f in grab(run(b), "extreme_returns")
                if f.date == day(b, -1)]
        assert len(hits) == len(TICKERS), hits
        assert not any("persisted" in f.detail for f in hits), hits
        return hits
    confirmed = last_bar(lambda idx: _gspc(idx, {len(idx) - 1: -0.25}))
    assert {f.severity for f in confirmed} == {"INFO"}, \
        "an index-confirmed crash on the latest bar must not spam WARN"
    quiet = last_bar(_gspc)
    assert {f.severity for f in quiet} == {"WARN"}, quiet
    assert all("row-level corruption" in f.detail for f in quiet), quiet


def test_row_level_corruption_fails_when_the_index_is_quiet():
    b = copy_bundle(CLEAN)
    b.indices = _gspc(b.adj_close.index)
    _scale_row(b, 700, 0.70)
    findings = grab(run(b), "extreme_returns")
    hits = [f for f in findings if f.severity == "FAIL" and f.date == day(b, 700)]
    assert {f.ticker for f in hits} == set(TICKERS), hits
    assert all("bad print" in f.detail and "row-level corruption" in f.detail
               for f in hits), hits
    assert not any("crisis whipsaw" in f.detail for f in findings), \
        "the panels must not certify their own exemption"


def test_crisis_whipsaw_confirmed_by_the_index_stays_info():
    b = copy_bundle(CLEAN)
    b.indices = _gspc(b.adj_close.index, {700: -0.30, 701: 1 / 0.70 - 1})
    _scale_row(b, 700, 0.70)
    findings = grab(run(b), "extreme_returns")
    assert not [f for f in findings if f.severity == "FAIL"], findings
    whipsaw = {f.ticker for f in findings if "crisis whipsaw" in f.detail}
    assert whipsaw == set(TICKERS), whipsaw


def test_market_wide_reversal_without_an_index_keeps_the_panel_only_rule():
    b = copy_bundle(CLEAN)
    _scale_row(b, 700, 0.70)
    findings = grab(run(b), "extreme_returns")
    assert not [f for f in findings if f.severity == "FAIL"], findings
    assert any("crisis whipsaw" in f.detail for f in findings), findings


# ---------------------------------------------- returns across missing bars
def test_bridged_returns_equal_pct_change_where_there_is_no_hole():
    px = CLEAN.adj_close  # HHH starts late: leading NaNs, no internal hole
    pd.testing.assert_frame_equal(quality._bridged_returns(px),
                                  px.pct_change(fill_method=None))
    holed = px.copy()
    holed.iloc[399, 1] = np.nan
    r = quality._bridged_returns(holed)
    assert np.isnan(r.iloc[399, 1])
    assert np.isclose(r.iloc[400, 1], px.iloc[400, 1] / px.iloc[398, 1] - 1.0)


def test_missed_split_right_after_a_missing_bar_still_fails():
    b = copy_bundle(CLEAN)
    blank(b, "BBB", 399)
    scale_prices(b, "BBB", slice(400, None), 0.5)
    hits = grab(run(b), "split_adjustment", "FAIL", "BBB")
    assert any(f.date == day(b, 400) and "NOT adjusted" in f.detail
               and "1-day data gap" in f.detail for f in hits), hits


def test_phantom_adjustment_right_after_a_missing_bar_still_fails():
    b = copy_bundle(CLEAN)
    c = b.adj_close.columns.get_loc("CCC")
    b.adj_close.iloc[499, c] = np.nan
    b.adj_close.iloc[500:, c] *= 0.5
    hits = grab(run(b), "split_adjustment", "FAIL", "CCC")
    assert any(f.date == day(b, 500) and "phantom" in f.detail
               for f in hits), hits


def test_spike_next_to_a_missing_bar_is_still_a_bad_print():
    for hole in (699, 701):  # the bar before the spike, then the bar after
        b = copy_bundle(CLEAN)
        blank(b, "BBB", hole)
        scale_prices(b, "BBB", 700, 1.8)
        hits = grab(run(b), "extreme_returns", "FAIL", "BBB")
        assert any(f.date == day(b, 700) and "bad print" in f.detail
                   and "1-day data gap" in f.detail for f in hits), (hole, hits)


def test_handled_split_right_after_a_missing_bar_is_still_info():
    b = copy_bundle(CLEAN)
    blank(b, SPLIT_TICKER, SPLIT_POS - 1)
    findings = run(b)
    assert grab(findings, "split_adjustment", "INFO", SPLIT_TICKER), findings
    assert not grab(findings, "split_adjustment", "FAIL"), findings


def test_plain_missing_bar_adds_only_missing_price_warnings():
    b = copy_bundle(CLEAN)
    blank(b, "BBB", 399)
    bad = [f for f in run(b) if f.severity in ("FAIL", "WARN")]
    assert bad and {f.check for f in bad} == {"missing_prices"}, bad


# ------------------------------------------------- split-shaped WARN hints
def test_missed_split_on_a_big_move_day_warns_and_names_the_ratio():
    b = copy_bundle(CLEAN)
    scale_prices(b, "BBB", slice(400, None), 0.5 * 1.05)
    findings = run(b)
    assert not grab(findings, "split_adjustment", "FAIL", "BBB"), \
        "outside SPLIT_RATIO_TOL a missed split is not provable from prices"
    hits = grab(findings, "extreme_returns", "WARN", "BBB")
    assert any("unadjusted 2:1 split" in f.detail for f in hits), hits


def test_missed_three_for_two_split_warns_and_names_the_ratio():
    b = copy_bundle(CLEAN)
    scale_prices(b, "BBB", slice(400, None), 2 / 3)
    hits = grab(run(b), "extreme_returns", "WARN", "BBB")
    assert any("unadjusted 3:2 split" in f.detail for f in hits), hits


def test_persistent_jump_far_from_any_ratio_gets_no_split_hint():
    b = copy_bundle(CLEAN)
    scale_prices(b, "CCC", slice(710, None), 1.6)
    hits = grab(run(b), "extreme_returns", "WARN", "CCC")
    assert hits and not any("split" in f.detail for f in hits), hits


def test_split_ratio_match_is_nearest_not_first():
    match, hint = quality._match_split_ratio, quality.SPLIT_HINT_TOL
    assert match(2.0) == 2.0 and match(0.125) == 0.125
    assert match(2.06) is None, "the FAIL tolerance stays at SPLIT_RATIO_TOL"
    assert match(2.06, hint) == 2.0
    assert match(1.30, hint) == 4 / 3, "5:4 is also within 8% but further away"


# ------------------------------------------------------------ volume scale
def test_volume_step_between_warn_and_fail_warns():
    b = copy_bundle(CLEAN)
    b.volume.iloc[300:, b.volume.columns.get_loc("FFF")] *= 60.0
    findings = run(b)
    assert grab(findings, "volume_scale", "WARN", "FFF"), findings
    assert not grab(findings, "volume_scale", "FAIL"), \
        "a 60x step is below VOLUME_STEP_FAIL"


def test_split_sized_volume_step_is_out_of_scope():
    # documented limit: a 20:1 split left in old shares (the largest in the
    # default universe) stays under VOLUME_STEP_WARN, so the check is silent;
    # lowering the threshold must be a deliberate change, not an accident
    for ratio in (4.0, 10.0, 20.0):
        b = copy_bundle(CLEAN)
        b.volume.iloc[:300, b.volume.columns.get_loc("FFF")] /= ratio
        assert not grab(run(b), "volume_scale"), ratio


def _indices(idx):
    rng = np.random.default_rng(3)
    return pd.DataFrame({"^VIX": 18 + np.cumsum(rng.normal(0, 0.3, len(idx))).clip(-10, 30),
                         "^IRX": 1.5 + 0.01 * (np.arange(len(idx)) % 7)}, index=idx)


def test_indices_pinned_vix_warns_and_out_of_range_rate_fails():
    b = copy_bundle(CLEAN)
    b.indices = _indices(b.adj_close.index)
    assert not grab(run(b), "indices")  # clean series: no findings
    b.indices.iloc[300:310, 0] = 21.5  # VIX pinned for 10 days
    b.indices.iloc[400, 1] = 60.0  # T-bill yield of 60%
    found = run(b)
    assert grab(found, "indices", "WARN", "^VIX"), found
    assert grab(found, "indices", "FAIL", "^IRX"), found


def test_indices_holes_and_closed_day_rows_warn():
    b = copy_bundle(CLEAN)
    b.indices = _indices(b.adj_close.index)
    b.indices.iloc[500:502, 0] = np.nan  # two missing VIX prints
    columbus = pd.Timestamp("2021-10-11")  # Treasury market closed, NYSE open
    b.indices.loc[columbus, "^IRX"] = np.nan
    closed = pd.Timestamp("2021-05-31")  # Memorial Day row with a value
    b.indices.loc[closed] = [20.0, 1.5]
    b.indices = b.indices.sort_index()
    found = grab(run(b), "indices", "WARN")
    assert any(f.ticker == "^VIX" and "2 missing" in f.detail for f in found), found
    assert not [f for f in found if f.ticker == "^IRX"], "bond holiday is not a hole"
    assert any(f.date == "2021-05-31" and "NYSE-closed" in f.detail for f in found), found


def test_header_only_indices_fail_per_series_without_crashing():
    # e.g. every index download failed: download_data writes a header-only file
    b = copy_bundle(CLEAN)
    b.indices = _indices(b.adj_close.index).iloc[0:0]
    found = grab(run(b), "indices", "FAIL")
    assert {f.ticker for f in found} == {"^VIX", "^IRX"}, found
    assert all(f.detail == "no data" for f in found)


def test_index_series_lagging_the_price_cache_warns_except_on_bond_holidays():
    b = copy_bundle(CLEAN)
    b.indices = _indices(b.adj_close.index)
    b.indices.iloc[-1, 0] = np.nan  # no ^VIX print on the latest bar
    hits = grab(run(b), "indices", "WARN", "^VIX")
    assert any("ends 1 days before price cache" in f.detail for f in hits), hits
    # a cache that ends on Columbus Day: the Treasury market was closed, so
    # a missing ^IRX print is not a lag - a missing ^VIX print still is
    columbus = pd.Timestamp("2021-10-11")
    b = copy_bundle(CLEAN)
    for f in PriceBundle.FIELDS:
        setattr(b, f, getattr(b, f).loc[:columbus])
    b.indices = _indices(b.adj_close.index)
    b.indices.loc[columbus] = np.nan
    found = grab(run(b), "indices")
    assert not [f for f in found if f.ticker == "^IRX"], found
    assert any(f.ticker == "^VIX" and "ends 1 days" in f.detail
               for f in found), found


def test_required_index_series_missing_or_file_absent_fails():
    b = copy_bundle(CLEAN)
    assert not grab(run(b), "indices"), "hand-built bundles may omit indices"
    b.required_indices = ("^VIX", "^IRX")
    hits = grab(run(b), "indices")
    assert [(f.severity, f.ticker) for f in hits] == [("FAIL", "")], hits
    assert "indices.csv missing" in hits[0].detail, hits
    b.indices = _indices(b.adj_close.index)
    assert not grab(run(b), "indices")
    b.indices = b.indices.drop(columns=["^IRX"])
    hits = grab(run(b), "indices")
    assert [(f.severity, f.ticker) for f in hits] == [("FAIL", "^IRX")], hits


def test_indices_duplicate_unsorted_and_weekend_dates_fail():
    def alignment(mutate):
        b = copy_bundle(CLEAN)
        b.indices = mutate(_indices(b.adj_close.index))
        return [f.detail for f in grab(run(b), "calendar_alignment", "FAIL")]

    def weekend(x):
        row = x.iloc[[100]].copy()
        row.index = [x.index[100] + pd.Timedelta(5 - x.index[100].dayofweek, unit="D")]
        return pd.concat([x, row]).sort_index()

    def swapped(x):
        order = list(range(len(x)))
        order[-1], order[-2] = order[-2], order[-1]
        return x.iloc[order]

    assert alignment(lambda x: x) == []
    assert alignment(lambda x: x.iloc[::2]) == [], \
        "a sparser index calendar is not misalignment"
    assert alignment(lambda x: pd.concat([x, x.iloc[[100]]]).sort_index()) \
        == ["indices.csv: duplicate date row"]
    assert alignment(swapped) == ["indices.csv: index not sorted"]
    assert alignment(weekend) == ["indices.csv: weekend date in index"]


def test_load_requires_the_registered_index_universe():
    saved = qdata.DATA_DIR
    with tempfile.TemporaryDirectory() as tmp:
        qdata.DATA_DIR = Path(tmp)
        try:
            for f in PriceBundle.FIELDS:
                getattr(CLEAN, f).to_csv(Path(tmp) / f"{f}.csv")
            b = PriceBundle.load()
            assert b.indices is None
            assert b.required_indices == tuple(qdata.INDEX_UNIVERSE)
            rep = run_all(b, universe=UNIVERSE)
            hits = grab(rep.findings, "indices")
            assert [(f.severity, f.ticker) for f in hits] == [("FAIL", "")], \
                "a cache without indices.csv must not pass"
            assert rep.worst() == "FAIL"
            _indices(CLEAN.adj_close.index).to_csv(Path(tmp) / "indices.csv")
            found = run_all(PriceBundle.load(), universe=UNIVERSE).findings
            missing = {f.ticker for f in grab(found, "indices", "FAIL")}
            assert missing == set(qdata.INDEX_UNIVERSE) - {"^VIX", "^IRX"}, missing
        finally:
            qdata.DATA_DIR = saved


def _rate_findings(irx_level, mutate):
    b = copy_bundle(CLEAN)
    b.indices = _indices(b.adj_close.index)
    tick = np.arange(len(b.indices))
    b.indices["^IRX"] = irx_level + 0.001 * (tick % 7)
    b.indices["^TNX"] = 4.0 + 0.01 * (tick % 5)
    mutate(b.indices)
    return [f for f in grab(run(b), "indices") if f.ticker in ("^IRX", "^TNX")]


def test_frozen_rate_series_warn_except_short_zirp_pins():
    def pins(*spec):
        def mutate(ind):
            for col, start, length, value in spec:
                ind.iloc[start:start + length, ind.columns.get_loc(col)] = value
        return mutate
    assert not _rate_findings(1.5, pins(("^IRX", 100, 9, 1.52),
                                        ("^TNX", 300, 4, 4.005)))
    hits = _rate_findings(1.5, pins(("^IRX", 100, 10, 1.52),
                                    ("^TNX", 300, 5, 4.005)))
    assert sorted((f.severity, f.ticker) for f in hits) == \
        [("WARN", "^IRX"), ("WARN", "^TNX")], hits
    assert all(("for 10 days" if f.ticker == "^IRX" else "for 5 days")
               in f.detail for f in hits), hits
    # at the zero bound bills really do pin for weeks
    assert not _rate_findings(0.02, pins(("^IRX", 500, 29, 0.012)))
    hits = _rate_findings(0.02, pins(("^IRX", 500, 30, 0.012)))
    assert [(f.severity, f.ticker) for f in hits] == [("WARN", "^IRX")], hits
    assert "for 30 days" in hits[0].detail, hits


def test_rate_unit_break_fails_and_large_jump_warns():
    def step(col, pos, delta):
        def mutate(ind):
            ind.iloc[pos:, ind.columns.get_loc(col)] += delta
        return mutate

    def to_decimal(ind):
        ind.iloc[600:, ind.columns.get_loc("^IRX")] /= 100.0

    assert not _rate_findings(5.0, step("^IRX", 400, -0.85)), \
        "a crisis-sized one-day move in the bill yield is real"
    hits = _rate_findings(5.0, step("^TNX", 500, 2.0))
    assert [(f.severity, f.ticker, f.date) for f in hits] == \
        [("WARN", "^TNX", day(CLEAN, 500))], hits
    hits = _rate_findings(5.0, to_decimal)
    assert [(f.severity, f.ticker, f.date) for f in hits] == \
        [("FAIL", "^IRX", day(CLEAN, 600))], hits
    assert "unit change" in hits[0].detail, hits


# -------------------------------------------------------------- known events
def test_known_events_downgrade_warn_but_never_fail():
    rep = DQReport(findings=[
        Finding("extreme_returns", "WARN", "MS", "2008-10-13", "x"),
        Finding("extreme_returns", "WARN", "C", "2008-11-24", "y"),
        Finding("split_adjustment", "FAIL", "MS", "2008-10-13", "z"),
    ])
    known = pd.DataFrame([
        {"check": "extreme_returns", "ticker": "MS", "date": "2008-10-13",
         "note": "real +87% day"},
        {"check": "split_adjustment", "ticker": "MS", "date": "2008-10-13",
         "note": "trying to silence a FAIL"},
        {"check": "extreme_returns", "ticker": "GONE", "date": "1999-01-04",
         "note": "stale row that matches nothing"},
    ])
    n, dead = apply_known_events(rep, known)
    assert n == 1
    sev = {(f.check, f.ticker): f.severity for f in rep.findings}
    assert sev[("extreme_returns", "MS")] == "INFO"
    assert sev[("extreme_returns", "C")] == "WARN", "unlisted stays WARN"
    assert sev[("split_adjustment", "MS")] == "FAIL", \
        "a FAIL must never be acknowledgeable"
    assert rep.worst() == "FAIL"
    assert ("extreme_returns", "GONE", "1999-01-04") in dead, \
        "dead acknowledgments must be surfaced, not silently ignored"


def test_known_events_survive_nan_and_numeric_keys():
    rep = DQReport(findings=[
        Finding("stale_prices", "WARN", "SHY", "", "aggregate"),
    ])
    known = pd.DataFrame([{"check": "stale_prices", "ticker": "SHY",
                           "date": np.nan, "note": "quantized ticks"}])
    n, _ = apply_known_events(rep, known)
    assert n == 1, "NaN date in the CSV must match a finding with date=''"


def test_known_events_match_bare_year_keys_however_read_csv_types_them():
    def findings():
        return DQReport(findings=[
            Finding("calendar_gaps", "WARN", "", "2001", "short year"),
            Finding("stale_prices", "WARN", "SHY", "", "aggregate")])
    body = "check,ticker,date,note\ncalendar_gaps,,2001,closure year\n"
    as_int = pd.read_csv(io.StringIO(body))
    assert as_int["date"].dtype.kind == "i"
    assert apply_known_events(findings(), as_int) == (1, [])
    more = body + "stale_prices,SHY,,quantized\n"
    as_float = pd.read_csv(io.StringIO(more))
    assert as_float["date"].dtype.kind == "f", "the empty cell forces float"
    rep = findings()
    assert apply_known_events(rep, as_float) == (2, []), \
        "2001.0 must still match the finding keyed '2001'"
    assert all(f.severity == "INFO" for f in rep.findings)
    as_str = pd.read_csv(io.StringIO(more), dtype=str, keep_default_na=False)
    assert apply_known_events(findings(), as_str) == (2, [])


def test_known_event_is_pinned_to_the_adjudicated_value():
    key = ("extreme_returns", "NFLX", "2022-04-20")

    def outcome(value, expect, with_column=True):
        rep = DQReport(findings=[Finding(key[0], "WARN", key[1], key[2],
                                         "moved", value=value)])
        row = {"check": key[0], "ticker": key[1], "date": key[2],
               "note": "real earnings move"}
        if with_column:
            row["expect"] = expect
        n, dead = apply_known_events(rep, pd.DataFrame([row]))
        assert dead == [], "a changed event is not a dead acknowledgment"
        return n, rep.findings[0]

    n, f = outcome(-0.3512, "-0.3512")
    assert (n, f.severity) == (1, "INFO") and "[acknowledged: " in f.detail
    # the same key now measures a different move (a halved series)
    n, f = outcome(-0.6756, "-0.3512")
    assert (n, f.severity) == (0, "WARN"), "a changed event is a NEW anomaly"
    assert "CHANGED" in f.detail and "-0.3512" in f.detail \
        and "-0.6756" in f.detail, f.detail
    # tolerance: max(1 point, 5% of the adjudicated value) = 1.76 points here
    assert outcome(-0.3352, "-0.3512")[1].severity == "INFO"
    assert outcome(-0.3712, "-0.3512")[1].severity == "WARN"
    # run lengths are effectively exact (5% of 5 is a quarter of a day)
    assert outcome(5.0, "5")[1].severity == "INFO"
    assert outcome(6.0, "5")[1].severity == "WARN"
    # anything unreadable fails closed
    assert outcome(-0.3512, "-35.1%")[1].severity == "WARN"
    assert outcome(None, "-0.3512")[1].severity == "WARN"
    # a blank expect, or no expect column, matches on the key alone
    assert outcome(-0.6756, "")[1].severity == "INFO"
    assert outcome(-0.6756, None, with_column=False)[1].severity == "INFO"


def test_acknowledged_jump_does_not_hide_a_rescaled_series():
    def jump(mult):
        b = copy_bundle(CLEAN)
        scale_prices(b, "CCC", slice(710, None), mult)
        rep = run_all(b, universe=UNIVERSE)
        hit, = [f for f in rep.findings if f.check == "extreme_returns"
                and f.ticker == "CCC" and f.date == day(b, 710)]
        return rep, hit

    _, seen = jump(1.6)
    assert seen.severity == "WARN" and abs(seen.value - 0.6) < 0.05, seen
    known = pd.DataFrame([{"check": "extreme_returns", "ticker": "CCC",
                           "date": seen.date, "note": "verified real move",
                           "expect": f"{seen.value:.4f}"}])
    rep, hit = jump(1.6)
    assert apply_known_events(rep, known) == (1, []) and hit.severity == "INFO"
    rep, hit = jump(2.2)  # the same bar now carries a different move
    assert apply_known_events(rep, known) == (0, [])
    assert hit.severity == "WARN" and "CHANGED" in hit.detail, hit


def test_finding_value_is_reported_only_when_measured():
    assert "value" not in Finding("delisting", "INFO", "HHH", "", "x").as_dict()
    assert Finding("extreme_returns", "WARN", "CCC", "2022-10-27", "y",
                   value=0.6).as_dict()["value"] == 0.6


# -------------------------------------------------------------- fundamentals
def build_clean_fundamentals() -> pd.DataFrame:
    rows = []
    for t in ["AAA", "BBB", "CCC"]:
        for k, pe in enumerate(pd.date_range("2020-03-31", periods=8,
                                             freq="QE")):
            rows.append({"ticker": t, "period_end": pe,
                         "report_date": pe + pd.Timedelta(45, unit="D"),
                         "revenue": 1e9 * (1.02 ** k) * (1 + hash(t) % 3),
                         "eps": 1.0 + 0.05 * k})
    return pd.DataFrame(rows)


def test_clean_fundamentals_pass():
    f = check_fundamentals(build_clean_fundamentals(),
                           snapshot_date="2022-06-30")
    bad = [x for x in f if x.severity in ("FAIL", "WARN")]
    assert not bad, [x.detail for x in bad]


def test_fund_duplicate_period_fails():
    df = build_clean_fundamentals()
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    assert grab(check_fundamentals(df), "fund_duplicates", "FAIL")


def test_fund_lookahead_fails():
    df = build_clean_fundamentals()
    df.loc[2, "report_date"] = df.loc[2, "period_end"] - pd.Timedelta(10, unit="D")
    hits = grab(check_fundamentals(df), "fund_lookahead", "FAIL")
    assert any("look-ahead" in f.detail for f in hits), hits


def test_fund_zero_filing_lag_fails():
    df = build_clean_fundamentals()
    df["report_date"] = df["period_end"]  # vendor default: no real date
    hits = grab(check_fundamentals(df), "fund_lookahead", "FAIL")
    assert len(hits) == len(df), "every row is usable the day the books close"
    assert all("equals period end" in f.detail for f in hits), hits


def test_fund_implausibly_short_filing_lag_warns():
    df = build_clean_fundamentals()
    df.loc[2, "report_date"] = df.loc[2, "period_end"] + pd.Timedelta(1, unit="D")
    hits = grab(check_fundamentals(df), "fund_lookahead")
    assert [(f.severity, f.ticker) for f in hits] == [("WARN", "AAA")], hits
    assert "1d filing lag" in hits[0].detail, hits
    df.loc[2, "report_date"] = df.loc[2, "period_end"] \
        + pd.Timedelta(quality.REPORT_LAG_MIN_D, unit="D")
    assert not grab(check_fundamentals(df), "fund_lookahead")


def test_fund_missing_period_end_warns():
    df = build_clean_fundamentals()
    df.loc[7, "period_end"] = pd.NaT  # AAA's latest row
    hits = grab(check_fundamentals(df), "fund_gaps", "WARN", "AAA")
    assert any("missing period_end" in f.detail for f in hits), hits


def test_fund_unit_break_after_missing_value_fails():
    df = build_clean_fundamentals()
    df.loc[3, "revenue"] = np.nan
    assert not grab(check_fundamentals(df), "fund_units"), \
        "a lone missing value is not a unit break"
    df.loc[4:7, "revenue"] *= 1000  # AAA switches units right after the gap
    hits = grab(check_fundamentals(df), "fund_units", "FAIL", "AAA")
    assert len(hits) == 1 and "unit change" in hits[0].detail, hits


def test_fund_missing_report_date_column_warns():
    df = build_clean_fundamentals().drop(columns=["report_date"])
    assert grab(check_fundamentals(df), "fund_lookahead", "WARN")


def test_fund_unit_break_fails():
    df = build_clean_fundamentals()
    df.loc[(df.ticker == "AAA") & (df.period_end == "2021-03-31"),
           "revenue"] *= 1000  # millions -> thousands switch
    hits = grab(check_fundamentals(df), "fund_units", "FAIL", "AAA")
    assert any("unit change" in f.detail for f in hits), hits


def test_fund_negative_revenue_fails():
    df = build_clean_fundamentals()
    df.loc[5, "revenue"] = -df.loc[5, "revenue"]
    assert grab(check_fundamentals(df), "fund_signs", "FAIL")


def test_fund_copy_forward_warns():
    df = build_clean_fundamentals()
    mask = df.ticker == "BBB"
    idx = df.index[mask][2:5]
    df.loc[idx, ["revenue", "eps"]] = df.loc[df.index[mask][2],
                                             ["revenue", "eps"]].values
    assert grab(check_fundamentals(df), "fund_stale", "WARN", "BBB")


def test_fund_skipped_quarter_warns():
    df = build_clean_fundamentals()
    df = df[~((df.ticker == "CCC") & (df.period_end == "2020-12-31"))]
    assert grab(check_fundamentals(df), "fund_gaps", "WARN", "CCC")


def test_fund_future_period_fails():
    df = build_clean_fundamentals()
    hits = grab(check_fundamentals(df, snapshot_date="2021-06-30"),
                "fund_lookahead", "FAIL")
    assert any("after snapshot" in f.detail for f in hits), hits


def test_fund_unpublished_at_snapshot_fails():
    df = build_clean_fundamentals()
    # snapshot after Q4-2020 period end but before its report date
    hits = grab(check_fundamentals(df, snapshot_date="2021-01-15"),
                "fund_lookahead", "FAIL")
    assert any("not public" in f.detail for f in hits), \
        "period ended but numbers unpublished at snapshot = look-ahead"


def test_fund_eps_sign_swing_not_a_unit_break():
    df = build_clean_fundamentals()
    df.loc[df.ticker == "AAA", "eps"] = \
        [-5.59, 0.31, -0.02, 2.1, -8.0, 0.05, 1.0, 1.1]
    assert not grab(check_fundamentals(df), "fund_units", ticker="AAA"), \
        "signed per-share metrics must be exempt from the unit-ratio test"


def test_fund_copy_forward_detected_despite_nan_column():
    df = build_clean_fundamentals()
    df["one_metric_always_nan"] = np.nan
    mask = df.ticker == "BBB"
    idx = df.index[mask][2:5]
    df.loc[idx, ["revenue", "eps"]] = df.loc[df.index[mask][2],
                                             ["revenue", "eps"]].values
    assert grab(check_fundamentals(df), "fund_stale", "WARN", "BBB"), \
        "an always-NaN column must not disable copy-forward detection"


def test_malformed_fundamentals_contained():
    rep = run_all(copy_bundle(CLEAN), universe=UNIVERSE,
                  fundamentals=pd.DataFrame({"ticker": ["A"],
                                             "periodend": ["2020-03-31"]}))
    crash = [f for f in rep.findings if f.check == "check_fundamentals"]
    assert crash and crash[0].severity == "FAIL", \
        "a malformed fundamentals file must FAIL the report, not crash it"


# --------------------------------------------------------------------- main
def main() -> None:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()


def test_infinity_in_raw_close_fails_even_with_missing_ohlc():
    b = copy_bundle(CLEAN)
    b.close.iloc[20, 0] = np.inf
    b.open.iloc[20, 0] = np.nan
    assert grab(run(b), "numeric_values", "FAIL", "AAA")


def test_infinite_volume_fails():
    b = copy_bundle(CLEAN)
    b.volume.iloc[20, 0] = np.inf
    assert grab(run(b), "numeric_values", "FAIL", "AAA")
