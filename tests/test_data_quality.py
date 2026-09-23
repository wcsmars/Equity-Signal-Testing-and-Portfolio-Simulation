"""Synthetic-corruption tests for qcore.quality.

Strategy: build a CLEAN synthetic OHLCV bundle (GBM paths, consistent
adjusted OHLC, real dividend/split adjustment mechanics), assert it produces
zero FAIL/WARN findings (false-positive guard), then inject one known defect
per test into a fresh copy and assert the matching check - and only a
sensible set of checks - fires. A DQ framework that cries wolf gets ignored;
one that misses a planted defect is worse than none.

Run: python3 tests/test_data_quality.py   (also pytest-compatible)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    sat = b.adj_close.index[100] + pd.Timedelta(days=5 - b.adj_close.index[100].dayofweek)
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


def test_shifted_series_fails_lead_lag():
    b = copy_bundle(CLEAN)
    for f in PriceBundle.FIELDS:
        df = getattr(b, f)
        df["AAA"] = df["AAA"].shift(1)
    hits = grab(run(b), "lead_lag", "FAIL", "AAA")
    assert any("shifted" in f.detail for f in hits), \
        "a whole-history one-day shift is look-ahead poison"


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
    b = copy_bundle(CLEAN)
    d_pos = 600
    for f in ["open", "high", "low", "adj_close", "close"]:
        df = getattr(b, f)
        df.iloc[d_pos, df.columns.get_loc("DDD")] *= 0.75  # -25%: below gate
    glitch_day = str(b.adj_close.index[d_pos].date())
    hits = grab(run(b), "extreme_returns", "FAIL", "DDD")
    assert any(f.date == glitch_day and "bad print" in f.detail
               for f in hits), \
        "the FAIL must land on the glitch day, not the bounce day"


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


# -------------------------------------------------------------- fundamentals
def build_clean_fundamentals() -> pd.DataFrame:
    rows = []
    for t in ["AAA", "BBB", "CCC"]:
        for k, pe in enumerate(pd.date_range("2020-03-31", periods=8,
                                             freq="QE")):
            rows.append({"ticker": t, "period_end": pe,
                         "report_date": pe + pd.Timedelta(days=45),
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
    df.loc[2, "report_date"] = df.loc[2, "period_end"] - pd.Timedelta(days=10)
    hits = grab(check_fundamentals(df), "fund_lookahead", "FAIL")
    assert any("look-ahead" in f.detail for f in hits), hits


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
