"""Validate cached OHLCV data and optional point-in-time fundamentals.

The expected cache contains adjusted open/high/low/adj_close, a dividend-
unadjusted close, volume, and index series. For yfinance, close is still
split-adjusted; set PriceBundle.raw_split_adjusted=False for actual tape
prices from another source. Checks use close/adj_close adjustment factors,
price/volume consistency, calendars, stale data, gaps and filing dates.

FAIL identifies defects that should block analysis; WARN requires review;
INFO records expected artifacts. Passing these checks is not a guarantee
that vendor data is accurate or free of every bias.

Run scripts/data_quality.py for a report, or call run_all(PriceBundle.load()).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
    USMartinLutherKingJr, USMemorialDay, USPresidentsDay, USThanksgivingDay,
    nearest_workday, sunday_to_monday)

# ---------------------------------------------------------------- thresholds
# Registered defaults; every check takes overrides for research use.
MISSING_INTERNAL_FAIL_PCT = 0.01   # >1% of a ticker's life missing mid-series
MISSING_GAP_FAIL_DAYS = 5          # one contiguous internal hole this long
STALE_RUN_WARN = 5                 # identical raw closes in a row
STALE_RUN_FAIL = 10
FROZEN_ROW_WARN = 3                # identical full OHLCV rows in a row
ZERO_RET_FRAC_INFO = 0.10          # share of exactly-unchanged closes (marks
                                   # are quantized/illiquid - count, don't gate)
VOL_COLLAPSE_FRAC = 0.10           # 10d vol under 10% of 1y median = smoothed
VOL_COLLAPSE_MIN_SIGMA = 0.003     # ...only for instruments that usually move
EXTREME_IDIO_WARN = 0.30           # |idiosyncratic daily return|
EXTREME_SIGMA_K = 15               # vol-scaled companion gate (low-vol names)
EXTREME_SIGMA_FLOOR = 0.01
SPLIT_RATIO_TOL = 0.025            # relative distance to a candidate ratio
SPLIT_RAW_JUMP = 0.40              # |raw return| that demands an explanation
SPLIT_ADJ_QUIET = 0.10             # adj return small => adjustment worked
FACTOR_NOISE_TOL = 1e-5            # adjustment-factor noise tolerance
FACTOR_ANCHOR_TOL = 1e-3           # |factor-1| allowed on the final bar
FACTOR_DRIFT_TOL = 0.005           # unexplained cumulative factor drift
DIVIDEND_MAX_YIELD = 0.25          # one-day payout above this is implausible
TTM_DIV_YIELD_WARN = 0.20          # trailing-12m implied payout cap
DIV_MAX_GAP_DAYS = 550             # expected payers: max days between events
ZERO_VOL_MOVE_PCT = 0.001          # price moved despite zero volume
ZERO_VOL_FAIL_DAYS = 10
VOLUME_STEP_WARN = 30.0            # 20d median volume ratio across a break
VOLUME_STEP_FAIL = 100.0           # ...thousands-vs-shares style unit error
DELIST_GRACE_DAYS = 5              # trailing rows a ticker may lag the file
OHLC_REL_TOL = 1e-3                # rounding slack for low<=px<=high
CAL_YEAR_DAYS = (249, 254)         # plausible trading days per full year
LEADLAG_MARGIN = 0.10              # lagged |corr| beats contemporaneous by
LEADLAG_MIN_CORR = 0.25            # ...and is material -> series is shifted

# Tickers whose adj_close MUST show dividend events (bond ETFs and broad
# index funds always distribute); a dividend-free stretch here means the
# cache silently degraded from total-return to price-return data.
EXPECT_DIVIDENDS = {
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG", "EMB",
    "SPY", "QQQ", "IWM", "DIA", "MDY", "EFA", "EEM", "VGK",
}

# Ratios a real split can take. Forward splits include 3:2 / 4:3 / 5:4;
# reverse splits are INTEGER-only (1:8 GE, 1:10 C) - fractional reverse
# splits do not exist, and admitting them (e.g. 2:3) would misread real
# +50%/+33% earnings moves as splits (AMD +52.3% on 2016-04-22 is a real
# print, not a botched 2:3 reverse split).
_INT_RATIOS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50]
_SPLIT_CANDIDATES = sorted(
    _INT_RATIOS + [1.5, 4 / 3, 5 / 4] + [1.0 / r for r in _INT_RATIOS])

# Full-day closures that are NOT holidays and must not count as data gaps.
SPECIAL_CLOSURES = pd.DatetimeIndex([
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",  # 9/11
    "2004-06-11",  # Reagan mourning
    "2007-01-02",  # Ford mourning
    "2012-10-29", "2012-10-30",  # hurricane Sandy
    "2018-12-05",  # G.H.W. Bush mourning
    "2025-01-09",  # Carter mourning
])

SEVERITIES = ["ok", "INFO", "WARN", "FAIL"]


class _NYSEHolidays(AbstractHolidayCalendar):
    """Approximate NYSE holiday rules (see calendar.py: no official file yet).

    Correct for regular holidays 2000+; special closures live in
    SPECIAL_CLOSURES. New Year's uses sunday_to_monday because the NYSE
    does not observe Jan 1 falling on a Saturday (e.g. stayed open
    2021-12-31).
    """
    rules = [
        Holiday("NewYears", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19",
                observance=nearest_workday),
        Holiday("July4", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def nyse_holidays(start, end) -> pd.DatetimeIndex:
    return _NYSEHolidays().holidays(pd.Timestamp(start), pd.Timestamp(end))


def nyse_bdays(start, end) -> pd.DatetimeIndex:
    """Expected NYSE trading days (approximate holiday rules)."""
    days = pd.bdate_range(start, end)
    drop = nyse_holidays(start, end).union(SPECIAL_CLOSURES)
    return days.difference(drop)


# ------------------------------------------------------------------ findings
@dataclass
class Finding:
    check: str
    severity: str            # INFO | WARN | FAIL
    ticker: str              # "" for file-level findings
    date: str                # ISO date or range, "" if not date-specific
    detail: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PriceBundle:
    """The six wide frames, index-aligned as stored (NOT reindexed here -
    misalignment between files is itself a finding).

    raw_split_adjusted: True for this repo's yfinance cache, where
    close.csv already has splits applied, so the adj/close factor may only
    move on dividends. Set False for vendors whose raw close is the actual
    tape price (then split-shaped factor jumps are the healthy signature
    of a handled split)."""
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    adj_close: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    indices: pd.DataFrame | None = None
    raw_split_adjusted: bool = True

    FIELDS = ["open", "high", "low", "adj_close", "close", "volume"]

    @classmethod
    def load(cls) -> "PriceBundle":
        from qcore.data import load
        kw = {f: load(f) for f in cls.FIELDS}
        try:
            kw["indices"] = load("indices")
        except FileNotFoundError:
            pass
        return cls(**kw)

    def frames(self) -> dict[str, pd.DataFrame]:
        return {f: getattr(self, f) for f in self.FIELDS}


@dataclass
class DQReport:
    findings: list[Finding] = field(default_factory=list)

    def worst(self) -> str:
        sev = {"ok"} | {f.severity for f in self.findings}
        return max(sev, key=SEVERITIES.index)

    def counts(self) -> pd.DataFrame:
        """check x severity counts for the dashboard."""
        if not self.findings:
            return pd.DataFrame(columns=["FAIL", "WARN", "INFO"])
        df = pd.DataFrame([f.as_dict() for f in self.findings])
        out = df.pivot_table(index="check", columns="severity", values="detail",
                             aggfunc="count", fill_value=0)
        return out.reindex(columns=["FAIL", "WARN", "INFO"], fill_value=0)

    def by_severity(self, severity: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    def as_dict(self) -> dict:
        return {"worst": self.worst(),
                "n_findings": len(self.findings),
                "findings": [f.as_dict() for f in self.findings]}


# ------------------------------------------------------------------- helpers
def _runs_of_equal(s: pd.Series) -> pd.DataFrame:
    """Runs of consecutive identical values in s (NaNs break runs - a gap
    must not weld two shorter runs into one long one).
    Returns DataFrame[value, start, end, length] for runs of length >= 2."""
    if s.notna().sum() < 2:
        return pd.DataFrame(columns=["value", "start", "end", "length"])
    brk = (s != s.shift()) | s.isna() | s.shift().isna()
    grp = brk.cumsum()[s.notna()]
    sv = s.dropna()
    agg = sv.groupby(grp).agg(["first", "size"])
    idx = sv.index.to_series().groupby(grp).agg(["first", "last"])
    out = pd.DataFrame({"value": agg["first"], "start": idx["first"],
                        "end": idx["last"], "length": agg["size"]})
    return out[out["length"] >= 2]


def _match_split_ratio(x: float, tol: float = SPLIT_RATIO_TOL) -> float | None:
    """The candidate split ratio nearest to x (>1 forward, <1 reverse),
    or None if nothing is within tol."""
    if not np.isfinite(x) or x <= 0:
        return None
    for cand in _SPLIT_CANDIDATES:
        if abs(x - cand) / cand < tol:
            return cand
    return None


def _life(s: pd.Series) -> pd.Series:
    """The slice of s between its first and last valid observations."""
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None:
        return s.iloc[0:0]
    return s.loc[first:last]


def _fmt_d(ts) -> str:
    return str(pd.Timestamp(ts).date())


# -------------------------------------------------------------------- checks
def check_calendar_alignment(b: PriceBundle) -> list[Finding]:
    """All six files must share one index and one column set; the index must
    be unique, sorted, weekday-only. Misalignment silently breaks any code
    that combines fields positionally."""
    out = []
    ref_name = "adj_close"
    ref = b.adj_close
    for name, df in b.frames().items():
        dup = df.index[df.index.duplicated()]
        for d in dup.unique()[:10]:
            out.append(Finding("calendar_alignment", "FAIL", "", _fmt_d(d),
                               f"{name}.csv: duplicate date row"))
        if not df.index.is_monotonic_increasing:
            out.append(Finding("calendar_alignment", "FAIL", "", "",
                               f"{name}.csv: index not sorted"))
        wk = df.index[df.index.dayofweek >= 5]
        for d in wk[:10]:
            out.append(Finding("calendar_alignment", "FAIL", "", _fmt_d(d),
                               f"{name}.csv: weekend date in index"))
        if name == ref_name:
            continue
        if not df.index.equals(ref.index):
            miss = ref.index.difference(df.index)
            extra = df.index.difference(ref.index)
            out.append(Finding(
                "calendar_alignment", "FAIL", "", "",
                f"{name}.csv index != {ref_name}.csv: "
                f"{len(miss)} missing, {len(extra)} extra dates "
                f"(e.g. {[_fmt_d(d) for d in miss[:3].tolist() + extra[:3].tolist()]})"))
    return out


def check_calendar_gaps(b: PriceBundle) -> list[Finding]:
    """Every expected NYSE trading day must be present, and no non-trading
    day may be present. A silently dropped day shifts every lagged signal;
    an extra day (half-session artifacts, bad merges) double-counts."""
    out = []
    idx = b.adj_close.index
    if len(idx) == 0:
        return [Finding("calendar_gaps", "FAIL", "", "", "empty price index")]
    expected = nyse_bdays(idx[0], idx[-1])
    missing = expected.difference(idx)
    for d in missing:
        out.append(Finding("calendar_gaps", "FAIL", "", _fmt_d(d),
                           "expected trading day absent from cache"))
    hol = nyse_holidays(idx[0], idx[-1]).union(SPECIAL_CLOSURES)
    extra = idx.intersection(hol)
    for d in extra:
        out.append(Finding("calendar_gaps", "WARN", "", _fmt_d(d),
                           "row exists on a market holiday/closure"))
    # per-year row-count sanity catches systematic calendar drift
    counts = idx.to_series().groupby(idx.year).size()
    for yr, n in counts.items():
        if yr in (idx[0].year, idx[-1].year):
            continue  # partial edge years
        lo, hi = CAL_YEAR_DAYS
        special = ((SPECIAL_CLOSURES.year == yr).sum()
                   if len(SPECIAL_CLOSURES) else 0)
        if not (lo - special <= n <= hi):
            out.append(Finding("calendar_gaps", "WARN", "", str(yr),
                               f"{n} trading days in year (expected "
                               f"{lo - special}-{hi})"))
    return out


def check_missing_prices(b: PriceBundle) -> list[Finding]:
    """NaN holes INSIDE a ticker's listed life, in EVERY field. Pre-inception
    and post-delisting NaN blocks are structural (see check_delisting); a
    hole in the middle means the feed dropped data the backtest will
    forward-fill or misalign over. A hole in close.csv alone silently breaks
    every raw-close consumer (dividend_yields, withholding); a hole in
    open/high/low alone breaks range/gap logic - so each field is scanned
    against the adj_close-defined life. Severity: any hole is WARN; a hole
    exceeding 1% of life OR a single contiguous gap of 5+ days (a feed
    outage, however old the ticker) is FAIL."""
    out = []
    frames = b.frames()
    for t in b.adj_close.columns:
        life = _life(b.adj_close[t])
        if len(life) == 0:
            continue  # all-NaN column -> symbol_mapping finding
        for name, df in frames.items():
            if t not in df.columns:
                continue  # symbol_mapping's finding
            s = df[t].reindex(life.index)
            holes = s.isna()
            n = int(holes.sum())
            if n == 0:
                continue
            gap = int(holes.groupby((~holes).cumsum()).sum().max())
            pct = n / len(life)
            sev = ("FAIL" if pct > MISSING_INTERNAL_FAIL_PCT
                   or gap >= MISSING_GAP_FAIL_DAYS else "WARN")
            d = holes[holes].index
            out.append(Finding(
                "missing_prices", sev, t,
                f"{_fmt_d(d[0])}..{_fmt_d(d[-1])}",
                f"{name}: {n} missing values inside listed life "
                f"({pct:.2%}, longest gap {gap} days)"))
    return out


def check_duplicate_rows(b: PriceBundle) -> list[Finding]:
    """Frozen full rows: identical (open, high, low, close, volume) on
    consecutive days. One coincidence is possible; a run means the feed
    re-served yesterday's record. Requires volume > 0 - an untraded holiday
    carry-forward with zero volume is the zero_volume check's business."""
    out = []
    frames = b.frames()
    common = b.adj_close.columns
    for f in frames.values():
        common = common.intersection(f.columns)
    for t in common:
        same = pd.Series(True, index=b.adj_close.index)
        for name in ["open", "high", "low", "close", "volume"]:
            s = frames[name][t]
            same &= s.eq(s.shift()) & s.notna()
        same &= b.volume[t] > 0
        if not same.any():
            continue
        # extend runs: `same` marks the 2nd+ day of a frozen streak
        grp = (~same).cumsum()
        streaks = same.groupby(grp).sum()
        streaks = streaks[streaks >= FROZEN_ROW_WARN - 1]
        idx = b.adj_close.index
        for g, n in streaks.items():
            days = same[(grp == g) & same].index
            start = idx[max(idx.get_loc(days[0]) - 1, 0)]
            out.append(Finding(
                "duplicate_rows", "WARN", t,
                f"{_fmt_d(start)}..{_fmt_d(days[-1])}",
                f"full OHLCV row repeated {int(n) + 1} consecutive days "
                f"with nonzero volume"))
    return out


def check_stale_prices(b: PriceBundle) -> list[Finding]:
    """Staleness in three flavors.

    1. Runs of identical RAW closes: a frozen feed manufactures fake
       zero-vol returns and understates risk (raw close is used because
       adjustment noise would mask repeats in adj_close).
    2. Realized-vol collapse: an interpolated/smoothed segment never
       repeats a price exactly, but its 10-day vol drops to a fraction of
       the ticker's normal vol. Only applied to instruments that normally
       move (trailing median vol >= 0.3%/day) - SHY sitting still is life,
       SPY sitting still is a defect.
    3. Aggregate: the share of exactly-unchanged closes over the whole
       life (INFO only - tick-quantized/illiquid marks are real data, but
       a backtest should know the marks are coarse)."""
    out = []
    ret = b.adj_close.pct_change(fill_method=None)
    for t in b.close.columns:
        life = _life(b.close[t])
        runs = _runs_of_equal(life)
        runs = runs[runs["length"] >= STALE_RUN_WARN]
        for _, r in runs.iterrows():
            sev = "FAIL" if r["length"] >= STALE_RUN_FAIL else "WARN"
            out.append(Finding(
                "stale_prices", sev, t,
                f"{_fmt_d(r['start'])}..{_fmt_d(r['end'])}",
                f"close pinned at {r['value']:.4g} for {int(r['length'])} days"))
        if len(life) > 40:
            frac = float((life.diff() == 0).sum() / max(len(life) - 1, 1))
            if frac > ZERO_RET_FRAC_INFO:
                out.append(Finding(
                    "stale_prices", "INFO", t, "",
                    f"{frac:.0%} of all closes exactly unchanged: coarse/"
                    "illiquid marks, treat backtest fills with suspicion"))
        if t not in ret.columns:
            continue
        r = ret[t].loc[life.index[0]:life.index[-1]] if len(life) else ret[t]
        sig10 = r.rolling(10, min_periods=8).std()
        med1y = r.rolling(252, min_periods=126).std().median()
        if med1y and np.isfinite(med1y) and med1y >= VOL_COLLAPSE_MIN_SIGMA:
            calm = sig10 < VOL_COLLAPSE_FRAC * med1y
            runs2 = _runs_of_equal(calm.astype(float))
            runs2 = runs2[(runs2["value"] == 1.0) & (runs2["length"] >= 5)]
            for _, rr in runs2.iterrows():
                out.append(Finding(
                    "stale_prices", "WARN", t,
                    f"{_fmt_d(rr['start'])}..{_fmt_d(rr['end'])}",
                    f"10d vol collapsed to <{VOL_COLLAPSE_FRAC:.0%} of the "
                    f"1y norm ({med1y:.2%}/d) for {int(rr['length'])} days: "
                    "smoothed/interpolated segment?"))
    return out


def check_zero_volume(b: PriceBundle) -> list[Finding]:
    """Zero volume with a MOVING price is contradictory (price changes
    require prints); zero-volume runs mean the instrument wasn't trading -
    fine for a tiny 2000s country ETF, alarming for AAPL. Move-with-no-volume
    gets the harsher treatment."""
    out = []
    common = b.volume.columns.intersection(b.adj_close.columns)
    ret = b.adj_close[common].pct_change(fill_method=None)
    for t in common:
        life = _life(b.adj_close[t])
        if len(life) == 0:
            continue
        v = b.volume[t].reindex(life.index)
        moved = (v == 0) & (ret[t].reindex(life.index).abs() > ZERO_VOL_MOVE_PCT)
        n_moved = int(moved.sum())
        if n_moved:
            sev = "FAIL" if n_moved > ZERO_VOL_FAIL_DAYS else "WARN"
            d = moved[moved].index
            out.append(Finding(
                "zero_volume", sev, t, f"{_fmt_d(d[0])}..{_fmt_d(d[-1])}",
                f"{n_moved} days with zero volume but price moved"))
        runs = _runs_of_equal((v == 0).astype(int))
        runs = runs[(runs["value"] == 1) & (runs["length"] >= 3)]
        for _, r in runs.iterrows():
            out.append(Finding(
                "zero_volume", "INFO", t,
                f"{_fmt_d(r['start'])}..{_fmt_d(r['end'])}",
                f"zero volume for {int(r['length'])} consecutive days"))
    return out


def check_extreme_returns(b: PriceBundle) -> list[Finding]:
    """Implausible one-day ADJUSTED returns, after removing the market-wide
    component (cross-sectional median return that day) so crash days don't
    light up the whole board. Split-ratio-shaped moves are excluded here -
    they belong to check_split_adjustment, which classifies them properly.

    Severity hinges on what happened NEXT: a bad print spikes and fully
    reverts the following day (FAIL - it is not a price, it is a glitch),
    while a genuine repricing sticks (WARN - verify against the tape;
    2008 financials really did move 30-90% in a day: MS +87% 2008-10-13,
    C +58% 2008-11-24, and those must not be 'cleaned' away). A hit that
    itself REVERSES a sub-threshold spike is reported as a bad print on
    the PREVIOUS day (the glitch), not as a move on the bounce day.

    A vol-scaled companion gate (15x the trailing 63d robust vol, floor
    1%) catches glitches that are huge for a low-vol instrument yet far
    below the absolute threshold; its persistent hits log as INFO so
    genuine vol outliers never spam the WARN board.

    Split-ratio-shaped moves are skipped ONLY when check_split_adjustment
    will actually claim the day (raw close co-jumped past its gate);
    otherwise a split-shaped bad print would fall between two checks."""
    out = []
    ret = b.adj_close.pct_change(fill_method=None)
    ret_raw = b.close.pct_change(fill_method=None)
    idio = ret.sub(ret.median(axis=1), axis=0)
    # pass 1: gates for every ticker, so classification can see how much
    # of the UNIVERSE was extreme on each day
    hit_cols = {}
    for t in ret.columns:
        r = ret[t]
        sig = (r - r.rolling(63, min_periods=40).median()).abs() \
            .rolling(63, min_periods=40).median() * 1.4826
        scaled_gate = np.maximum(EXTREME_SIGMA_K * sig.shift(),
                                 EXTREME_SIGMA_FLOOR)
        hit_cols[t] = ((idio[t].abs() > EXTREME_IDIO_WARN)
                       | (r.abs() > scaled_gate)) & r.notna()
    hit_df = pd.DataFrame(hit_cols, index=ret.index)
    # bad prints are idiosyncratic by nature - they do not synchronize.
    # When a chunk of the board is extreme on the same day (2020-03-16,
    # 14 tickers) OR the market median itself moved >3% (Lehman days when
    # thin bond ETFs dislocated alone: LQD -9.1% on 2008-09-29's -8%
    # tape), a V-reversal is crisis whipsaw, not a glitch.
    day_hits = hit_df.sum(axis=1)
    crisis = (day_hits >= max(5, int(0.05 * max(hit_df.shape[1], 1)))) \
        | (ret.median(axis=1).abs() > 0.03)
    crisis = (crisis | crisis.shift(1, fill_value=False)
              | crisis.shift(-1, fill_value=False))
    for t in ret.columns:
        r = ret[t]
        for d in hit_df.index[hit_df[t]]:
            r_adj, r_i = r.at[d], idio.at[d, t]
            raw_r = ret_raw[t].get(d, np.nan) if t in ret_raw.columns else np.nan
            ratio = _match_split_ratio(1.0 / (1.0 + r_adj))
            claimed_by_split = (
                ratio is not None and np.isfinite(raw_r)
                and abs(raw_r) > SPLIT_RAW_JUMP
                and abs(raw_r - r_adj) < 0.05)
            if claimed_by_split:
                continue  # split check adjudicates (missed-split FAIL)
            pos = ret.index.get_loc(d)
            r_prev = r.iloc[pos - 1] if pos > 0 else np.nan
            r_next = r.iloc[pos + 1] if pos + 1 < len(ret.index) else np.nan
            rt_next = (1.0 + r_adj) * (1.0 + r_next) - 1.0 \
                if np.isfinite(r_next) else np.nan
            rt_prev = (1.0 + r_prev) * (1.0 + r_adj) - 1.0 \
                if np.isfinite(r_prev) else np.nan
            in_crisis = bool(crisis.loc[d])
            if np.isfinite(rt_prev) and abs(rt_prev) < 0.25 * abs(r_adj) \
                    and abs(r_prev) > 0.5 * EXTREME_IDIO_WARN:
                if in_crisis:
                    out.append(Finding(
                        "extreme_returns", "INFO", t,
                        _fmt_d(ret.index[pos - 1]),
                        f"{r_prev:+.1%} then {r_adj:+.1%} inside a "
                        f"market-wide extreme window ({int(day_hits.loc[d])} "
                        "tickers): crisis whipsaw, not a bad print"))
                else:
                    out.append(Finding(
                        "extreme_returns", "FAIL", t,
                        _fmt_d(ret.index[pos - 1]),
                        f"return {r_prev:+.1%} fully reversed by the next "
                        f"day's {r_adj:+.1%}: bad print, not a price"))
            elif np.isfinite(rt_next) and abs(rt_next) < 0.25 * abs(r_adj):
                if in_crisis:
                    out.append(Finding(
                        "extreme_returns", "INFO", t, _fmt_d(d),
                        f"{r_adj:+.1%} then {r_next:+.1%} inside a "
                        f"market-wide extreme window ({int(day_hits.loc[d])} "
                        "tickers): crisis whipsaw, not a bad print"))
                else:
                    out.append(Finding(
                        "extreme_returns", "FAIL", t, _fmt_d(d),
                        f"adjusted return {r_adj:+.1%} fully reversed next "
                        f"day ({r_next:+.1%}): bad print, not a price"))
            else:
                sev = "WARN" if abs(r_i) > EXTREME_IDIO_WARN else "INFO"
                what = (f"adjusted return {r_adj:+.1%} (idiosyncratic "
                        f"{r_i:+.1%})" if abs(r_i) > EXTREME_IDIO_WARN else
                        f"return {r_adj:+.1%} is >{EXTREME_SIGMA_K}x this "
                        f"instrument's typical daily move")
                out.append(Finding(
                    "extreme_returns", sev, t, _fmt_d(d),
                    f"{what}, persisted next day: verify against the tape"))
    # one finding per (ticker, date): prev-day attribution can duplicate
    seen, dedup = set(), []
    for f in out:
        if (f.ticker, f.date, f.severity) not in seen:
            seen.add((f.ticker, f.date, f.severity))
            dedup.append(f)
    return dedup


def check_split_adjustment(b: PriceBundle) -> list[Finding]:
    """Split handling, verified from both sides.

    A real split moves the RAW close by (almost exactly) a simple ratio
    while the ADJUSTED close barely moves. So for every raw jump that looks
    like a split (or is simply huge):
      raw jumps, adj quiet          -> handled split (INFO, counted)
      raw jumps, adj jumps the same -> the adjustment MISSED the split (FAIL)
    and the mirror image:
      adj split-sized jump, raw quiet -> phantom adjustment applied to a
                                         split that never happened (FAIL)
    """
    out = []
    common = b.close.columns.intersection(b.adj_close.columns)
    r_raw = b.close[common].pct_change(fill_method=None)
    r_adj = b.adj_close[common].pct_change(fill_method=None)
    for t in common:
        raw_jump = r_raw.index[
            r_raw[t].notna() & (r_raw[t].abs() > SPLIT_RAW_JUMP)]
        for d in raw_jump:
            ratio = _match_split_ratio(1.0 / (1.0 + r_raw.at[d, t]))
            adj_r = r_adj.at[d, t]
            if np.isnan(adj_r):
                out.append(Finding("split_adjustment", "WARN", t, _fmt_d(d),
                                   f"raw close jumped {r_raw.at[d, t]:+.1%} "
                                   "but adjusted close is missing"))
            elif ratio is None:
                pass  # big but not split-shaped: extreme_returns adjudicates
                # (real crisis moves show up identically in raw and adjusted)
            elif abs(adj_r) <= SPLIT_ADJ_QUIET:
                if b.raw_split_adjusted:
                    # in this cache close.csv is split-adjusted at source,
                    # so a split-sized jump ONLY in raw means close.csv
                    # failed to apply a split that adj_close absorbed
                    out.append(Finding(
                        "split_adjustment", "FAIL", t, _fmt_d(d),
                        f"raw close jumped {r_raw.at[d, t]:+.1%} "
                        f"({ratio:g}:1-shaped) while adjusted stayed quiet: "
                        "close.csv missed a split adj_close applied"))
                else:
                    out.append(Finding(
                        "split_adjustment", "INFO", t, _fmt_d(d),
                        f"{ratio:g}:1 split correctly adjusted "
                        f"(raw {r_raw.at[d, t]:+.1%}, adj {adj_r:+.1%})"))
            elif abs(adj_r - r_raw.at[d, t]) < 0.05:
                out.append(Finding(
                    "split_adjustment", "FAIL", t, _fmt_d(d),
                    f"{ratio:g}:1 split NOT adjusted: raw "
                    f"{r_raw.at[d, t]:+.1%} and adjusted {adj_r:+.1%} "
                    "moved together"))
            else:
                out.append(Finding(
                    "split_adjustment", "WARN", t, _fmt_d(d),
                    f"raw {r_raw.at[d, t]:+.1%} vs adjusted {adj_r:+.1%}: "
                    "partial/unclear adjustment"))
        phantom = r_adj.index[
            r_adj[t].notna() & (r_adj[t].abs() > SPLIT_RAW_JUMP)
            & (r_raw[t].abs() < SPLIT_ADJ_QUIET)]
        for d in phantom:
            out.append(Finding(
                "split_adjustment", "FAIL", t, _fmt_d(d),
                f"adjusted close jumped {r_adj.at[d, t]:+.1%} while raw "
                f"moved {r_raw.at[d, t]:+.1%}: phantom adjustment"))
    return out


def check_adjustment_factor(b: PriceBundle) -> list[Finding]:
    """The cumulative adjustment factor f = adj_close / close and its
    day-over-day ratio g = f(t) / f(t-1) are the most sensitive
    bad-adjusted-close detector we have. Four invariants:

    1. Every g != 1 must be a legitimate event. With raw_split_adjusted
       (this cache) the ONLY legitimate event is a dividend:
       1 < g <= 1/(1-25%). Split-shaped g here means the two files
       disagree about a split - a defect, not an exemption. Without the
       premise, split-ratio g values are the healthy split signature.
    2. g < 1 is a NEGATIVE dividend: always corrupt (bar a genuine
       reverse split under the non-premise mode).
    3. The factor is anchored: yfinance sets adjusted == raw on the
       latest bar, so |f(last) - 1| > 0.1% means a whole-block rescale /
       mid-history anchor (silent total-return distortion).
    4. Conservation: total drift log(f_end/f_start) must be explained by
       the flagged events. A sub-tolerance daily drip (e.g. x1.0002/day)
       hides from the per-day test but not from the ledger."""
    out = []
    common = b.close.columns.intersection(b.adj_close.columns)
    f = b.adj_close[common] / b.close[common]
    for t in common:
        ft = f[t].dropna()
        if len(ft) < 2:
            continue
        bad = ft[ft <= 0]
        for d in bad.index[:5]:
            out.append(Finding("adjustment_factor", "FAIL", t, _fmt_d(d),
                               "non-positive adjustment factor"))
        if len(bad):
            continue
        g = (ft / ft.shift()).dropna()
        events = g[(g - 1.0).abs() > FACTOR_NOISE_TOL]
        dy_by_date = {}
        for d, gv in events.items():
            split_like = _match_split_ratio(gv) is not None
            if split_like and not b.raw_split_adjusted:
                continue  # split signature: adjudicated by split check
            if gv > 1.0:
                dy = 1.0 - 1.0 / gv
                if dy > DIVIDEND_MAX_YIELD:
                    out.append(Finding(
                        "adjustment_factor", "WARN", t, _fmt_d(d),
                        f"factor jump implies a {dy:.1%} one-day payout"
                        + (" (split-shaped, but close.csv is split-adjusted"
                           " at source - files disagree about a split)"
                           if split_like else "")))
                else:
                    dy_by_date[d] = dy
            else:
                out.append(Finding(
                    "adjustment_factor", "FAIL", t, _fmt_d(d),
                    f"adjustment factor fell x{gv:.4f}: negative-dividend "
                    "artifact"
                    + (" (split-shaped, but close.csv is split-adjusted at"
                       " source)" if split_like else "")))
        if b.raw_split_adjusted and abs(ft.iloc[-1] - 1.0) > FACTOR_ANCHOR_TOL:
            out.append(Finding(
                "adjustment_factor", "FAIL", t, _fmt_d(ft.index[-1]),
                f"factor on the final bar is {ft.iloc[-1]:.4f}, not 1.0: "
                "adjusted block rescaled / anchored mid-history"))
        drift = float(np.log(ft.iloc[-1] / ft.iloc[0]))
        explained = float(np.log(g.reindex(events.index)).sum()) \
            if len(events) else 0.0
        if abs(drift - explained) > FACTOR_DRIFT_TOL:
            out.append(Finding(
                "adjustment_factor", "FAIL", t, "",
                f"cumulative factor drift {np.exp(drift - explained) - 1:+.2%}"
                " unexplained by dividend/split events: creeping adjustment "
                "corruption"))
        if dy_by_date:
            dy_s = pd.Series(0.0, index=ft.index)
            for d, dy in dy_by_date.items():
                dy_s.loc[d] = dy
            ttm = dy_s.rolling(252, min_periods=1).sum()
            if (ttm > TTM_DIV_YIELD_WARN).any():
                d = ttm.index[ttm > TTM_DIV_YIELD_WARN][0]
                out.append(Finding(
                    "adjustment_factor", "WARN", t, _fmt_d(d),
                    f"implied dividend yield {ttm.loc[d]:.1%} over the "
                    "trailing year: individually-plausible payouts summing "
                    "to an implausible stream"))
    return out


def check_dividend_presence(b: PriceBundle,
                            expect_dividends: set[str] | None = None
                            ) -> list[Finding]:
    """Bond ETFs and broad index funds ALWAYS distribute. If the factor
    shows no dividend events for one of them over a long stretch, the
    cache has silently degraded from total-return to price-return data -
    a few percent a year of phantom underperformance that no other check
    can see (each daily factor ratio is a perfectly innocent 1.0).

    Gaps are measured from the FIRST observed payout onward: a young fund
    that has not started distributing yet (QQQ paid nothing 1999-2002 -
    expenses exceeded income) is not 'degraded'; one that paid and then
    went silent is."""
    out = []
    if expect_dividends is None:
        expect_dividends = EXPECT_DIVIDENDS
    common = b.close.columns.intersection(b.adj_close.columns)
    for t in sorted(expect_dividends & set(common)):
        ft = (b.adj_close[t] / b.close[t]).dropna()
        if len(ft) < 504:  # need ~2y of life to judge a payout cadence
            continue
        g = (ft / ft.shift()).dropna()
        ev = g.index[(g - 1.0) > FACTOR_NOISE_TOL]
        if len(ev) == 0:
            out.append(Finding(
                "dividend_presence", "FAIL", t,
                f"{_fmt_d(ft.index[0])}..{_fmt_d(ft.index[-1])}",
                f"zero dividend events over the whole {len(ft)}-day life "
                "of an instrument that always distributes: total-return "
                "data degraded to price-return"))
            continue
        marks = ev.append(pd.DatetimeIndex([ft.index[-1]]))
        gaps = np.diff(marks.values).astype("timedelta64[D]").astype(int)
        if len(gaps) and gaps.max() > DIV_MAX_GAP_DAYS:
            i = int(np.argmax(gaps))
            out.append(Finding(
                "dividend_presence", "FAIL", t,
                f"{_fmt_d(marks[i])}..{_fmt_d(marks[i + 1])}",
                f"payouts went silent for {int(gaps.max())} days after "
                "distributing regularly: total-return data degraded to "
                "price-return mid-history"))
    return out


def check_volume_scale(b: PriceBundle) -> list[Finding]:
    """A step change in volume MAGNITUDE (share count vs thousands, or a
    split whose volume was never re-based) wrecks liquidity filters and
    cost models while every price check stays green. Compares the 20-day
    median volume across each date; only breaks that persist (both
    medians well-formed) are flagged, and consecutive flag-days collapse
    into one finding."""
    out = []
    for t in b.volume.columns:
        v = _life(b.volume[t]).replace(0.0, np.nan)
        if v.notna().sum() < 60:
            continue
        m = v.rolling(20, min_periods=10).median()
        ratio = m / m.shift(20)
        hot = (ratio > VOLUME_STEP_WARN) | (ratio < 1.0 / VOLUME_STEP_WARN)
        runs = _runs_of_equal(hot.astype(float))
        runs = runs[(runs["value"] == 1.0) & (runs["length"] >= 5)]
        for _, r in runs.iterrows():
            seg = ratio.loc[r["start"]:r["end"]].dropna()
            worst = float(max(seg.max(), 1.0 / seg.min()))
            sev = "FAIL" if worst > VOLUME_STEP_FAIL else "WARN"
            out.append(Finding(
                "volume_scale", sev, t,
                f"{_fmt_d(r['start'])}..{_fmt_d(r['end'])}",
                f"20d median volume stepped ~x{worst:.0f} across this "
                "window: unit/scale break, not a liquidity regime"))
    return out


def check_lead_lag(b: PriceBundle) -> list[Finding]:
    """A ticker whose whole history is shifted by one day is catastrophic
    look-ahead for every cross-sectional signal, yet every per-ticker
    check passes (the series itself is pristine - it is just in the wrong
    place). Signature: its returns correlate more with YESTERDAY's (or
    tomorrow's) market than with today's."""
    out = []
    ret = b.adj_close.pct_change(fill_method=None)
    if ret.shape[1] < 5:
        return out  # need a market to compare against
    for t in ret.columns:
        r = ret[t]
        if r.notna().sum() < 250:
            continue
        mkt = ret.drop(columns=[t]).median(axis=1)
        c0 = abs(r.corr(mkt))
        cm = abs(r.corr(mkt.shift(1)))   # r reacts to yesterday's market
        cp = abs(r.corr(mkt.shift(-1)))  # r anticipates tomorrow's market
        best, day = max((cm, "previous"), (cp, "next"))
        if best > LEADLAG_MIN_CORR and best > c0 + LEADLAG_MARGIN:
            out.append(Finding(
                "lead_lag", "FAIL", t, "",
                f"returns correlate with the {day} day's market "
                f"({best:.2f}) better than the same day's ({c0:.2f}): "
                "series shifted by one day"))
    return out


def check_ohlc_consistency(b: PriceBundle) -> list[Finding]:
    """low <= {open, close} <= high, and all prices positive. open/high/low
    in this cache are ADJUSTED, so they are compared against adj_close -
    comparing them to the raw close would flag every dividend ever paid."""
    out = []
    common = b.low.columns
    for df in (b.high, b.open, b.adj_close):
        common = common.intersection(df.columns)
    lo, hi = b.low[common], b.high[common]
    tol = 1.0 + OHLC_REL_TOL
    for t in common:
        rows = pd.DataFrame({"lo": lo[t], "hi": hi[t],
                             "op": b.open[t], "cl": b.adj_close[t]}).dropna()
        for name, px in [("open", rows["op"]), ("close", rows["cl"])]:
            viol = rows.index[(px > rows["hi"] * tol) | (px < rows["lo"] / tol)]
            for d in viol[:10]:
                out.append(Finding(
                    "ohlc_consistency", "WARN", t, _fmt_d(d),
                    f"{name} {px.at[d]:.4g} outside [low {rows.at[d, 'lo']:.4g}, "
                    f"high {rows.at[d, 'hi']:.4g}]"))
        inverted = rows.index[rows["lo"] > rows["hi"] * tol]
        for d in inverted[:10]:
            out.append(Finding("ohlc_consistency", "FAIL", t, _fmt_d(d),
                               f"low {rows.at[d, 'lo']:.4g} > high "
                               f"{rows.at[d, 'hi']:.4g}"))
        nonpos = rows.index[(rows[["lo", "hi", "op", "cl"]] <= 0).any(axis=1)]
        for d in nonpos[:10]:
            out.append(Finding("ohlc_consistency", "FAIL", t, _fmt_d(d),
                               "non-positive price"))
    return out


def check_delisting(b: PriceBundle) -> list[Finding]:
    """A ticker whose data stops before the end of the file has either
    delisted (backtests must know: its 'flat forever' tail is survivorship
    poison if treated as tradeable) or silently fell out of the feed. Late
    inceptions are expected (ABBV 2013, META 2012) and only counted."""
    out = []
    idx = b.adj_close.index
    if len(idx) == 0:
        return out
    last_day = idx[-1]
    for t in b.adj_close.columns:
        s = b.adj_close[t]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            continue  # symbol_mapping's finding
        if last is not None:
            lag = len(idx[(idx > last)])
            if lag > DELIST_GRACE_DAYS:
                out.append(Finding(
                    "delisting", "FAIL", t, _fmt_d(last),
                    f"data stops {lag} trading days before file end "
                    f"({_fmt_d(last_day)}): delisted or feed broke"))
        if first != idx[0]:
            out.append(Finding(
                "delisting", "INFO", t, _fmt_d(first),
                f"first data {len(idx[idx < first])} days after file start "
                "(later inception/IPO)"))
    return out


def check_symbol_mapping(b: PriceBundle,
                         universe: set[str] | None = None) -> list[Finding]:
    """Columns must agree across the six files, match the registered
    universe in qcore.data, and actually contain data. A ticker that is
    all-NaN, or present in close.csv but missing from volume.csv, is a
    download/rename casualty (the classic: FB->META) that reindex() will
    silently turn into a column of NaNs downstream."""
    out = []
    ref = set(b.adj_close.columns)
    for name, df in b.frames().items():
        dups = df.columns[df.columns.duplicated()]
        for c in dups:
            out.append(Finding("symbol_mapping", "FAIL", str(c), "",
                               f"{name}.csv: duplicate column"))
        if name == "adj_close":
            continue
        missing = ref - set(df.columns)
        extra = set(df.columns) - ref
        for c in sorted(missing):
            out.append(Finding("symbol_mapping", "FAIL", c, "",
                               f"column missing from {name}.csv"))
        for c in sorted(extra):
            out.append(Finding("symbol_mapping", "WARN", c, "",
                               f"column only in {name}.csv"))
    if universe is None:
        from qcore.data import ETF_UNIVERSE, STOCK_UNIVERSE
        universe = set(ETF_UNIVERSE) | set(STOCK_UNIVERSE)
    for c in sorted(universe - ref):
        out.append(Finding("symbol_mapping", "FAIL", c, "",
                           "in registered universe but absent from cache"))
    for c in sorted(ref - universe):
        out.append(Finding("symbol_mapping", "WARN", c, "",
                           "in cache but not in registered universe"))
    for t in b.adj_close.columns:
        if b.adj_close[t].notna().sum() == 0:
            out.append(Finding("symbol_mapping", "FAIL", t, "",
                               "column exists but holds no data (bad symbol "
                               "or failed download)"))
    return out


def check_indices(b: PriceBundle) -> list[Finding]:
    """Sanity ranges and staleness for the auxiliary index series - a
    frozen VIX silently disables every vol-regime gate in the program."""
    out = []
    if b.indices is None:
        return out
    ranges = {"^VIX": (5, 150), "^VIX3M": (5, 150), "^IRX": (-2, 25),
              "^GSPC": (500, 50_000), "^TNX": (-1, 20)}
    # rates legitimately pin for weeks under ZIRP (^IRX at 0.00, 2011-15);
    # only diffusive series get the staleness treatment
    stale_applies = {"^VIX", "^VIX3M", "^GSPC"}
    for t in b.indices.columns:
        s = _life(b.indices[t])
        if len(s) == 0:
            out.append(Finding("indices", "FAIL", t, "", "no data"))
            continue
        lo, hi = ranges.get(t, (0, np.inf))
        bad = s[(s < lo) | (s > hi)]
        for d in bad.index[:5]:
            out.append(Finding("indices", "FAIL", t, _fmt_d(d),
                               f"value {bad.at[d]:.4g} outside sane range "
                               f"[{lo}, {hi}]"))
        if t in stale_applies:
            runs = _runs_of_equal(s)
            runs = runs[runs["length"] >= STALE_RUN_WARN]
            for _, r in runs.iterrows():
                out.append(Finding(
                    "indices", "WARN", t,
                    f"{_fmt_d(r['start'])}..{_fmt_d(r['end'])}",
                    f"value pinned at {r['value']:.4g} for "
                    f"{int(r['length'])} days"))
        lag = len(b.adj_close.index[b.adj_close.index > s.index[-1]])
        if lag > DELIST_GRACE_DAYS:
            out.append(Finding("indices", "WARN", t, _fmt_d(s.index[-1]),
                               f"series ends {lag} days before price cache"))
    return out


# -------------------------------------------------------------- fundamentals
# Metrics that can never be negative; anything else is a sign/parse error.
NONNEG_METRICS = {"revenue", "total_assets", "shares_out", "total_equity_abs",
                  "cash", "market_cap"}
UNIT_BREAK_RATIO = 100.0    # 100x between periods => thousands vs millions
UNIT_WARN_RATIO = 10.0
REPORT_LAG_WARN_D = 180
PERIOD_GAP_FACTOR = 1.6     # gap > 1.6x the ticker's median period spacing


def check_fundamentals(fund: pd.DataFrame,
                       snapshot_date=None) -> list[Finding]:
    """Validate a long-format fundamentals table.

    Expected schema (one row per ticker-period):
      ticker        str
      period_end    fiscal period end date
      report_date   date the numbers became PUBLIC (filing/press release).
                    This is the column that prevents look-ahead: a backtest
                    may use a row only after report_date.
      <metrics...>  numeric columns (revenue, eps, total_assets, ...)

    Checks: duplicate (ticker, period_end); report_date before period_end
    (impossible - guarantees look-ahead); missing report_date; implausible
    filing lag; skipped fiscal quarters; unit breaks (a 100x jump between
    consecutive periods is a thousands-vs-millions switch, not growth);
    negative values in nonneg metrics; identical metric vectors repeated
    across periods (vendor copy-forward); periods ending after the snapshot
    date (rows from the future)."""
    out = []
    fund = fund.copy()
    fund["period_end"] = pd.to_datetime(fund["period_end"])
    has_report = "report_date" in fund.columns
    if has_report:
        fund["report_date"] = pd.to_datetime(fund["report_date"])
    metric_cols = [c for c in fund.columns
                   if c not in ("ticker", "period_end", "report_date")
                   and pd.api.types.is_numeric_dtype(fund[c])]

    dup = fund.duplicated(["ticker", "period_end"], keep=False)
    for (t, p), _ in fund[dup].groupby(["ticker", "period_end"]):
        out.append(Finding("fund_duplicates", "FAIL", t, _fmt_d(p),
                           "duplicate (ticker, period_end) rows"))

    if not has_report:
        out.append(Finding("fund_lookahead", "WARN", "", "",
                           "no report_date column: backtests cannot know "
                           "when numbers became public"))
    else:
        bad = fund[fund["report_date"] < fund["period_end"]]
        for _, r in bad.iterrows():
            out.append(Finding(
                "fund_lookahead", "FAIL", r["ticker"], _fmt_d(r["period_end"]),
                f"report_date {_fmt_d(r['report_date'])} precedes period end: "
                "look-ahead guaranteed"))
        nan_rep = fund[fund["report_date"].isna()]
        for t, grp in nan_rep.groupby("ticker"):
            out.append(Finding("fund_lookahead", "WARN", t, "",
                               f"{len(grp)} rows missing report_date"))
        lag = (fund["report_date"] - fund["period_end"]).dt.days
        late = fund[lag > REPORT_LAG_WARN_D]
        for _, r in late.iterrows():
            out.append(Finding(
                "fund_lookahead", "WARN", r["ticker"], _fmt_d(r["period_end"]),
                f"{(r['report_date'] - r['period_end']).days}d filing lag"))

    if snapshot_date is not None:
        snap = pd.Timestamp(snapshot_date)
        fut = fund[fund["period_end"] > snap]
        for _, r in fut.iterrows():
            out.append(Finding("fund_lookahead", "FAIL", r["ticker"],
                               _fmt_d(r["period_end"]),
                               f"period ends after snapshot {_fmt_d(snap)}"))
        if has_report:
            # numbers not yet PUBLIC at the snapshot are just as much
            # look-ahead as periods from the future
            unpub = fund[(fund["report_date"] > snap)
                         & ~(fund["period_end"] > snap)]
            for _, r in unpub.iterrows():
                out.append(Finding(
                    "fund_lookahead", "FAIL", r["ticker"],
                    _fmt_d(r["period_end"]),
                    f"report_date {_fmt_d(r['report_date'])} is after "
                    f"snapshot {_fmt_d(snap)}: numbers were not public"))

    for t, grp in fund.sort_values("period_end").groupby("ticker"):
        gaps = grp["period_end"].diff().dt.days
        # adapt to the ticker's own cadence (quarterly ~91d, annual ~365d)
        med = gaps.median()
        if len(gaps.dropna()) >= 2 and np.isfinite(med):
            limit = max(PERIOD_GAP_FACTOR * med, 100)
            for pe, gap in zip(grp["period_end"].iloc[1:], gaps.iloc[1:]):
                if gap > limit:
                    out.append(Finding(
                        "fund_gaps", "WARN", t, _fmt_d(pe),
                        f"{gap:.0f}d since prior period (typical "
                        f"{med:.0f}d): skipped period(s)"))
        for m in metric_cols:
            v = grp[m].reset_index(drop=True)
            if m.lower() not in NONNEG_METRICS:
                # only nonneg SCALE metrics can betray a unit switch;
                # signed per-share flows (eps, net income) legitimately
                # swing 10-100x across a zero crossing
                continue
            prev = v.shift()
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (v.abs() / prev.abs()).replace([np.inf], np.nan)
            for i in ratio.index[(ratio > UNIT_BREAK_RATIO)
                                 | (ratio < 1 / UNIT_BREAK_RATIO)]:
                out.append(Finding(
                    "fund_units", "FAIL", t,
                    _fmt_d(grp["period_end"].iloc[i]),
                    f"{m} jumped x{ratio[i]:.3g} vs prior period: "
                    "suspected unit change"))
            for i in ratio.index[((ratio > UNIT_WARN_RATIO)
                                  & (ratio <= UNIT_BREAK_RATIO))
                                 | ((ratio >= 1 / UNIT_BREAK_RATIO)
                                    & (ratio < 1 / UNIT_WARN_RATIO))]:
                out.append(Finding(
                    "fund_units", "WARN", t,
                    _fmt_d(grp["period_end"].iloc[i]),
                    f"{m} jumped x{ratio[i]:.3g} vs prior period"))
            if m.lower() in NONNEG_METRICS:
                for i in v.index[v < 0]:
                    out.append(Finding(
                        "fund_signs", "FAIL", t,
                        _fmt_d(grp["period_end"].iloc[i]),
                        f"negative {m}: {v[i]:.4g}"))
        if len(metric_cols) and len(grp) >= 3:
            vec = grp[metric_cols].reset_index(drop=True)
            # NaN-tolerant equality: a permanently-missing metric column
            # must not disable copy-forward detection (NaN == NaN is False)
            eq = vec.eq(vec.shift()) | (vec.isna() & vec.shift().isna())
            same = eq.all(axis=1) & vec.notna().any(axis=1)
            run = 0
            for i, flag in enumerate(same):
                run = run + 1 if flag else 0
                if run == 2:  # 3 identical periods
                    out.append(Finding(
                        "fund_stale", "WARN", t,
                        _fmt_d(grp["period_end"].iloc[i]),
                        "identical metrics for 3+ consecutive periods: "
                        "vendor copy-forward"))
    return out


# -------------------------------------------------------------- known events
def apply_known_events(report: DQReport,
                       known: pd.DataFrame) -> tuple[int, list[tuple]]:
    """Downgrade findings a human has already adjudicated as real market
    events (not data errors) to INFO, so the steady-state dashboard is
    quiet and only NEW anomalies alarm. `known` needs columns
    check,ticker,date,note - see data/dq_known_events.csv. Never silences
    FAILs: a FAIL is a data defect by construction and must be fixed in
    the data, not acknowledged away.

    Keys are normalized to strings (read_csv turns empty cells into NaN
    and bare years into ints; either would silently dead-letter the row).
    Returns (n_downgraded, unmatched_rows) - surface unmatched rows to the
    operator: a dead acknowledgment means the finding key drifted and the
    allowlist is silently not protecting what they think it protects."""
    def norm(x) -> str:
        return "" if pd.isna(x) else str(x).strip()

    notes = {(norm(c), norm(t), norm(d)): norm(n)
             for c, t, d, n in zip(known["check"], known["ticker"],
                                   known["date"], known["note"])}
    n, matched = 0, set()
    for f in report.findings:
        k = (f.check, f.ticker, f.date)
        if k in notes:
            matched.add(k)
            if f.severity == "WARN":
                f.severity = "INFO"
                f.detail += f" [acknowledged: {notes[k]}]"
                n += 1
    unmatched = [k for k in notes if k not in matched]
    return n, unmatched


# ------------------------------------------------------------------- run_all
PRICE_CHECKS: list[Callable[[PriceBundle], list[Finding]]] = [
    check_calendar_alignment,
    check_calendar_gaps,
    check_symbol_mapping,
    check_missing_prices,
    check_duplicate_rows,
    check_stale_prices,
    check_zero_volume,
    check_volume_scale,
    check_extreme_returns,
    check_split_adjustment,
    check_adjustment_factor,
    check_dividend_presence,
    check_ohlc_consistency,
    check_delisting,
    check_lead_lag,
    check_indices,
]


def _sanitized(b: PriceBundle) -> PriceBundle:
    """Best-effort view for content checks when the index itself is broken:
    duplicate dates dropped (keep first), index sorted. The structural
    damage is still reported from the RAW bundle by the calendar checks -
    this only keeps one bad row from crashing every other check."""
    def fix(df):
        if df is None:
            return None
        if df.index.duplicated().any() or not df.index.is_monotonic_increasing:
            df = df[~df.index.duplicated(keep="first")].sort_index()
        return df
    return PriceBundle(**{f: fix(getattr(b, f)) for f in PriceBundle.FIELDS},
                       indices=fix(b.indices),
                       raw_split_adjusted=b.raw_split_adjusted)


def run_all(bundle: PriceBundle,
            fundamentals: pd.DataFrame | None = None,
            snapshot_date=None,
            universe: set[str] | None = None,
            expect_dividends: set[str] | None = None) -> DQReport:
    """Run every check; a crashing check becomes a FAIL finding rather than
    killing the report (partially-validated data must never look clean)."""
    report = DQReport()
    structural = (check_calendar_alignment, check_calendar_gaps)
    clean_view = _sanitized(bundle)
    extra_kw = {check_symbol_mapping: {"universe": universe},
                check_dividend_presence: {"expect_dividends": expect_dividends}}
    for chk in PRICE_CHECKS:
        target = bundle if chk in structural else clean_view
        try:
            report.findings.extend(chk(target, **extra_kw.get(chk, {})))
        except Exception as e:  # noqa: BLE001
            report.findings.append(Finding(
                chk.__name__, "FAIL", "", "", f"check crashed: {e!r}"))
    if fundamentals is not None:
        try:
            report.findings.extend(
                check_fundamentals(fundamentals, snapshot_date))
        except Exception as e:  # noqa: BLE001
            report.findings.append(Finding(
                "check_fundamentals", "FAIL", "", "",
                f"check crashed: {e!r} (malformed fundamentals file?)"))
    return report
