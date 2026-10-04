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
    AbstractHolidayCalendar, GoodFriday, Holiday, USColumbusDay, USLaborDay,
    USMartinLutherKingJr, USMemorialDay, USPresidentsDay, USThanksgivingDay,
    nearest_workday, sunday_to_monday)

# ---------------------------------------------------------------- thresholds
# Registered defaults; every check takes overrides for research use.
MISSING_INTERNAL_FAIL_PCT = 0.01   # >1% of a ticker's life missing mid-series
MISSING_GAP_FAIL_DAYS = 5          # one contiguous internal hole this long
STALE_RUN_WARN = 5                 # identical raw closes in a row
STALE_RUN_FAIL = 10
# Treasury-bill funds: with bill yields near zero the close does not move for
# weeks (SGOV printed the same close for 11-13 sessions three times in 2020),
# so a pinned close there is reported as WARN and is never an automatic FAIL.
CASH_LIKE_FUNDS = frozenset({"SGOV", "BIL", "SHV"})
FROZEN_ROW_WARN = 2                # identical full OHLCV rows in a row (with
                                   # nonzero volume even ONE exact repeat of
                                   # all five fields is a re-served record)
CARRIED_ROW_FAIL_FRAC = 0.25       # share of live tickers whose raw close is
                                   # exactly the previous session's (2000-2026
                                   # maximum on the default universe: 10.6%)
CARRIED_ROW_MIN_TICKERS = 5        # ...and at least this many of them
ZERO_RET_FRAC_INFO = 0.10          # share of exactly-unchanged closes (marks
                                   # are quantized/illiquid - count, don't gate)
VOL_COLLAPSE_FRAC = 0.10           # 10d vol under 10% of 1y median = smoothed
VOL_COLLAPSE_MIN_SIGMA = 0.003     # ...only for instruments that usually move
EXTREME_IDIO_WARN = 0.30           # |idiosyncratic daily return|
EXTREME_SIGMA_K = 15               # vol-scaled companion gate (low-vol names)
EXTREME_SIGMA_FLOOR = 0.01
REVERSAL_SIGMA_K = 6               # whole-bar prints below both gates: a move
REVERSAL_FLOOR = 0.04              # this many trailing sigmas (at least 4%)
                                   # that the next bar undoes, on a bar whose
                                   # range clears both neighbours' ranges
                                   # (2000-2026 on the default universe: none)
CRISIS_INDEX = "^GSPC"             # independent series that must confirm a
CRISIS_INDEX_MOVE = 0.02           # market-wide extreme day: |index return|
                                   # that day (smallest on a real such day,
                                   # 2000-2026: 2.6%)
SPLIT_RATIO_TOL = 0.025            # relative distance to a candidate ratio
SPLIT_HINT_TOL = 0.08              # wider band used ONLY to name the nearest
                                   # ratio in a WARN, never to FAIL
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
VOLUME_STEP_WARN = 30.0            # 20d median volume ratio across a break;
                                   # above any real split (<= 20:1) on purpose
VOLUME_STEP_FAIL = 100.0           # ...thousands-vs-shares style unit error
DELIST_GRACE_DAYS = 5              # trailing rows a ticker may lag the file
                                   # before it counts as delisted (a shorter
                                   # lag still WARNs: no price on the latest
                                   # bars)
LIVE_EDGE_FAIL_FRAC = 0.25         # share of live tickers with no price on the
                                   # final row that makes it a partial bar
OHLC_REL_TOL = 1e-3                # rounding slack for low<=px<=high
WICK_ABS = 0.10                    # high/low this far outside the open-close
                                   # body...
WICK_RANGE_K = 5.0                 # ...and this many trailing-median ranges
WICK_RANGE_WINDOW = 252            # sessions in that trailing median
WICK_SYSTEMIC_TICKERS = 3          # this many tickers on one day = a market
                                   # dislocation, reported once for the day
CAL_YEAR_DAYS = (249, 254)         # plausible trading days per full year
LEADLAG_MARGIN = 0.10              # lagged |corr| beats contemporaneous by
LEADLAG_MIN_CORR = 0.25            # ...and is material -> series is shifted
LEADLAG_BLOCK_MIN_OBS = 200        # returns a calendar year needs to be tested
                                   # on its own
LEADLAG_BLOCK_TAIL = 250           # trailing rows tested as one more window
LEADLAG_BLOCK_MARGIN = 0.30        # one window: lagged |rank corr| beats the
                                   # same-day one by this much (largest margin
                                   # in any window a cache of the default
                                   # universe ending on any day of 2000-2026
                                   # would have tested: 0.23)
HISTORY_LOSS_FAIL_ROWS = 5         # recorded observations a series may lose
                                   # before the loss is a FAIL
RATE_ZIRP_LEVEL = 0.25             # yield (%) at/below which bills pin for weeks
RATE_STALE_RUN_WARN = 10           # identical ^IRX prints above that level
                                   # (2000-2026 maximum: 5)
RATE_ZIRP_STALE_RUN_WARN = 30      # ...and at any level (maximum: 19, at
                                   # 0.005% in Dec 2011)
RATE_JUMP_WARN_PP = 1.5            # |change between prints|, percentage points
                                   # (2000-2026 maxima: ^IRX 0.85, ^TNX 0.47)
RATE_JUMP_FAIL_PP = 3.0            # ...percent-vs-decimal style unit break
KNOWN_EVENT_ABS_TOL = 0.01         # an acknowledgement carrying an `expect`
KNOWN_EVENT_REL_TOL = 0.05         # value holds while |value - expect| <=
                                   # max(abs, rel x |expect|)

# Tickers whose adj_close MUST show dividend events (bond, broad index,
# sector, real-estate and single-country funds that always distribute); a
# dividend-free stretch here means the cache silently degraded from
# total-return to price-return data. Every name below has no payout gap
# above DIV_MAX_GAP_DAYS in a 2000-2026 vendor history (longest: 486 days).
# Deliberately left out: EWJ, EWT, EWY, XBI and GDX (real multi-year payout
# gaps) and the commodity / currency funds (GLD, SLV, USO, UNG, DBC, UUP,
# FXE, FXY).
EXPECT_DIVIDENDS = {
    # bonds / credit
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG", "EMB",
    # broad equity
    "SPY", "QQQ", "IWM", "DIA", "MDY", "EFA", "EEM", "VGK",
    # single-country funds
    "EWA", "EWC", "EWG", "EWH", "EWU", "EWZ", "FXI",
    # US sectors / industries / real estate
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "SMH", "KRE", "XME", "XOP", "IYR", "VNQ",
}

# Expected payers whose FIRST payout arrives late in the vendor history,
# mapped to the date by which it must have appeared. Any other expected
# payer must show a payout within DIV_MAX_GAP_DAYS of its first row. A date,
# not a blanket exemption: history stripped beyond the known quiet period
# still fails.
DIV_FIRST_PAYOUT_BY = {
    "QQQ": "2003-12-31",  # paid nothing 1999-2002: expenses exceeded income
    "XLK": "2002-12-31",  # same dot-com-era pattern, first payout 2002-12
    "SMH": "2012-12-31",  # vendor history shows no payout before 2012-12
}

# Ratios a real split can take. Forward splits include 3:2 / 4:3 / 5:4.
# Their raw moves (-33% / -25% / -20%) sit below SPLIT_RAW_JUMP, so
# check_split_adjustment never sees them; check_adjustment_factor matches
# factor jumps against these candidates instead. A close.csv that misses a
# 3:2 split raises its split-shaped WARN, and a missed 4:3 its trailing-year
# yield WARN, but a missed 5:4 on a non-dividend payer passes as a 20%
# dividend. Closing that gap needs the vendor's split events.
# A split missed by BOTH series leaves the factor untouched. It is a
# split_adjustment FAIL only when the raw move lands within SPLIT_RATIO_TOL
# of a ratio (that day's own return inside about +/-2.5%); otherwise
# check_extreme_returns reports a WARN that names the nearest ratio, and a
# missed 4:3 or 5:4 cannot be told from a real -25% / -20% print at all.
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

# rate series that follow the bond-market calendar (see _BondOnlyHolidays)
BOND_ONLY_SERIES = {"^IRX", "^TNX"}


class _NYSEHolidays(AbstractHolidayCalendar):
    """Approximate NYSE holiday rules; no official exchange calendar file is
    used. Also drives qcore.calendar.confirmed_month_ends.

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


class _BondOnlyHolidays(AbstractHolidayCalendar):
    """Days the Treasury market closes while the NYSE trades: rate series
    (^IRX, ^TNX) legitimately have no print on these dates."""
    rules = [USColumbusDay,
             Holiday("VeteransDay", month=11, day=11, observance=nearest_workday)]


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
    value: float | None = None   # the number the check measured (return, run
                                 # length, yield, ratio) where there is one;
                                 # lets an acknowledgement pin what it covers

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        if d["value"] is None:
            del d["value"]
        return d


@dataclass
class PriceBundle:
    """The six wide frames, index-aligned as stored (NOT reindexed here -
    misalignment between files is itself a finding).

    raw_split_adjusted: True for this repo's yfinance cache, where
    close.csv already has splits applied, so the adj/close factor may only
    move on dividends. Set False for vendors whose raw close is the actual
    tape price (then split-shaped factor jumps are the healthy signature
    of a handled split).

    required_indices: index series that MUST be present. load() sets it to
    the registered INDEX_UNIVERSE, so a cache whose indices.csv (or one of
    its series) is missing fails; hand-built bundles default to no
    requirement and may omit indices or pass a subset."""
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    adj_close: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    indices: pd.DataFrame | None = None
    raw_split_adjusted: bool = True
    required_indices: tuple[str, ...] = ()

    FIELDS = ["open", "high", "low", "adj_close", "close", "volume"]

    @classmethod
    def load(cls) -> "PriceBundle":
        from qcore.data import INDEX_UNIVERSE, load
        # strict=False: duplicate or unsorted dates and a file with no rows
        # are findings for the checks to locate, not reasons to refuse the
        # bundle
        kw = {f: load(f, strict=False) for f in cls.FIELDS}
        try:
            kw["indices"] = load("indices", strict=False)
        except FileNotFoundError:
            pass  # check_indices reports it; the price checks still run
        return cls(**kw, required_indices=tuple(INDEX_UNIVERSE))

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
    best = min(_SPLIT_CANDIDATES, key=lambda cand: abs(x - cand) / cand)
    return best if abs(x - best) / best < tol else None


def _ratio_label(ratio: float) -> str:
    """'2:1', '3:2' or '1:8 reverse' for a split candidate."""
    if ratio < 1.0:
        return f"1:{1.0 / ratio:g} reverse"
    for num, den in ((3, 2), (4, 3), (5, 4)):
        if abs(ratio - num / den) < 1e-9:
            return f"{num}:{den}"
    return f"{ratio:g}:1"


def _life(s: pd.Series) -> pd.Series:
    """The slice of s between its first and last valid observations."""
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None:
        return s.iloc[0:0]
    return s.loc[first:last]


def _fmt_d(ts) -> str:
    return str(pd.Timestamp(ts).date())


def _bridged_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Bar-to-bar returns measured from each series' PREVIOUS VALID bar, so
    the level change across a NaN hole lands on the first bar after it
    instead of vanishing (pct_change leaves that bar NaN). Identical to
    pct_change(fill_method=None) wherever there is no hole; hole bars stay
    NaN."""
    return px / px.ffill().shift() - 1.0


def _gap_note(px: pd.Series, d) -> str:
    """' (across an N-day data gap)' when the N bars before d are missing
    inside the series' life, else ''."""
    pos = px.index.get_loc(d)
    n = 0
    while pos - n - 1 >= 0 and pd.isna(px.iloc[pos - n - 1]):
        n += 1
    if n == 0 or pos - n - 1 < 0:
        return ""  # leading NaNs are pre-inception, not a gap
    return f" (across a {n}-day data gap)"


# -------------------------------------------------------------------- checks
def check_calendar_alignment(b: PriceBundle) -> list[Finding]:
    """All six files must share one index and one column set; the index must
    be unique, sorted, weekday-only. Misalignment silently breaks any code
    that combines fields positionally. The index file gets the same
    uniqueness / order / weekday tests but not the cross-file equality (its
    calendar legitimately differs: bond-market holidays, vendor rows on
    closed days - check_indices reports those)."""
    out = []
    ref_name = "adj_close"
    ref = b.adj_close
    frames = b.frames()
    if b.indices is not None:
        frames["indices"] = b.indices
    for name, df in frames.items():
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
        if name in (ref_name, "indices"):
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
    # a row that exists but holds no price for ANY ticker is the same defect
    # as a dropped day: every lagged signal sees a hole there
    if b.adj_close.shape[1]:
        for d in idx[b.adj_close.isna().all(axis=1).to_numpy()]:
            out.append(Finding("calendar_gaps", "FAIL", "", _fmt_d(d),
                               "row present but no ticker has a price "
                               "(empty trading day)"))
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
    consecutive days. With nonzero volume even ONE exact repeat of all five
    fields means the feed re-served yesterday's record (a coincidence would
    need four prices AND the share count to match). Requires volume > 0 -
    an untraded carry-forward with zero volume is left to check_zero_volume
    and, when the whole board is carried, check_carried_rows."""
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


def check_carried_rows(b: PriceBundle) -> list[Finding]:
    """A whole session carried forward from the previous one: a large share
    of the board closes at EXACTLY the previous session's raw close on the
    same date. A stale feed or a forward-filling merge does this, and no
    per-ticker check can see a single such day - whatever the volume field
    says (a placeholder bar reports zero volume). On the latest bar it means
    targets would be sized on yesterday's prices. Keyed on the raw close
    only; universes under CARRIED_ROW_MIN_TICKERS names are left to
    check_duplicate_rows."""
    out = []
    c = b.close
    if len(c) < 2 or c.shape[1] < CARRIED_ROW_MIN_TICKERS:
        return out
    prev = c.shift()
    live = c.notna() & prev.notna()
    n_live = live.sum(axis=1)
    n_same = (c.eq(prev) & live).sum(axis=1)
    hit = (n_same >= CARRIED_ROW_MIN_TICKERS) \
        & (n_same >= CARRIED_ROW_FAIL_FRAC * n_live)
    for d in c.index[hit.to_numpy()]:
        out.append(Finding(
            "carried_rows", "FAIL", "", _fmt_d(d),
            f"{int(n_same.loc[d])} of {int(n_live.loc[d])} live tickers "
            f"({n_same.loc[d] / n_live.loc[d]:.0%}) have a raw close identical "
            "to the previous session's: row carried forward"
            + (" (this is the latest bar)" if d == c.index[-1] else "")))
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
            cash_like = t in CASH_LIKE_FUNDS
            sev = "FAIL" if r["length"] >= STALE_RUN_FAIL and not cash_like else "WARN"
            note = (" (Treasury-bill fund: plausible when bill yields are near zero)"
                    if cash_like and r["length"] >= STALE_RUN_FAIL else "")
            out.append(Finding(
                "stale_prices", sev, t,
                f"{_fmt_d(r['start'])}..{_fmt_d(r['end'])}",
                f"close pinned at {r['value']:.4g} for {int(r['length'])} days{note}",
                value=float(r["length"])))
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
                    "smoothed/interpolated segment?",
                    value=float(rr["length"])))
    return out


def check_zero_volume(b: PriceBundle) -> list[Finding]:
    """Zero volume with a MOVING price is contradictory (price changes
    require prints); zero-volume runs mean the instrument wasn't trading -
    fine for a tiny 2000s country ETF, alarming for AAPL. Move-with-no-volume
    gets the harsher treatment. The move is measured from the previous valid
    bar, so a zero-volume bar right after a missing one is still judged."""
    out = []
    common = b.volume.columns.intersection(b.adj_close.columns)
    ret = _bridged_returns(b.adj_close[common])
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
                f"{n_moved} days with zero volume but price moved",
                value=float(n_moved)))
        runs = _runs_of_equal((v == 0).astype(int))
        runs = runs[(runs["value"] == 1) & (runs["length"] >= 3)]
        for _, r in runs.iterrows():
            out.append(Finding(
                "zero_volume", "INFO", t,
                f"{_fmt_d(r['start'])}..{_fmt_d(r['end'])}",
                f"zero volume for {int(r['length'])} consecutive days"))
    return out


def check_extreme_returns(b: PriceBundle) -> list[Finding]:
    """Implausible bar-to-bar ADJUSTED returns (measured from the previous
    valid bar, so a move across a missing bar is still seen), after removing
    the market-wide component (cross-sectional median return that day) so
    crash days don't light up the whole board. Split-ratio-shaped moves are
    excluded here - they belong to check_split_adjustment, which classifies
    them properly.

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

    A WHOLE bar printed at the wrong level (open, high, low and close
    scaled together) that stays below both gates leaves every other check
    green: the bar is internally consistent and the adjustment factor is
    untouched. Its shape gives it away instead: a move beyond
    REVERSAL_SIGMA_K trailing sigmas (floor REVERSAL_FLOOR, in raw and
    idiosyncratic return alike) that the next bar undoes, on a bar whose
    entire range lies outside both neighbours' ranges. Real one-day
    whipsaws of that size trade through at least one neighbour's range, so
    they are not flagged; a genuine gap-and-return would be, which is why
    this is a WARN (it can be adjudicated in the known-events file) and not
    a FAIL. It is not applied inside a market-wide extreme window, and it
    cannot judge the latest bar.

    A hit with NO later print (the latest bar) can be neither confirmed nor
    reversed yet - a bad print looks exactly like this on the day it
    arrives. It is WARN and says so, whichever gate fired, unless the whole
    board moved with it (confirmed market-wide window).

    The market-wide ("crisis") exemption must not be certified by the data
    it excuses: a garbled or mis-scaled ROW synchronizes by construction.
    When the bundle carries CRISIS_INDEX, a market-wide extreme day counts
    as a crisis only if that index moved more than CRISIS_INDEX_MOVE the
    same day; otherwise its V-reversals stay bad prints. Without the index
    (or without its print that day) the panel-only rule applies. A corrupt
    row inside a genuine, index-confirmed window is still excused.

    Split-ratio-shaped moves are skipped ONLY when check_split_adjustment
    will actually claim the day (raw close co-jumped past its gate);
    otherwise a split-shaped bad print would fall between two checks. A
    persistent move that the raw close shared and that sits near a split
    ratio (SPLIT_HINT_TOL) but outside the split check's tolerance keeps
    its WARN and names the ratio: an unadjusted split that coincided with a
    real move looks exactly like it."""
    out = []
    ret = _bridged_returns(b.adj_close)
    ret_raw = _bridged_returns(b.close)
    idio = ret.sub(ret.median(axis=1), axis=0)
    # pass 1: gates for every ticker, so classification can see how much
    # of the UNIVERSE was extreme on each day
    hit_cols, rev_cols = {}, {}
    for t in ret.columns:
        r = ret[t]
        sig = (r - r.rolling(63, min_periods=40).median()).abs() \
            .rolling(63, min_periods=40).median() * 1.4826
        scaled_gate = np.maximum(EXTREME_SIGMA_K * sig.shift(),
                                 EXTREME_SIGMA_FLOOR)
        hit_cols[t] = ((idio[t].abs() > EXTREME_IDIO_WARN)
                       | (r.abs() > scaled_gate)) & r.notna()
        rev_gate = np.maximum(REVERSAL_SIGMA_K * sig.shift(), REVERSAL_FLOOR)
        rev_cols[t] = ((r.abs() > rev_gate) & (idio[t].abs() > rev_gate)
                       & r.notna() & ~hit_cols[t])
    hit_df = pd.DataFrame(hit_cols, index=ret.index)
    # bad prints are idiosyncratic by nature - they do not synchronize.
    # When a chunk of the board is extreme on the same day (2020-03-16,
    # 14 tickers) OR the market median itself moved >3% (Lehman days when
    # thin bond ETFs dislocated alone: LQD -9.1% on 2008-09-29's -8%
    # tape), a V-reversal is crisis whipsaw, not a glitch.
    day_hits = hit_df.sum(axis=1)
    broad = (day_hits >= max(5, int(0.05 * max(hit_df.shape[1], 1)))) \
        | (ret.median(axis=1).abs() > 0.03)
    # ...but the panels must not certify their own exemption (see docstring)
    unconfirmed = pd.Series(False, index=ret.index)
    idx_ret = None
    if b.indices is not None and CRISIS_INDEX in b.indices.columns:
        idx_ret = pd.to_numeric(b.indices[CRISIS_INDEX], errors="coerce") \
            .dropna().pct_change(fill_method=None).reindex(ret.index)
        unconfirmed = broad & idx_ret.notna() \
            & (idx_ret.abs() <= CRISIS_INDEX_MOVE)
    crisis = broad & ~unconfirmed
    crisis = (crisis | crisis.shift(1, fill_value=False)
              | crisis.shift(-1, fill_value=False))

    def row_note(day) -> str:
        if not bool(unconfirmed.loc[day]):
            return ""
        return (f" [{int(day_hits.loc[day])} tickers extreme that day while "
                f"{CRISIS_INDEX} moved {idx_ret.loc[day]:+.1%}: row-level "
                "corruption?]")

    for t in ret.columns:
        r = ret[t]
        rv = r.dropna()  # neighbours are the adjacent VALID bars
        px = b.adj_close[t]
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
            pos = rv.index.get_loc(d)
            r_prev = rv.iloc[pos - 1] if pos > 0 else np.nan
            r_next = rv.iloc[pos + 1] if pos + 1 < len(rv) else np.nan
            d_prev = rv.index[pos - 1] if pos > 0 else d
            gap = _gap_note(px, d)
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
                        _fmt_d(d_prev),
                        f"{r_prev:+.1%} then {r_adj:+.1%} inside a "
                        f"market-wide extreme window ({int(day_hits.loc[d])} "
                        "tickers): crisis whipsaw, not a bad print"))
                else:
                    out.append(Finding(
                        "extreme_returns", "FAIL", t,
                        _fmt_d(d_prev),
                        f"return {r_prev:+.1%} fully reversed by the next "
                        f"day's {r_adj:+.1%}: bad print, not a price"
                        + (_gap_note(px, d_prev) or gap)
                        + (row_note(d_prev) or row_note(d))))
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
                        f"day ({r_next:+.1%}): bad print, not a price"
                        + (gap or _gap_note(px, rv.index[pos + 1]))
                        + row_note(d)))
            else:
                big = abs(r_i) > EXTREME_IDIO_WARN
                what = (f"adjusted return {r_adj:+.1%} (idiosyncratic "
                        f"{r_i:+.1%})" if big else
                        f"return {r_adj:+.1%} is >{EXTREME_SIGMA_K}x this "
                        f"instrument's typical daily move")
                if np.isfinite(r_next):
                    sev = "WARN" if big else "INFO"
                    tail = "persisted next day: verify against the tape"
                else:
                    # no later print: the reversal tests above could not run
                    sev = "WARN" if (big or not in_crisis) else "INFO"
                    where = ("the latest bar" if d == ret.index[-1]
                             else "this series' final bar")
                    tail = (f"on {where}: no later print has confirmed or "
                            "reversed it yet - verify against the tape "
                            "before trading on it" + row_note(d))
                hint = ""
                if (sev == "WARN" and np.isfinite(raw_r) and r_adj > -1.0
                        and abs(raw_r - r_adj) < 0.05):
                    near = _match_split_ratio(1.0 / (1.0 + r_adj),
                                              SPLIT_HINT_TOL)
                    if near is not None:
                        resid = (1.0 + r_adj) * near - 1.0
                        hint = (" (raw close moved with it: an unadjusted "
                                f"{_ratio_label(near)} split plus a "
                                f"{resid:+.1%} real move would look the same"
                                " - check corporate actions)")
                out.append(Finding(
                    "extreme_returns", sev, t, _fmt_d(d),
                    f"{what}{gap}, {tail}{hint}", value=float(r_adj)))
    # whole-bar prints below both gates (see docstring). A day that already
    # carries a finding (previous-day attribution) is not reported twice.
    reported = {(f.ticker, f.date) for f in out}
    for t in ret.columns:
        if t not in b.high.columns or t not in b.low.columns:
            continue
        r, hi, lo = ret[t], b.high[t], b.low[t]
        bars = b.adj_close[t].dropna().index  # neighbours = adjacent VALID bars
        for d in ret.index[rev_cols[t]]:
            pos = bars.get_loc(d)
            if pos + 1 >= len(bars) or bool(crisis.loc[d]) \
                    or (t, _fmt_d(d)) in reported:
                continue
            d_prev, d_next = bars[pos - 1], bars[pos + 1]
            r_adj, r_next = r.at[d], r.at[d_next]
            if not abs((1.0 + r_adj) * (1.0 + r_next) - 1.0) \
                    < 0.25 * abs(r_adj):
                continue
            if r_adj > 0:
                island = (lo.get(d, np.nan) > hi.get(d_prev, np.nan)
                          and lo.get(d, np.nan) > hi.get(d_next, np.nan))
            else:
                island = (hi.get(d, np.nan) < lo.get(d_prev, np.nan)
                          and hi.get(d, np.nan) < lo.get(d_next, np.nan))
            if island:
                out.append(Finding(
                    "extreme_returns", "WARN", t, _fmt_d(d),
                    f"adjusted return {r_adj:+.1%} (>{REVERSAL_SIGMA_K}x this "
                    "instrument's typical daily move) fully reversed next "
                    f"day ({r_next:+.1%}) and the whole bar printed outside "
                    "both neighbours' ranges: possible bad print, verify "
                    "against the tape" + _gap_note(b.adj_close[t], d),
                    value=float(r_adj)))
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
    Only raw moves beyond SPLIT_RAW_JUMP (40%) are examined, so 3:2, 4:3
    and 5:4 splits are left to check_adjustment_factor (see
    _SPLIT_CANDIDATES) - which sees them only when the two files disagree.

    The co-move FAIL needs the raw move within SPLIT_RATIO_TOL of a
    candidate ratio. A split missed by BOTH series on a day whose own return
    pushes the jump outside that band is not provable from prices (a real
    -52% print looks the same): check_extreme_returns reports it as a WARN
    that names the nearest ratio. Read the WARN list, not just the exit code.

    Returns are measured from each series' previous valid bar, so a split
    right after a missing bar is still classified.
    """
    out = []
    common = b.close.columns.intersection(b.adj_close.columns)
    r_raw = _bridged_returns(b.close[common])
    r_adj = _bridged_returns(b.adj_close[common])
    for t in common:
        raw_jump = r_raw.index[
            r_raw[t].notna() & (r_raw[t].abs() > SPLIT_RAW_JUMP)]
        for d in raw_jump:
            gap = _gap_note(b.close[t], d)
            factor = 1.0 + r_raw.at[d, t]
            ratio = _match_split_ratio(1.0 / factor) if np.isfinite(factor) and factor > 0 else None
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
                        "close.csv missed a split adj_close applied" + gap))
                else:
                    out.append(Finding(
                        "split_adjustment", "INFO", t, _fmt_d(d),
                        f"{ratio:g}:1 split correctly adjusted "
                        f"(raw {r_raw.at[d, t]:+.1%}, adj {adj_r:+.1%})"
                        + gap))
            elif abs(adj_r - r_raw.at[d, t]) < 0.05:
                out.append(Finding(
                    "split_adjustment", "FAIL", t, _fmt_d(d),
                    f"{ratio:g}:1 split NOT adjusted: raw "
                    f"{r_raw.at[d, t]:+.1%} and adjusted {adj_r:+.1%} "
                    "moved together" + gap))
            else:
                out.append(Finding(
                    "split_adjustment", "WARN", t, _fmt_d(d),
                    f"raw {r_raw.at[d, t]:+.1%} vs adjusted {adj_r:+.1%}: "
                    "partial/unclear adjustment" + gap))
        phantom = r_adj.index[
            r_adj[t].notna() & (r_adj[t].abs() > SPLIT_RAW_JUMP)
            & (r_raw[t].abs() < SPLIT_ADJ_QUIET)]
        for d in phantom:
            out.append(Finding(
                "split_adjustment", "FAIL", t, _fmt_d(d),
                f"adjusted close jumped {r_adj.at[d, t]:+.1%} while raw "
                f"moved {r_raw.at[d, t]:+.1%}: phantom adjustment"
                + _gap_note(b.adj_close[t], d)))
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
                           if split_like else ""), value=float(dy)))
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
                    "to an implausible stream", value=float(ttm.loc[d])))
    return out


def check_dividend_presence(b: PriceBundle,
                            expect_dividends: set[str] | None = None,
                            first_payout_by: dict[str, str] | None = None
                            ) -> list[Finding]:
    """The funds in EXPECT_DIVIDENDS ALWAYS distribute. If the factor
    shows no dividend events for one of them over a long stretch, the
    cache has silently degraded from total-return to price-return data -
    a few percent a year of phantom underperformance that no other check
    can see (each daily factor ratio is a perfectly innocent 1.0).

    Three rules, each a FAIL:
      1. no payout over the whole life (a history that ends on or before
         the fund's date in `first_payout_by` is not judged: its first
         payout was not due yet);
      2. the first payout arrives more than DIV_MAX_GAP_DAYS after the
         first row - an early history that lost its dividends. A fund
         that really started paying late (QQQ paid nothing 1999-2002 -
         expenses exceeded income) is listed in `first_payout_by`
         (default DIV_FIRST_PAYOUT_BY) with the date its first payout
         must have appeared by;
      3. a later gap above DIV_MAX_GAP_DAYS between payouts, or after the
         last one: the fund paid and then went silent."""
    out = []
    if expect_dividends is None:
        expect_dividends = EXPECT_DIVIDENDS
    if first_payout_by is None:
        first_payout_by = DIV_FIRST_PAYOUT_BY
    common = b.close.columns.intersection(b.adj_close.columns)
    for t in sorted(expect_dividends & set(common)):
        ft = (b.adj_close[t] / b.close[t]).dropna()
        if len(ft) < 504:  # need ~2y of life to judge a payout cadence
            continue
        g = (ft / ft.shift()).dropna()
        ev = g.index[(g - 1.0) > FACTOR_NOISE_TOL]
        if len(ev) == 0:
            if (t in first_payout_by
                    and ft.index[-1] <= pd.Timestamp(first_payout_by[t])):
                continue  # the cache ends before its first payout was due
            out.append(Finding(
                "dividend_presence", "FAIL", t,
                f"{_fmt_d(ft.index[0])}..{_fmt_d(ft.index[-1])}",
                f"zero dividend events over the whole {len(ft)}-day life "
                "of an instrument that always distributes: total-return "
                "data degraded to price-return"))
            continue
        lead = int((ev[:1].values - ft.index[:1].values)
                   .astype("timedelta64[D]").astype(int)[0])
        late_ok = (t in first_payout_by
                   and ev[0] <= pd.Timestamp(first_payout_by[t]))
        if lead > DIV_MAX_GAP_DAYS and not late_ok:
            out.append(Finding(
                "dividend_presence", "FAIL", t,
                f"{_fmt_d(ft.index[0])}..{_fmt_d(ev[0])}",
                f"no payout for the first {lead} days of an instrument "
                "that always distributes: total-return data degraded to "
                "price-return at the start of the history"))
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
    """A step change in volume MAGNITUDE (shares vs thousands or lots)
    wrecks liquidity filters and cost models while every price check stays
    green. Compares the 20-day median volume across each date; only breaks
    that persist (both medians well-formed) are flagged, and consecutive
    flag-days collapse into one finding.

    Scope: unit errors of roughly VOLUME_STEP_WARN (30x) and above. A split
    whose volume was never re-based is NOT covered: real splits are 2:1 to
    20:1, and steps that size cannot be told apart from genuine liquidity
    ramps (thin early ETF histories step 8x and more); with close and
    volume both split-adjusted at source there are no split dates to test
    against either."""
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
                "window: unit/scale break, not a liquidity regime",
                value=worst))
    return out


def check_lead_lag(b: PriceBundle) -> list[Finding]:
    """A ticker whose whole history is shifted by one day is catastrophic
    look-ahead for every cross-sectional signal, yet every per-ticker
    check passes (the series itself is pristine - it is just in the wrong
    place). Signature: its returns correlate more with YESTERDAY's (or
    tomorrow's) market than with today's.

    The whole-history test cannot see a shift confined to a SEGMENT (a bad
    merge, a partial re-download): the unshifted years dominate the
    full-sample correlation. So each calendar year with at least
    LEADLAG_BLOCK_MIN_OBS returns, and the trailing LEADLAG_BLOCK_TAIL
    rows, is also tested on its own. One window is a small sample, and in
    a crisis thinly traded bond and currency funds really do trail the
    equity market for weeks, so the window test uses rank correlations (a
    handful of crash days cannot carry it) and the wider
    LEADLAG_BLOCK_MARGIN. A shifted stretch much shorter than a window, and
    instruments that barely correlate with the market, stay out of reach of
    both tests."""
    out = []
    ret = b.adj_close.pct_change(fill_method=None)
    if ret.shape[1] < 5:
        return out  # need a market to compare against
    years = ret.index.year
    windows = [np.flatnonzero(years == y) for y in np.unique(years)]
    windows.append(np.arange(max(len(ret) - LEADLAG_BLOCK_TAIL, 0), len(ret)))
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
            continue  # the windows would only repeat it
        w = pd.DataFrame({"r": r, "same": mkt, "previous": mkt.shift(1),
                          "next": mkt.shift(-1)})
        shifted = []
        for rows in windows:
            seg = w.iloc[rows]
            seg = seg[seg["r"].notna()]
            if len(seg) < LEADLAG_BLOCK_MIN_OBS:
                continue
            c = seg.rank().corr()["r"].abs()
            day = "previous" if c["previous"] >= c["next"] else "next"
            if c[day] > c["same"] + LEADLAG_BLOCK_MARGIN:
                shifted.append((c[day] - c["same"], day, c[day], c["same"],
                                seg.index[0], seg.index[-1]))
        if shifted:
            _, day, best, c0, _, _ = max(shifted)
            out.append(Finding(
                "lead_lag", "FAIL", t,
                f"{_fmt_d(min(s[4] for s in shifted))}.."
                f"{_fmt_d(max(s[5] for s in shifted))}",
                f"in {len(shifted)} window(s) of this span returns rank-"
                f"correlate with the {day} day's market ({best:.2f}) better "
                f"than the same day's ({c0:.2f}): segment shifted by one "
                "day"))
    return out


def check_numeric_values(b: PriceBundle) -> list[Finding]:
    """Infinity and impossible individual fields must fail even when another
    OHLC field is missing on the same row."""
    out = []
    frames = b.frames()
    if b.indices is not None:
        frames["indices"] = b.indices
    for name, df in frames.items():
        for col in df.columns:
            values = df[col]
            if not pd.api.types.is_numeric_dtype(values):
                out.append(Finding("numeric_values", "FAIL", str(col), "",
                                   f"{name}: nonnumeric values"))
                continue
            invalid = values.notna() & ~np.isfinite(values)
            if name not in {"indices", "volume"}:
                invalid |= values <= 0
            elif name == "volume":
                invalid |= values < 0
            for date in values.index[invalid][:10]:
                out.append(Finding("numeric_values", "FAIL", str(col), _fmt_d(date),
                                   f"{name}: invalid numeric value {values.loc[date]}") )
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


def check_range_plausibility(b: PriceBundle) -> list[Finding]:
    """High or low far outside the day's open-close body (a "wick").

    check_ohlc_consistency only proves that open and close sit inside
    [low, high]; it cannot say whether the extremes themselves ever traded.
    Vendor highs and lows keep single erroneous prints, and those sit next
    to genuine dislocations (flash-crash and crisis sessions) that only the
    tape can tell apart. A wick counts when it is more than WICK_ABS beyond
    min/max(open, close) AND more than WICK_RANGE_K times the ticker's own
    trailing-median daily range (so a habitually wide-ranging name is not
    flagged for behaving normally; a ticker's first 60 sessions have no
    trailing median and are not judged).

    Everything here is INFO: close-to-close signals never read high or low,
    and a bad print cannot be told from a real one without the tape, so
    this check counts and locates the bars instead of gating on them. One
    finding per isolated (ticker, day); a day on which
    WICK_SYSTEMIC_TICKERS or more tickers wick together is a market-wide
    dislocation and is reported once for the day. Anything that uses high
    or low as a fill trigger, stop level or range estimator should review
    these bars first."""
    out = []
    common = b.low.columns
    for df in (b.high, b.open, b.adj_close):
        common = common.intersection(df.columns)
    if not len(common):
        return out
    idx = b.low.index
    lo, hi, op, cl = (
        df.reindex(index=idx, columns=common).apply(pd.to_numeric, errors="coerce")
        for df in (b.low, b.high, b.open, b.adj_close))
    ok = (lo > 0) & (hi > 0) & (op > 0) & (cl > 0)
    lo, hi, op, cl = (df.where(ok) for df in (lo, hi, op, cl))
    body_lo, body_hi = op.where(op < cl, cl), op.where(op > cl, cl)
    typical = (hi / lo - 1.0).rolling(
        WICK_RANGE_WINDOW, min_periods=60).median().shift(1)
    sides = {"low": (1.0 - lo / body_lo, lo, "below"),
             "high": (hi / body_hi - 1.0, hi, "above")}
    hot = {side: (wick > WICK_ABS) & (wick > WICK_RANGE_K * typical)
           for side, (wick, _, _) in sides.items()}
    n_day = sum(h.sum(axis=1) for h in hot.values())
    for d in n_day.index[n_day >= WICK_SYSTEMIC_TICKERS]:
        worst = max(float(sides[s][0].loc[d].where(hot[s].loc[d]).max())
                    for s in sides if hot[s].loc[d].any())
        out.append(Finding(
            "range_plausibility", "INFO", "", _fmt_d(d),
            f"{int(n_day.loc[d])} high/low wicks beyond {WICK_ABS:.0%} on one "
            f"day (largest {worst:.1%}): market-wide dislocation"))
    for side, (wick, px, word) in sides.items():
        isolated = hot[side].loc[(n_day < WICK_SYSTEMIC_TICKERS).to_numpy()]
        for t in common:
            for d in isolated.index[isolated[t]]:
                usual = typical.at[d, t]
                times = (f"{wick.at[d, t] / usual:.0f}x" if usual > 0
                         else "far beyond")
                out.append(Finding(
                    "range_plausibility", "INFO", t, _fmt_d(d),
                    f"{side} {px.at[d, t]:.4g} is {wick.at[d, t]:.1%} {word} "
                    f"open {op.at[d, t]:.4g} / close {cl.at[d, t]:.4g}, "
                    f"{times} its median daily range: single-print wick or "
                    "real dislocation, verify against the tape before "
                    f"relying on this {side}"))
    return out


def check_delisting(b: PriceBundle) -> list[Finding]:
    """A ticker whose data stops before the end of the file has either
    delisted (backtests must know: its 'flat forever' tail is survivorship
    poison if treated as tradeable) or silently fell out of the feed. Late
    inceptions are expected (ABBV 2013, META 2012) and only counted.

    The live edge: a ticker with no adjusted close on the latest 1 to
    DELIST_GRACE_DAYS rows is not delisted yet, but its newest bars are
    missing - check_missing_prices only looks inside the listed life, so
    this is the check that must say so (WARN). When LIVE_EDGE_FAIL_FRAC of
    the live tickers (and more than one) lack the final row, the row is a
    partial bar and FAILs: signals would rank a broken cross-section."""
    out = []
    idx = b.adj_close.index
    if len(idx) == 0:
        return out
    last_day = idx[-1]
    live, lagging = 0, []
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
            else:
                live += 1
                if lag > 0:
                    lagging.append(t)
                    out.append(Finding(
                        "delisting", "WARN", t, _fmt_d(last),
                        f"no price on the latest {lag} row(s) (file ends "
                        f"{_fmt_d(last_day)}): late print, halt or partial "
                        "download - this ticker's signals are stale on the "
                        "decision bar", value=float(lag)))
        if first != idx[0]:
            out.append(Finding(
                "delisting", "INFO", t, _fmt_d(first),
                f"first data {len(idx[idx < first])} days after file start "
                "(later inception/IPO)"))
    if len(lagging) > 1 and len(lagging) >= LIVE_EDGE_FAIL_FRAC * live:
        out.append(Finding(
            "delisting", "FAIL", "", _fmt_d(last_day),
            f"final row has no price for {len(lagging)} of {live} live "
            f"tickers (e.g. {', '.join(lagging[:5])}): partial or "
            "in-progress bar"))
    return out


def check_history_loss(b: PriceBundle,
                       reference: pd.DataFrame | None = None
                       ) -> list[Finding]:
    """History a series HELD when `reference` was recorded and no longer
    has. A cache that holds only recent years for a ticker looks, to every
    other check, exactly like a later inception (check_delisting counts it
    as INFO) while every backtest on it quietly changes.

    `reference` is a coverage table recorded for the same cache: index =
    series name; columns first, last (dates) and rows (observation count) -
    the layout of the coverage.csv the downloader writes beside the price
    files. For each adjusted-close column or index series it lists, the
    observations dated inside the recorded [first, last] span are counted;
    fewer than the recorded `rows` means observations vanished: WARN up to
    HISTORY_LOSS_FAIL_ROWS of them, FAIL beyond. Growth (new rows, earlier
    history) is never a finding, and a series that is in only one of the
    two is left to check_symbol_mapping. No reference, no check; a
    reference that cannot be read is a WARN, because "could not compare"
    must not read as "nothing lost"."""
    out = []
    if reference is None:
        return out
    try:
        # the first ten characters: a date written with a time of day
        # (2000-01-03 00:00:00) is still that date
        ref = pd.DataFrame({
            "first": pd.to_datetime(reference["first"].astype(str).str[:10],
                                    format="%Y-%m-%d", errors="coerce"),
            "last": pd.to_datetime(reference["last"].astype(str).str[:10],
                                   format="%Y-%m-%d", errors="coerce"),
            "rows": pd.to_numeric(reference["rows"], errors="coerce"),
        }).dropna()
    except (AttributeError, KeyError, TypeError, ValueError):
        return [Finding("history_loss", "WARN", "", "",
                        "recorded coverage is unreadable (needs first, last "
                        "and rows for each series): lost history cannot be "
                        "ruled out")]
    if len(ref) < len(reference):
        out.append(Finding(
            "history_loss", "WARN", "", "",
            f"{len(reference) - len(ref)} of {len(reference)} recorded "
            "series have an unreadable first, last or rows: lost history "
            "cannot be ruled out for them"))
    series = {}
    for df in [b.adj_close] + ([b.indices] if b.indices is not None else []):
        df = df.loc[:, ~df.columns.duplicated()]
        series.update({str(t): df[t] for t in df.columns})
    ref = ref[~ref.index.duplicated()]
    for t, (first, last, rows) in ref.iterrows():
        if str(t) not in series:
            continue
        s = series[str(t)].dropna()
        kept = int(((s.index >= first) & (s.index <= last)).sum())
        lost, rows = int(rows) - kept, int(rows)
        if lost <= 0:
            continue
        what, date = "", ""
        if len(s) == 0:
            what = "; no observations are left"
        elif s.index[0] > first:
            date = _fmt_d(s.index[0])
            what = f"; history now starts {date}"
        elif s.index[-1] < last:
            what = f"; history now ends {_fmt_d(s.index[-1])}"
        out.append(Finding(
            "history_loss",
            "FAIL" if lost > HISTORY_LOSS_FAIL_ROWS else "WARN", str(t), date,
            f"{lost} of the {rows} observations recorded for "
            f"{_fmt_d(first)}..{_fmt_d(last)} are gone{what}: results "
            "computed on the earlier cache no longer reproduce",
            value=float(lost)))
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
    from qcore.data import OPERATIONAL_UNIVERSE
    for c in sorted(ref - universe - set(OPERATIONAL_UNIVERSE)):
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
    frozen VIX silently disables every vol-regime gate in the program, and
    a frozen or mis-scaled ^IRX feeds straight into the cash credit and the
    risk-free rate. Series named in b.required_indices must exist."""
    out = []
    if b.indices is None:
        if b.required_indices:
            out.append(Finding(
                "indices", "FAIL", "", "",
                "indices.csv missing: required series "
                f"{', '.join(b.required_indices)} unavailable (no cash "
                "rate: idle cash would be credited 0%)"))
        return out
    for t in b.required_indices:
        if t not in b.indices.columns:
            out.append(Finding("indices", "FAIL", t, "",
                               "required series missing from indices.csv"))
    ranges = {"^VIX": (5, 150), "^VIX3M": (5, 150), "^IRX": (-2, 25),
              "^GSPC": (500, 50_000), "^TNX": (-1, 20)}
    # ^IRX legitimately pins for weeks under ZIRP (0.005% for 19 sessions in
    # Dec 2011), so it gets level-aware run limits instead of STALE_RUN_WARN;
    # the 10-year yield never repeats for long and takes the plain test
    stale_applies = {"^VIX", "^VIX3M", "^GSPC", "^TNX"}
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
                    f"{int(r['length'])} days", value=float(r["length"])))
        if t == "^IRX":
            runs = _runs_of_equal(s)
            limit = np.where(runs["value"] > RATE_ZIRP_LEVEL,
                             RATE_STALE_RUN_WARN, RATE_ZIRP_STALE_RUN_WARN)
            for _, r in runs[runs["length"] >= limit].iterrows():
                out.append(Finding(
                    "indices", "WARN", t,
                    f"{_fmt_d(r['start'])}..{_fmt_d(r['end'])}",
                    f"value pinned at {r['value']:.4g} for "
                    f"{int(r['length'])} days (frozen feed? the cash credit "
                    "and risk-free rate read this series)",
                    value=float(r["length"])))
        if t in BOND_ONLY_SERIES:
            # a yield does not move several points between prints; a switch
            # between percent and decimal units does exactly that. (A switch
            # while the level is under RATE_JUMP_WARN_PP stays invisible.)
            step = s.dropna().diff()
            big = step[step.abs() > RATE_JUMP_WARN_PP]
            for d in big.index[:5]:
                sev = "FAIL" if abs(big.at[d]) > RATE_JUMP_FAIL_PP else "WARN"
                out.append(Finding(
                    "indices", sev, t, _fmt_d(d),
                    f"yield moved {big.at[d]:+.3g} pp in one print to "
                    f"{s.at[d]:.4g}: unit change or bad print?",
                    value=float(big.at[d])))
        # the newest prints matter most: a series that stops even one row
        # before the price cache is forward-filled on the decision bar
        after = b.adj_close.index[b.adj_close.index > s.index[-1]]
        if t in BOND_ONLY_SERIES and len(after):
            after = after.difference(
                _BondOnlyHolidays().holidays(after[0], after[-1]))
        lag = len(after)
        if lag > 0:
            out.append(Finding("indices", "WARN", t, _fmt_d(s.index[-1]),
                               f"series ends {lag} days before price cache",
                               value=float(lag)))
        # holes inside the series' life on days the NYSE traded (a missing
        # ^IRX/^VIX3M print is silently forward-filled downstream)
        expected = nyse_bdays(s.index[0], s.index[-1]).intersection(b.adj_close.index)
        if t in BOND_ONLY_SERIES:
            expected = expected.difference(
                _BondOnlyHolidays().holidays(s.index[0], s.index[-1]))
        holes = expected[s.reindex(expected).isna().to_numpy()]
        if len(holes):
            out.append(Finding(
                "indices", "WARN", t, _fmt_d(holes[-1]),
                f"{len(holes)} missing value(s) inside the series on NYSE "
                f"trading days, latest {', '.join(_fmt_d(d) for d in holes[-3:])}"))
    # rows on days the NYSE was closed (vendor rows on holidays)
    idx = b.indices.index
    if len(idx) == 0:
        return out  # header-only file: the per-series "no data" FAILs stand
    off = idx.difference(nyse_bdays(idx[0], idx[-1]))
    off = off[b.indices.loc[off].notna().any(axis=1).to_numpy()] if len(off) else off
    if len(off):
        out.append(Finding(
            "indices", "WARN", "", _fmt_d(off[-1]),
            f"{len(off)} row(s) with values on NYSE-closed days, latest "
            f"{', '.join(_fmt_d(d) for d in off[-3:])}"))
    return out


# -------------------------------------------------------------- fundamentals
# Metrics that can never be negative; anything else is a sign/parse error.
NONNEG_METRICS = {"revenue", "total_assets", "shares_out", "total_equity_abs",
                  "cash", "market_cap"}
UNIT_BREAK_RATIO = 100.0    # 100x between periods => thousands vs millions
UNIT_WARN_RATIO = 10.0
REPORT_LAG_WARN_D = 180
REPORT_LAG_MIN_D = 7        # nothing is published within a week of period end
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

    Checks: missing period_end; duplicate (ticker, period_end); report_date
    on or before period_end (impossible - guarantees look-ahead; a vendor
    that defaults report_date to the period end is the commonest case);
    missing report_date; implausible filing lag (under REPORT_LAG_MIN_D or
    over REPORT_LAG_WARN_D days); skipped fiscal quarters; unit breaks (a
    100x jump from the last REPORTED period - missing values are bridged -
    is a thousands-vs-millions switch, not growth);
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

    # a row with no period_end cannot be placed in time: report it, then
    # keep it out of the per-period checks (NaT sorts last and would be
    # compared against the real latest period)
    no_pe = fund["period_end"].isna()
    for t, grp in fund[no_pe].groupby("ticker"):
        out.append(Finding("fund_gaps", "WARN", t, "",
                           f"{len(grp)} row(s) missing period_end"))
    fund = fund[~no_pe]

    dup = fund.duplicated(["ticker", "period_end"], keep=False)
    for (t, p), _ in fund[dup].groupby(["ticker", "period_end"]):
        out.append(Finding("fund_duplicates", "FAIL", t, _fmt_d(p),
                           "duplicate (ticker, period_end) rows"))

    if not has_report:
        out.append(Finding("fund_lookahead", "WARN", "", "",
                           "no report_date column: backtests cannot know "
                           "when numbers became public"))
    else:
        # whole days: a same-day timestamp (16:00 on the period-end date)
        # is as impossible as an earlier one
        lag = (fund["report_date"] - fund["period_end"]).dt.days
        for (_, r), n_days in zip(fund[lag <= 0].iterrows(), lag[lag <= 0]):
            how = "precedes" if n_days < 0 else "equals"
            out.append(Finding(
                "fund_lookahead", "FAIL", r["ticker"], _fmt_d(r["period_end"]),
                f"report_date {_fmt_d(r['report_date'])} {how} period end: "
                "look-ahead guaranteed"))
        nan_rep = fund[fund["report_date"].isna()]
        for t, grp in nan_rep.groupby("ticker"):
            out.append(Finding("fund_lookahead", "WARN", t, "",
                               f"{len(grp)} rows missing report_date"))
        late = fund[lag > REPORT_LAG_WARN_D]
        for _, r in late.iterrows():
            out.append(Finding(
                "fund_lookahead", "WARN", r["ticker"], _fmt_d(r["period_end"]),
                f"{(r['report_date'] - r['period_end']).days}d filing lag"))
        early = fund[(lag > 0) & (lag < REPORT_LAG_MIN_D)]
        for _, r in early.iterrows():
            out.append(Finding(
                "fund_lookahead", "WARN", r["ticker"], _fmt_d(r["period_end"]),
                f"{(r['report_date'] - r['period_end']).days}d filing lag "
                f"(under {REPORT_LAG_MIN_D}d): report_date is probably not "
                "the publication date"))

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
            prev = v.ffill().shift()  # the last REPORTED value
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
def _known_event_changed(value, expect: str) -> str:
    """'' while a finding's measured value still matches the adjudicated
    `expect` (or no value was pinned); otherwise the reason it does not.
    Anything unreadable fails closed: no downgrade."""
    if expect == "":
        return ""
    try:
        want = float(expect)
    except ValueError:
        want = np.nan
    if not np.isfinite(want):
        return f"unreadable expect {expect!r}"
    if value is None or not np.isfinite(value):
        return f"adjudicated {want:g}, but this finding measures no value"
    if abs(value - want) > max(KNOWN_EVENT_ABS_TOL,
                               KNOWN_EVENT_REL_TOL * abs(want)):
        return f"adjudicated {want:g}, now {value:.4g}"
    return ""


def apply_known_events(report: DQReport,
                       known: pd.DataFrame) -> tuple[int, list[tuple]]:
    """Downgrade findings a human has already adjudicated as real market
    events (not data errors) to INFO, so the steady-state dashboard is
    quiet and only NEW anomalies alarm. `known` needs columns
    check,ticker,date,note, with `date` equal to the finding's date or range
    string (e.g. 2000-07-21..2000-07-27). scripts/data_quality.py reads it
    from a local data/dq_known_events.csv, or --known; none is distributed
    with the project, and the report runs without one. Never silences
    FAILs: a FAIL is a data defect by construction and must be fixed in
    the data, not acknowledged away.

    Optional column `expect`: the value that was adjudicated (Finding.value
    - the return, yield, run length or ratio the check measured; it is in
    the JSON report). A row with `expect` acknowledges the finding only
    while its value stays within max(KNOWN_EVENT_ABS_TOL,
    KNOWN_EVENT_REL_TOL x |expect|) of it; otherwise the finding stays WARN
    and says the acknowledged event CHANGED - a revised bar or a re-scaled
    series on an acknowledged date is a new anomaly, not the old one. A
    blank `expect` (or no such column) matches on the key alone.

    Keys are normalized to strings (read_csv turns empty cells into NaN
    and bare years into ints - or into floats such as 2001.0 when the same
    column also has an empty cell; any of these would dead-letter the row).
    Returns (n_downgraded, unmatched_rows) - surface unmatched rows to the
    operator: a dead acknowledgment means the finding key drifted and the
    allowlist is silently not protecting what they think it protects."""
    def norm(x) -> str:
        if pd.isna(x):
            return ""
        if isinstance(x, (float, np.floating)) and float(x).is_integer():
            x = int(x)  # 2001.0 -> "2001", the finding's year key
        return str(x).strip()

    expects = (known["expect"] if "expect" in known.columns
               else [""] * len(known))
    notes = {(norm(c), norm(t), norm(d)): (norm(n), norm(e))
             for c, t, d, n, e in zip(known["check"], known["ticker"],
                                      known["date"], known["note"], expects)}
    n, matched = 0, set()
    for f in report.findings:
        k = (f.check, f.ticker, f.date)
        if k in notes:
            matched.add(k)  # a changed event is not a dead acknowledgment
            if f.severity == "WARN":
                note, expect = notes[k]
                changed = _known_event_changed(f.value, expect)
                if changed:
                    f.detail += (f" [acknowledged event CHANGED: {changed} - "
                                 f"re-verify before trusting: {note}]")
                    continue
                f.severity = "INFO"
                f.detail += f" [acknowledged: {note}]"
                n += 1
    unmatched = [k for k in notes if k not in matched]
    return n, unmatched


# ------------------------------------------------------------------- run_all
PRICE_CHECKS: list[Callable[[PriceBundle], list[Finding]]] = [
    check_calendar_alignment,
    check_calendar_gaps,
    check_symbol_mapping,
    check_numeric_values,
    check_missing_prices,
    check_duplicate_rows,
    check_carried_rows,
    check_stale_prices,
    check_zero_volume,
    check_volume_scale,
    check_extreme_returns,
    check_split_adjustment,
    check_adjustment_factor,
    check_dividend_presence,
    check_ohlc_consistency,
    check_range_plausibility,
    check_delisting,
    check_history_loss,
    check_lead_lag,
    check_indices,
]


def _sanitized(b: PriceBundle) -> PriceBundle:
    """Best-effort view for content checks when the index itself is broken:
    duplicate dates dropped (keep first), index sorted. The structural
    damage is still reported from the RAW bundle by the calendar checks
    (the index file included) - this only keeps one bad row from crashing
    every other check."""
    def fix(df):
        if df is None:
            return None
        if df.index.duplicated().any() or not df.index.is_monotonic_increasing:
            df = df[~df.index.duplicated(keep="first")].sort_index()
        return df
    return PriceBundle(**{f: fix(getattr(b, f)) for f in PriceBundle.FIELDS},
                       indices=fix(b.indices),
                       raw_split_adjusted=b.raw_split_adjusted,
                       required_indices=b.required_indices)


def run_all(bundle: PriceBundle,
            fundamentals: pd.DataFrame | None = None,
            snapshot_date=None,
            universe: set[str] | None = None,
            expect_dividends: set[str] | None = None,
            reference_coverage: pd.DataFrame | None = None) -> DQReport:
    """Run every check; a crashing check becomes a FAIL finding rather than
    killing the report (partially-validated data must never look clean).
    reference_coverage: a coverage table recorded for this cache (see
    check_history_loss); without one that check is skipped."""
    report = DQReport()
    structural = (check_calendar_alignment, check_calendar_gaps)
    clean_view = _sanitized(bundle)
    extra_kw = {check_symbol_mapping: {"universe": universe},
                check_dividend_presence: {"expect_dividends": expect_dividends},
                check_history_loss: {"reference": reference_coverage}}
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
