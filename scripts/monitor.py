"""Risk controls: kill and review rules for the blended portfolio.

The thresholds are registered in advance of live trading, so that a decision
to stop is not improvised in the middle of a drawdown. They are judged on
the ENSEMBLE column of a daily-returns file:

  KILL    drawdown from the running peak below -15%
  KILL    rolling two-year (504-session) Sharpe ratio, in excess of the
          T-bill return, below -0.30
  REVIEW  underwater (time since the last equity peak) for 32 months or more
  KILL    underwater for 42 months or more

One further review trigger is not computed here: fill slippage against the
backtest's assumption left unreconciled for three months. scripts/reconcile.py
measures that slippage from a fills file.

Changing a threshold after going live is itself treated as a KILL-level
event: the response to "the rule feels wrong now" is to stop trading, redo
the research and register again, not to move the threshold during a loss.

Calibration caveat. The thresholds were calibrated to the worst cases of the
backtest at the time they were registered, and the current backtest no
longer matches those bases. On the saved blended backtest (2000-01-03 to
2026-07-01, 6,663 sessions) the maximum drawdown is -11.3% and the longest
underwater spell is just under 23 months, both inside the limits, but the
rolling two-year excess Sharpe bottoms at -0.64 (2020-03-18) and is below
-0.30 on 20 sessions (February-March 2020 and October-November 2023). The
two-year rule therefore fires inside the backtest itself. The thresholds
are shown here as registered; whether to register them again is an open
question that this script does not settle.

Source. The returns file must be named; there is no default. A live check
reads a hand-kept CSV of the account's own daily returns (no script writes
it): a Date column written YYYY-MM-DD and an ENSEMBLE column holding each
session's net return as a fraction (0.0012 = +0.12%), deposits and
withdrawals excluded. examples/live_returns.example.csv is a made-up file in
that format. --backtest judges results/sleeve_returns.csv instead, the
blended backtest stream that src/ensemble.py saves; that says nothing about
an account and is a calibration run only. Only the ENSEMBLE column sets the
verdict and the exit status. Other columns (the sleeves) are printed for
information when they breach; no thresholds are registered for them. A file
without an ENSEMBLE column is judged on all of its columns.

Window. The monitor keeps no state between runs, so it judges every session
of the window, not only the last row: a breach on any judged session sets
the status, and the first and worst breach dates are printed. A live file
is judged from its first row; --since narrows the window to the sessions on
or after a date (for example the previous check). The backtest calibration
run reports the end state only (its last session) unless --since is given.
Rows before the window still seed the equity curve and the two-year window.

Two readings. Every run prints both, because they differ whenever a breach
has reverted:
  at the last session     the three rules at the final row alone;
  on any judged session   the worst status over the judged window.
The verdict (the "overall:" line) and the exit status follow the second, so
a breach that later reverted stays KILL or REVIEW until --since moves past
it, and its rule row is tagged "(last session <status>)". Judging every
session trips more often than judging one row per weekly run, at the same
thresholds. Passing the last row's date to --since gives the last-row-only
verdict.

Input rules. No verdict is issued (exit 3) when
  - a date is not a plain YYYY-MM-DD day, or dates repeat or are unordered;
  - a row falls on a day the NYSE was closed (calendar-day exports lay the
    same returns on more rows, shortening the two-year window);
  - an NYSE session between the first and last row has no row (its profit
    or loss would be missing from every rule), unless --allow-gaps is passed
    for a verified gap;
  - a live file ends more than MAX_STALE_SESSIONS completed NYSE sessions
    before today's New York date, or after it. --as-of replays an old file
    against a stated date;
  - the two-year rule is in force (504 observations) and the cash benchmark
    cannot price it: ^IRX missing from data/indices.csv, or a return date in
    a judged window with no ^IRX print in the previous MAX_RATE_AGE_DAYS
    calendar days. Without this check the engine would credit 0% or carry
    the last cached yield forward, and the "excess" Sharpe would silently
    become something else;
  - --backtest is asked for and results/sleeve_returns.csv does not exist
    (the message names the command that creates it).

Usage:
  python scripts/monitor.py examples/live_returns.example.csv --as-of 2025-03-31
  python scripts/monitor.py state/live_returns.csv      # live check, whole file
  python scripts/monitor.py state/live_returns.csv --since YYYY-MM-DD
  python scripts/monitor.py old.csv --as-of YYYY-MM-DD   # offline replay
  python scripts/monitor.py --backtest                   # calibration only
  python scripts/monitor.py --backtest --since 2002-01-08
      # every backtest session whose two-year window the cached ^IRX prices
      # (a cache that starts on 2000-01-03 has no earlier print for that day)

Exit status (0-2 match scripts/data_quality.py):
  0  ok
  1  REVIEW
  2  KILL
  3  no verdict: invalid, stale or unreadable input, or a usage error
  4  no verdict: unexpected internal error (traceback on stderr)
0-2 report the judged-window reading, the same status as the "overall:" line.
Only an "overall:" line is a verdict. A Python start-up failure (for example
a missing dependency) exits 1 before the monitor runs and prints none.

Limits: three rules on one return series. The monitor reads returns, not
positions or account value, and no script here writes the live returns file.
Session dates are checked against the rule-based NYSE calendar in
qcore.quality, which cannot anticipate an unscheduled closure. A new live file
leaves the two-year rule pending for its first 504 sessions.
"""

import argparse
import csv
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import (TBILL_HAIRCUT_BPS, TRADING_DAYS, _irx_series,  # noqa: E402
                            cash_daily_return)
from qcore.quality import nyse_bdays  # noqa: E402

KILL_MAX_DRAWDOWN = -0.15
KILL_SHARPE_2Y = -0.30
REVIEW_UNDERWATER_M = 32
KILL_UNDERWATER_M = 42
WINDOW_2Y = 2 * TRADING_DAYS

BACKTEST_RETURNS = ROOT / "results" / "sleeve_returns.csv"
MAX_STALE_SESSIONS = 5  # one trading week, the check cadence
# Oldest ^IRX print allowed to price a return date. The longest gap in the
# 2000-2026 cache is 7 calendar days (2001-09-10 -> 2001-09-17); ordinary
# long weekends and bond-market holidays leave 4-5.
MAX_RATE_AGE_DAYS = 7

EXIT_OK = 0
EXIT_REVIEW = 1
EXIT_KILL = 2
EXIT_INVALID_INPUT = 3  # distinct from KILL
EXIT_CRASH = 4          # distinct from REVIEW, which shares 1 with a Python traceback


def current_ny_date() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()


def sessions_behind(last: pd.Timestamp, today: pd.Timestamp) -> int:
    """Completed NYSE sessions after `last` and before `today`. Today's own
    session is not counted: its close may not have printed yet."""
    if last >= today:
        return 0
    days = nyse_bdays(last, today)
    return int(((days > last) & (days < today)).sum())


def session_mismatch(index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """(rows on non-session dates, sessions with no row) between the first
    and last date of a valid, ordered index."""
    expected = nyse_bdays(index[0], index[-1])
    return index.difference(expected), expected.difference(index)


def _show_dates(dates: pd.DatetimeIndex, limit: int = 5) -> str:
    shown = ", ".join(str(d.date()) for d in dates[:limit])
    return shown + (f" (+{len(dates) - limit} more)" if len(dates) > limit else "")


def _irx() -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # absence is reported by the callers
        return _irx_series()


def benchmark_span() -> str:
    irx = _irx()
    if irx.empty:
        return "^IRX unavailable"
    return (f"^IRX {irx.index[0].date()} .. {irx.index[-1].date()} "
            f"less {TBILL_HAIRCUT_BPS:.0f}bp")


def cash_benchmark(index: pd.DatetimeIndex) -> pd.Series:
    """Daily T-bill return in force on each date of `index`, or ValueError
    when ^IRX is unavailable or does not cover those dates. The engine's
    cash_daily_return degrades to 0% (with a warning) or forward-fills the
    last cached yield; either silently changes an excess Sharpe."""
    irx = _irx()
    if irx.empty:
        raise ValueError("cash benchmark unavailable: data/indices.csv has no ^IRX "
                         "history (rerun src/download_data.py)")
    seen = pd.Series(irx.index, index=irx.index)
    prior = seen.reindex(irx.index.union(index)).ffill().shift(1).reindex(index)
    age = np.asarray((index - pd.DatetimeIndex(prior)).days, dtype=float)
    bad = ~(age <= MAX_RATE_AGE_DAYS)  # NaN (no earlier print) is bad
    if bad.any():
        # first session whose own 2y window starts after the last unpriced date
        clear = int(np.flatnonzero(bad)[-1]) + WINDOW_2Y
        remedy = "refresh data/indices.csv" + (
            f", or judge only the fully priced windows with --since {index[clear].date()} "
            f"or later" if clear < len(index) else "")
        raise ValueError(
            f"cash benchmark does not cover the returns: ^IRX spans "
            f"{irx.index[0].date()} .. {irx.index[-1].date()}, but {int(bad.sum())} of "
            f"{len(index)} dates in the judged 2y windows (first {index[bad][0].date()}) "
            f"have no print in the prior {MAX_RATE_AGE_DAYS} days ({remedy})")
    rf = cash_daily_return(index)
    if not np.isfinite(rf.to_numpy(dtype=float)).all():
        raise ValueError("monitor cash benchmark contains nonfinite returns")
    return rf


def evaluate(returns: pd.Series, since=None) -> list[tuple[str, str, str]]:
    """The three registered rules for one return series.

    Every session on or after `since` is judged (all of them when `since` is
    None): a breach on any judged session sets the status. Each detail
    string leads with the value at the last session; when more than one
    session is judged it also names the first and the worst breach."""
    r = returns
    if (r.empty or not isinstance(r.index, pd.DatetimeIndex)
            or not r.index.is_unique or not r.index.is_monotonic_increasing
            or r.index.hasnans or r.index.tz is not None
            or not r.index.equals(r.index.normalize())
            or not np.isfinite(r.to_numpy(dtype=float)).all()
            or (r < -1).any()):
        raise ValueError("monitor returns must be nonempty, finite and >= -100%, on unique ordered dates")
    n = len(r)
    judged = (np.ones(n, dtype=bool) if since is None
              else np.asarray(r.index >= pd.Timestamp(since)))
    if not judged.any():
        raise ValueError(f"no session on or after {pd.Timestamp(since).date()} to judge")
    eq = (1 + r).cumprod()
    peak = eq.cummax().clip(lower=1.0)

    dd = (eq / peak - 1).to_numpy(dtype=float)
    sharpe = np.full(n, np.nan)
    if n >= WINDOW_2Y:
        # first judged session whose 2y window is complete; the benchmark
        # only has to cover the dates those windows use
        first = max(WINDOW_2Y - 1, int(np.argmax(judged)))
        used = r.iloc[first - WINDOW_2Y + 1:]
        ex = (used - cash_benchmark(used.index)).to_numpy(dtype=float)
        windows = np.lib.stride_tricks.sliding_window_view(ex, WINDOW_2Y)
        mean, std = windows.mean(axis=1), windows.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe[first:] = np.where(
                std > 0, mean / std * np.sqrt(TRADING_DAYS),
                np.where(mean != 0, np.sign(mean) * np.inf, np.nan))
    at_peak = (eq >= peak - 1e-12).to_numpy()
    last_peak = np.maximum.accumulate(np.where(at_peak, np.arange(n), 0))
    underwater_m = np.asarray((r.index - r.index[last_peak]).days, dtype=float) / 30.44

    dd_kill = judged & (dd < KILL_MAX_DRAWDOWN)
    sharpe_kill = judged & (sharpe < KILL_SHARPE_2Y)  # NaN never breaches
    uw_review = judged & (underwater_m >= REVIEW_UNDERWATER_M)
    uw_kill = judged & (underwater_m >= KILL_UNDERWATER_M)

    def status(breach_kill, breach_review=False):
        return "KILL" if breach_kill else ("REVIEW" if breach_review else "ok")

    def history(breach, values, worst, fmt):
        if judged.sum() < 2 or not breach.any():
            return ""
        at = np.flatnonzero(breach)
        w = at[worst(values[at])]
        return (f"; breached on {len(at)} session(s), first {r.index[at[0]].date()}, "
                f"worst {format(values[w], fmt)} on {r.index[w].date()}")

    return [
        ("current drawdown", f"{dd[-1]:.1%} (kill < {KILL_MAX_DRAWDOWN:.0%})"
         + history(dd_kill, dd, np.argmin, ".1%"),
         status(dd_kill.any())),
        ("rolling 2y excess Sharpe", (f"{sharpe[-1]:.2f} (kill < {KILL_SHARPE_2Y})"
         if n >= WINDOW_2Y else f"pending: {n}/{WINDOW_2Y} observations")
         + history(sharpe_kill, sharpe, np.argmin, ".2f"),
         status(sharpe_kill.any())),
        ("months underwater", f"{underwater_m[-1]:.1f} (review >= {REVIEW_UNDERWATER_M}, "
         f"kill >= {KILL_UNDERWATER_M})" + history(uw_review, underwater_m, np.argmax, ".1f"),
         status(uw_kill.any(), uw_review.any())),
    ]


def read_returns(src: Path) -> pd.DataFrame:
    """The returns CSV with its first column parsed as strict YYYY-MM-DD
    dates (no format inference, so no silent month/day swap)."""
    with src.open(newline="", encoding="utf-8-sig") as stream:
        header = [name.strip() for name in next(csv.reader(stream), [])]
    if not header or len(header) != len(set(header)):
        raise ValueError("return CSV must have a nonempty unique header")
    df = pd.read_csv(src, index_col=0, converters={0: str})
    try:
        df.index = pd.to_datetime(df.index.str.strip(), format="%Y-%m-%d")
    except ValueError as exc:
        reason = str(exc).splitlines()[0].rstrip(".")
        raise ValueError(f"dates must be written YYYY-MM-DD ({reason})") from None
    return df


def _day(text: str, flag: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(text, format="%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{flag} must be a YYYY-MM-DD date, got {text!r}") from None


def _invalid(message) -> int:
    print(f"invalid monitor input: {message}")
    return EXIT_INVALID_INPUT


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # argparse's own status 2 would read as KILL
        self.print_usage(sys.stderr)
        print(f"invalid monitor input: {message}")
        sys.exit(EXIT_INVALID_INPUT)


def _run() -> int:
    ap = _Parser(allow_abbrev=False, description="Pre-registered decay/kill monitor for the deployed ensemble.")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("returns", nargs="?", type=Path, default=None,
                        help="live daily-returns CSV (Date YYYY-MM-DD, ENSEMBLE column)")
    source.add_argument("--backtest", action="store_true",
                        help="calibration run on the frozen backtest stream; not an account check")
    ap.add_argument("--since", metavar="YYYY-MM-DD", default=None,
                    help="judge the sessions on or after this date (default: the whole live "
                         "file; the last session of the backtest)")
    ap.add_argument("--as-of", metavar="YYYY-MM-DD", default=None,
                    help="offline replay: judge a live file's freshness against this date "
                         "instead of today (New York)")
    ap.add_argument("--allow-gaps", action="store_true",
                    help="accept verified missing NYSE sessions (rows on closed days are "
                         "always rejected)")
    args = ap.parse_args()
    if args.backtest and args.as_of is not None:
        ap.error("--as-of applies to a live file, not to --backtest")
    src = BACKTEST_RETURNS if args.backtest else args.returns
    if args.backtest and not src.is_file():
        return _invalid("results/sleeve_returns.csv not found: --backtest judges the blended "
                        "backtest stream, which the ensemble build saves. Create it first: "
                        "python src/ensemble.py")
    try:
        today = current_ny_date() if args.as_of is None else _day(args.as_of, "--as-of")
        since = None if args.since is None else _day(args.since, "--since")
        df = read_returns(src)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return _invalid(exc)
    if df.empty or df.columns.empty or not df.columns.is_unique:
        return _invalid("need nonempty unique return columns")
    if (not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans
            or not df.index.is_unique or not df.index.is_monotonic_increasing
            or df.index.tz is not None or not df.index.equals(df.index.normalize())):
        return _invalid("need valid, unique, ordered session dates without time-of-day")
    last = df.index[-1]
    print(f"monitor: {src.name}, {df.index[0].date()} .. {last.date()}")
    if args.backtest:
        print("mode: BACKTEST calibration - the frozen research stream, NOT the live account")
    else:
        if last > today:
            return _invalid(f"last row {last.date()} is dated after {today.date()} (New York)")
        lag = sessions_behind(last, today)
        if lag > MAX_STALE_SESSIONS:
            return _invalid(f"stale live file: last row {last.date()} is {lag} completed NYSE "
                            f"sessions before {today.date()} (limit {MAX_STALE_SESSIONS}); "
                            f"bring it up to date")
        when = (f"{today.date()} New York" if args.as_of is None
                else f"--as-of {today.date()}, an offline replay and not today's date")
        print(f"mode: LIVE account, checked against {when} "
              f"({lag} completed session(s) not yet recorded)")
    off_session, missing = session_mismatch(df.index)
    if len(off_session):
        return _invalid(f"{len(off_session)} row(s) on non-NYSE-session dates: "
                        f"{_show_dates(off_session)}")
    if len(missing) and not args.allow_gaps:
        return _invalid(f"{len(missing)} NYSE session(s) missing between the first and last "
                        f"row: {_show_dates(missing)}; pass --allow-gaps only for a verified gap")
    if len(missing):
        print(f" ? warning: {len(missing)} NYSE session(s) missing ({_show_dates(missing)}); "
              f"their P&L is absent from every rule below")
    if since is None:
        since = last if args.backtest else df.index[0]
    if since > last:
        return _invalid(f"--since {since.date()} is after the last row {last.date()}")
    in_window = df.index[df.index >= since]
    print(f"judged: {len(in_window)} of {len(df)} session(s), "
          f"{in_window[0].date()} .. {last.date()}")
    try:
        print("cash benchmark: " + (benchmark_span() if len(df) >= WINDOW_2Y
                                    else "not used yet (2y Sharpe pending)") + "\n")
    except (OSError, ValueError) as exc:
        return _invalid(f"cash benchmark unreadable: {exc}")
    worst = at_last = "ok"
    rank = ["ok", "REVIEW", "KILL"].index
    judged = ["ENSEMBLE"] if "ENSEMBLE" in df.columns else list(df.columns)
    for col in df.columns:
        try:
            rows = evaluate(df[col], since)
            last_rows = evaluate(df[col], last)  # the same rules at the final row alone
        except (OSError, ValueError) as exc:
            print(f" ! {col}: invalid monitor input: {exc}")
            return EXIT_INVALID_INPUT
        badge = {"ok": " ", "REVIEW": "?", "KILL": "!"}
        for (name, detail, st), (_, _, st_last) in zip(rows, last_rows):
            if col in judged:
                worst = max(worst, st, key=rank)
                at_last = max(at_last, st_last, key=rank)
            if col == "ENSEMBLE" or st != "ok":
                notes = ([f"last session {st_last}"] if st_last != st else []) \
                    + ([] if col in judged else ["info"])
                tag = st + (f" ({', '.join(notes)})" if notes else "")
                print(f" {badge[st]} {col:18s} {name:26s} {detail:38s} {tag}")
    # Two readings, so a breach that has since reverted is not mistaken for
    # the current state (or the reverse). The verdict is the window reading.
    print(f"\nat the last session ({last.date()}): {at_last}")
    print(f"on any judged session ({in_window[0].date()} .. {last.date()}): {worst}"
          f"  <- verdict and exit status")
    print(f"overall: {worst}")
    return {"ok": EXIT_OK, "REVIEW": EXIT_REVIEW, "KILL": EXIT_KILL}[worst]


def main() -> None:
    try:
        code = _run()
    except Exception as exc:  # a crash must not exit 1, which means REVIEW
        traceback.print_exc()
        print(f"monitor not evaluated, no verdict: {type(exc).__name__}: {exc}")
        code = EXIT_CRASH
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
